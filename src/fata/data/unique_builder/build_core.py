"""Shared unique-image construction engine.

The engine enforces the selection invariant in this order for every source
representative: load and canonically hash the source image, apply ID/hash
deduplication, run the black-image blind control, deterministically encode a
staging JPEG, reopen that exact JPEG, and only then run Full inference.  The
mapping is committed only if the reopened final bytes remain Full-correct.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from PIL import Image

from .common import (
    CANONICAL_HASH_DESCRIPTION,
    DATASET_NAMES,
    JPEG_SETTINGS,
    SEED,
    atomic_write_json,
    atomic_write_jsonl,
    append_jsonl_durable,
    canonical_rgb_sha256,
    canonical_rgb_sha256_path,
    dhash_path,
    is_correct_mc,
    is_correct_open,
    read_jsonl_tolerant,
    save_deterministic_jpeg,
    sha256_file,
    model_directory_fingerprint,
)
from .sources import Candidate, RecoverableSourceError, SourceBundle
from fata.utils.paths import resolve_dataset_relative_path


CHECKPOINT_FORMAT_VERSION = 2
CONSTRUCTION_PROTOCOL_VERSION = "fata-unique-image-selection-v4"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_TRANSIENT_SOURCE_METADATA_KEYS = frozenset(
    {"cache_status", "attempt", "cached_source"}
)
_SCREENING_BASE_FIELDS = frozenset(
    {
        "candidate_position",
        "source_dataset",
        "source_split",
        "source_index",
        "source_split_index",
        "source_question_id",
        "source_image_id",
        "reason",
        "accepted",
    }
)
_SCREENING_SOURCE_HASH_FIELDS = frozenset({"source_canonical_rgb_sha256"})
_SCREENING_BLIND_FIELDS = frozenset(
    {"blind_prediction", "blind_correct", "control"}
)
_SCREENING_SAVED_FIELDS = frozenset(
    {"saved_image_sha256", "saved_canonical_rgb_sha256", "perceptual_hash"}
)
_SCREENING_FULL_FIELDS = frozenset({"full_prediction", "full_correct"})
_SCREENING_ACCEPTED_FIELDS = frozenset({"retained_index", "image_filename"})
_SCREENING_FIELDS_BY_REASON = {
    "duplicate_source_image_id": _SCREENING_BASE_FIELDS,
    "duplicate_source_canonical_rgb_sha256": (
        _SCREENING_BASE_FIELDS | _SCREENING_SOURCE_HASH_FIELDS
    ),
    "excluded_prior_dataset_source_image_id": (
        _SCREENING_BASE_FIELDS | _SCREENING_SOURCE_HASH_FIELDS
    ),
    "excluded_prior_dataset_source_hash": (
        _SCREENING_BASE_FIELDS | _SCREENING_SOURCE_HASH_FIELDS
    ),
    "blind_black_image_control_correct": (
        _SCREENING_BASE_FIELDS
        | _SCREENING_SOURCE_HASH_FIELDS
        | _SCREENING_BLIND_FIELDS
    ),
    "duplicate_saved_canonical_rgb_sha256": (
        _SCREENING_BASE_FIELDS
        | _SCREENING_SOURCE_HASH_FIELDS
        | _SCREENING_BLIND_FIELDS
        | _SCREENING_SAVED_FIELDS
    ),
    "excluded_prior_dataset_saved_hash": (
        _SCREENING_BASE_FIELDS
        | _SCREENING_SOURCE_HASH_FIELDS
        | _SCREENING_BLIND_FIELDS
        | _SCREENING_SAVED_FIELDS
    ),
    "full_visual_incorrect": (
        _SCREENING_BASE_FIELDS
        | _SCREENING_SOURCE_HASH_FIELDS
        | _SCREENING_BLIND_FIELDS
        | _SCREENING_SAVED_FIELDS
        | _SCREENING_FULL_FIELDS
    ),
    "accepted": (
        _SCREENING_BASE_FIELDS
        | _SCREENING_SOURCE_HASH_FIELDS
        | _SCREENING_BLIND_FIELDS
        | _SCREENING_SAVED_FIELDS
        | _SCREENING_FULL_FIELDS
        | _SCREENING_ACCEPTED_FIELDS
    ),
}


@dataclass(frozen=True)
class BuildExclusions:
    """Frozen identities accepted by earlier datasets in a build-all run."""

    source_image_ids: frozenset[str] = frozenset()
    source_canonical_rgb_sha256: frozenset[str] = frozenset()
    saved_canonical_rgb_sha256: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        fields = {
            "source_image_ids": self.source_image_ids,
            "source_canonical_rgb_sha256": self.source_canonical_rgb_sha256,
            "saved_canonical_rgb_sha256": self.saved_canonical_rgb_sha256,
        }
        for name, values in fields.items():
            if not isinstance(values, frozenset):
                raise TypeError(f"BuildExclusions.{name} must be a frozenset")
            if any(not isinstance(value, str) or not value for value in values):
                raise TypeError(
                    f"BuildExclusions.{name} must contain non-empty strings"
                )
        for name in (
            "source_canonical_rgb_sha256",
            "saved_canonical_rgb_sha256",
        ):
            if any(_SHA256_RE.fullmatch(value) is None for value in fields[name]):
                raise ValueError(
                    f"BuildExclusions.{name} must contain lowercase SHA-256 digests"
                )

    @classmethod
    def from_iterables(
        cls,
        source_image_ids: Iterable[Any] = (),
        source_hashes: Iterable[str] = (),
        saved_hashes: Iterable[str] = (),
    ) -> "BuildExclusions":
        raw_values = {
            "source_image_ids": source_image_ids,
            "source_hashes": source_hashes,
            "saved_hashes": saved_hashes,
        }
        for name, values in raw_values.items():
            if isinstance(values, (str, bytes)):
                raise TypeError(f"{name} must be a collection, not a string")
        return cls(
            frozenset(str(value) for value in source_image_ids if value is not None),
            frozenset(str(value) for value in source_hashes),
            frozenset(str(value) for value in saved_hashes),
        )


@dataclass(frozen=True)
class BuildConfig:
    output_root: Path
    target: int = 1000
    seed: int = SEED
    resume: bool = False
    exclusions: BuildExclusions = field(default_factory=BuildExclusions)

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_root", Path(self.output_root).resolve())
        if type(self.target) is not int or self.target <= 0:
            raise ValueError("target must be positive")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if type(self.resume) is not bool:
            raise TypeError("resume must be a boolean")
        if not isinstance(self.exclusions, BuildExclusions):
            raise TypeError("exclusions must be a BuildExclusions instance")


@dataclass(frozen=True)
class BuildResult:
    dataset_name: str
    output_root: Path
    dataset_dir: Path
    mapping_path: Path
    sidecar_path: Path
    screening_path: Path
    checkpoint_path: Path
    stats_path: Path
    stats: dict[str, Any]
    samples: tuple[dict[str, Any], ...]
    target_reached: bool

    @property
    def accepted_source_image_ids(self) -> frozenset[str]:
        return frozenset(
            str(sample["source_image_id"])
            for sample in self.samples
            if sample.get("source_image_id") is not None
        )

    @property
    def accepted_source_hashes(self) -> frozenset[str]:
        return frozenset(
            str(sample["source_canonical_rgb_sha256"]) for sample in self.samples
        )

    @property
    def accepted_saved_hashes(self) -> frozenset[str]:
        return frozenset(
            str(sample["saved_canonical_rgb_sha256"]) for sample in self.samples
        )


class InsufficientEligibleSamples(RuntimeError):
    """The complete prepared source pool did not satisfy the requested target."""

    def __init__(self, result: BuildResult) -> None:
        self.result = result
        super().__init__(
            f"{result.dataset_name}: complete candidate pool retained "
            f"{result.stats['retained_count']} of target {result.stats['target']}"
        )


def resolve_output_root(
    requested: Path | str,
    *,
    resume: bool,
    dataset_name: str | None = None,
) -> Path:
    """Choose a non-overwriting output root, or the exact root for resume.

    A single-dataset invocation may safely join a root already populated by the
    other datasets, but never replaces artifacts for its own dataset.  A
    build-all invocation (``dataset_name=None``) treats any non-empty existing
    root as a prior run and creates a UTC-timestamped sibling.
    """

    requested = Path(requested).resolve()
    if resume:
        if not requested.is_dir():
            raise FileNotFoundError(f"resume output root does not exist: {requested}")
        return requested
    if not requested.exists():
        requested.mkdir(parents=True)
        return requested
    if not requested.is_dir():
        raise NotADirectoryError(requested)
    if dataset_name is not None:
        own_artifacts = (
            requested / dataset_name,
            requested / "manifests" / f"{dataset_name}_checkpoint.json",
            requested / "manifests" / f"{dataset_name}_samples.jsonl",
        )
        if not any(path.exists() for path in own_artifacts):
            return requested
    else:
        # Reports, source_cache, and build-environment captures are commonly
        # prepared before construction.  They do not constitute an existing
        # dataset build.  Any dataset directory or checkpoint does.
        has_build_artifacts = any((requested / name).exists() for name in DATASET_NAMES)
        manifests = requested / "manifests"
        if manifests.is_dir() and any(manifests.glob("*_checkpoint.json")):
            has_build_artifacts = True
        if not has_build_artifacts:
            return requested
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = requested.with_name(f"{requested.name}_{timestamp}")
    selected = base
    suffix = 1
    while selected.exists():
        selected = base.with_name(f"{base.name}_{suffix:02d}")
        suffix += 1
    selected.mkdir(parents=True)
    return selected


def _stable_json_digest(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _stable_source_metadata(value: Any, *, _parent_key: str | None = None) -> Any:
    """Normalize only known VQA archive materialization details.

    Key names such as ``attempt`` can be meaningful provenance elsewhere, so the
    exemption is deliberately limited to the two records emitted by
    :func:`prepare_vqav2_source` and only when their stable artifact identity is
    structurally present.
    """

    if isinstance(value, dict):
        is_vqa_archive_record = (
            _parent_key in {"question_zip", "annotation_zip"}
            and {"path", "url", "size", "sha256"}.issubset(value)
        )
        return {
            key: _stable_source_metadata(child, _parent_key=key)
            for key, child in value.items()
            if not (
                is_vqa_archive_record
                and key in _TRANSIENT_SOURCE_METADATA_KEYS
            )
        }
    if isinstance(value, list):
        return [
            _stable_source_metadata(child, _parent_key=_parent_key)
            for child in value
        ]
    return copy.deepcopy(value)


def _candidate_pool_fingerprint(bundle: SourceBundle) -> str:
    # Image bytes are verified independently as candidates are scanned.  This
    # fingerprint locks ordering, representatives, prompts, labels, and options.
    rows = [
        {
            "source_dataset": candidate.source_dataset,
            "source_split": candidate.source_split,
            "source_index": candidate.source_index,
            "source_split_index": candidate.source_split_index,
            "source_question_id": candidate.source_question_id,
            "source_image_id": candidate.source_image_id,
            "source_canonical_rgb_sha256": candidate.source_canonical_rgb_sha256,
            "task_type": candidate.task_type,
            "question": candidate.question,
            "answers": list(candidate.answers),
            "options": list(candidate.options),
            "ground_truth_letter": candidate.ground_truth_letter,
            "ground_truth_text": candidate.ground_truth_text,
            "prompt": candidate.prompt,
            "mapping_payload": candidate.mapping_payload,
            "legacy_temperature_zero": candidate.legacy_temperature_zero,
            "provenance": candidate.provenance,
        }
        for candidate in bundle.candidates
    ]
    return _stable_json_digest(rows)


def _construction_protocol_manifest(runner: Any) -> dict[str, Any]:
    """Freeze every selection-relevant protocol input available from the runner."""

    runner_manifest_provider = getattr(runner, "construction_manifest", None)
    if callable(runner_manifest_provider):
        runner_manifest = runner_manifest_provider()
    else:
        # Test/custom runners remain supported, but are still locked to an
        # explicit class identity plus the protocol/scorer version.
        runner_manifest = {
            "runner": (
                f"{runner.__class__.__module__}.{runner.__class__.__qualname__}"
            )
        }
    if not isinstance(runner_manifest, dict):
        raise TypeError("runner construction_manifest() must return a dictionary")
    return {
        "protocol_version": CONSTRUCTION_PROTOCOL_VERSION,
        "scorers": {
            "open": "legacy-official-vqa-process-plus-truth-substring-v1",
            "multiple_choice": (
                "legacy-letter-regex-then-ground-truth-substring-v1;"
                "VQAv2=A-D;ScienceQA=A-F"
            ),
        },
        "canonical_hash": CANONICAL_HASH_DESCRIPTION,
        "jpeg_settings": JPEG_SETTINGS,
        "runner": runner_manifest,
    }


def _exclusions_fingerprint(exclusions: BuildExclusions) -> str:
    return _stable_json_digest(
        {
            "source_image_ids": sorted(exclusions.source_image_ids),
            "source_canonical_rgb_sha256": sorted(
                exclusions.source_canonical_rgb_sha256
            ),
            "saved_canonical_rgb_sha256": sorted(
                exclusions.saved_canonical_rgb_sha256
            ),
        }
    )


def _correct(candidate: Candidate, prediction: str) -> bool:
    if candidate.task_type == "open":
        return is_correct_open(prediction, candidate.answers)
    if candidate.task_type == "multiple_choice":
        if (
            candidate.ground_truth_letter is None
            or candidate.ground_truth_text is None
            or not candidate.options
        ):
            raise RuntimeError("MC candidate is missing options or ground truth")
        # Preserve the original task-specific parser bounds exactly.  The
        # legacy ScienceQA script searched A-F even when a row exposed fewer
        # choices; VQAv2 searched A-D.
        mapping_dataset = candidate.mapping_payload.get("dataset")
        if mapping_dataset == "ScienceQA_MC":
            max_letter = "F"
        elif mapping_dataset == "VQAv2_MC":
            max_letter = "D"
        else:
            raise RuntimeError(
                f"unknown multiple-choice dataset {mapping_dataset!r}"
            )
        return is_correct_mc(
            prediction,
            candidate.ground_truth_letter,
            candidate.ground_truth_text,
            max_letter,
        )
    raise RuntimeError(f"unknown candidate task type {candidate.task_type!r}")


def _paths(config: BuildConfig, dataset_name: str) -> dict[str, Path]:
    root = config.output_root
    if not isinstance(dataset_name, str) or dataset_name not in DATASET_NAMES:
        raise ValueError(
            f"dataset_name must be one of the formal datasets: {DATASET_NAMES!r}"
        )
    dataset_dir = root / dataset_name
    manifests_dir = root / "manifests"
    paths = {
        "dataset_dir": dataset_dir,
        "images_dir": dataset_dir / "images",
        "stage_dir": dataset_dir / ".stage",
        "mapping": dataset_dir / f"{dataset_name}_mapping.jsonl",
        "sidecar": manifests_dir / f"{dataset_name}_samples.jsonl",
        "checkpoint": manifests_dir / f"{dataset_name}_checkpoint.json",
        "source": manifests_dir / f"{dataset_name}_source.json",
        "candidate_pool": manifests_dir / f"{dataset_name}_candidate_pool.jsonl",
        "screening": manifests_dir / f"{dataset_name}_screening.jsonl",
        "stats": manifests_dir / f"{dataset_name}_selection_stats.json",
    }
    directory_keys = {"dataset_dir", "images_dir", "stage_dir"}
    for key, path in paths.items():
        try:
            relative = path.relative_to(root)
        except ValueError as exc:
            raise ValueError("builder artifact path escapes output_root") from exc
        current = root
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                raise ValueError(
                    f"builder artifact path contains a symlink: {current}"
                )
        resolved = path.resolve()
        if not resolved.is_relative_to(root):
            raise ValueError("resolved builder artifact path escapes output_root")
        paths[key] = resolved
        if key in directory_keys and path.exists() and not path.is_dir():
            raise ValueError(f"builder directory path is not a directory: {path}")
        if key not in directory_keys and path.exists() and not path.is_file():
            raise ValueError(f"builder file path is not a regular file: {path}")
    if manifests_dir.exists() and not manifests_dir.is_dir():
        raise ValueError(f"builder manifests path is not a directory: {manifests_dir}")
    return paths


def _initial_stats(bundle: SourceBundle, config: BuildConfig) -> dict[str, Any]:
    stats = copy.deepcopy(bundle.selection_stats)
    required = (
        "source_splits",
        "source_qa_records_total",
        "source_unique_images_total",
        "task_prefilter_unique_images",
        "invalid_metadata_count",
        "yes_no_removed_count",
        "missing_or_corrupt_images",
        "source_id_duplicate_count",
        "exact_image_hash_duplicate_count",
        "unique_images_evaluated",
        "blind_wrong_count",
        "full_correct_count",
        "target_reached_source_index",
        "retained_count",
        "acceptance_rate",
    )
    missing = [key for key in required if key not in stats]
    if missing:
        raise RuntimeError(f"source bundle selection_stats missing {missing!r}")
    stats.update(
        {
            "dataset": bundle.dataset_name,
            "target": config.target,
            "seed": config.seed,
            "control": "blind_black_image_control",
            "candidate_representatives_total": len(bundle.candidates),
            "scanned_candidates": int(stats.get("scanned_candidates", 0)),
            "source_hash_candidates_scanned": 0,
            "excluded_source_image_id_count": 0,
            "excluded_source_hash_count": 0,
            "excluded_saved_hash_count": 0,
            "engine_source_id_duplicate_count": 0,
            "engine_exact_source_hash_duplicate_count": 0,
            "saved_exact_hash_duplicate_count": 0,
            "blind_correct_count": 0,
            "full_incorrect_count": 0,
            "target_reached_candidate_position": None,
            "candidate_pool_exhausted": False,
        }
    )
    return stats


def _checkpoint_payload(
    bundle: SourceBundle,
    config: BuildConfig,
    pool_fingerprint: str,
    exclusions_fingerprint: str,
    construction_protocol_fingerprint: str,
    next_position: int,
    stats: dict[str, Any],
    samples: list[dict[str, Any]],
    seen_source_ids: set[str],
    seen_source_hashes: set[str],
    seen_saved_hashes: set[str],
) -> dict[str, Any]:
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "dataset": bundle.dataset_name,
        "target": config.target,
        "seed": config.seed,
        "candidate_pool_fingerprint": pool_fingerprint,
        "exclusions_fingerprint": exclusions_fingerprint,
        "construction_protocol_fingerprint": construction_protocol_fingerprint,
        "next_candidate_position": next_position,
        "selection_stats": stats,
        "samples": samples,
        "seen_source_image_ids": sorted(seen_source_ids),
        "seen_source_canonical_rgb_sha256": sorted(seen_source_hashes),
        "seen_saved_canonical_rgb_sha256": sorted(seen_saved_hashes),
    }


def _write_public_state(
    paths: dict[str, Path],
    stats: dict[str, Any],
    samples: list[dict[str, Any]],
) -> None:
    atomic_write_jsonl(paths["mapping"], [sample["mapping"] for sample in samples])
    atomic_write_jsonl(paths["sidecar"], samples)
    atomic_write_json(paths["stats"], stats)


def _screening_base(position: int, candidate: Candidate) -> dict[str, Any]:
    return {
        "candidate_position": position,
        "source_dataset": candidate.source_dataset,
        "source_split": candidate.source_split,
        "source_index": candidate.source_index,
        "source_split_index": candidate.source_split_index,
        "source_question_id": candidate.source_question_id,
        "source_image_id": candidate.source_image_id,
        "reason": None,
        "accepted": False,
    }


def _candidate_pool_rows(bundle: SourceBundle) -> list[dict[str, Any]]:
    """Serialize every pre-model representative, independent of predictions."""

    return [
        {
            "candidate_position": position,
            "source_dataset": candidate.source_dataset,
            "source_split": candidate.source_split,
            "source_index": candidate.source_index,
            "source_split_index": candidate.source_split_index,
            "source_question_id": candidate.source_question_id,
            "source_image_id": candidate.source_image_id,
            "source_canonical_rgb_sha256": candidate.source_canonical_rgb_sha256,
            "task_type": candidate.task_type,
            "question": candidate.question,
            "answers": list(candidate.answers),
            "options": list(candidate.options),
            "ground_truth_letter": candidate.ground_truth_letter,
            "ground_truth_text": candidate.ground_truth_text,
            "prompt": candidate.prompt,
            "mapping_template": candidate.mapping_payload,
            "legacy_temperature_zero": candidate.legacy_temperature_zero,
            "provenance": candidate.provenance,
        }
        for position, candidate in enumerate(bundle.candidates)
    ]


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise RuntimeError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _checkpoint_string_set(
    checkpoint: dict[str, Any], key: str, *, sha256: bool = False
) -> set[str]:
    raw = checkpoint.get(key)
    if not isinstance(raw, list):
        raise RuntimeError(f"checkpoint {key} must be a list")
    values: list[str] = []
    for index, value in enumerate(raw):
        if not isinstance(value, str) or not value:
            raise RuntimeError(
                f"checkpoint {key}[{index}] must be a non-empty string"
            )
        if sha256:
            _require_sha256(value, label=f"checkpoint {key}[{index}]")
        values.append(value)
    if len(set(values)) != len(values):
        raise RuntimeError(f"checkpoint {key} contains duplicate entries")
    if values != sorted(values):
        raise RuntimeError(f"checkpoint {key} is not in canonical sorted order")
    return set(values)


def _require_exact_checkpoint_set(
    *, key: str, actual: set[str], expected: set[str]
) -> None:
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise RuntimeError(
            f"checkpoint {key} does not match committed state: "
            f"missing={missing[:10]!r}, extra={extra[:10]!r}"
        )


def _canonical_json(value: Any, *, label: str) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{label} is not strict JSON data") from exc


def _read_jsonl_strict(path: Path, *, label: str) -> list[Any]:
    """Read an atomically published JSONL artifact without tail tolerance."""

    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise RuntimeError(f"cannot read strict {label}: {path}") from exc
    if raw and not raw.endswith(b"\n"):
        raise RuntimeError(f"strict {label} is missing its final newline: {path}")
    records: list[Any] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise RuntimeError(
                f"strict {label} contains a blank line at {line_number}: {path}"
            )
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"strict {label} contains invalid JSON at line {line_number}: {path}"
            ) from exc
    return records


def _read_json_object_strict(path: Path, *, label: str) -> dict[str, Any]:
    """Read one strict JSON object from a builder-owned artifact."""

    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read strict {label}: {path}") from exc
    if not raw.endswith(b"\n"):
        raise RuntimeError(f"strict {label} is missing its final newline: {path}")
    if not isinstance(value, dict):
        raise RuntimeError(f"strict {label} must contain one JSON object: {path}")
    _canonical_json(value, label=label)
    return value


def _require_exact_json(*, label: str, actual: Any, expected: Any) -> None:
    if _canonical_json(actual, label=label) != _canonical_json(
        expected, label=f"expected {label}"
    ):
        raise RuntimeError(f"{label} does not match the reconstructed committed state")


def _validated_completed_sibling_samples(
    root: Path,
    dataset_name: str,
    *,
    target: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Validate a completed sibling build before trusting its identities."""

    paths = _paths(BuildConfig(root, target=target, seed=seed), dataset_name)
    indicators = (
        paths["dataset_dir"],
        paths["mapping"],
        paths["sidecar"],
        paths["checkpoint"],
        paths["source"],
        paths["candidate_pool"],
        paths["screening"],
        paths["stats"],
    )
    if not any(path.exists() for path in indicators):
        return []
    missing = [str(path) for path in indicators if not path.exists()]
    if missing or not paths["images_dir"].is_dir():
        raise RuntimeError(
            f"shared output root contains an incomplete {dataset_name} build; "
            f"missing required artifacts: {missing!r}"
        )

    checkpoint = _read_json_object_strict(
        paths["checkpoint"], label=f"{dataset_name} sibling checkpoint"
    )
    stats = _read_json_object_strict(
        paths["stats"], label=f"{dataset_name} sibling selection stats"
    )
    source = _read_json_object_strict(
        paths["source"], label=f"{dataset_name} sibling source manifest"
    )
    samples = _read_jsonl_strict(
        paths["sidecar"], label=f"{dataset_name} sibling sample sidecar"
    )
    mappings = _read_jsonl_strict(
        paths["mapping"], label=f"{dataset_name} sibling mapping"
    )
    candidates = _read_jsonl_strict(
        paths["candidate_pool"], label=f"{dataset_name} sibling candidate pool"
    )
    screening = _read_jsonl_strict(
        paths["screening"], label=f"{dataset_name} sibling screening"
    )

    for key, expected in (
        ("format_version", CHECKPOINT_FORMAT_VERSION),
        ("dataset", dataset_name),
        ("target", target),
        ("seed", seed),
    ):
        if checkpoint.get(key) != expected:
            raise RuntimeError(
                f"shared output root has incompatible {dataset_name} checkpoint {key}"
            )
    _require_exact_json(
        label=f"{dataset_name} sibling checkpoint/public samples",
        actual=checkpoint.get("samples"),
        expected=samples,
    )
    _require_exact_json(
        label=f"{dataset_name} sibling checkpoint/public selection stats",
        actual=checkpoint.get("selection_stats"),
        expected=stats,
    )
    if (
        len(samples) != target
        or len(mappings) != target
        or stats.get("retained_count") != target
        or stats.get("dataset") != dataset_name
        or stats.get("target") != target
        or stats.get("seed") != seed
    ):
        raise RuntimeError(
            f"shared output root {dataset_name} checkpoint is not complete/compatible"
        )
    next_position = checkpoint.get("next_candidate_position")
    if (
        type(next_position) is not int
        or not 0 < next_position <= len(candidates)
        or len(screening) != next_position
        or stats.get("scanned_candidates") != next_position
    ):
        raise RuntimeError(f"{dataset_name} sibling checkpoint position is invalid")

    source_pool_fingerprint = source.get("candidate_pool_fingerprint")
    if (
        source.get("dataset") != dataset_name
        or not isinstance(source_pool_fingerprint, str)
        or _SHA256_RE.fullmatch(source_pool_fingerprint) is None
        or source_pool_fingerprint != checkpoint.get("candidate_pool_fingerprint")
    ):
        raise RuntimeError(f"{dataset_name} sibling candidate-pool fingerprint mismatch")
    protocol = source.get("construction_protocol")
    if (
        not isinstance(protocol, dict)
        or protocol.get("protocol_version") != CONSTRUCTION_PROTOCOL_VERSION
    ):
        raise RuntimeError(f"{dataset_name} sibling source protocol is invalid")
    protocol_fingerprint = _stable_json_digest(protocol)
    if (
        source.get("construction_protocol_fingerprint") != protocol_fingerprint
        or checkpoint.get("construction_protocol_fingerprint")
        != protocol_fingerprint
    ):
        raise RuntimeError(f"{dataset_name} sibling construction protocol mismatch")
    initial_stats = source.get("initial_selection_stats")
    if not isinstance(initial_stats, dict) or source.get(
        "initial_selection_stats_fingerprint"
    ) != _stable_json_digest(initial_stats):
        raise RuntimeError(f"{dataset_name} sibling initial stats fingerprint mismatch")
    source_metadata = source.get("source_metadata")
    if not isinstance(source_metadata, dict) or source.get(
        "source_metadata_stable_fingerprint"
    ) != _stable_json_digest(_stable_source_metadata(source_metadata)):
        raise RuntimeError(f"{dataset_name} sibling source metadata fingerprint mismatch")
    exclusions = source.get("build_exclusions")
    exclusion_fields = {
        "source_image_ids",
        "source_canonical_rgb_sha256",
        "saved_canonical_rgb_sha256",
    }
    if not isinstance(exclusions, dict) or set(exclusions) != exclusion_fields:
        raise RuntimeError(f"{dataset_name} sibling build exclusions are invalid")
    for key, values in exclusions.items():
        if (
            not isinstance(values, list)
            or any(not isinstance(value, str) or not value for value in values)
            or values != sorted(set(values))
        ):
            raise RuntimeError(
                f"{dataset_name} sibling build exclusion {key} is not canonical"
            )
    exclusions_fingerprint = _stable_json_digest(exclusions)
    if (
        source.get("exclusions_fingerprint") != exclusions_fingerprint
        or checkpoint.get("exclusions_fingerprint") != exclusions_fingerprint
    ):
        raise RuntimeError(f"{dataset_name} sibling exclusions fingerprint mismatch")

    candidate_fields = {
        "candidate_position",
        "source_dataset",
        "source_split",
        "source_index",
        "source_split_index",
        "source_question_id",
        "source_image_id",
        "source_canonical_rgb_sha256",
        "task_type",
        "question",
        "answers",
        "options",
        "ground_truth_letter",
        "ground_truth_text",
        "prompt",
        "mapping_template",
        "legacy_temperature_zero",
        "provenance",
    }
    fingerprint_rows: list[dict[str, Any]] = []
    reconstructed_candidates: list[Candidate] = []
    for position, candidate in enumerate(candidates):
        if (
            not isinstance(candidate, dict)
            or set(candidate) != candidate_fields
            or candidate.get("candidate_position") != position
        ):
            raise RuntimeError(
                f"{dataset_name} sibling candidate pool schema/ordering is invalid"
            )
        fingerprint_rows.append(
            {
                key: candidate["mapping_template"] if key == "mapping_payload" else candidate[key]
                for key in (
                    "source_dataset",
                    "source_split",
                    "source_index",
                    "source_split_index",
                    "source_question_id",
                    "source_image_id",
                    "source_canonical_rgb_sha256",
                    "task_type",
                    "question",
                    "answers",
                    "options",
                    "ground_truth_letter",
                    "ground_truth_text",
                    "prompt",
                    "mapping_payload",
                    "legacy_temperature_zero",
                    "provenance",
                )
            }
        )
        try:
            reconstructed_candidates.append(
                Candidate(
                    source_dataset=candidate["source_dataset"],
                    source_split=candidate["source_split"],
                    source_index=candidate["source_index"],
                    source_split_index=candidate["source_split_index"],
                    source_question_id=candidate["source_question_id"],
                    source_image_id=candidate["source_image_id"],
                    question=candidate["question"],
                    answers=tuple(candidate["answers"]),
                    task_type=candidate["task_type"],
                    prompt=candidate["prompt"],
                    mapping_payload=candidate["mapping_template"],
                    image_loader=lambda: None,
                    source_canonical_rgb_sha256=candidate[
                        "source_canonical_rgb_sha256"
                    ],
                    options=tuple(candidate["options"]),
                    ground_truth_letter=candidate["ground_truth_letter"],
                    ground_truth_text=candidate["ground_truth_text"],
                    legacy_temperature_zero=candidate["legacy_temperature_zero"],
                    provenance=candidate["provenance"],
                )
            )
        except (KeyError, TypeError) as exc:
            raise RuntimeError(
                f"{dataset_name} sibling candidate values are invalid"
            ) from exc
    if _stable_json_digest(fingerprint_rows) != source_pool_fingerprint:
        raise RuntimeError(f"{dataset_name} sibling candidate-pool body mismatch")

    accepted_rows: dict[int, tuple[int, dict[str, Any]]] = {}
    committed: dict[int, dict[str, Any]] = {}
    lineage_keys = (
        "source_dataset",
        "source_split",
        "source_index",
        "source_split_index",
        "source_question_id",
        "source_image_id",
    )
    for position, row in enumerate(screening):
        if not isinstance(row, dict) or row.get("candidate_position") != position:
            raise RuntimeError(f"{dataset_name} sibling screening is not contiguous")
        committed[position] = row
        candidate = candidates[position]
        if any(row.get(key) != candidate.get(key) for key in lineage_keys):
            raise RuntimeError(
                f"{dataset_name} sibling screening/candidate lineage mismatch"
            )
        accepted = row.get("accepted")
        reason = row.get("reason")
        if type(accepted) is not bool or accepted != (reason == "accepted"):
            raise RuntimeError(f"{dataset_name} sibling screening decision is invalid")
        if accepted:
            retained_index = row.get("retained_index")
            if type(retained_index) is not int or retained_index in accepted_rows:
                raise RuntimeError(
                    f"{dataset_name} sibling retained screening index is invalid"
                )
            accepted_rows[retained_index] = (position, row)
    if (
        set(accepted_rows) != set(range(target))
        or accepted_rows[target - 1][0] != next_position - 1
        or stats.get("target_reached_candidate_position") != next_position - 1
    ):
        raise RuntimeError(f"{dataset_name} sibling accepted cohort is invalid")

    try:
        sibling_exclusions = BuildExclusions(
            source_image_ids=frozenset(exclusions["source_image_ids"]),
            source_canonical_rgb_sha256=frozenset(
                exclusions["source_canonical_rgb_sha256"]
            ),
            saved_canonical_rgb_sha256=frozenset(
                exclusions["saved_canonical_rgb_sha256"]
            ),
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{dataset_name} sibling exclusions are invalid") from exc
    reconstructed_bundle = SourceBundle(
        dataset_name=dataset_name,
        candidates=reconstructed_candidates,
        selection_stats=initial_stats,
        source_metadata=source_metadata,
    )
    (
        reconstructed_stats,
        reconstructed_accepted,
        reconstructed_source_ids,
        reconstructed_source_hashes,
        reconstructed_saved_hashes,
    ) = _reconstruct_committed_state(
        reconstructed_bundle,
        BuildConfig(
            root,
            target=target,
            seed=seed,
            resume=True,
            exclusions=sibling_exclusions,
        ),
        committed,
        next_position,
    )
    _require_exact_json(
        label=f"{dataset_name} sibling reconstructed selection stats",
        actual=stats,
        expected=reconstructed_stats,
    )
    if set(reconstructed_accepted) != set(accepted_rows):
        raise RuntimeError(f"{dataset_name} sibling accepted reconstruction mismatch")

    accepted_source_hashes: set[str] = set()
    accepted_saved_hashes: set[str] = set()
    candidate_sample_keys = lineage_keys + (
        "prompt",
        "options",
        "ground_truth_letter",
        "ground_truth_text",
        "legacy_temperature_zero",
        "provenance",
    )
    for retained_index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise RuntimeError(f"{dataset_name} sibling sample is not an object")
        position, row = accepted_rows[retained_index]
        candidate = candidates[position]
        filename = f"images/{dataset_name}_{retained_index:04d}.jpg"
        if (
            sample.get("dataset") != dataset_name
            or sample.get("retained_index") != retained_index
            or sample.get("image_filename") != filename
            or any(sample.get(key) != candidate.get(key) for key in candidate_sample_keys)
            or sample.get("source_canonical_rgb_sha256")
            != row.get("source_canonical_rgb_sha256")
        ):
            raise RuntimeError(f"{dataset_name} sibling sample lineage is invalid")
        mapping = sample.get("mapping")
        if not isinstance(mapping, dict) or mapping.get("image_filename") != filename:
            raise RuntimeError(f"{dataset_name} sibling sample mapping is invalid")
        _require_exact_json(
            label=f"{dataset_name} sibling public mapping {retained_index}",
            actual=mappings[retained_index],
            expected=mapping,
        )
        source_hash = _require_sha256(
            sample.get("source_canonical_rgb_sha256"),
            label=f"{dataset_name} sibling source hash {retained_index}",
        )
        byte_hash = _require_sha256(
            sample.get("saved_image_sha256"),
            label=f"{dataset_name} sibling byte hash {retained_index}",
        )
        saved_hash = _require_sha256(
            sample.get("saved_canonical_rgb_sha256"),
            label=f"{dataset_name} sibling saved hash {retained_index}",
        )
        if (
            sample.get("canonical_rgb_sha256") != saved_hash
            or row.get("saved_image_sha256") != byte_hash
            or row.get("saved_canonical_rgb_sha256") != saved_hash
            or row.get("image_filename") != filename
        ):
            raise RuntimeError(f"{dataset_name} sibling artifact lineage is invalid")
        image_path = resolve_dataset_relative_path(paths["dataset_dir"], filename)
        if image_path.is_symlink() or not image_path.is_file():
            raise RuntimeError(f"{dataset_name} sibling saved image is missing/symlinked")
        if sha256_file(image_path) != byte_hash:
            raise RuntimeError(f"{dataset_name} sibling saved image byte hash changed")
        if canonical_rgb_sha256_path(image_path) != saved_hash:
            raise RuntimeError(f"{dataset_name} sibling saved image RGB hash changed")
        if source_hash in accepted_source_hashes or saved_hash in accepted_saved_hashes:
            raise RuntimeError(f"{dataset_name} sibling accepted identities are duplicated")
        accepted_source_hashes.add(source_hash)
        accepted_saved_hashes.add(saved_hash)
    if reconstructed_source_ids != _checkpoint_string_set(
        checkpoint, "seen_source_image_ids"
    ):
        raise RuntimeError(f"{dataset_name} sibling checkpoint source IDs mismatch")
    if reconstructed_source_hashes != _checkpoint_string_set(
        checkpoint, "seen_source_canonical_rgb_sha256", sha256=True
    ):
        raise RuntimeError(f"{dataset_name} sibling checkpoint source hashes mismatch")
    if reconstructed_saved_hashes != _checkpoint_string_set(
        checkpoint, "seen_saved_canonical_rgb_sha256", sha256=True
    ):
        raise RuntimeError(f"{dataset_name} sibling checkpoint saved hashes mismatch")
    return samples


def collect_completed_sibling_exclusions(
    output_root: Path | str,
    *,
    dataset_name: str,
    target: int,
    seed: int,
) -> BuildExclusions:
    """Collect verified identities from completed datasets in a shared root."""

    if dataset_name not in DATASET_NAMES:
        raise ValueError(f"unknown formal dataset {dataset_name!r}")
    if type(target) is not int or target <= 0:
        raise ValueError("target must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    root = Path(output_root).resolve()
    source_ids: set[str] = set()
    source_hashes: set[str] = set()
    saved_hashes: set[str] = set()
    source_owners: dict[str, str] = {}
    saved_owners: dict[str, str] = {}
    for sibling in DATASET_NAMES:
        if sibling == dataset_name:
            continue
        samples = _validated_completed_sibling_samples(
            root, sibling, target=target, seed=seed
        )
        for sample in samples:
            source_hash = str(sample["source_canonical_rgb_sha256"])
            saved_hash = str(sample["saved_canonical_rgb_sha256"])
            if source_hash in source_owners:
                raise RuntimeError(
                    "shared output root already contains a cross-dataset source RGB "
                    f"duplicate in {source_owners[source_hash]} and {sibling}"
                )
            if saved_hash in saved_owners:
                raise RuntimeError(
                    "shared output root already contains a cross-dataset saved RGB "
                    f"duplicate in {saved_owners[saved_hash]} and {sibling}"
                )
            source_owners[source_hash] = sibling
            saved_owners[saved_hash] = sibling
            source_hashes.add(source_hash)
            saved_hashes.add(saved_hash)
            if dataset_name in {"VQAv2_Open", "VQAv2_MC"} and sibling in {
                "VQAv2_Open",
                "VQAv2_MC",
            }:
                source_id = sample.get("source_image_id")
                if source_id is not None:
                    source_ids.add(str(source_id))
    return BuildExclusions.from_iterables(
        source_image_ids=source_ids,
        source_hashes=source_hashes,
        saved_hashes=saved_hashes,
    )


def load_stored_build_exclusions(
    output_root: Path | str, *, dataset_name: str
) -> BuildExclusions:
    """Load the immutable exclusion set recorded when a dataset build began."""

    if dataset_name not in DATASET_NAMES:
        raise ValueError(f"unknown formal dataset {dataset_name!r}")
    root = Path(output_root).resolve()
    paths = _paths(BuildConfig(root), dataset_name)
    if not paths["source"].is_file():
        raise RuntimeError(
            f"resume source manifest does not exist for {dataset_name}: "
            f"{paths['source']}"
        )
    source = _read_json_object_strict(
        paths["source"], label=f"{dataset_name} resume source manifest"
    )
    exclusions = source.get("build_exclusions")
    fields = {
        "source_image_ids",
        "source_canonical_rgb_sha256",
        "saved_canonical_rgb_sha256",
    }
    if (
        source.get("dataset") != dataset_name
        or not isinstance(exclusions, dict)
        or set(exclusions) != fields
        or source.get("exclusions_fingerprint") != _stable_json_digest(exclusions)
    ):
        raise RuntimeError(f"invalid stored build exclusions for {dataset_name}")
    if any(
        not isinstance(values, list)
        or any(not isinstance(value, str) or not value for value in values)
        or values != sorted(set(values))
        for values in exclusions.values()
    ):
        raise RuntimeError(f"non-canonical stored build exclusions for {dataset_name}")
    try:
        return BuildExclusions(
            source_image_ids=frozenset(exclusions["source_image_ids"]),
            source_canonical_rgb_sha256=frozenset(
                exclusions["source_canonical_rgb_sha256"]
            ),
            saved_canonical_rgb_sha256=frozenset(
                exclusions["saved_canonical_rgb_sha256"]
            ),
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"invalid stored build exclusions for {dataset_name}") from exc


def _require_screening_value(
    row: dict[str, Any], key: str, expected: Any, *, position: int
) -> None:
    if key not in row:
        raise RuntimeError(f"committed screening row {position} is missing {key}")
    _require_exact_json(
        label=f"committed screening {key} at position {position}",
        actual=row[key],
        expected=expected,
    )


def _require_prediction(row: dict[str, Any], key: str, *, position: int) -> str:
    value = row.get(key)
    if not isinstance(value, str):
        raise RuntimeError(
            f"committed screening {key} at position {position} must be a string"
        )
    return value


def _require_reason(row: dict[str, Any], expected: str, *, position: int) -> None:
    if row.get("reason") != expected:
        raise RuntimeError(
            f"committed screening row {position} is inconsistent with the "
            f"builder state: expected reason {expected!r}, got {row.get('reason')!r}"
        )


def _require_screening_schema(
    row: dict[str, Any], reason: str, *, position: int
) -> None:
    expected_fields = _SCREENING_FIELDS_BY_REASON.get(reason)
    if expected_fields is None:
        raise RuntimeError(
            f"committed screening row {position} has invalid schema: "
            f"unknown reason {reason!r}"
        )
    actual_fields = set(row)
    if actual_fields != expected_fields:
        missing = sorted(expected_fields - actual_fields)
        extra = sorted(actual_fields - expected_fields)
        raise RuntimeError(
            f"committed screening row {position} has invalid schema for reason "
            f"{reason!r}: missing={missing!r}, extra={extra!r}"
        )


def _reconstruct_committed_state(
    bundle: SourceBundle,
    config: BuildConfig,
    committed: dict[int, dict[str, Any]],
    next_position: int,
) -> tuple[
    dict[str, Any],
    dict[int, tuple[int, dict[str, Any], Candidate]],
    set[str],
    set[str],
    set[str],
]:
    """Rebuild every mutable checkpoint field from committed screening rows."""

    stats = _initial_stats(bundle, config)
    seen_source_ids: set[str] = set()
    seen_source_hashes: set[str] = set()
    seen_saved_hashes: set[str] = set()
    accepted_rows: dict[int, tuple[int, dict[str, Any], Candidate]] = {}

    for position in range(next_position):
        row = committed[position]
        candidate = bundle.candidates[position]
        expected_base = _screening_base(position, candidate)
        for key in (
            "candidate_position",
            "source_dataset",
            "source_split",
            "source_index",
            "source_split_index",
            "source_question_id",
            "source_image_id",
        ):
            _require_screening_value(row, key, expected_base[key], position=position)

        reason = row.get("reason")
        if not isinstance(reason, str) or not reason:
            raise RuntimeError(
                f"committed screening row {position} has invalid reason"
            )
        accepted = row.get("accepted")
        if type(accepted) is not bool:
            raise RuntimeError(
                f"committed screening row {position} has invalid accepted flag"
            )
        if accepted != (reason == "accepted"):
            raise RuntimeError(
                f"committed screening row {position} accepted flag and reason disagree"
            )
        _require_screening_schema(row, reason, position=position)

        stats["scanned_candidates"] += 1
        source_id = (
            str(candidate.source_image_id)
            if candidate.source_image_id is not None
            else None
        )
        if source_id is not None and source_id in seen_source_ids:
            _require_reason(row, "duplicate_source_image_id", position=position)
            if row.get("source_canonical_rgb_sha256") is not None:
                raise RuntimeError(
                    f"committed screening row {position} has a source RGB hash "
                    "despite duplicate-source-ID rejection occurring before hashing"
                )
            stats["engine_source_id_duplicate_count"] += 1
            stats["source_id_duplicate_count"] += 1
        else:
            if source_id is not None:
                if not source_id:
                    raise RuntimeError(
                        f"committed screening row {position} has empty source_image_id"
                    )
                seen_source_ids.add(source_id)

            if reason == "source_network_or_decode_error":
                if row.get("source_canonical_rgb_sha256") is not None:
                    raise RuntimeError(
                        f"committed screening row {position} has a source RGB hash "
                        "despite source loading failing before hashing"
                    )
                stats["missing_or_corrupt_images"] += 1
            else:
                source_hash = _require_sha256(
                    row.get("source_canonical_rgb_sha256"),
                    label=(
                        "committed screening source_canonical_rgb_sha256 at "
                        f"position {position}"
                    ),
                )
                if (
                    candidate.source_canonical_rgb_sha256 is not None
                    and source_hash != candidate.source_canonical_rgb_sha256
                ):
                    raise RuntimeError(
                        "committed screening source RGB hash does not match the "
                        f"candidate at position {position}"
                    )
                stats["source_hash_candidates_scanned"] += 1
                if source_hash in seen_source_hashes:
                    _require_reason(
                        row,
                        "duplicate_source_canonical_rgb_sha256",
                        position=position,
                    )
                    stats["engine_exact_source_hash_duplicate_count"] += 1
                    stats["exact_image_hash_duplicate_count"] += 1
                else:
                    seen_source_hashes.add(source_hash)
                    if (
                        source_id is not None
                        and source_id in config.exclusions.source_image_ids
                    ):
                        _require_reason(
                            row,
                            "excluded_prior_dataset_source_image_id",
                            position=position,
                        )
                        stats["excluded_source_image_id_count"] += 1
                    elif source_hash in config.exclusions.source_canonical_rgb_sha256:
                        _require_reason(
                            row,
                            "excluded_prior_dataset_source_hash",
                            position=position,
                        )
                        stats["excluded_source_hash_count"] += 1
                    else:
                        stats["unique_images_evaluated"] += 1
                        blind_prediction = _require_prediction(
                            row, "blind_prediction", position=position
                        )
                        blind_correct = row.get("blind_correct")
                        if type(blind_correct) is not bool:
                            raise RuntimeError(
                                "committed screening blind_correct at position "
                                f"{position} must be boolean"
                            )
                        recomputed_blind_correct = _correct(
                            candidate, blind_prediction
                        )
                        if blind_correct is not recomputed_blind_correct:
                            raise RuntimeError(
                                "committed screening blind_correct does not match "
                                "the stored prediction and candidate at position "
                                f"{position}"
                            )
                        _require_screening_value(
                            row,
                            "control",
                            "blind_black_image_control",
                            position=position,
                        )
                        if blind_correct:
                            _require_reason(
                                row,
                                "blind_black_image_control_correct",
                                position=position,
                            )
                            stats["blind_correct_count"] += 1
                        else:
                            stats["blind_wrong_count"] += 1
                            saved_byte_hash = _require_sha256(
                                row.get("saved_image_sha256"),
                                label=(
                                    "committed screening saved_image_sha256 at "
                                    f"position {position}"
                                ),
                            )
                            saved_rgb_hash = _require_sha256(
                                row.get("saved_canonical_rgb_sha256"),
                                label=(
                                    "committed screening saved_canonical_rgb_sha256 "
                                    f"at position {position}"
                                ),
                            )
                            # The byte digest is validated here even though only the
                            # canonical RGB digest participates in deduplication.
                            assert saved_byte_hash
                            perceptual_hash = row.get("perceptual_hash")
                            if not isinstance(perceptual_hash, str) or not perceptual_hash:
                                raise RuntimeError(
                                    "committed screening perceptual_hash at position "
                                    f"{position} must be a non-empty string"
                                )
                            if saved_rgb_hash in seen_saved_hashes:
                                _require_reason(
                                    row,
                                    "duplicate_saved_canonical_rgb_sha256",
                                    position=position,
                                )
                                stats["saved_exact_hash_duplicate_count"] += 1
                            elif (
                                saved_rgb_hash
                                in config.exclusions.saved_canonical_rgb_sha256
                            ):
                                _require_reason(
                                    row,
                                    "excluded_prior_dataset_saved_hash",
                                    position=position,
                                )
                                stats["excluded_saved_hash_count"] += 1
                            else:
                                full_prediction = _require_prediction(
                                    row, "full_prediction", position=position
                                )
                                full_correct = row.get("full_correct")
                                if type(full_correct) is not bool:
                                    raise RuntimeError(
                                        "committed screening full_correct at position "
                                        f"{position} must be boolean"
                                    )
                                recomputed_full_correct = _correct(
                                    candidate, full_prediction
                                )
                                if full_correct is not recomputed_full_correct:
                                    raise RuntimeError(
                                        "committed screening full_correct does not "
                                        "match the stored prediction and candidate at "
                                        f"position {position}"
                                    )
                                if full_correct:
                                    _require_reason(row, "accepted", position=position)
                                    retained_index = row.get("retained_index")
                                    if (
                                        type(retained_index) is not int
                                        or retained_index != len(accepted_rows)
                                    ):
                                        raise RuntimeError(
                                            "committed screening accepted rows have "
                                            "non-contiguous retained indices"
                                        )
                                    if retained_index >= config.target:
                                        raise RuntimeError(
                                            "checkpoint contains more accepted samples "
                                            "than its target"
                                        )
                                    accepted_rows[retained_index] = (
                                        position,
                                        row,
                                        candidate,
                                    )
                                    seen_saved_hashes.add(saved_rgb_hash)
                                    stats["full_correct_count"] += 1
                                    if bundle.dataset_name == "ScienceQA_MC":
                                        split_counts = stats.setdefault(
                                            "split_composition",
                                            {
                                                "train": 0,
                                                "validation": 0,
                                                "test": 0,
                                            },
                                        )
                                        split_counts[candidate.source_split] += 1
                                else:
                                    _require_reason(
                                        row, "full_visual_incorrect", position=position
                                    )
                                    stats["full_incorrect_count"] += 1

        stats["retained_count"] = len(accepted_rows)
        evaluated = stats["unique_images_evaluated"]
        stats["acceptance_rate"] = (
            len(accepted_rows) / evaluated if evaluated else 0.0
        )
        if len(accepted_rows) == config.target:
            if position != next_position - 1:
                raise RuntimeError(
                    "checkpoint continued scanning after the target was first reached"
                )
            stats["target_reached_source_index"] = candidate.source_index
            stats["target_reached_candidate_position"] = position

    stats["candidate_pool_exhausted"] = (
        next_position >= len(bundle.candidates)
        and len(accepted_rows) < config.target
    )
    return (
        stats,
        accepted_rows,
        seen_source_ids,
        seen_source_hashes,
        seen_saved_hashes,
    )


def _load_resume_state(
    bundle: SourceBundle,
    config: BuildConfig,
    paths: dict[str, Path],
    pool_fingerprint: str,
    exclusions_fingerprint: str,
    construction_protocol_fingerprint: str,
    *,
    repair_public_state: bool = True,
) -> tuple[int, dict[str, Any], list[dict[str, Any]], set[str], set[str], set[str]]:
    checkpoint = json.loads(paths["checkpoint"].read_text(encoding="utf-8"))
    if not isinstance(checkpoint, dict):
        raise RuntimeError("resume checkpoint must be a JSON object")
    expected = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "dataset": bundle.dataset_name,
        "target": config.target,
        "seed": config.seed,
        "candidate_pool_fingerprint": pool_fingerprint,
        "exclusions_fingerprint": exclusions_fingerprint,
        "construction_protocol_fingerprint": construction_protocol_fingerprint,
    }
    mismatches = {}
    for key, value in expected.items():
        actual = checkpoint.get(key)
        if _canonical_json(actual, label=f"checkpoint {key}") != _canonical_json(
            value, label=f"expected checkpoint {key}"
        ):
            mismatches[key] = (actual, value)
    if mismatches:
        raise RuntimeError(f"refusing incompatible resume checkpoint: {mismatches!r}")
    next_position = checkpoint.get("next_candidate_position")
    if type(next_position) is not int or not 0 <= next_position <= len(bundle.candidates):
        raise RuntimeError("checkpoint has invalid next_candidate_position")
    stats = checkpoint.get("selection_stats")
    samples = checkpoint.get("samples")
    if not isinstance(stats, dict) or not isinstance(samples, list):
        raise RuntimeError("checkpoint has invalid stats/samples")
    if len(samples) > config.target:
        raise RuntimeError("checkpoint contains more samples than its target")
    if len(samples) != stats.get("retained_count"):
        raise RuntimeError("checkpoint retained_count does not match samples")
    checkpoint_seen_source_ids = _checkpoint_string_set(
        checkpoint, "seen_source_image_ids"
    )
    checkpoint_seen_source_hashes = _checkpoint_string_set(
        checkpoint, "seen_source_canonical_rgb_sha256", sha256=True
    )
    checkpoint_seen_saved_hashes = _checkpoint_string_set(
        checkpoint, "seen_saved_canonical_rgb_sha256", sha256=True
    )

    # The checkpoint is authoritative.  Verify every committed final file, then
    # rebuild the two public JSONL views atomically in case a crash separated the
    # checkpoint commit from a public-file rewrite.
    expected_filenames: set[str] = set()
    expected_seen_saved_hashes: set[str] = set()
    for retained_index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise RuntimeError("checkpoint samples must contain objects")
        sample_retained_index = sample.get("retained_index")
        if (
            type(sample_retained_index) is not int
            or sample_retained_index != retained_index
        ):
            raise RuntimeError("checkpoint retained indices are not contiguous")
        relative = sample.get("image_filename")
        if not isinstance(relative, str):
            raise RuntimeError("checkpoint sample has invalid image_filename")
        lexical_image_path = paths["dataset_dir"].joinpath(
            *PurePosixPath(relative).parts
        )
        if lexical_image_path.is_symlink():
            raise RuntimeError(
                f"checkpoint sample image_filename is a symlink: {relative!r}"
            )
        try:
            image_path = resolve_dataset_relative_path(
                paths["dataset_dir"], relative
            )
        except ValueError as exc:
            raise RuntimeError(
                f"checkpoint sample has unsafe image_filename: {relative!r}"
            ) from exc
        expected_relative = (
            f"images/{bundle.dataset_name}_{retained_index:04d}.jpg"
        )
        if relative != expected_relative:
            raise RuntimeError(
                "checkpoint sample image_filename does not match its dataset/"
                f"retained index: expected {expected_relative!r}, got {relative!r}"
            )
        mapping = sample.get("mapping")
        if not isinstance(mapping, dict) or mapping.get("image_filename") != relative:
            raise RuntimeError(
                "checkpoint sample mapping image_filename does not match the "
                f"validated sample path at retained index {retained_index}"
            )
        _require_sha256(
            sample.get("source_canonical_rgb_sha256"),
            label=(
                "checkpoint sample source_canonical_rgb_sha256 at retained "
                f"index {retained_index}"
            ),
        )
        saved_byte_hash = _require_sha256(
            sample.get("saved_image_sha256"),
            label=f"checkpoint sample saved_image_sha256 at retained index {retained_index}",
        )
        saved_rgb_hash = _require_sha256(
            sample.get("saved_canonical_rgb_sha256"),
            label=(
                "checkpoint sample saved_canonical_rgb_sha256 at retained "
                f"index {retained_index}"
            ),
        )
        if saved_rgb_hash in expected_seen_saved_hashes:
            raise RuntimeError(
                "checkpoint accepted samples contain duplicate saved canonical "
                f"RGB hash {saved_rgb_hash}"
            )
        expected_seen_saved_hashes.add(saved_rgb_hash)
        expected_filenames.add(relative)
        if not image_path.is_file():
            raise RuntimeError(f"resume image is missing: {image_path}")
        if sha256_file(image_path) != saved_byte_hash:
            raise RuntimeError(f"resume image byte hash changed: {image_path}")
        if canonical_rgb_sha256_path(image_path) != saved_rgb_hash:
            raise RuntimeError(f"resume image RGB hash changed: {image_path}")

    # append+fsync precedes checkpoint.  Discard only an uncommitted tail (or a
    # duplicate retry tail) and insist on one explicit reason per committed scan.
    screening = read_jsonl_tolerant(paths["screening"])
    committed: dict[int, dict[str, Any]] = {}
    for row_index, row in enumerate(screening):
        if not isinstance(row, dict):
            raise RuntimeError(
                f"screening row {row_index} is not a JSON object"
            )
        position = row.get("candidate_position")
        if isinstance(position, int) and 0 <= position < next_position:
            if position in committed:
                raise RuntimeError(f"duplicate committed screening row {position}")
            committed[position] = row
    if set(committed) != set(range(next_position)):
        missing = sorted(set(range(next_position)) - set(committed))
        raise RuntimeError(f"checkpoint is missing committed screening rows {missing[:10]!r}")

    (
        expected_stats,
        accepted_rows,
        expected_seen_source_ids,
        expected_seen_source_hashes,
        reconstructed_seen_saved_hashes,
    ) = _reconstruct_committed_state(bundle, config, committed, next_position)

    if set(accepted_rows) != set(range(len(samples))):
        raise RuntimeError(
            "committed accepted screening rows do not match checkpoint samples"
        )
    for retained_index, sample in enumerate(samples):
        position, row, candidate = accepted_rows[retained_index]
        relative_name = f"images/{bundle.dataset_name}_{retained_index:04d}.jpg"
        final_mapping = copy.deepcopy(candidate.mapping_payload)
        final_mapping["image_filename"] = relative_name
        expected_sample = {
            "dataset": bundle.dataset_name,
            "retained_index": retained_index,
            "image_filename": relative_name,
            "source_dataset": candidate.source_dataset,
            "source_split": candidate.source_split,
            "source_index": candidate.source_index,
            "source_split_index": candidate.source_split_index,
            "source_question_id": candidate.source_question_id,
            "source_image_id": candidate.source_image_id,
            "source_canonical_rgb_sha256": row["source_canonical_rgb_sha256"],
            "canonical_rgb_sha256": row["saved_canonical_rgb_sha256"],
            "saved_image_sha256": row["saved_image_sha256"],
            "saved_canonical_rgb_sha256": row["saved_canonical_rgb_sha256"],
            "perceptual_hash": row["perceptual_hash"],
            "prompt": candidate.prompt,
            "blind_control": "blind_black_image_control",
            "blind_prediction": row["blind_prediction"],
            "blind_correct": False,
            "full_prediction": row["full_prediction"],
            "full_correct": True,
            "blind_black_image_control_prediction": row["blind_prediction"],
            "blind_black_image_control_correct": False,
            "full_visual_prediction": row["full_prediction"],
            "full_visual_correct": True,
            "mapping": final_mapping,
            "options": list(candidate.options),
            "ground_truth_letter": candidate.ground_truth_letter,
            "ground_truth_text": candidate.ground_truth_text,
            "legacy_temperature_zero": candidate.legacy_temperature_zero,
            "jpeg_settings": JPEG_SETTINGS,
            "provenance": candidate.provenance,
        }
        _require_exact_json(
            label=f"checkpoint sample at retained index {retained_index}",
            actual=sample,
            expected=expected_sample,
        )
        for key in (
            "image_filename",
            "source_image_id",
            "source_canonical_rgb_sha256",
            "saved_image_sha256",
            "saved_canonical_rgb_sha256",
            "perceptual_hash",
        ):
            _require_screening_value(
                row, key, expected_sample[key], position=position
            )
        image_path = resolve_dataset_relative_path(
            paths["dataset_dir"], relative_name
        )
        if dhash_path(image_path) != row["perceptual_hash"]:
            raise RuntimeError(
                f"resume image perceptual hash changed: {image_path}"
            )

    if reconstructed_seen_saved_hashes != expected_seen_saved_hashes:
        raise RuntimeError(
            "checkpoint samples and committed screening rows disagree on saved hashes"
        )

    _require_exact_json(
        label="checkpoint selection_stats",
        actual=stats,
        expected=expected_stats,
    )

    _require_exact_checkpoint_set(
        key="seen_source_image_ids",
        actual=checkpoint_seen_source_ids,
        expected=expected_seen_source_ids,
    )
    _require_exact_checkpoint_set(
        key="seen_source_canonical_rgb_sha256",
        actual=checkpoint_seen_source_hashes,
        expected=expected_seen_source_hashes,
    )
    _require_exact_checkpoint_set(
        key="seen_saved_canonical_rgb_sha256",
        actual=checkpoint_seen_saved_hashes,
        expected=reconstructed_seen_saved_hashes,
    )

    if repair_public_state:
        # Mutate the resume tree only after every checkpoint path, digest,
        # committed decision, and reconstructed set has passed. Remove only
        # builder-owned, explicitly matched orphan names left by a crash.
        for path in paths["images_dir"].glob(f"{bundle.dataset_name}_*.jpg"):
            relative = path.relative_to(paths["dataset_dir"]).as_posix()
            if relative not in expected_filenames:
                path.unlink()
        for path in paths["stage_dir"].glob("*.jpg"):
            path.unlink()
        atomic_write_jsonl(
            paths["screening"],
            [committed[index] for index in range(next_position)],
        )
        _write_public_state(paths, stats, samples)
    return (
        next_position,
        stats,
        samples,
        checkpoint_seen_source_ids,
        checkpoint_seen_source_hashes,
        checkpoint_seen_saved_hashes,
    )


def _build_result(
    bundle: SourceBundle,
    config: BuildConfig,
    paths: dict[str, Path],
    stats: dict[str, Any],
    samples: list[dict[str, Any]],
) -> BuildResult:
    return BuildResult(
        dataset_name=bundle.dataset_name,
        output_root=config.output_root,
        dataset_dir=paths["dataset_dir"],
        mapping_path=paths["mapping"],
        sidecar_path=paths["sidecar"],
        screening_path=paths["screening"],
        checkpoint_path=paths["checkpoint"],
        stats_path=paths["stats"],
        stats=copy.deepcopy(stats),
        samples=tuple(copy.deepcopy(samples)),
        target_reached=len(samples) >= config.target,
    )


class _StoredConstructionProtocolRunner:
    """Manifest-only runner used to validate an already-complete resume."""

    def __init__(self, runner_manifest: dict[str, Any]) -> None:
        self._runner_manifest = copy.deepcopy(runner_manifest)

    def construction_manifest(self) -> dict[str, Any]:
        return copy.deepcopy(self._runner_manifest)

    def infer_blind(self, *_args: Any, **_kwargs: Any) -> str:
        raise RuntimeError("complete-resume preflight must not perform inference")

    def infer(self, *_args: Any, **_kwargs: Any) -> str:
        raise RuntimeError("complete-resume preflight must not perform inference")


def load_completed_resume(
    bundle: SourceBundle,
    config: BuildConfig,
    *,
    repair_public_state: bool = True,
    model_path: Path | str | None = None,
    current_runner: Any | None = None,
) -> BuildResult | None:
    """Validate and return a complete checkpoint without loading model weights.

    Incomplete checkpoints return ``None`` so the caller can construct its real
    runner and continue. A checkpoint that claims completion is passed through
    the same strict source/pool/screening/sample reconstruction as a normal
    resume; only its already-recorded runner manifest is reused because no new
    inference will occur.
    """

    if not config.resume:
        return None
    paths = _paths(config, bundle.dataset_name)
    if not paths["checkpoint"].is_file():
        raise RuntimeError(
            f"--resume requested but checkpoint does not exist: {paths['checkpoint']}"
        )
    checkpoint = json.loads(paths["checkpoint"].read_text(encoding="utf-8"))
    if not isinstance(checkpoint, dict):
        raise RuntimeError("resume checkpoint must be a JSON object")
    for key, expected in (
        ("format_version", CHECKPOINT_FORMAT_VERSION),
        ("dataset", bundle.dataset_name),
        ("target", config.target),
        ("seed", config.seed),
    ):
        if _canonical_json(checkpoint.get(key), label=f"checkpoint {key}") != _canonical_json(
            expected, label=f"expected checkpoint {key}"
        ):
            raise RuntimeError(f"refusing incompatible resume checkpoint field {key}")
    samples = checkpoint.get("samples")
    stats = checkpoint.get("selection_stats")
    if not isinstance(samples, list) or not isinstance(stats, dict):
        raise RuntimeError("checkpoint has invalid stats/samples")
    retained_count = stats.get("retained_count")
    if type(retained_count) is not int or retained_count != len(samples):
        raise RuntimeError("checkpoint retained_count does not match samples")
    if len(samples) > config.target:
        raise RuntimeError("checkpoint contains more samples than its target")
    if len(samples) < config.target:
        return None
    if not paths["source"].is_file():
        raise RuntimeError(
            f"complete resume source manifest does not exist: {paths['source']}"
        )
    source = json.loads(paths["source"].read_text(encoding="utf-8"))
    if not isinstance(source, dict):
        raise RuntimeError("complete resume source manifest must be a JSON object")
    protocol = source.get("construction_protocol")
    if not isinstance(protocol, dict) or not isinstance(protocol.get("runner"), dict):
        raise RuntimeError("complete resume source manifest has invalid protocol")
    stored_runner_manifest = protocol["runner"]
    if model_path is not None and current_runner is not None:
        raise ValueError("pass only one complete-resume identity source")
    if current_runner is not None:
        current_manifest = _construction_protocol_manifest(current_runner)["runner"]
        _require_exact_json(
            label="complete resume current runner manifest",
            actual=current_manifest,
            expected=stored_runner_manifest,
        )
    elif model_path is not None:
        stored_model_manifest = stored_runner_manifest.get("model")
        if not isinstance(stored_model_manifest, dict):
            raise RuntimeError(
                "complete resume stored runner has no model artifact manifest"
            )
        current_model_manifest = model_directory_fingerprint(Path(model_path))
        _require_exact_json(
            label="complete resume current model artifact manifest",
            actual=current_model_manifest,
            expected=stored_model_manifest,
        )
    else:
        raise RuntimeError(
            "complete resume requires current_runner or model_path identity"
        )
    runner = _StoredConstructionProtocolRunner(stored_runner_manifest)
    return build_dataset(
        bundle,
        runner,
        config,
        repair_resume_state=repair_public_state,
    )


def build_dataset(
    bundle: SourceBundle,
    runner: Any,
    config: BuildConfig,
    *,
    fail_if_short: bool = True,
    repair_resume_state: bool = True,
) -> BuildResult:
    """Build one dataset serially and durably from prepared representatives."""

    paths = _paths(config, bundle.dataset_name)
    pool_fingerprint = _candidate_pool_fingerprint(bundle)
    exclusions_fingerprint = _exclusions_fingerprint(config.exclusions)
    construction_protocol = _construction_protocol_manifest(runner)
    construction_protocol_fingerprint = _stable_json_digest(construction_protocol)
    artifact_paths = (
        paths["mapping"],
        paths["sidecar"],
        paths["checkpoint"],
        paths["screening"],
        paths["stats"],
        paths["candidate_pool"],
    )
    if config.resume:
        required_resume_artifacts = (
            paths["checkpoint"],
            paths["source"],
            paths["candidate_pool"],
        )
        missing_resume_artifacts = [
            str(path) for path in required_resume_artifacts if not path.is_file()
        ]
        if missing_resume_artifacts:
            raise RuntimeError(
                "--resume requested but required atomic build artifacts are missing: "
                f"{missing_resume_artifacts!r}"
            )
    elif any(path.exists() for path in artifact_paths):
        existing = [str(path) for path in artifact_paths if path.exists()]
        raise FileExistsError(
            f"refusing to overwrite existing build artifacts without --resume: {existing!r}"
        )

    if config.resume and not repair_resume_state:
        required_directories = (
            paths["dataset_dir"],
            paths["images_dir"],
            paths["checkpoint"].parent,
        )
        missing_directories = [
            str(path) for path in required_directories if not path.is_dir()
        ]
        if missing_directories:
            raise RuntimeError(
                "read-only resume requires existing build directories: "
                f"{missing_directories!r}"
            )
    else:
        for key in ("images_dir", "stage_dir"):
            paths[key].mkdir(parents=True, exist_ok=True)
        for key in (
            "mapping",
            "sidecar",
            "checkpoint",
            "source",
            "candidate_pool",
            "screening",
            "stats",
        ):
            paths[key].parent.mkdir(parents=True, exist_ok=True)

    stable_source_metadata = _stable_source_metadata(bundle.source_metadata)
    exclusions_manifest = {
        "source_image_ids": sorted(config.exclusions.source_image_ids),
        "source_canonical_rgb_sha256": sorted(
            config.exclusions.source_canonical_rgb_sha256
        ),
        "saved_canonical_rgb_sha256": sorted(
            config.exclusions.saved_canonical_rgb_sha256
        ),
    }
    source_manifest = {
        "dataset": bundle.dataset_name,
        "candidate_pool_fingerprint": pool_fingerprint,
        "canonical_hash": CANONICAL_HASH_DESCRIPTION,
        "jpeg_settings": JPEG_SETTINGS,
        "construction_protocol": construction_protocol,
        "construction_protocol_fingerprint": construction_protocol_fingerprint,
        "source_metadata": bundle.source_metadata,
        "source_metadata_stable_fingerprint": _stable_json_digest(
            stable_source_metadata
        ),
        "initial_selection_stats": bundle.selection_stats,
        "initial_selection_stats_fingerprint": _stable_json_digest(
            bundle.selection_stats
        ),
        "build_exclusions": exclusions_manifest,
        "exclusions_fingerprint": _stable_json_digest(exclusions_manifest),
    }
    if paths["source"].exists() and config.resume:
        old_source = json.loads(paths["source"].read_text(encoding="utf-8"))
        if not isinstance(old_source, dict):
            raise RuntimeError("resume source manifest must be a JSON object")
        old_protocol = old_source.get("construction_protocol")
        if not isinstance(old_protocol, dict):
            raise RuntimeError(
                "resume source manifest is missing construction_protocol"
            )
        old_protocol_fingerprint = old_source.get(
            "construction_protocol_fingerprint"
        )
        recomputed_protocol_fingerprint = _stable_json_digest(old_protocol)
        if old_protocol_fingerprint != recomputed_protocol_fingerprint:
            raise RuntimeError(
                "resume source manifest construction_protocol body does not match "
                "its fingerprint"
            )
        _require_exact_json(
            label="resume source manifest construction_protocol",
            actual=old_protocol,
            expected=construction_protocol,
        )
        if "source_metadata" not in old_source:
            raise RuntimeError("resume source manifest is missing source_metadata")
        old_stable_source_metadata = _stable_source_metadata(
            old_source["source_metadata"]
        )
        old_metadata_fingerprint = old_source.get(
            "source_metadata_stable_fingerprint"
        )
        recomputed_metadata_fingerprint = _stable_json_digest(
            old_stable_source_metadata
        )
        if old_metadata_fingerprint != recomputed_metadata_fingerprint:
            raise RuntimeError(
                "resume source manifest stable source_metadata does not match "
                "its fingerprint"
            )
        _require_exact_json(
            label="resume source manifest stable source_metadata",
            actual=old_stable_source_metadata,
            expected=stable_source_metadata,
        )
        old_initial_stats = old_source.get("initial_selection_stats")
        if not isinstance(old_initial_stats, dict):
            raise RuntimeError(
                "resume source manifest is missing initial_selection_stats"
            )
        if old_source.get("initial_selection_stats_fingerprint") != _stable_json_digest(
            old_initial_stats
        ):
            raise RuntimeError(
                "resume source manifest initial_selection_stats body does not "
                "match its fingerprint"
            )
        _require_exact_json(
            label="resume source manifest initial_selection_stats",
            actual=old_initial_stats,
            expected=bundle.selection_stats,
        )
        old_exclusions = old_source.get("build_exclusions")
        if not isinstance(old_exclusions, dict):
            raise RuntimeError("resume source manifest is missing build_exclusions")
        if old_source.get("exclusions_fingerprint") != _stable_json_digest(
            old_exclusions
        ):
            raise RuntimeError(
                "resume source manifest build_exclusions body does not match "
                "its fingerprint"
            )
        _require_exact_json(
            label="resume source manifest build_exclusions",
            actual=old_exclusions,
            expected=exclusions_manifest,
        )
        # Cache materialization status legitimately changes from
        # "copied/downloaded" on the first run to "existing" on resume.  Lock
        # the semantic source/pool fields here; candidate_pool and per-image
        # hashes independently protect the actual inputs.
        stable_keys = (
            "dataset",
            "candidate_pool_fingerprint",
            "canonical_hash",
            "jpeg_settings",
            "construction_protocol_fingerprint",
            "source_metadata_stable_fingerprint",
            "initial_selection_stats_fingerprint",
            "exclusions_fingerprint",
        )
        changed_keys = [
            key
            for key in stable_keys
            if old_source.get(key) != source_manifest.get(key)
        ]
        if changed_keys:
            raise RuntimeError(
                "source manifest changed since checkpoint: "
                + ", ".join(changed_keys)
            )
    else:
        atomic_write_json(paths["source"], source_manifest)

    candidate_pool_rows = _candidate_pool_rows(bundle)
    if paths["candidate_pool"].exists() and config.resume:
        existing_pool = _read_jsonl_strict(
            paths["candidate_pool"], label="candidate-pool manifest"
        )
        _require_exact_json(
            label="candidate-pool manifest",
            actual=existing_pool,
            expected=candidate_pool_rows,
        )
    else:
        atomic_write_jsonl(paths["candidate_pool"], candidate_pool_rows)

    if config.resume:
        (
            next_position,
            stats,
            samples,
            seen_source_ids,
            seen_source_hashes,
            seen_saved_hashes,
        ) = _load_resume_state(
            bundle,
            config,
            paths,
            pool_fingerprint,
            exclusions_fingerprint,
            construction_protocol_fingerprint,
            repair_public_state=repair_resume_state,
        )
    else:
        next_position = 0
        stats = _initial_stats(bundle, config)
        samples: list[dict[str, Any]] = []
        seen_source_ids: set[str] = set()
        seen_source_hashes: set[str] = set()
        seen_saved_hashes: set[str] = set()
        stats["candidate_pool_exhausted"] = not bundle.candidates
        atomic_write_jsonl(paths["screening"], [])
        checkpoint = _checkpoint_payload(
            bundle,
            config,
            pool_fingerprint,
            exclusions_fingerprint,
            construction_protocol_fingerprint,
            next_position,
            stats,
            samples,
            seen_source_ids,
            seen_source_hashes,
            seen_saved_hashes,
        )
        atomic_write_json(paths["checkpoint"], checkpoint)
        _write_public_state(paths, stats, samples)

    if len(samples) >= config.target:
        return _build_result(bundle, config, paths, stats, samples)

    print(
        json.dumps(
            {
                "phase": "build_start",
                "dataset": bundle.dataset_name,
                "candidate_position": next_position,
                "candidate_pool": len(bundle.candidates),
                "retained": len(samples),
                "target": config.target,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    for position in range(next_position, len(bundle.candidates)):
        candidate = bundle.candidates[position]
        screening = _screening_base(position, candidate)
        source_id = (
            str(candidate.source_image_id)
            if candidate.source_image_id is not None
            else None
        )
        stats["scanned_candidates"] += 1

        # Reliable source-ID grouping was already applied by the loader.  This is
        # an independent guard immediately before any model call.
        if source_id is not None and source_id in seen_source_ids:
            stats["engine_source_id_duplicate_count"] += 1
            stats["source_id_duplicate_count"] += 1
            screening["reason"] = "duplicate_source_image_id"
        else:
            if source_id is not None:
                seen_source_ids.add(source_id)
            try:
                source_image = candidate.image_loader()
            except RecoverableSourceError as error:
                # A bounded source/network failure is not a scientific
                # rejection. Leave the durable checkpoint at this candidate so
                # a later --resume retries it instead of making the selected
                # cohort depend on transient availability.
                raise RecoverableSourceError(
                    f"{bundle.dataset_name} candidate {position} is temporarily "
                    f"unavailable; resume will retry it: {error}"
                ) from error
            else:
                source_hash = canonical_rgb_sha256(source_image)
                stats["source_hash_candidates_scanned"] += 1
                screening["source_canonical_rgb_sha256"] = source_hash
                if (
                    candidate.source_canonical_rgb_sha256 is not None
                    and source_hash != candidate.source_canonical_rgb_sha256
                ):
                    raise RuntimeError(
                        "source image RGB changed after source preparation for "
                        f"candidate {position}: {source_hash} != "
                        f"{candidate.source_canonical_rgb_sha256}"
                    )
                if source_hash in seen_source_hashes:
                    stats["engine_exact_source_hash_duplicate_count"] += 1
                    stats["exact_image_hash_duplicate_count"] += 1
                    screening["reason"] = "duplicate_source_canonical_rgb_sha256"
                else:
                    seen_source_hashes.add(source_hash)
                    if source_id is not None and source_id in config.exclusions.source_image_ids:
                        stats["excluded_source_image_id_count"] += 1
                        screening["reason"] = "excluded_prior_dataset_source_image_id"
                    elif source_hash in config.exclusions.source_canonical_rgb_sha256:
                        stats["excluded_source_hash_count"] += 1
                        screening["reason"] = "excluded_prior_dataset_source_hash"
                    else:
                        stats["unique_images_evaluated"] += 1
                        blind_prediction = runner.infer_blind(
                            candidate.prompt, candidate.legacy_temperature_zero
                        )
                        blind_correct = _correct(candidate, blind_prediction)
                        screening.update(
                            blind_prediction=blind_prediction,
                            blind_correct=blind_correct,
                            control="blind_black_image_control",
                        )
                        if blind_correct:
                            stats["blind_correct_count"] += 1
                            screening["reason"] = "blind_black_image_control_correct"
                        else:
                            stats["blind_wrong_count"] += 1
                            retained_index = len(samples)
                            stage_path = paths["stage_dir"] / (
                                f"candidate_{position:06d}_{source_hash[:16]}.jpg"
                            )
                            save_deterministic_jpeg(
                                source_image, stage_path, exclusive=True
                            )
                            # Every hash below and Full inference use the reopened
                            # deterministic JPEG, never the in-memory source image.
                            saved_byte_hash = sha256_file(stage_path)
                            saved_rgb_hash = canonical_rgb_sha256_path(stage_path)
                            saved_dhash = dhash_path(stage_path)
                            screening.update(
                                saved_image_sha256=saved_byte_hash,
                                saved_canonical_rgb_sha256=saved_rgb_hash,
                                perceptual_hash=saved_dhash,
                            )
                            if saved_rgb_hash in seen_saved_hashes:
                                stats["saved_exact_hash_duplicate_count"] += 1
                                screening["reason"] = (
                                    "duplicate_saved_canonical_rgb_sha256"
                                )
                                stage_path.unlink()
                            elif saved_rgb_hash in config.exclusions.saved_canonical_rgb_sha256:
                                stats["excluded_saved_hash_count"] += 1
                                screening["reason"] = "excluded_prior_dataset_saved_hash"
                                stage_path.unlink()
                            else:
                                with Image.open(stage_path) as reopened:
                                    reopened.load()
                                    full_prediction = runner.infer(
                                        reopened,
                                        candidate.prompt,
                                        candidate.legacy_temperature_zero,
                                    )
                                full_correct = _correct(candidate, full_prediction)
                                screening.update(
                                    full_prediction=full_prediction,
                                    full_correct=full_correct,
                                )
                                if not full_correct:
                                    stats["full_incorrect_count"] += 1
                                    screening["reason"] = "full_visual_incorrect"
                                    stage_path.unlink()
                                else:
                                    stats["full_correct_count"] += 1
                                    final_name = (
                                        f"{bundle.dataset_name}_{retained_index:04d}.jpg"
                                    )
                                    relative_name = f"images/{final_name}"
                                    final_path = paths["images_dir"] / final_name
                                    if final_path.is_symlink():
                                        stage_path.unlink()
                                        raise RuntimeError(
                                            "refusing to replace a symlinked final image: "
                                            f"{final_path}"
                                        )
                                    os.replace(stage_path, final_path)
                                    mapping = copy.deepcopy(candidate.mapping_payload)
                                    mapping["image_filename"] = relative_name
                                    sample = {
                                        "dataset": bundle.dataset_name,
                                        "retained_index": retained_index,
                                        "image_filename": relative_name,
                                        "source_dataset": candidate.source_dataset,
                                        "source_split": candidate.source_split,
                                        "source_index": candidate.source_index,
                                        "source_split_index": candidate.source_split_index,
                                        "source_question_id": candidate.source_question_id,
                                        "source_image_id": candidate.source_image_id,
                                        "source_canonical_rgb_sha256": source_hash,
                                        # The unqualified canonical hash is always
                                        # the final JPEG reopened from disk.  Source
                                        # pixels are tracked under their explicit key.
                                        "canonical_rgb_sha256": saved_rgb_hash,
                                        "saved_image_sha256": saved_byte_hash,
                                        "saved_canonical_rgb_sha256": saved_rgb_hash,
                                        "perceptual_hash": saved_dhash,
                                        "prompt": candidate.prompt,
                                        "blind_control": "blind_black_image_control",
                                        "blind_prediction": blind_prediction,
                                        "blind_correct": False,
                                        "full_prediction": full_prediction,
                                        "full_correct": True,
                                        "blind_black_image_control_prediction": (
                                            blind_prediction
                                        ),
                                        "blind_black_image_control_correct": False,
                                        "full_visual_prediction": full_prediction,
                                        "full_visual_correct": True,
                                        "mapping": mapping,
                                        "options": list(candidate.options),
                                        "ground_truth_letter": candidate.ground_truth_letter,
                                        "ground_truth_text": candidate.ground_truth_text,
                                        "legacy_temperature_zero": (
                                            candidate.legacy_temperature_zero
                                        ),
                                        "jpeg_settings": JPEG_SETTINGS,
                                        "provenance": candidate.provenance,
                                    }
                                    samples.append(sample)
                                    seen_saved_hashes.add(saved_rgb_hash)
                                    screening.update(
                                        reason="accepted",
                                        accepted=True,
                                        retained_index=retained_index,
                                        image_filename=relative_name,
                                    )
                                    if bundle.dataset_name == "ScienceQA_MC":
                                        split_counts = stats.setdefault(
                                            "split_composition",
                                            {"train": 0, "validation": 0, "test": 0},
                                        )
                                        split_counts[candidate.source_split] += 1

        if screening["reason"] is None:
            raise RuntimeError(f"candidate {position} ended without a screening reason")
        stats["retained_count"] = len(samples)
        evaluated = stats["unique_images_evaluated"]
        stats["acceptance_rate"] = len(samples) / evaluated if evaluated else 0.0
        if len(samples) >= config.target:
            stats["target_reached_source_index"] = candidate.source_index
            stats["target_reached_candidate_position"] = position
        stats["candidate_pool_exhausted"] = (
            position + 1 >= len(bundle.candidates)
            and len(samples) < config.target
        )

        # A durable explicit decision comes first.  Then the atomic checkpoint
        # commits it; resume discards only any append tail beyond that checkpoint.
        append_jsonl_durable(paths["screening"], screening)
        next_position = position + 1
        checkpoint = _checkpoint_payload(
            bundle,
            config,
            pool_fingerprint,
            exclusions_fingerprint,
            construction_protocol_fingerprint,
            next_position,
            stats,
            samples,
            seen_source_ids,
            seen_source_hashes,
            seen_saved_hashes,
        )
        atomic_write_json(paths["checkpoint"], checkpoint)
        _write_public_state(paths, stats, samples)
        if stats["scanned_candidates"] % 25 == 0 or len(samples) >= config.target:
            print(
                json.dumps(
                    {
                        "phase": "build_progress",
                        "dataset": bundle.dataset_name,
                        "candidate_position": position,
                        "candidate_pool": len(bundle.candidates),
                        "scanned": stats["scanned_candidates"],
                        "model_evaluated": stats["unique_images_evaluated"],
                        "retained": len(samples),
                        "target": config.target,
                        "last_reason": screening["reason"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        if len(samples) >= config.target:
            break

    stats["candidate_pool_exhausted"] = (
        next_position >= len(bundle.candidates) and len(samples) < config.target
    )
    # Persist the terminal exhaustion bit (or simply refresh the reached state).
    checkpoint = _checkpoint_payload(
        bundle,
        config,
        pool_fingerprint,
        exclusions_fingerprint,
        construction_protocol_fingerprint,
        next_position,
        stats,
        samples,
        seen_source_ids,
        seen_source_hashes,
        seen_saved_hashes,
    )
    atomic_write_json(paths["checkpoint"], checkpoint)
    _write_public_state(paths, stats, samples)
    result = _build_result(bundle, config, paths, stats, samples)
    if not result.target_reached and fail_if_short:
        raise InsufficientEligibleSamples(result)
    return result


__all__ = [
    "BuildConfig",
    "BuildExclusions",
    "BuildResult",
    "InsufficientEligibleSamples",
    "build_dataset",
    "collect_completed_sibling_exclusions",
    "load_completed_resume",
    "load_stored_build_exclusions",
    "resolve_output_root",
]
