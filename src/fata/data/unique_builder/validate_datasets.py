#!/usr/bin/env python3
"""Independent, strict validation for the rebuilt FATA evaluation datasets.

The validator deliberately recomputes image identities from the files on disk;
it does not trust counters or hashes emitted by the construction process.  It
also audits dHash-near pairs with a conservative pixel-level confirmation step.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import struct
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Sequence

from PIL import Image, ImageChops, ImageOps
from fata.utils.paths import assert_output_separate, resolve_owned_output_path


DATASET_NAMES = ("TextVQA_Open", "VQAv2_Open", "VQAv2_MC", "ScienceQA_MC")
MC_DATASETS = frozenset({"VQAv2_MC", "ScienceQA_MC"})
EXPECTED_SOURCE_DATASET = {
    "TextVQA_Open": "textvqa/textvqa",
    "VQAv2_Open": "official VQAv2",
    "VQAv2_MC": "official VQAv2",
    "ScienceQA_MC": "derek-thomas/ScienceQA",
}
EXPECTED_SOURCE_SPLITS = {
    "TextVQA_Open": frozenset({"validation"}),
    "VQAv2_Open": frozenset({"validation"}),
    "VQAv2_MC": frozenset({"validation"}),
    "ScienceQA_MC": frozenset({"train", "validation", "test"}),
}
EXPECTED_SOURCE_SPLIT_ORDER = {
    "TextVQA_Open": ("validation",),
    "VQAv2_Open": ("validation",),
    "VQAv2_MC": ("validation",),
    "ScienceQA_MC": ("train", "validation", "test"),
}
EXPECTED_SOURCE_QA_RECORDS = {
    "TextVQA_Open": 5000,
    "VQAv2_Open": 214354,
    "VQAv2_MC": 214354,
    "ScienceQA_MC": 21208,
}
EXPECTED_LEGACY_TEMPERATURE_ZERO = {
    "TextVQA_Open": True,
    "VQAv2_Open": False,
    "VQAv2_MC": False,
    "ScienceQA_MC": False,
}
EXPECTED_JPEG_SETTINGS = {
    "format": "JPEG",
    "quality": 75,
    "subsampling": 2,
    "optimize": False,
    "progressive": False,
}
RELIABLE_SOURCE_ID_DATASETS = frozenset(
    {"TextVQA_Open", "VQAv2_Open", "VQAv2_MC"}
)
REQUIRED_SAMPLE_FIELDS = (
    "dataset",
    "retained_index",
    "image_filename",
    "source_dataset",
    "source_split",
    "source_index",
    "source_split_index",
    "source_question_id",
    "source_image_id",
    "source_canonical_rgb_sha256",
    "canonical_rgb_sha256",
    "saved_image_sha256",
    "saved_canonical_rgb_sha256",
    "perceptual_hash",
    "prompt",
    "blind_control",
    "blind_prediction",
    "blind_correct",
    "full_prediction",
    "full_correct",
    "blind_black_image_control_prediction",
    "blind_black_image_control_correct",
    "full_visual_prediction",
    "full_visual_correct",
    "legacy_temperature_zero",
    "jpeg_settings",
    "provenance",
    "mapping",
)
SCREENING_IDENTITY_FIELDS = (
    "source_dataset",
    "source_split",
    "source_index",
    "source_split_index",
    "source_question_id",
    "source_image_id",
)
SCREENING_REASONS = frozenset(
    {
        "duplicate_source_image_id",
        "duplicate_source_canonical_rgb_sha256",
        "excluded_prior_dataset_source_image_id",
        "excluded_prior_dataset_source_hash",
        "blind_black_image_control_correct",
        "duplicate_saved_canonical_rgb_sha256",
        "excluded_prior_dataset_saved_hash",
        "full_visual_incorrect",
        "accepted",
    }
)
EVALUATED_SCREENING_REASONS = frozenset(
    {
        "blind_black_image_control_correct",
        "duplicate_saved_canonical_rgb_sha256",
        "excluded_prior_dataset_saved_hash",
        "full_visual_incorrect",
        "accepted",
    }
)
BLIND_WRONG_SCREENING_REASONS = EVALUATED_SCREENING_REASONS.difference(
    {"blind_black_image_control_correct"}
)
SAVED_SCREENING_REASONS = BLIND_WRONG_SCREENING_REASONS
FULL_SCREENING_REASONS = frozenset({"full_visual_incorrect", "accepted"})
FROZEN_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions. "
)
FROZEN_OPEN_SUFFIX = "\nAnswer the question using a single word or phrase. ASSISTANT:"
FROZEN_MC_SUFFIX = "Answer with the option's letter from the given choices directly. ASSISTANT:"
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
HEX16_RE = re.compile(r"^[0-9a-f]{16}$")
OPTION_LINE_RE = re.compile(r"^\s*([A-Z])\s*[.)]\s*(.*?)\s*$")
STAT_ALIASES: dict[str, tuple[str, ...]] = {
    "source_splits": ("source_splits", "source_split", "splits"),
    "source_qa_records_total": (
        "source_qa_records_total",
        "source_qa_total",
        "source_records_total",
        "source_qa_records",
    ),
    "source_unique_images_total": (
        "source_unique_images_total",
        "source_unique_images",
        "unique_source_images",
    ),
    "task_prefilter_unique_images": (
        "task_prefilter_unique_images",
        "eligible_unique_images",
        "prefilter_unique_images",
        "post_prefilter_unique_images",
    ),
    "invalid_metadata_count": ("invalid_metadata_count", "invalid_metadata"),
    "yes_no_removed_count": ("yes_no_removed_count", "yes_no_removed"),
    "missing_or_corrupt_images": (
        "missing_or_corrupt_images",
        "missing_or_corrupt_image_count",
        "image_errors",
    ),
    "source_id_duplicate_count": (
        "source_id_duplicate_count",
        "duplicate_source_image_ids",
        "source_id_duplicates",
    ),
    "exact_image_hash_duplicate_count": (
        "exact_image_hash_duplicate_count",
        "exact_hash_duplicates",
        "duplicate_rgb_images",
    ),
    "unique_images_evaluated": (
        "unique_images_evaluated",
        "evaluated_unique_images",
        "model_evaluated_unique_images",
        "model_evaluated",
    ),
    "blind_wrong_count": (
        "blind_wrong_count",
        "blind_control_wrong_count",
        "blind_wrong",
    ),
    "full_correct_count": (
        "full_correct_count",
        "full_correct_within_blind_wrong",
        "blind_wrong_full_correct_count",
        "full_correct",
    ),
    "target_reached_source_index": (
        "target_reached_source_index",
        "target_source_index",
        "source_index_at_target",
    ),
    "retained_count": ("retained_count", "final_retained", "retained"),
    "acceptance_rate": ("acceptance_rate", "retention_rate"),
    "split_composition": (
        "split_composition",
        "scienceqa_split_composition",
        "retained_by_split",
    ),
}
REQUESTED_NUMERIC_STATS = (
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


@dataclass(slots=True)
class ImageRecord:
    dataset: str
    mapping_index: int
    image_filename: str
    path: Path
    width: int
    height: int
    file_size: int
    saved_image_sha256: str
    canonical_rgb_sha256: str
    perceptual_hash: str
    source_dataset: Any = None
    source_split: Any = None
    source_index: Any = None
    source_question_id: Any = None
    source_image_id: Any = None
    source_canonical_rgb_sha256: Any = None

    @property
    def key(self) -> str:
        return f"{self.dataset}:{self.mapping_index}"

    def manifest_row(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "mapping_index": self.mapping_index,
            "image_filename": self.image_filename,
            "width": self.width,
            "height": self.height,
            "file_size": self.file_size,
            "saved_image_sha256": self.saved_image_sha256,
            "canonical_rgb_sha256": self.canonical_rgb_sha256,
            "perceptual_hash": self.perceptual_hash,
            "source_dataset": self.source_dataset,
            "source_split": self.source_split,
            "source_index": self.source_index,
            "source_question_id": self.source_question_id,
            "source_image_id": self.source_image_id,
            "source_canonical_rgb_sha256": self.source_canonical_rgb_sha256,
        }


class IssueLog:
    """Count every issue while keeping the JSON report reasonably small."""

    def __init__(self, max_examples: int = 100) -> None:
        self.counts: Counter[str] = Counter()
        self.examples: list[dict[str, Any]] = []
        self.max_examples = max_examples

    def add(self, code: str, message: str, **context: Any) -> None:
        self.counts[code] += 1
        if len(self.examples) < self.max_examples:
            example = {"code": code, "message": message}
            example.update(context)
            self.examples.append(example)

    def as_dict(self) -> dict[str, Any]:
        return {
            "total": sum(self.counts.values()),
            "counts": dict(sorted(self.counts.items())),
            "examples": self.examples,
            "examples_truncated": sum(self.counts.values()) > len(self.examples),
        }


def canonical_rgb_sha256(image: Image.Image) -> str:
    """Hash EXIF-transposed RGB pixels using the project's normative format."""

    normalized = ImageOps.exif_transpose(image).convert("RGB")
    normalized.load()
    width, height = normalized.size
    digest = hashlib.sha256()
    digest.update(b"RGB\0")
    digest.update(struct.pack(">QQ", width, height))
    digest.update(normalized.tobytes())
    return digest.hexdigest()


def saved_file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def dhash(image: Image.Image) -> str:
    """Return the 64-bit dHash used by the builder, as 16 lowercase hex digits."""

    normalized = ImageOps.exif_transpose(image).convert("RGB")
    gray = normalized.resize((9, 8), Image.Resampling.LANCZOS).convert("L")
    pixels = gray.tobytes()
    value = 0
    for row in range(8):
        offset = row * 9
        for column in range(8):
            value = (value << 1) | int(
                pixels[offset + column + 1] > pixels[offset + column]
            )
    return f"{value:016x}"


def hamming_distance(hash_a: str, hash_b: str) -> int:
    return (int(hash_a, 16) ^ int(hash_b, 16)).bit_count()


def _atomic_write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path = resolve_owned_output_path(path.parent, path.name)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        fd = -1
        path = resolve_owned_output_path(path.parent, path.name)
        os.replace(temporary, path)
    finally:
        if fd >= 0:
            os.close(fd)
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(path: Path, value: Any) -> None:
    _atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    _atomic_write_text(
        path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    )


def _atomic_write_csv(
    path: Path, rows: Sequence[dict[str, Any]], columns: Sequence[str]
) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(columns), extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    _atomic_write_text(path, buffer.getvalue())


def _load_jsonl(path: Path, issues: IssueLog, prefix: str) -> list[dict[str, Any]]:
    if path.is_symlink():
        issues.add(
            f"symlink_{prefix}_file",
            f"required JSONL artifact must not be a symlink: {path}",
        )
        return []
    if not path.is_file():
        issues.add(f"missing_{prefix}_file", f"required JSONL file does not exist: {path}")
        return []
    rows: list[dict[str, Any]] = []
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeError) as error:
        issues.add(f"unreadable_{prefix}_file", str(error), path=str(path))
        return rows
    if raw and not raw.endswith(b"\n"):
        issues.add(
            f"missing_final_newline_{prefix}",
            "strict JSONL artifact must end with a newline",
            path=str(path),
        )
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            issues.add(
                f"blank_{prefix}_line",
                "blank JSONL lines are not permitted in strict validation",
                line=line_number,
            )
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            issues.add(
                f"invalid_{prefix}_json",
                str(error),
                line=line_number,
            )
            continue
        if not isinstance(value, dict):
            issues.add(
                f"non_object_{prefix}_row",
                "each JSONL row must be an object",
                line=line_number,
            )
            continue
        rows.append(value)
    return rows


def _load_json(path: Path, issues: IssueLog, prefix: str) -> dict[str, Any]:
    if path.is_symlink():
        issues.add(
            f"symlink_{prefix}_file",
            f"required JSON artifact must not be a symlink: {path}",
        )
        return {}
    if not path.is_file():
        issues.add(f"missing_{prefix}_file", f"required JSON file does not exist: {path}")
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        issues.add(f"invalid_{prefix}_json", str(error), path=str(path))
        return {}
    if not isinstance(value, dict):
        issues.add(f"non_object_{prefix}", "top-level JSON value must be an object")
        return {}
    return value


def _normalize_text(value: Any) -> str:
    return " ".join(str(value).split()).casefold()


def _is_nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _mapping_signature(mapping: dict[str, Any]) -> str:
    return json.dumps(mapping, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _stable_json_digest(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _stable_source_metadata(value: Any, parent_key: str | None = None) -> Any:
    if isinstance(value, dict):
        is_vqa_archive = (
            parent_key in {"question_zip", "annotation_zip"}
            and {"path", "url", "size", "sha256"}.issubset(value)
        )
        return {
            key: _stable_source_metadata(child, key)
            for key, child in value.items()
            if not (
                is_vqa_archive
                and key in {"cache_status", "attempt", "cached_source"}
            )
        }
    if isinstance(value, list):
        return [_stable_source_metadata(child, parent_key) for child in value]
    return value


def _validate_builder_manifests(
    dataset: str,
    expected_count: int,
    source: dict[str, Any],
    checkpoint: dict[str, Any],
    stats: dict[str, Any],
    candidates: list[dict[str, Any]],
    screening: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    mappings: list[dict[str, Any]],
    issues: IssueLog,
) -> None:
    expected_source_fields = {
        "dataset",
        "candidate_pool_fingerprint",
        "canonical_hash",
        "jpeg_settings",
        "construction_protocol",
        "construction_protocol_fingerprint",
        "source_metadata",
        "source_metadata_stable_fingerprint",
        "initial_selection_stats",
        "initial_selection_stats_fingerprint",
        "build_exclusions",
        "exclusions_fingerprint",
    }
    expected_checkpoint_fields = {
        "format_version",
        "dataset",
        "target",
        "seed",
        "candidate_pool_fingerprint",
        "exclusions_fingerprint",
        "construction_protocol_fingerprint",
        "next_candidate_position",
        "selection_stats",
        "samples",
        "seen_source_image_ids",
        "seen_source_canonical_rgb_sha256",
        "seen_saved_canonical_rgb_sha256",
    }
    if set(source) != expected_source_fields:
        issues.add(
            "source_manifest_schema_mismatch",
            "source manifest must have the exact frozen field set",
            actual=sorted(source),
            expected=sorted(expected_source_fields),
        )
    if set(checkpoint) != expected_checkpoint_fields:
        issues.add(
            "checkpoint_schema_mismatch",
            "checkpoint must have the exact frozen field set",
            actual=sorted(checkpoint),
            expected=sorted(expected_checkpoint_fields),
        )
    if type(checkpoint.get("format_version")) is not int or checkpoint.get(
        "format_version"
    ) != 2:
        issues.add(
            "checkpoint_format_version_mismatch",
            "checkpoint format_version must equal the frozen version 2",
            actual=checkpoint.get("format_version"),
        )
    if source.get("canonical_hash") != (
        "SHA256(b'RGB\\0' + width_uint64_be + height_uint64_be + "
        "EXIF-transposed RGB pixel bytes)"
    ):
        issues.add(
            "source_canonical_hash_contract_mismatch",
            "source manifest canonical-hash contract is not frozen",
        )
    if source.get("jpeg_settings") != EXPECTED_JPEG_SETTINGS:
        issues.add(
            "source_jpeg_contract_mismatch",
            "source manifest JPEG contract is not frozen",
        )
    for label, document in (("source", source), ("checkpoint", checkpoint)):
        if document.get("dataset") != dataset:
            issues.add(
                f"{label}_dataset_mismatch",
                f"{label} manifest dataset does not match directory",
                actual=document.get("dataset"),
            )
    for label, value in (
        ("checkpoint target", checkpoint.get("target")),
        ("selection_stats target", stats.get("target")),
    ):
        if type(value) is not int or value != expected_count:
            issues.add(
                "builder_target_mismatch",
                f"{label} must equal the exact validation target",
                actual=value,
                expected=expected_count,
            )
    checkpoint_seed = checkpoint.get("seed")
    stats_seed = stats.get("seed")
    if (
        type(checkpoint_seed) is not int
        or checkpoint_seed < 0
        or type(stats_seed) is not int
        or stats_seed != checkpoint_seed
    ):
        issues.add(
            "builder_seed_mismatch",
            "checkpoint and selection_stats must contain the same non-negative integer seed",
            checkpoint=checkpoint_seed,
            selection_stats=stats_seed,
        )
    try:
        checkpoint_samples_equal = _mapping_signature(
            {"samples": checkpoint.get("samples")}
        ) == _mapping_signature({"samples": samples})
        checkpoint_stats_equal = _mapping_signature(
            {"stats": checkpoint.get("selection_stats")}
        ) == _mapping_signature({"stats": stats})
    except (TypeError, ValueError):
        checkpoint_samples_equal = False
        checkpoint_stats_equal = False
    if not checkpoint_samples_equal:
        issues.add(
            "checkpoint_samples_mismatch",
            "checkpoint samples must exactly equal the public sample sidecar",
        )
    if not checkpoint_stats_equal:
        issues.add(
            "checkpoint_selection_stats_mismatch",
            "checkpoint selection_stats must exactly equal the public stats",
        )
    next_position = checkpoint.get("next_candidate_position")
    if (
        type(next_position) is not int
        or next_position != len(screening)
        or not 0 <= next_position <= len(candidates)
    ):
        issues.add(
            "checkpoint_position_mismatch",
            "next_candidate_position must equal the exact committed screening prefix",
            actual=next_position,
            screening_rows=len(screening),
            candidate_rows=len(candidates),
        )

    candidate_rows: list[dict[str, Any]] = []
    candidate_positions_valid = True
    for position, candidate in enumerate(candidates):
        if candidate.get("candidate_position") != position or isinstance(
            candidate.get("candidate_position"), bool
        ):
            candidate_positions_valid = False
        candidate_rows.append(
            {
                "source_dataset": candidate.get("source_dataset"),
                "source_split": candidate.get("source_split"),
                "source_index": candidate.get("source_index"),
                "source_split_index": candidate.get("source_split_index"),
                "source_question_id": candidate.get("source_question_id"),
                "source_image_id": candidate.get("source_image_id"),
                "source_canonical_rgb_sha256": candidate.get(
                    "source_canonical_rgb_sha256"
                ),
                "task_type": candidate.get("task_type"),
                "question": candidate.get("question"),
                "answers": candidate.get("answers"),
                "options": candidate.get("options"),
                "ground_truth_letter": candidate.get("ground_truth_letter"),
                "ground_truth_text": candidate.get("ground_truth_text"),
                "prompt": candidate.get("prompt"),
                "mapping_payload": candidate.get("mapping_template"),
                "legacy_temperature_zero": candidate.get(
                    "legacy_temperature_zero"
                ),
                "provenance": candidate.get("provenance"),
            }
        )
    if not candidate_positions_valid:
        issues.add(
            "candidate_position_discontinuity",
            "candidate pool positions must be exact zero-based order",
        )
    recomputed_pool = _stable_json_digest(candidate_rows)
    source_pool = source.get("candidate_pool_fingerprint")
    checkpoint_pool = checkpoint.get("candidate_pool_fingerprint")
    if (
        not isinstance(source_pool, str)
        or not HEX64_RE.fullmatch(source_pool)
        or source_pool != checkpoint_pool
        or source_pool != recomputed_pool
    ):
        issues.add(
            "candidate_pool_fingerprint_mismatch",
            "source/checkpoint fingerprint must equal the strict candidate-pool body",
        )

    protocol = source.get("construction_protocol")
    protocol_fingerprint = source.get("construction_protocol_fingerprint")
    if (
        not isinstance(protocol, dict)
        or not isinstance(protocol_fingerprint, str)
        or not HEX64_RE.fullmatch(protocol_fingerprint)
        or protocol_fingerprint != checkpoint.get("construction_protocol_fingerprint")
        or _stable_json_digest(protocol) != protocol_fingerprint
    ):
        issues.add(
            "construction_protocol_fingerprint_mismatch",
            "source/checkpoint protocol fingerprint must equal the protocol body",
        )

    for body_key, fingerprint_key in (
        ("source_metadata", "source_metadata_stable_fingerprint"),
        ("initial_selection_stats", "initial_selection_stats_fingerprint"),
        ("build_exclusions", "exclusions_fingerprint"),
    ):
        body = source.get(body_key)
        fingerprint = source.get(fingerprint_key)
        stable_body = (
            _stable_source_metadata(body) if body_key == "source_metadata" else body
        )
        if (
            not isinstance(body, dict)
            or not isinstance(fingerprint, str)
            or not HEX64_RE.fullmatch(fingerprint)
            or _stable_json_digest(stable_body) != fingerprint
        ):
            issues.add(
                f"{body_key}_fingerprint_mismatch",
                f"{body_key} must be present and match its stable fingerprint",
            )
    if source.get("exclusions_fingerprint") != checkpoint.get(
        "exclusions_fingerprint"
    ):
        issues.add(
            "checkpoint_exclusions_fingerprint_mismatch",
            "checkpoint exclusions fingerprint must match source manifest",
        )

    expected_source_ids = {
        str(row["source_image_id"])
        for row in screening
        if row.get("source_image_id") is not None
    }
    expected_source_hashes = {
        row["source_canonical_rgb_sha256"]
        for row in screening
        if isinstance(row.get("source_canonical_rgb_sha256"), str)
        and HEX64_RE.fullmatch(row["source_canonical_rgb_sha256"])
    }
    expected_saved_hashes = {
        sample["saved_canonical_rgb_sha256"]
        for sample in samples
        if isinstance(sample.get("saved_canonical_rgb_sha256"), str)
        and HEX64_RE.fullmatch(sample["saved_canonical_rgb_sha256"])
    }
    for key, expected in (
        ("seen_source_image_ids", expected_source_ids),
        ("seen_source_canonical_rgb_sha256", expected_source_hashes),
        ("seen_saved_canonical_rgb_sha256", expected_saved_hashes),
    ):
        raw = checkpoint.get(key)
        valid = (
            isinstance(raw, list)
            and all(isinstance(value, str) and value for value in raw)
            and raw == sorted(set(raw))
            and set(raw) == expected
        )
        if not valid:
            issues.add(
                f"checkpoint_{key}_mismatch",
                f"checkpoint {key} must exactly match committed artifacts",
            )

    if len(samples) == expected_count and screening:
        accepted_positions = [
            position
            for position, row in enumerate(screening)
            if row.get("accepted") is True and row.get("reason") == "accepted"
        ]
        if len(accepted_positions) != expected_count or accepted_positions[-1] != len(
            screening
        ) - 1:
            issues.add(
                "checkpoint_completion_boundary_mismatch",
                "the target must first be reached on the final committed row",
            )


def _safe_image_reference(
    dataset_dir: Path, value: Any
) -> tuple[str | None, Path | None, str | None]:
    if not isinstance(value, str) or not value:
        return None, None, "image_filename must be a non-empty string"
    if "\\" in value:
        return None, None, "image_filename must use POSIX separators"
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or "." in pure.parts:
        return None, None, "image_filename must be a safe relative path"
    if len(pure.parts) != 2 or pure.parts[0] != "images":
        return None, None, "image_filename must have the form images/<name>.jpg"
    if pure.suffix.casefold() != ".jpg":
        return None, None, "image_filename must have the exact .jpg suffix"
    path = dataset_dir.joinpath(*pure.parts)
    try:
        path.resolve(strict=False).relative_to(dataset_dir.resolve(strict=False))
    except ValueError:
        return None, None, "image_filename escapes its dataset directory"
    return pure.as_posix(), path, None


def _validate_open_mapping(mapping: dict[str, Any], index: int, issues: IssueLog) -> bool:
    valid = True
    if mapping.get("type") != "open":
        issues.add(
            "invalid_mapping_type", "open dataset row must have type='open'", row=index
        )
        valid = False
    if not _is_nonempty_string(mapping.get("question")):
        issues.add("invalid_question", "question must be a non-empty string", row=index)
        valid = False
    answers = mapping.get("answers")
    if not isinstance(answers, list) or not answers or not all(
        _is_nonempty_string(answer) for answer in answers
    ):
        issues.add(
            "invalid_answers",
            "open answers must be a non-empty list of non-empty strings",
            row=index,
        )
        valid = False
    return valid


def _question_options(question: str) -> list[tuple[str, str]]:
    markers = list(re.finditer(r"(?im)^\s*Options\s*:\s*$", question))
    if not markers:
        return []
    suffix = question[markers[-1].end() :]
    parsed: list[tuple[str, str]] = []
    for line in suffix.splitlines():
        if not line.strip():
            continue
        match = OPTION_LINE_RE.match(line)
        if not match:
            return []
        parsed.append((match.group(1), match.group(2)))
    return parsed


def _validate_mc_mapping(
    dataset: str, mapping: dict[str, Any], index: int, issues: IssueLog
) -> bool:
    valid = True
    if mapping.get("type") != "multiple_choice":
        issues.add(
            "invalid_mapping_type",
            "MC dataset row must have type='multiple_choice'",
            row=index,
        )
        valid = False
    question = mapping.get("question")
    if not _is_nonempty_string(question):
        issues.add("invalid_question", "question must be a non-empty string", row=index)
        return False
    options = mapping.get("options")
    expected_count = 4 if dataset == "VQAv2_MC" else None
    if not isinstance(options, list) or not all(
        _is_nonempty_string(option) for option in options
    ):
        issues.add(
            "invalid_options", "options must be a list of non-empty strings", row=index
        )
        return False
    if expected_count is not None and len(options) != expected_count:
        issues.add(
            "invalid_option_count",
            f"{dataset} requires exactly {expected_count} options",
            row=index,
            actual=len(options),
        )
        valid = False
    if dataset == "ScienceQA_MC" and not 2 <= len(options) <= 6:
        issues.add(
            "invalid_option_count",
            "ScienceQA_MC requires between two and six options",
            row=index,
            actual=len(options),
        )
        valid = False
    displayed_options = [_normalize_text(option) for option in options]
    construction_normalized_options = [_legacy_vqa_process(option) for option in options]
    if any(not value for value in construction_normalized_options) or len(
        set(construction_normalized_options)
    ) != len(construction_normalized_options):
        issues.add(
            "duplicate_normalized_options",
            "MC options must be non-empty and unique under the frozen construction normalizer",
            row=index,
        )
        valid = False
    expected_letters = [chr(ord("A") + offset) for offset in range(len(options))]
    displayed = _question_options(question)
    if [letter for letter, _ in displayed] != expected_letters or [
        _normalize_text(text) for _, text in displayed
    ] != displayed_options:
        issues.add(
            "question_option_mismatch",
            "question's rendered option block does not match options",
            row=index,
        )
        valid = False
    answers = mapping.get("answers")
    if not isinstance(answers, list) or not answers:
        issues.add("invalid_answers", "MC answers must contain an answer letter", row=index)
        return False
    answer_letters = [str(answer).strip().upper() for answer in answers]
    if any(letter not in expected_letters for letter in answer_letters):
        issues.add(
            "answer_letter_out_of_range",
            "MC answer letter is outside the option range",
            row=index,
            answers=answer_letters,
        )
        return False
    if len(set(answer_letters)) != 1:
        issues.add(
            "inconsistent_answer_letters",
            "all MC answer entries must identify the same option",
            row=index,
        )
        valid = False
    answer_index = ord(answer_letters[0]) - ord("A")
    ground_truth = mapping.get("ground_truth_text")
    if not _is_nonempty_string(ground_truth):
        issues.add(
            "invalid_ground_truth_text",
            "ground_truth_text must be a non-empty string",
            row=index,
        )
        valid = False
    elif _legacy_vqa_process(ground_truth) != construction_normalized_options[answer_index]:
        issues.add(
            "answer_text_letter_mismatch",
            "ground_truth_text does not match the option selected by answers",
            row=index,
        )
        valid = False
    if (
        construction_normalized_options.count(
            construction_normalized_options[answer_index]
        )
        != 1
    ):
        issues.add(
            "correct_answer_not_unique",
            "the normalized correct answer must occur exactly once",
            row=index,
        )
        valid = False
    return valid


def _legacy_vqa_process(text: Any) -> str:
    """Independent copy of the construction-time open-answer normalizer."""

    value = str(text).lower().replace("\n", " ").replace("\r", " ")
    value = re.sub(r"([^\w\s])", r" ", value)
    words = [word for word in value.split() if word not in ("a", "an", "the")]
    number_map = {
        "zero": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
    }
    return " ".join(number_map.get(word, word) for word in words)


def _construction_correct(
    dataset: str, mapping: dict[str, Any], prediction: str
) -> bool | None:
    """Independently recompute the frozen construction scorer."""

    if dataset not in MC_DATASETS:
        answers = mapping.get("answers")
        if not isinstance(answers, list) or not answers:
            return None
        pred = _legacy_vqa_process(prediction)
        truths = [_legacy_vqa_process(answer) for answer in answers]
        if pred in truths:
            return True
        return any(truth and truth in pred for truth in truths)

    options = mapping.get("options")
    answers = mapping.get("answers")
    ground_truth = mapping.get("ground_truth_text")
    if (
        not isinstance(options, list)
        or not options
        or not isinstance(answers, list)
        or not answers
        or not isinstance(ground_truth, str)
    ):
        return None
    letter = str(answers[0]).strip().lower()
    # Preserve the original per-script parser ranges, including ScienceQA's
    # deliberately wider A--F parser when a question has fewer than six choices.
    max_letter = "f" if dataset == "ScienceQA_MC" else "d"
    pred = str(prediction).strip().lower()
    match = re.search(
        rf"(?i)(?:^|\s|\()(option\s+)?([a-{max_letter}])(?:\)|\.|:|\s|$)", pred
    )
    if match:
        extracted = match.group(2).lower()
    else:
        if ground_truth.lower() in pred:
            return True
        extracted = pred[0] if pred else ""
    return extracted == letter


def _expected_construction_prompt(
    dataset: str, mapping_template: dict[str, Any]
) -> str | None:
    """Reconstruct the frozen legacy prompt from candidate mapping content."""

    question = mapping_template.get("question")
    if not isinstance(question, str):
        return None
    if dataset not in MC_DATASETS:
        return (
            f"{FROZEN_SYSTEM_PROMPT}USER: <image>\n{question}"
            f"{FROZEN_OPEN_SUFFIX}"
        )
    if dataset == "ScienceQA_MC":
        # ScienceQA's rendered question intentionally ends in a newline.
        return f"{FROZEN_SYSTEM_PROMPT}USER: <image>\n{question}{FROZEN_MC_SUFFIX}"
    return (
        f"{FROZEN_SYSTEM_PROMPT}USER: <image>\n{question}\n"
        f"{FROZEN_MC_SUFFIX}"
    )


def _flatten_keys(value: Any, output: dict[str, Any] | None = None) -> dict[str, Any]:
    output = {} if output is None else output
    if not isinstance(value, dict):
        return output
    for key, child in value.items():
        normalized = re.sub(r"[^a-z0-9]+", "_", str(key).casefold()).strip("_")
        output.setdefault(normalized, child)
        if isinstance(child, dict):
            _flatten_keys(child, output)
    return output


def extract_selection_stats(stats: dict[str, Any]) -> dict[str, Any]:
    """Normalize known selection-stat spellings without inventing values."""

    flattened = _flatten_keys(stats)
    extracted: dict[str, Any] = {}
    for canonical, aliases in STAT_ALIASES.items():
        for alias in aliases:
            normalized = re.sub(r"[^a-z0-9]+", "_", alias.casefold()).strip("_")
            if normalized in flattened:
                extracted[canonical] = flattened[normalized]
                break
    return extracted


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip().rstrip("%")
        try:
            parsed = float(stripped)
        except ValueError:
            return None
        return parsed if math.isfinite(parsed) else None
    return None


def _validate_stats(
    dataset: str,
    stats: dict[str, Any],
    retained_count: int,
    issues: IssueLog,
) -> dict[str, Any]:
    extracted = extract_selection_stats(stats)
    if "source_splits" not in extracted:
        issues.add(
            "missing_selection_stat",
            "selection statistics are missing requested field 'source_splits'",
            field="source_splits",
        )
    elif not (
        _is_nonempty_string(extracted["source_splits"])
        or (
            isinstance(extracted["source_splits"], list)
            and extracted["source_splits"]
            and all(_is_nonempty_string(value) for value in extracted["source_splits"])
        )
    ):
        issues.add(
            "invalid_selection_stat",
            "source_splits must be a non-empty string or list of non-empty strings",
            field="source_splits",
        )
    for field in REQUESTED_NUMERIC_STATS:
        if field not in extracted:
            issues.add(
                "missing_selection_stat",
                f"selection statistics are missing requested field {field!r}",
                field=field,
            )
            continue
        value = extracted[field]
        if field == "target_reached_source_index" and value is None:
            if retained_count > 0:
                issues.add(
                    "invalid_selection_stat",
                    "target_reached_source_index is null despite retained samples",
                    field=field,
                )
            continue
        parsed = _number(value)
        if parsed is None:
            issues.add(
                "invalid_selection_stat",
                f"selection statistic {field!r} must be numeric",
                field=field,
                value=value,
            )
        elif field == "acceptance_rate":
            if parsed < 0 or parsed > 100:
                issues.add(
                    "invalid_selection_stat",
                    "acceptance_rate must be in [0, 1] or percentage form in [0, 100]",
                    field=field,
                    value=value,
                )
        elif parsed < 0 or not parsed.is_integer():
            issues.add(
                "invalid_selection_stat",
                f"selection statistic {field!r} must be a non-negative integer",
                field=field,
                value=value,
            )

    numeric = {key: _number(value) for key, value in extracted.items()}
    reported_retained = numeric.get("retained_count")
    if reported_retained is not None and reported_retained != retained_count:
        issues.add(
            "selection_retained_count_mismatch",
            "selection stats retained_count differs from sidecar/mapping count",
            reported=reported_retained,
            actual=retained_count,
        )
    ordered_relations = (
        ("source_qa_records_total", "source_unique_images_total"),
        ("source_unique_images_total", "task_prefilter_unique_images"),
        ("task_prefilter_unique_images", "unique_images_evaluated"),
        ("unique_images_evaluated", "blind_wrong_count"),
        ("blind_wrong_count", "full_correct_count"),
        ("full_correct_count", "retained_count"),
    )
    for larger, smaller in ordered_relations:
        left, right = numeric.get(larger), numeric.get(smaller)
        if left is not None and right is not None and left < right:
            issues.add(
                "selection_count_invariant_failure",
                f"expected {larger} >= {smaller}",
                larger_value=left,
                smaller_value=right,
            )
    evaluated = numeric.get("unique_images_evaluated")
    acceptance = numeric.get("acceptance_rate")
    if evaluated is not None and acceptance is not None and evaluated > 0:
        expected = retained_count / evaluated
        normalized_acceptance = acceptance / 100.0 if acceptance > 1.0 else acceptance
        if not math.isclose(normalized_acceptance, expected, rel_tol=1e-4, abs_tol=1e-4):
            issues.add(
                "acceptance_rate_mismatch",
                "acceptance_rate must equal retained_count / unique_images_evaluated",
                reported=acceptance,
                expected=expected,
            )
    if dataset == "ScienceQA_MC":
        composition = extracted.get("split_composition")
        if not isinstance(composition, dict):
            issues.add(
                "missing_scienceqa_split_composition",
                "ScienceQA selection stats require a split_composition object",
            )
        else:
            expected_splits = {"train", "validation", "test"}
            missing = expected_splits.difference(composition)
            if missing:
                issues.add(
                    "incomplete_scienceqa_split_composition",
                    "ScienceQA split composition must include train, validation, and test",
                    missing=sorted(missing),
                )
            values = [_number(composition.get(split)) for split in expected_splits]
            if any(
                value is None or value < 0 or not value.is_integer()
                for value in values
            ):
                issues.add(
                    "invalid_scienceqa_split_composition",
                    "ScienceQA split counts must be non-negative integers",
                    split_composition=composition,
                )
            if all(value is not None for value in values) and sum(values) != retained_count:
                issues.add(
                    "scienceqa_split_composition_mismatch",
                    "ScienceQA retained split counts do not sum to retained_count",
                    split_composition=composition,
                    retained_count=retained_count,
                )
    return extracted


def _retained_screening_record(row: dict[str, Any]) -> tuple[bool, bool]:
    """Return (decision_was_expressible, row_is_retained)."""

    for key in ("retained", "accepted", "selected"):
        if key in row and isinstance(row[key], bool):
            return True, row[key]
    for key in ("decision", "status", "outcome", "reason"):
        if key in row and isinstance(row[key], str):
            normalized = row[key].strip().casefold().replace("-", "_").replace(" ", "_")
            accepted = normalized in {
                "accepted",
                "accept",
                "retained",
                "selected",
                "full_correct",
            }
            return True, accepted
    return False, False


def _valid_source_question_id(value: Any) -> bool:
    return (isinstance(value, int) and not isinstance(value, bool)) or _is_nonempty_string(
        value
    )


def _validate_candidate_pool(
    dataset: str,
    candidates: Sequence[dict[str, Any]],
    issues: IssueLog,
) -> None:
    """Validate the immutable pre-model pool independently of screening rows."""

    if not candidates:
        issues.add(
            "empty_candidate_pool_manifest",
            "candidate-pool manifest must contain every pre-model representative",
        )
        return
    seen_source_ids: set[str] = set()
    seen_source_hashes: set[str] = set()
    expected_task_type = "multiple_choice" if dataset in MC_DATASETS else "open"
    for index, candidate in enumerate(candidates):
        position = candidate.get("candidate_position")
        if isinstance(position, bool) or position != index:
            issues.add(
                "candidate_pool_position_discontinuity",
                "candidate_position must be the zero-based JSONL row index",
                row=index,
                candidate_position=position,
            )
        if candidate.get("source_dataset") != EXPECTED_SOURCE_DATASET[dataset]:
            issues.add(
                "candidate_pool_source_dataset_mismatch",
                "candidate source_dataset does not match the frozen upstream source",
                row=index,
                expected=EXPECTED_SOURCE_DATASET[dataset],
                actual=candidate.get("source_dataset"),
            )
        split = candidate.get("source_split")
        if split not in EXPECTED_SOURCE_SPLITS[dataset]:
            issues.add(
                "candidate_pool_source_split_mismatch",
                "candidate source_split is outside the frozen source split(s)",
                row=index,
                actual=split,
                expected=sorted(EXPECTED_SOURCE_SPLITS[dataset]),
            )
        for field in ("source_index", "source_split_index"):
            value = candidate.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                issues.add(
                    "candidate_pool_invalid_source_index",
                    f"candidate {field} must be a non-negative integer",
                    row=index,
                    field=field,
                    value=value,
                )
        if not _valid_source_question_id(candidate.get("source_question_id")):
            issues.add(
                "candidate_pool_invalid_question_id",
                "candidate source_question_id must be a non-empty string or integer",
                row=index,
                value=candidate.get("source_question_id"),
            )
        source_id = candidate.get("source_image_id")
        if dataset in RELIABLE_SOURCE_ID_DATASETS:
            if source_id is None or not str(source_id).strip():
                issues.add(
                    "candidate_pool_missing_source_image_id",
                    "candidate requires a reliable non-empty source_image_id",
                    row=index,
                )
            else:
                normalized_source_id = str(source_id)
                if normalized_source_id in seen_source_ids:
                    issues.add(
                        "candidate_pool_duplicate_source_image_id",
                        "pre-model representatives repeat a reliable source image ID",
                        row=index,
                        source_image_id=source_id,
                    )
                seen_source_ids.add(normalized_source_id)
        elif source_id is not None:
            issues.add(
                "candidate_pool_unexpected_source_image_id",
                "ScienceQA candidates must use source RGB identity, not a fabricated image ID",
                row=index,
                source_image_id=source_id,
            )
        source_hash = candidate.get("source_canonical_rgb_sha256")
        if source_hash is not None:
            if not isinstance(source_hash, str) or not HEX64_RE.fullmatch(source_hash):
                issues.add(
                    "candidate_pool_invalid_source_hash",
                    "candidate source hash must be null or lowercase SHA-256",
                    row=index,
                    value=source_hash,
                )
            else:
                if source_hash in seen_source_hashes:
                    issues.add(
                        "candidate_pool_duplicate_source_hash",
                        "pre-model representatives repeat canonical source RGB content",
                        row=index,
                        source_canonical_rgb_sha256=source_hash,
                    )
                seen_source_hashes.add(source_hash)
        if candidate.get("task_type") != expected_task_type:
            issues.add(
                "candidate_pool_task_type_mismatch",
                "candidate task_type does not match the dataset",
                row=index,
                expected=expected_task_type,
                actual=candidate.get("task_type"),
            )
        mapping_template = candidate.get("mapping_template")
        if not isinstance(mapping_template, dict):
            issues.add(
                "candidate_pool_invalid_mapping_template",
                "candidate mapping_template must be an object",
                row=index,
            )
            continue
        if mapping_template.get("image_filename") != "":
            issues.add(
                "candidate_pool_nonempty_image_filename",
                "pre-model mapping_template image_filename must be empty",
                row=index,
                value=mapping_template.get("image_filename"),
            )
        expected_prompt = _expected_construction_prompt(dataset, mapping_template)
        if expected_prompt is None or candidate.get("prompt") != expected_prompt:
            issues.add(
                "candidate_pool_prompt_mismatch",
                "candidate prompt does not exactly match the frozen legacy prompt bytes",
                row=index,
            )
        expected_fields = {
            "question": mapping_template.get("question"),
            "answers": mapping_template.get("answers"),
            "options": mapping_template.get("options", []),
            "ground_truth_text": mapping_template.get("ground_truth_text"),
        }
        for field, expected in expected_fields.items():
            if candidate.get(field) != expected:
                issues.add(
                    "candidate_pool_mapping_field_mismatch",
                    "candidate scoring field differs from its mapping_template",
                    row=index,
                    field=field,
                    candidate_value=candidate.get(field),
                    mapping_value=expected,
                )
        answers = mapping_template.get("answers")
        expected_letter = (
            str(answers[0]).strip().upper()
            if dataset in MC_DATASETS and isinstance(answers, list) and answers
            else None
        )
        if candidate.get("ground_truth_letter") != expected_letter:
            issues.add(
                "candidate_pool_mapping_field_mismatch",
                "candidate ground_truth_letter differs from its mapping_template answer",
                row=index,
                field="ground_truth_letter",
                candidate_value=candidate.get("ground_truth_letter"),
                mapping_value=expected_letter,
            )
        expected_temperature_flag = EXPECTED_LEGACY_TEMPERATURE_ZERO[dataset]
        if candidate.get("legacy_temperature_zero") is not expected_temperature_flag:
            issues.add(
                "candidate_pool_invalid_temperature_flag",
                "candidate legacy_temperature_zero does not match the frozen task setting",
                row=index,
                expected=expected_temperature_flag,
                actual=candidate.get("legacy_temperature_zero"),
            )


def _issue_unexpected_fields(
    row: dict[str, Any],
    fields: Sequence[str],
    issues: IssueLog,
    *,
    row_index: int,
    phase: str,
) -> None:
    unexpected = [field for field in fields if field in row]
    if unexpected:
        issues.add(
            "screening_reason_state_mismatch",
            f"{phase} screening row contains fields that the builder cannot emit",
            row=row_index,
            fields=unexpected,
            reason=row.get("reason"),
        )


def _validate_screening_manifest(
    dataset: str,
    screening: Sequence[dict[str, Any]],
    candidates: Sequence[dict[str, Any]],
    samples: Sequence[dict[str, Any]],
    issues: IssueLog,
) -> dict[str, Any]:
    """Validate every committed decision and independently derive its counters."""

    reason_counts: Counter[str] = Counter()
    accepted_count = 0
    accepted_indices: set[int] = set()
    evaluated_count = 0
    blind_wrong_count = 0
    full_correct_count = 0
    source_hash_count = 0
    last_accepted_position: int | None = None
    last_accepted_source_index: int | None = None
    seen_source_ids: set[str] = set()
    seen_source_hashes: set[str] = set()
    accepted_saved_hashes: set[str] = set()

    if not screening:
        issues.add(
            "empty_screening_manifest",
            "screening manifest must record candidate acceptance/rejection decisions",
        )
    if len(screening) > len(candidates):
        issues.add(
            "screening_exceeds_candidate_pool",
            "screening contains more decisions than immutable pool candidates",
            screening_rows=len(screening),
            candidate_pool_rows=len(candidates),
        )

    for index, row in enumerate(screening):
        position = row.get("candidate_position")
        if isinstance(position, bool) or position != index:
            issues.add(
                "screening_candidate_position_discontinuity",
                "candidate_position must be the contiguous zero-based JSONL row index",
                row=index,
                candidate_position=position,
            )
        candidate = candidates[index] if index < len(candidates) else None
        if candidate is not None:
            for field in SCREENING_IDENTITY_FIELDS:
                if row.get(field) != candidate.get(field):
                    issues.add(
                        "screening_candidate_identity_mismatch",
                        "screening identity differs from the immutable candidate-pool row",
                        row=index,
                        field=field,
                        screening_value=row.get(field),
                        candidate_value=candidate.get(field),
                    )

        accepted = row.get("accepted")
        if not isinstance(accepted, bool):
            issues.add(
                "screening_invalid_accepted_flag",
                "screening accepted must be a boolean",
                row=index,
                value=accepted,
            )
            accepted = False
        reason = row.get("reason")
        if reason not in SCREENING_REASONS:
            issues.add(
                "screening_invalid_reason",
                "screening reason is missing or outside the frozen state machine",
                row=index,
                reason=reason,
            )
        else:
            reason_counts[reason] += 1
        if accepted != (reason == "accepted"):
            issues.add(
                "screening_acceptance_reason_mismatch",
                "accepted is true if and only if reason is exactly 'accepted'",
                row=index,
                accepted=accepted,
                reason=reason,
            )

        source_id = row.get("source_image_id")
        source_id_seen = source_id is not None and str(source_id) in seen_source_ids
        if source_id_seen != (reason == "duplicate_source_image_id"):
            issues.add(
                "screening_source_id_reason_mismatch",
                "duplicate_source_image_id reason disagrees with prior screened identities",
                row=index,
                reason=reason,
                previously_seen=source_id_seen,
            )
        if source_id is not None:
            seen_source_ids.add(str(source_id))

        source_loaded = reason not in {
            "duplicate_source_image_id",
            "source_network_or_decode_error",
        }
        source_hash = row.get("source_canonical_rgb_sha256")
        valid_source_hash = isinstance(source_hash, str) and bool(
            HEX64_RE.fullmatch(source_hash)
        )
        if source_loaded and not valid_source_hash:
            issues.add(
                "screening_missing_source_hash",
                "every successfully loaded candidate requires a canonical source RGB hash",
                row=index,
                reason=reason,
            )
        if not source_loaded and "source_canonical_rgb_sha256" in row:
            issues.add(
                "screening_reason_state_mismatch",
                "builder does not emit a source hash before source load succeeds",
                row=index,
                reason=reason,
            )
        if valid_source_hash:
            source_hash_count += 1
            if candidate is not None:
                candidate_hash = candidate.get("source_canonical_rgb_sha256")
                if candidate_hash is not None and candidate_hash != source_hash:
                    issues.add(
                        "screening_candidate_source_hash_mismatch",
                        "screening source hash differs from the precomputed candidate hash",
                        row=index,
                        screening_value=source_hash,
                        candidate_value=candidate_hash,
                    )
            source_hash_seen = source_hash in seen_source_hashes
            if source_hash_seen != (
                reason == "duplicate_source_canonical_rgb_sha256"
            ):
                issues.add(
                    "screening_source_hash_reason_mismatch",
                    "duplicate source-RGB reason disagrees with prior screened hashes",
                    row=index,
                    reason=reason,
                    previously_seen=source_hash_seen,
                )
            seen_source_hashes.add(source_hash)

        mapping_template = (
            candidate.get("mapping_template") if isinstance(candidate, dict) else None
        )
        mapping_for_score = mapping_template if isinstance(mapping_template, dict) else None
        is_evaluated = reason in EVALUATED_SCREENING_REASONS
        blind_prediction = row.get("blind_prediction")
        blind_declared = row.get("blind_correct")
        blind_rescored: bool | None = None
        if is_evaluated:
            evaluated_count += 1
            if row.get("control") != "blind_black_image_control":
                issues.add(
                    "screening_invalid_blind_control",
                    "model-evaluated row must name blind_black_image_control",
                    row=index,
                    value=row.get("control"),
                )
            if not isinstance(blind_prediction, str):
                issues.add(
                    "screening_missing_blind_prediction",
                    "model-evaluated row must preserve a string blind prediction",
                    row=index,
                )
            if not isinstance(blind_declared, bool):
                issues.add(
                    "screening_invalid_blind_correct_flag",
                    "model-evaluated row blind_correct must be boolean",
                    row=index,
                    value=blind_declared,
                )
            if mapping_for_score is not None and isinstance(blind_prediction, str):
                blind_rescored = _construction_correct(
                    dataset, mapping_for_score, blind_prediction
                )
            if blind_rescored is None:
                issues.add(
                    "screening_prediction_unscorable",
                    "blind prediction cannot be independently rescored from candidate pool",
                    row=index,
                    field="blind_prediction",
                )
            else:
                blind_wrong_count += int(not blind_rescored)
                if isinstance(blind_declared, bool) and blind_declared != blind_rescored:
                    issues.add(
                        "screening_blind_correct_flag_mismatch",
                        "blind_correct differs from independent frozen-scorer result",
                        row=index,
                        declared=blind_declared,
                        rescored=blind_rescored,
                    )
                expected_reason_class = (
                    "blind_black_image_control_correct"
                    if blind_rescored
                    else "blind-wrong continuation"
                )
                if blind_rescored != (
                    reason == "blind_black_image_control_correct"
                ):
                    issues.add(
                        "screening_blind_reason_mismatch",
                        "screening reason disagrees with independently rescored blind result",
                        row=index,
                        reason=reason,
                        expected=expected_reason_class,
                    )
        else:
            _issue_unexpected_fields(
                row,
                ("blind_prediction", "blind_correct", "control"),
                issues,
                row_index=index,
                phase="non-evaluated",
            )

        has_saved_file = reason in SAVED_SCREENING_REASONS
        saved_hash = row.get("saved_canonical_rgb_sha256")
        if has_saved_file:
            for field in ("saved_image_sha256", "saved_canonical_rgb_sha256"):
                value = row.get(field)
                if not isinstance(value, str) or not HEX64_RE.fullmatch(value):
                    issues.add(
                        "screening_invalid_saved_hash",
                        "blind-wrong row must preserve valid deterministic-JPEG hashes",
                        row=index,
                        field=field,
                        value=value,
                    )
            perceptual_hash = row.get("perceptual_hash")
            if not isinstance(perceptual_hash, str) or not HEX16_RE.fullmatch(
                perceptual_hash
            ):
                issues.add(
                    "screening_invalid_saved_perceptual_hash",
                    "blind-wrong row must preserve its 64-bit dHash",
                    row=index,
                    value=perceptual_hash,
                )
            if isinstance(saved_hash, str) and HEX64_RE.fullmatch(saved_hash):
                saved_hash_seen = saved_hash in accepted_saved_hashes
                if saved_hash_seen != (
                    reason == "duplicate_saved_canonical_rgb_sha256"
                ):
                    issues.add(
                        "screening_saved_hash_reason_mismatch",
                        "duplicate saved-RGB reason disagrees with prior accepted JPEG hashes",
                        row=index,
                        reason=reason,
                        previously_seen=saved_hash_seen,
                    )
        else:
            _issue_unexpected_fields(
                row,
                (
                    "saved_image_sha256",
                    "saved_canonical_rgb_sha256",
                    "perceptual_hash",
                ),
                issues,
                row_index=index,
                phase="pre-save",
            )

        is_full_evaluated = reason in FULL_SCREENING_REASONS
        full_prediction = row.get("full_prediction")
        full_declared = row.get("full_correct")
        full_rescored: bool | None = None
        if is_full_evaluated:
            if not isinstance(full_prediction, str):
                issues.add(
                    "screening_missing_full_prediction",
                    "full-evaluated row must preserve a string full prediction",
                    row=index,
                )
            if not isinstance(full_declared, bool):
                issues.add(
                    "screening_invalid_full_correct_flag",
                    "full-evaluated row full_correct must be boolean",
                    row=index,
                    value=full_declared,
                )
            if mapping_for_score is not None and isinstance(full_prediction, str):
                full_rescored = _construction_correct(
                    dataset, mapping_for_score, full_prediction
                )
            if full_rescored is None:
                issues.add(
                    "screening_prediction_unscorable",
                    "full prediction cannot be independently rescored from candidate pool",
                    row=index,
                    field="full_prediction",
                )
            else:
                full_correct_count += int(full_rescored)
                if isinstance(full_declared, bool) and full_declared != full_rescored:
                    issues.add(
                        "screening_full_correct_flag_mismatch",
                        "full_correct differs from independent frozen-scorer result",
                        row=index,
                        declared=full_declared,
                        rescored=full_rescored,
                    )
                if full_rescored != (reason == "accepted"):
                    issues.add(
                        "screening_full_reason_mismatch",
                        "screening reason disagrees with independently rescored full result",
                        row=index,
                        reason=reason,
                        rescored=full_rescored,
                    )
        else:
            _issue_unexpected_fields(
                row,
                ("full_prediction", "full_correct"),
                issues,
                row_index=index,
                phase="non-full-evaluated",
            )

        if reason == "accepted":
            retained_index = row.get("retained_index")
            if isinstance(retained_index, bool) or retained_index != accepted_count:
                issues.add(
                    "screening_retained_index_discontinuity",
                    "accepted rows must assign retained_index contiguously in acceptance order",
                    row=index,
                    retained_index=retained_index,
                    expected=accepted_count,
                )
            if isinstance(retained_index, int) and not isinstance(retained_index, bool):
                if retained_index in accepted_indices:
                    issues.add(
                        "screening_duplicate_retained_index",
                        "multiple accepted rows identify the same retained index",
                        row=index,
                        retained_index=retained_index,
                    )
                accepted_indices.add(retained_index)
            sidecar = samples[accepted_count] if accepted_count < len(samples) else None
            if sidecar is None:
                issues.add(
                    "screening_missing_retained_sidecar",
                    "accepted screening row has no sidecar row in acceptance order",
                    row=index,
                    retained_index=retained_index,
                )
            else:
                expected_filename = sidecar.get("image_filename")
                if row.get("image_filename") != expected_filename:
                    issues.add(
                        "screening_sidecar_identity_mismatch",
                        "accepted screening image_filename differs from sidecar",
                        row=index,
                        field="image_filename",
                    )
                for field in SCREENING_IDENTITY_FIELDS + (
                    "source_canonical_rgb_sha256",
                    "saved_image_sha256",
                    "saved_canonical_rgb_sha256",
                    "perceptual_hash",
                ):
                    if row.get(field) != sidecar.get(field):
                        issues.add(
                            "screening_sidecar_identity_mismatch",
                            "accepted screening provenance differs from sidecar",
                            row=index,
                            field=field,
                            screening_value=row.get(field),
                            sidecar_value=sidecar.get(field),
                        )
                prediction_pairs = (
                    ("blind_prediction", "blind_prediction"),
                    ("blind_prediction", "blind_black_image_control_prediction"),
                    ("blind_correct", "blind_correct"),
                    ("blind_correct", "blind_black_image_control_correct"),
                    ("full_prediction", "full_prediction"),
                    ("full_prediction", "full_visual_prediction"),
                    ("full_correct", "full_correct"),
                    ("full_correct", "full_visual_correct"),
                )
                for screening_field, sidecar_field in prediction_pairs:
                    if row.get(screening_field) != sidecar.get(sidecar_field):
                        issues.add(
                            "screening_sidecar_prediction_mismatch",
                            "accepted screening prediction/flag differs from sidecar",
                            row=index,
                            screening_field=screening_field,
                            sidecar_field=sidecar_field,
                        )
                if candidate is not None:
                    for field in (
                        "prompt",
                        "legacy_temperature_zero",
                        "provenance",
                        "options",
                        "ground_truth_letter",
                        "ground_truth_text",
                    ):
                        if sidecar.get(field) != candidate.get(field):
                            issues.add(
                                "sidecar_candidate_field_mismatch",
                                "retained sidecar differs from immutable candidate pool",
                                row=index,
                                field=field,
                            )
                    template = candidate.get("mapping_template")
                    if isinstance(template, dict):
                        expected_mapping = dict(template)
                        expected_mapping["image_filename"] = expected_filename
                        if sidecar.get("mapping") != expected_mapping:
                            issues.add(
                                "sidecar_candidate_mapping_mismatch",
                                "retained mapping is not the candidate mapping_template plus filename",
                                row=index,
                            )
            if isinstance(saved_hash, str) and HEX64_RE.fullmatch(saved_hash):
                accepted_saved_hashes.add(saved_hash)
            accepted_count += 1
            last_accepted_position = index
            source_index = row.get("source_index")
            last_accepted_source_index = (
                source_index
                if isinstance(source_index, int) and not isinstance(source_index, bool)
                else None
            )
        else:
            _issue_unexpected_fields(
                row,
                ("retained_index", "image_filename"),
                issues,
                row_index=index,
                phase="rejected",
            )

    if accepted_count != len(samples):
        issues.add(
            "screening_retained_count_mismatch",
            "accepted screening decisions do not equal sidecar rows",
            screening_retained=accepted_count,
            samples=len(samples),
        )
    if accepted_indices != set(range(len(samples))):
        issues.add(
            "screening_retained_index_coverage_mismatch",
            "accepted rows do not cover every retained index exactly once",
            covered=sorted(accepted_indices)[:100],
            expected_count=len(samples),
        )

    return {
        "scanned_candidates": len(screening),
        "source_hash_candidates_scanned": source_hash_count,
        "reason_counts": reason_counts,
        "unique_images_evaluated": evaluated_count,
        "blind_wrong_count": blind_wrong_count,
        "full_correct_count": full_correct_count,
        "retained_count": accepted_count,
        "target_reached_candidate_position": last_accepted_position,
        "target_reached_source_index": last_accepted_source_index,
    }


def _validate_recomputed_stats(
    dataset: str,
    stats: dict[str, Any],
    derived: dict[str, Any],
    candidates: Sequence[dict[str, Any]],
    samples: Sequence[dict[str, Any]],
    target: int,
    issues: IssueLog,
) -> None:
    """Cross-check all construction counters recoverable from immutable logs."""

    reason_counts: Counter[str] = derived["reason_counts"]
    expected_source_splits = list(EXPECTED_SOURCE_SPLIT_ORDER[dataset])
    if stats.get("source_splits") != expected_source_splits:
        issues.add(
            "selection_source_splits_mismatch",
            "source_splits must exactly match the frozen upstream split order",
            reported=stats.get("source_splits"),
            expected=expected_source_splits,
        )

    expected_source_records = EXPECTED_SOURCE_QA_RECORDS[dataset]
    source_records = stats.get("source_qa_records_total")
    if (
        isinstance(source_records, bool)
        or not isinstance(source_records, int)
        or source_records != expected_source_records
    ):
        issues.add(
            "selection_source_qa_total_mismatch",
            "source_qa_records_total must equal the frozen upstream QA-record total",
            reported=source_records,
            expected=expected_source_records,
        )

    task_prefilter_count = stats.get("task_prefilter_unique_images")
    if (
        isinstance(task_prefilter_count, bool)
        or not isinstance(task_prefilter_count, int)
        or task_prefilter_count != len(candidates)
    ):
        issues.add(
            "selection_task_prefilter_pool_mismatch",
            "task_prefilter_unique_images must equal the immutable candidate-pool row count",
            reported=task_prefilter_count,
            recomputed=len(candidates),
        )

    exact_expected: dict[str, int] = {
        "candidate_representatives_total": len(candidates),
        "scanned_candidates": derived["scanned_candidates"],
        "source_hash_candidates_scanned": derived["source_hash_candidates_scanned"],
        "excluded_source_image_id_count": reason_counts[
            "excluded_prior_dataset_source_image_id"
        ],
        "excluded_source_hash_count": reason_counts[
            "excluded_prior_dataset_source_hash"
        ],
        "excluded_saved_hash_count": reason_counts[
            "excluded_prior_dataset_saved_hash"
        ],
        "engine_source_id_duplicate_count": reason_counts[
            "duplicate_source_image_id"
        ],
        "engine_exact_source_hash_duplicate_count": reason_counts[
            "duplicate_source_canonical_rgb_sha256"
        ],
        "saved_exact_hash_duplicate_count": reason_counts[
            "duplicate_saved_canonical_rgb_sha256"
        ],
        "unique_images_evaluated": derived["unique_images_evaluated"],
        "blind_correct_count": reason_counts[
            "blind_black_image_control_correct"
        ],
        "blind_wrong_count": derived["blind_wrong_count"],
        "full_incorrect_count": reason_counts["full_visual_incorrect"],
        "full_correct_count": derived["full_correct_count"],
        "retained_count": derived["retained_count"],
    }
    for field, expected in exact_expected.items():
        value = stats.get(field)
        if isinstance(value, bool) or not isinstance(value, int):
            issues.add(
                "selection_stat_not_exact_integer",
                "recomputable selection statistic must be a JSON integer",
                field=field,
                value=value,
                expected=expected,
            )
        elif value != expected:
            issues.add(
                "selection_stat_screening_mismatch",
                "selection statistic disagrees with independent screening-log recount",
                field=field,
                reported=value,
                recomputed=expected,
            )

    reported_target = stats.get("target")
    if (
        isinstance(reported_target, bool)
        or not isinstance(reported_target, int)
        or reported_target != target
    ):
        issues.add(
            "selection_target_mismatch",
            "selection target must equal the validator's requested count",
            reported=reported_target,
            expected=target,
        )
    if stats.get("dataset") != dataset:
        issues.add(
            "selection_dataset_mismatch",
            "selection stats dataset must match the validated dataset",
            reported=stats.get("dataset"),
            expected=dataset,
        )
    if stats.get("control") != "blind_black_image_control":
        issues.add(
            "selection_control_mismatch",
            "selection stats must name the frozen blind-black-image control",
            reported=stats.get("control"),
        )

    reached = derived["retained_count"] >= target
    expected_position = (
        derived["target_reached_candidate_position"] if reached else None
    )
    expected_source_index = derived["target_reached_source_index"] if reached else None
    for field, expected in (
        ("target_reached_candidate_position", expected_position),
        ("target_reached_source_index", expected_source_index),
    ):
        value = stats.get(field)
        if value != expected or isinstance(value, bool):
            issues.add(
                "selection_target_position_mismatch",
                "target-reached position/index disagrees with the target-th accepted row",
                field=field,
                reported=value,
                recomputed=expected,
            )

    evaluated = derived["unique_images_evaluated"]
    expected_rate = derived["retained_count"] / evaluated if evaluated else 0.0
    reported_rate = _number(stats.get("acceptance_rate"))
    if reported_rate is None or not math.isclose(
        reported_rate, expected_rate, rel_tol=1e-12, abs_tol=1e-12
    ):
        issues.add(
            "selection_stat_screening_mismatch",
            "acceptance_rate disagrees with retained/evaluated screening recount",
            field="acceptance_rate",
            reported=stats.get("acceptance_rate"),
            recomputed=expected_rate,
        )

    expected_exhausted = (
        derived["scanned_candidates"] >= len(candidates)
        and derived["retained_count"] < target
    )
    if stats.get("candidate_pool_exhausted") is not expected_exhausted:
        issues.add(
            "selection_pool_exhaustion_mismatch",
            "candidate_pool_exhausted disagrees with pool/screening/target counts",
            reported=stats.get("candidate_pool_exhausted"),
            recomputed=expected_exhausted,
        )

    lower_bounds = {
        "missing_or_corrupt_images": reason_counts["source_network_or_decode_error"],
        "source_id_duplicate_count": reason_counts["duplicate_source_image_id"],
        "exact_image_hash_duplicate_count": reason_counts[
            "duplicate_source_canonical_rgb_sha256"
        ],
    }
    for field, minimum in lower_bounds.items():
        value = stats.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            issues.add(
                "selection_stat_below_screening_minimum",
                "source-preparation aggregate cannot be below its screening contribution",
                field=field,
                reported=value,
                minimum=minimum,
            )

    if dataset == "ScienceQA_MC":
        actual_composition = Counter(sample.get("source_split") for sample in samples)
        expected_composition = {
            split: actual_composition[split] for split in ("train", "validation", "test")
        }
        reported = stats.get("split_composition")
        valid_reported = (
            isinstance(reported, dict)
            and set(reported) == set(expected_composition)
            and all(
                isinstance(reported.get(split), int)
                and not isinstance(reported.get(split), bool)
                for split in expected_composition
            )
        )
        if not valid_reported or reported != expected_composition:
            issues.add(
                "scienceqa_split_composition_sidecar_mismatch",
                "ScienceQA split composition must exactly equal retained sidecar splits",
                reported=reported,
                recomputed=expected_composition,
            )


def validate_dataset(
    dataset_root: Path | str,
    dataset: str,
    expected_count: int = 1000,
    max_issue_examples: int = 100,
) -> tuple[dict[str, Any], list[ImageRecord]]:
    """Validate one dataset and return its JSON-ready result plus image records."""

    root = Path(dataset_root).resolve()
    if dataset not in DATASET_NAMES:
        raise ValueError(f"unsupported dataset {dataset!r}")
    dataset_dir = root / dataset
    manifests_dir = root / "manifests"
    mapping_path = dataset_dir / f"{dataset}_mapping.jsonl"
    samples_path = manifests_dir / f"{dataset}_samples.jsonl"
    stats_path = manifests_dir / f"{dataset}_selection_stats.json"
    candidate_pool_path = manifests_dir / f"{dataset}_candidate_pool.jsonl"
    screening_path = manifests_dir / f"{dataset}_screening.jsonl"
    checkpoint_path = manifests_dir / f"{dataset}_checkpoint.json"
    source_path = manifests_dir / f"{dataset}_source.json"
    issues = IssueLog(max_issue_examples)

    if not dataset_dir.is_dir():
        issues.add("missing_dataset_directory", f"dataset directory does not exist: {dataset_dir}")
    mappings = _load_jsonl(mapping_path, issues, "mapping")
    samples = _load_jsonl(samples_path, issues, "samples")
    candidates = _load_jsonl(candidate_pool_path, issues, "candidate_pool")
    screening = _load_jsonl(screening_path, issues, "screening")
    stats_raw = _load_json(stats_path, issues, "selection_stats")
    checkpoint_raw = _load_json(checkpoint_path, issues, "checkpoint")
    source_raw = _load_json(source_path, issues, "source")
    _validate_candidate_pool(dataset, candidates, issues)
    _validate_builder_manifests(
        dataset,
        expected_count,
        source_raw,
        checkpoint_raw,
        stats_raw,
        candidates,
        screening,
        samples,
        mappings,
        issues,
    )

    mapping_signatures = [_mapping_signature(mapping) for mapping in mappings]
    duplicate_mapping_rows = len(mapping_signatures) - len(set(mapping_signatures))
    if duplicate_mapping_rows:
        issues.add(
            "duplicate_mapping_rows",
            "mapping contains byte-semantically duplicate JSON objects",
            count=duplicate_mapping_rows,
        )

    referenced: list[str] = []
    resolved_paths: list[Path | None] = []
    invalid_qa_rows = 0
    for index, mapping in enumerate(mappings):
        if mapping.get("dataset") != dataset:
            issues.add(
                "mapping_dataset_mismatch",
                "mapping dataset field does not match its directory",
                row=index,
                actual=mapping.get("dataset"),
            )
        if dataset in MC_DATASETS:
            qa_valid = _validate_mc_mapping(dataset, mapping, index, issues)
        else:
            qa_valid = _validate_open_mapping(mapping, index, issues)
        invalid_qa_rows += int(not qa_valid)
        normalized, image_path, error = _safe_image_reference(
            dataset_dir, mapping.get("image_filename")
        )
        if error:
            issues.add("invalid_image_filename", error, row=index)
            referenced.append(f"<invalid:{index}>")
            resolved_paths.append(None)
        else:
            assert normalized is not None and image_path is not None
            referenced.append(normalized)
            resolved_paths.append(image_path)

    duplicate_referenced_filenames = len(referenced) - len(set(referenced))
    if duplicate_referenced_filenames:
        issues.add(
            "duplicate_referenced_filenames",
            "mapping references a filename more than once",
            count=duplicate_referenced_filenames,
        )

    image_dir = dataset_dir / "images"
    disk_files = {
        path.relative_to(dataset_dir).as_posix()
        for path in image_dir.rglob("*")
        if path.is_file()
    } if image_dir.is_dir() else set()
    if not image_dir.is_dir():
        issues.add("missing_images_directory", f"images directory does not exist: {image_dir}")
    valid_references = {value for value in referenced if not value.startswith("<invalid:")}
    orphan_files = sorted(disk_files.difference(valid_references))
    if orphan_files:
        issues.add(
            "orphan_image_files",
            "images directory contains unreferenced files",
            count=len(orphan_files),
            first=orphan_files[:10],
        )

    if len(mappings) != expected_count:
        issues.add(
            "mapping_count_mismatch",
            "mapping row count does not equal the requested target",
            actual=len(mappings),
            expected=expected_count,
        )
    if len(samples) != expected_count:
        issues.add(
            "sample_count_mismatch",
            "samples sidecar row count does not equal the requested target",
            actual=len(samples),
            expected=expected_count,
        )

    blind_correct_count = 0
    full_correct_count = 0
    missing_required_sample_fields = 0
    sidecar_mapping_mismatches = 0
    source_identities: list[str] = []
    for index, sample in enumerate(samples):
        missing_fields = [field for field in REQUIRED_SAMPLE_FIELDS if field not in sample]
        if missing_fields:
            missing_required_sample_fields += 1
            issues.add(
                "missing_required_sample_fields",
                "sample sidecar row lacks required provenance/filter fields",
                row=index,
                fields=missing_fields,
            )
        if sample.get("dataset") != dataset:
            issues.add(
                "sample_dataset_mismatch",
                "sidecar dataset field must match its directory",
                row=index,
                actual=sample.get("dataset"),
            )
        retained_index = sample.get("retained_index")
        if isinstance(retained_index, bool) or retained_index != index:
            issues.add(
                "sample_retained_index_discontinuity",
                "sidecar retained_index must equal its zero-based row index",
                row=index,
                retained_index=retained_index,
            )
        if index >= len(mappings) or sample.get("mapping") != mappings[index]:
            sidecar_mapping_mismatches += 1
            issues.add(
                "sidecar_mapping_mismatch",
                "sample.mapping must exactly equal the mapping row at the same index",
                row=index,
            )
        if index < len(mappings):
            expected_filename = mappings[index].get("image_filename")
            if sample.get("image_filename") != expected_filename:
                issues.add(
                    "sample_image_filename_mismatch",
                    "sidecar image_filename must equal its mapping image_filename",
                    row=index,
                    sidecar=sample.get("image_filename"),
                    mapping=expected_filename,
                )
        for field in ("source_canonical_rgb_sha256", "canonical_rgb_sha256", "saved_image_sha256"):
            value = sample.get(field)
            if not isinstance(value, str) or not HEX64_RE.fullmatch(value):
                issues.add(
                    "invalid_declared_hash",
                    f"sample {field} must be 64 lowercase hexadecimal characters",
                    row=index,
                    field=field,
                )
        saved_canonical = sample.get("saved_canonical_rgb_sha256")
        if not isinstance(saved_canonical, str) or not HEX64_RE.fullmatch(
            saved_canonical
        ):
            issues.add(
                "invalid_declared_hash",
                "sample saved_canonical_rgb_sha256 must be lowercase SHA-256",
                row=index,
                field="saved_canonical_rgb_sha256",
            )
        elif sample.get("canonical_rgb_sha256") != saved_canonical:
            issues.add(
                "sample_saved_canonical_hash_mismatch",
                "canonical_rgb_sha256 must identify the reopened saved JPEG",
                row=index,
            )
        phash = sample.get("perceptual_hash")
        if not isinstance(phash, str) or not HEX16_RE.fullmatch(phash):
            issues.add(
                "invalid_declared_perceptual_hash",
                "sample perceptual_hash must be 16 lowercase hexadecimal characters",
                row=index,
            )
        source_dataset = sample.get("source_dataset")
        expected_source_dataset = EXPECTED_SOURCE_DATASET[dataset]
        if source_dataset != expected_source_dataset:
            issues.add(
                "invalid_source_dataset",
                "source_dataset must match the frozen upstream source for this benchmark",
                row=index,
                expected=expected_source_dataset,
                actual=source_dataset,
            )
        if sample.get("source_split") not in EXPECTED_SOURCE_SPLITS[dataset]:
            issues.add(
                "invalid_source_split",
                "source_split is outside the frozen source split(s)",
                row=index,
                actual=sample.get("source_split"),
                expected=sorted(EXPECTED_SOURCE_SPLITS[dataset]),
            )
        source_index = sample.get("source_index")
        if isinstance(source_index, bool) or not isinstance(source_index, int) or source_index < 0:
            issues.add("invalid_source_index", "source_index must be a non-negative integer", row=index)
        source_split_index = sample.get("source_split_index")
        if (
            isinstance(source_split_index, bool)
            or not isinstance(source_split_index, int)
            or source_split_index < 0
        ):
            issues.add(
                "invalid_source_split_index",
                "source_split_index must be a non-negative integer",
                row=index,
            )
        if not _valid_source_question_id(sample.get("source_question_id")):
            issues.add(
                "invalid_source_question_id",
                "source_question_id must be a non-empty string or integer",
                row=index,
                value=sample.get("source_question_id"),
            )
        if not _is_nonempty_string(sample.get("prompt")):
            issues.add(
                "invalid_construction_prompt",
                "sample prompt must be a non-empty string",
                row=index,
            )
        if sample.get("blind_control") != "blind_black_image_control":
            issues.add(
                "invalid_sample_blind_control",
                "sample blind_control must name blind_black_image_control",
                row=index,
                actual=sample.get("blind_control"),
            )
        expected_temperature_flag = EXPECTED_LEGACY_TEMPERATURE_ZERO[dataset]
        if sample.get("legacy_temperature_zero") is not expected_temperature_flag:
            issues.add(
                "invalid_legacy_temperature_flag",
                "sample legacy_temperature_zero does not match the frozen task setting",
                row=index,
                expected=expected_temperature_flag,
                actual=sample.get("legacy_temperature_zero"),
            )
        jpeg_settings = sample.get("jpeg_settings")
        valid_jpeg_settings = (
            isinstance(jpeg_settings, dict)
            and set(jpeg_settings) == set(EXPECTED_JPEG_SETTINGS)
            and all(
                type(jpeg_settings[key]) is type(expected_value)
                and jpeg_settings[key] == expected_value
                for key, expected_value in EXPECTED_JPEG_SETTINGS.items()
            )
        )
        if not valid_jpeg_settings:
            issues.add(
                "sample_jpeg_settings_mismatch",
                "sample jpeg_settings must exactly match the frozen deterministic JPEG settings",
                row=index,
                expected=EXPECTED_JPEG_SETTINGS,
                actual=jpeg_settings,
            )
        if not isinstance(sample.get("provenance"), dict):
            issues.add(
                "invalid_sample_provenance",
                "sample provenance must be an object",
                row=index,
            )
        alias_pairs = (
            ("blind_prediction", "blind_black_image_control_prediction"),
            ("blind_correct", "blind_black_image_control_correct"),
            ("full_prediction", "full_visual_prediction"),
            ("full_correct", "full_visual_correct"),
        )
        for short_field, explicit_field in alias_pairs:
            if sample.get(short_field) != sample.get(explicit_field):
                issues.add(
                    "sample_prediction_alias_mismatch",
                    "short and explicit construction fields must be identical",
                    row=index,
                    short_field=short_field,
                    explicit_field=explicit_field,
                )
        for prediction_field in (
            "blind_black_image_control_prediction",
            "full_visual_prediction",
        ):
            if not isinstance(sample.get(prediction_field), str):
                issues.add(
                    "invalid_construction_prediction",
                    f"{prediction_field} must be a string",
                    row=index,
                )
        blind_declared = sample.get("blind_black_image_control_correct")
        full_declared = sample.get("full_visual_correct")
        if not isinstance(blind_declared, bool):
            issues.add(
                "invalid_blind_correct_flag",
                "blind_black_image_control_correct must be boolean",
                row=index,
            )
        if not isinstance(full_declared, bool):
            issues.add(
                "invalid_full_correct_flag", "full_visual_correct must be boolean", row=index
            )

        mapping_for_score = mappings[index] if index < len(mappings) else None
        blind_prediction = sample.get("blind_black_image_control_prediction")
        full_prediction = sample.get("full_visual_prediction")
        blind_rescored: bool | None = None
        full_rescored: bool | None = None
        if isinstance(mapping_for_score, dict) and isinstance(blind_prediction, str):
            blind_rescored = _construction_correct(
                dataset, mapping_for_score, blind_prediction
            )
        if isinstance(mapping_for_score, dict) and isinstance(full_prediction, str):
            full_rescored = _construction_correct(
                dataset, mapping_for_score, full_prediction
            )
        if blind_rescored is None:
            issues.add(
                "construction_prediction_unscorable",
                "blind prediction could not be independently rescored",
                row=index,
                field="blind_black_image_control_prediction",
            )
        else:
            blind_correct_count += int(blind_rescored)
            if isinstance(blind_declared, bool) and blind_declared != blind_rescored:
                issues.add(
                    "blind_correct_flag_mismatch",
                    "declared blind correctness differs from independent rescoring",
                    row=index,
                    declared=blind_declared,
                    rescored=blind_rescored,
                )
            if blind_rescored:
                issues.add(
                    "retained_sample_blind_correct",
                    "retained sample violates blind-black-image-wrong invariant",
                    row=index,
                )
        if full_rescored is None:
            issues.add(
                "construction_prediction_unscorable",
                "full prediction could not be independently rescored",
                row=index,
                field="full_visual_prediction",
            )
        else:
            full_correct_count += int(full_rescored)
            if isinstance(full_declared, bool) and full_declared != full_rescored:
                issues.add(
                    "full_correct_flag_mismatch",
                    "declared full correctness differs from independent rescoring",
                    row=index,
                    declared=full_declared,
                    rescored=full_rescored,
                )
            if not full_rescored:
                issues.add(
                    "retained_sample_full_incorrect",
                    "retained sample violates full-visual-correct invariant",
                    row=index,
                )
        source_image_id = sample.get("source_image_id")
        source_hash = sample.get("source_canonical_rgb_sha256")
        if dataset in RELIABLE_SOURCE_ID_DATASETS:
            if source_image_id is not None and str(source_image_id).strip():
                # Within one benchmark, the reliable upstream ID itself is the
                # identity.  A mutable provenance label must not namespace away
                # a duplicate raw ID.
                identity = f"id:{source_image_id}"
            else:
                issues.add(
                    "missing_source_image_identity",
                    "this benchmark requires a non-empty reliable source_image_id",
                    row=index,
                )
                identity = f"invalid-row:{index}"
        else:
            if source_image_id is not None and str(source_image_id).strip():
                issues.add(
                    "unexpected_source_image_id",
                    "ScienceQA has no frozen reliable source image ID and must use its source RGB hash",
                    row=index,
                    source_image_id=source_image_id,
                )
            if isinstance(source_hash, str) and HEX64_RE.fullmatch(source_hash):
                identity = f"sha256:{source_hash}"
            else:
                issues.add(
                    "missing_source_image_identity",
                    "no valid source RGB hash is available",
                    row=index,
                )
                identity = f"invalid-row:{index}"
        source_identities.append(identity)

    source_identity_duplicates = len(source_identities) - len(set(source_identities))
    if source_identity_duplicates:
        issues.add(
            "duplicate_source_image_identities",
            "retained samples repeat a source image identity",
            count=source_identity_duplicates,
        )

    image_records: list[ImageRecord] = []
    missing_referenced = 0
    unreadable = 0
    declared_hash_mismatches = 0
    for index, image_path in enumerate(resolved_paths):
        if image_path is None:
            continue
        if not image_path.is_file():
            missing_referenced += 1
            issues.add(
                "missing_referenced_image",
                "mapping references a file that does not exist",
                row=index,
                image_filename=referenced[index],
            )
            continue
        try:
            raw_sha = saved_file_sha256(image_path)
            with Image.open(image_path) as opened:
                opened.load()
                encoded_format = opened.format
                encoded_mode = opened.mode
                encoded_width, encoded_height = opened.size
                if encoded_format != "JPEG":
                    issues.add(
                        "invalid_encoded_image_format",
                        "retained image bytes must be JPEG",
                        row=index,
                        actual=encoded_format,
                    )
                if encoded_mode != "RGB":
                    issues.add(
                        "invalid_encoded_image_mode",
                        "retained JPEG must decode natively as RGB",
                        row=index,
                        actual=encoded_mode,
                    )
                if encoded_width <= 0 or encoded_height <= 0:
                    issues.add(
                        "invalid_encoded_image_dimensions",
                        "retained JPEG dimensions must be positive",
                        row=index,
                        width=encoded_width,
                        height=encoded_height,
                    )
                normalized = ImageOps.exif_transpose(opened).convert("RGB")
                normalized.load()
                width, height = normalized.size
                canonical_sha = canonical_rgb_sha256(normalized)
                perceptual = dhash(normalized)
        except (OSError, ValueError, Image.DecompressionBombError) as error:
            unreadable += 1
            issues.add(
                "unreadable_image",
                str(error),
                row=index,
                image_filename=referenced[index],
            )
            continue
        sample = samples[index] if index < len(samples) else {}
        comparisons = (
            ("saved_image_sha256", raw_sha),
            ("canonical_rgb_sha256", canonical_sha),
            ("perceptual_hash", perceptual),
        )
        for field, computed in comparisons:
            if sample.get(field) != computed:
                declared_hash_mismatches += 1
                issues.add(
                    "declared_hash_mismatch",
                    f"sidecar {field} differs from independently recomputed value",
                    row=index,
                    field=field,
                    declared=sample.get(field),
                    computed=computed,
                )
        image_records.append(
            ImageRecord(
                dataset=dataset,
                mapping_index=index,
                image_filename=referenced[index],
                path=image_path,
                width=width,
                height=height,
                file_size=image_path.stat().st_size,
                saved_image_sha256=raw_sha,
                canonical_rgb_sha256=canonical_sha,
                perceptual_hash=perceptual,
                source_dataset=sample.get("source_dataset"),
                source_split=sample.get("source_split"),
                source_index=sample.get("source_index"),
                source_question_id=sample.get("source_question_id"),
                source_image_id=sample.get("source_image_id"),
                source_canonical_rgb_sha256=sample.get("source_canonical_rgb_sha256"),
            )
        )

    canonical_hashes = [record.canonical_rgb_sha256 for record in image_records]
    canonical_duplicates = len(canonical_hashes) - len(set(canonical_hashes))
    if canonical_duplicates:
        groups: dict[str, list[str]] = defaultdict(list)
        for record in image_records:
            groups[record.canonical_rgb_sha256].append(record.image_filename)
        issues.add(
            "duplicate_canonical_rgb_images",
            "retained files contain duplicate decoded RGB content",
            count=canonical_duplicates,
            first_groups=[
                {"canonical_rgb_sha256": key, "files": value}
                for key, value in groups.items()
                if len(value) > 1
            ][:10],
        )

    source_canonical_hashes = [
        sample.get("source_canonical_rgb_sha256")
        for sample in samples
        if isinstance(sample.get("source_canonical_rgb_sha256"), str)
        and HEX64_RE.fullmatch(sample["source_canonical_rgb_sha256"])
    ]
    source_canonical_duplicates = len(source_canonical_hashes) - len(
        set(source_canonical_hashes)
    )
    if source_canonical_duplicates:
        issues.add(
            "duplicate_source_canonical_rgb_images",
            "retained samples repeat source RGB content, even if source IDs differ",
            count=source_canonical_duplicates,
        )

    derived_screening = _validate_screening_manifest(
        dataset, screening, candidates, samples, issues
    )

    # Retain the compact accepted-row checks below as a second, independent
    # audit of the strict state-machine validation above.
    expressed_decisions = 0
    screening_retained = 0
    screening_retained_indices: set[int] = set()
    for index, row in enumerate(screening):
        expressed, retained = _retained_screening_record(row)
        expressed_decisions += int(expressed)
        screening_retained += int(expressed and retained)
        if expressed and retained:
            retained_index = row.get("retained_index")
            if (
                isinstance(retained_index, bool)
                or not isinstance(retained_index, int)
                or retained_index < 0
                or retained_index >= len(mappings)
            ):
                issues.add(
                    "screening_invalid_retained_index",
                    "accepted screening row must identify an existing retained index",
                    row=index,
                    retained_index=retained_index,
                )
                mapping_for_score = None
            else:
                if retained_index in screening_retained_indices:
                    issues.add(
                        "screening_duplicate_retained_index",
                        "multiple accepted screening rows identify the same retained index",
                        row=index,
                        retained_index=retained_index,
                    )
                screening_retained_indices.add(retained_index)
                mapping_for_score = mappings[retained_index]

            blind = row.get(
                "blind_correct", row.get("blind_black_image_control_correct")
            )
            full = row.get("full_correct", row.get("full_visual_correct"))
            if blind is not False:
                issues.add(
                    "screening_retained_blind_invariant_failure",
                    "accepted screening row must declare blind correctness false",
                    row=index,
                )
            if full is not True:
                issues.add(
                    "screening_retained_full_invariant_failure",
                    "accepted screening row must declare full correctness true",
                    row=index,
                )
            blind_prediction = row.get(
                "blind_prediction",
                row.get("blind_black_image_control_prediction"),
            )
            full_prediction = row.get(
                "full_prediction", row.get("full_visual_prediction")
            )
            if not isinstance(blind_prediction, str):
                issues.add(
                    "screening_missing_blind_prediction",
                    "accepted screening row must preserve its blind prediction",
                    row=index,
                )
            if not isinstance(full_prediction, str):
                issues.add(
                    "screening_missing_full_prediction",
                    "accepted screening row must preserve its full prediction",
                    row=index,
                )
            if isinstance(mapping_for_score, dict) and isinstance(
                blind_prediction, str
            ):
                blind_rescored = _construction_correct(
                    dataset, mapping_for_score, blind_prediction
                )
                if blind_rescored is not False:
                    issues.add(
                        "screening_blind_rescore_invariant_failure",
                        "accepted screening blind prediction is not wrong when rescored",
                        row=index,
                        rescored=blind_rescored,
                    )
            if isinstance(mapping_for_score, dict) and isinstance(
                full_prediction, str
            ):
                full_rescored = _construction_correct(
                    dataset, mapping_for_score, full_prediction
                )
                if full_rescored is not True:
                    issues.add(
                        "screening_full_rescore_invariant_failure",
                        "accepted screening full prediction is not correct when rescored",
                        row=index,
                        rescored=full_rescored,
                    )
            if (
                isinstance(retained_index, int)
                and not isinstance(retained_index, bool)
                and 0 <= retained_index < len(samples)
            ):
                sidecar = samples[retained_index]
                if blind_prediction != sidecar.get(
                    "blind_black_image_control_prediction"
                ):
                    issues.add(
                        "screening_sidecar_prediction_mismatch",
                        "screening blind prediction differs from retained sidecar",
                        row=index,
                        retained_index=retained_index,
                        field="blind_prediction",
                    )
                if full_prediction != sidecar.get("full_visual_prediction"):
                    issues.add(
                        "screening_sidecar_prediction_mismatch",
                        "screening full prediction differs from retained sidecar",
                        row=index,
                        retained_index=retained_index,
                        field="full_prediction",
                    )
    if not screening:
        issues.add(
            "empty_screening_manifest",
            "screening manifest must record candidate acceptance/rejection decisions",
        )
    elif expressed_decisions == 0:
        issues.add(
            "screening_decisions_not_expressible",
            "screening rows contain no recognized retained/accepted decision field",
        )
    elif expressed_decisions and screening_retained != len(samples):
        issues.add(
            "screening_retained_count_mismatch",
            "recognized retained screening decisions do not equal samples sidecar count",
            screening_retained=screening_retained,
            samples=len(samples),
        )
    elif screening_retained_indices != set(range(len(samples))):
        issues.add(
            "screening_retained_index_coverage_mismatch",
            "accepted screening rows do not cover every retained index exactly once",
            covered=sorted(screening_retained_indices)[:100],
            expected_count=len(samples),
        )

    extracted_stats = _validate_stats(dataset, stats_raw, len(samples), issues)
    _validate_recomputed_stats(
        dataset,
        stats_raw,
        derived_screening,
        candidates,
        samples,
        expected_count,
        issues,
    )
    evaluated_from_stats = _number(extracted_stats.get("unique_images_evaluated"))
    if (
        evaluated_from_stats is not None
        and evaluated_from_stats > len(screening)
    ):
        issues.add(
            "screening_rows_fewer_than_evaluated_images",
            "screening manifest cannot account for every reported model-evaluated image",
            screening_rows=len(screening),
            unique_images_evaluated=evaluated_from_stats,
        )
    counts = {
        "mapping_rows": len(mappings),
        "sample_sidecar_rows": len(samples),
        "candidate_pool_rows": len(candidates),
        "screening_rows": len(screening),
        "actual_files_in_images_directory": len(disk_files),
        "existing_referenced_images": len(image_records),
        "unique_referenced_filenames": len(set(referenced)),
        "unique_source_image_identities": len(set(source_identities)),
        "unique_canonical_rgb_sha256": len(set(canonical_hashes)),
        "missing_referenced_images": missing_referenced,
        "orphan_image_files": len(orphan_files),
        "unreadable_images": unreadable,
        "duplicate_mapping_rows": duplicate_mapping_rows,
        "duplicate_referenced_filenames": duplicate_referenced_filenames,
        "duplicate_source_image_identities": source_identity_duplicates,
        "duplicate_source_canonical_rgb_images": source_canonical_duplicates,
        "duplicate_canonical_rgb_images": canonical_duplicates,
        "invalid_question_answer_rows": invalid_qa_rows,
        "sample_rows_missing_required_fields": missing_required_sample_fields,
        "sidecar_mapping_mismatches": sidecar_mapping_mismatches,
        "declared_hash_mismatches": declared_hash_mismatches,
        "blind_control_correct_retained": blind_correct_count,
        "full_visual_correct_retained": full_correct_count,
        "screening_retained_detected": screening_retained,
    }
    checks = {
        "mapping_rows_equal_expected": len(mappings) == expected_count,
        "sample_rows_equal_expected": len(samples) == expected_count,
        "referenced_files_equal_expected": len(image_records) == expected_count,
        "unique_filenames_equal_expected": len(set(referenced)) == expected_count,
        "unique_source_identities_equal_expected": len(set(source_identities))
        == expected_count,
        "unique_source_rgb_hashes_equal_expected": len(set(source_canonical_hashes))
        == expected_count,
        "unique_rgb_hashes_equal_expected": len(set(canonical_hashes)) == expected_count,
        "no_missing_images": missing_referenced == 0,
        "no_orphan_images": not orphan_files,
        "no_unreadable_images": unreadable == 0,
        "no_duplicate_mapping_rows": duplicate_mapping_rows == 0,
        "no_invalid_questions_or_answers": invalid_qa_rows == 0,
        "required_sample_schema_complete": missing_required_sample_fields == 0,
        "sidecar_mapping_exact": sidecar_mapping_mismatches == 0,
        "declared_hashes_match_files": declared_hash_mismatches == 0,
        "blind_correct_count_is_zero": blind_correct_count == 0,
        "full_correct_count_equals_expected": full_correct_count == expected_count,
        "selection_and_screening_invariants": sum(issues.counts.values()) == 0,
    }
    result = {
        "dataset": dataset,
        "pass": all(checks.values()) and sum(issues.counts.values()) == 0,
        "paths": {
            "mapping": str(mapping_path),
            "samples": str(samples_path),
            "candidate_pool": str(candidate_pool_path),
            "screening": str(screening_path),
            "selection_stats": str(stats_path),
        },
        "input_artifact_sha256": {
            name: saved_file_sha256(path)
            if path.is_file() and not path.is_symlink()
            else None
            for name, path in (
                ("mapping", mapping_path),
                ("samples", samples_path),
                ("candidate_pool", candidate_pool_path),
                ("screening", screening_path),
                ("selection_stats", stats_path),
                ("checkpoint", checkpoint_path),
                ("source", source_path),
            )
        },
        "expected_count": expected_count,
        "counts": counts,
        "checks": checks,
        "selection_stats": extracted_stats,
        "issues": issues.as_dict(),
    }
    return result, image_records


@dataclass(slots=True)
class _PixelComparisonRepresentation:
    """One decoded image reduced to the only resolution used by pair review."""

    size: tuple[int, int] | None
    normalized_128: Image.Image | None
    error: str | None = None


class _PixelComparisonCache:
    """Lazy per-audit cache, preventing repeated full-image decode and resize."""

    def __init__(self) -> None:
        self._items: dict[Path, _PixelComparisonRepresentation] = {}
        self.hits = 0
        self.load_attempts = 0

    def get(self, path: Path) -> _PixelComparisonRepresentation:
        cached = self._items.get(path)
        if cached is not None:
            self.hits += 1
            return cached
        self.load_attempts += 1
        try:
            with Image.open(path) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
                image.load()
                size = image.size
                normalized = image.resize((128, 128), Image.Resampling.LANCZOS)
                normalized.load()
            value = _PixelComparisonRepresentation(size, normalized)
        except (OSError, ValueError, Image.DecompressionBombError) as error:
            value = _PixelComparisonRepresentation(None, None, str(error))
        self._items[path] = value
        return value

    def report(self) -> dict[str, Any]:
        return {
            "representation": "EXIF-transposed RGB resized to 128x128 with LANCZOS",
            "unique_paths_loaded": self.load_attempts,
            "cache_hits": self.hits,
            "load_errors": sum(item.error is not None for item in self._items.values()),
        }


def _pixel_difference_metrics(
    normalized_a: Image.Image, normalized_b: Image.Image
) -> dict[str, float]:
    if normalized_a.size != normalized_b.size:
        raise ValueError("pixel-difference inputs must have identical dimensions")
    histogram = ImageChops.difference(normalized_a, normalized_b).histogram()
    channel_values = normalized_a.width * normalized_a.height * 3
    absolute_sum = 0
    squared_sum = 0
    channels_within_8 = 0
    channels_within_16 = 0
    for channel in range(3):
        for difference in range(256):
            count = histogram[channel * 256 + difference]
            absolute_sum += difference * count
            squared_sum += difference * difference * count
            if difference <= 8:
                channels_within_8 += count
            if difference <= 16:
                channels_within_16 += count
    return {
        "mean_absolute_error": absolute_sum / channel_values,
        "root_mean_square_error": math.sqrt(squared_sum / channel_values),
        "channel_fraction_abs_diff_le_8": channels_within_8 / channel_values,
        "channel_fraction_abs_diff_le_16": channels_within_16 / channel_values,
    }


_DIRECT_PIXEL_THRESHOLDS = {
    "aspect_relative_difference_max": 0.001,
    "mean_absolute_error_max": 2.0,
    "root_mean_square_error_max": 4.0,
    "channel_fraction_abs_diff_le_8_min": 0.99,
}
_REENCODE_RESIZE_PIXEL_THRESHOLDS = {
    "aspect_relative_difference_max": 0.012,
    "normalized_128_mean_absolute_error_max": 12.0,
    "normalized_128_root_mean_square_error_max": 24.0,
    "normalized_128_channel_fraction_abs_diff_le_16_min": 0.80,
    "low_frequency_32_mean_absolute_error_max": 4.0,
    "low_frequency_32_root_mean_square_error_max": 6.5,
    "low_frequency_32_channel_fraction_abs_diff_le_8_min": 0.88,
    "low_frequency_32_channel_fraction_abs_diff_le_16_min": 0.985,
}
_ALIGNED_CROP_PIXEL_THRESHOLDS = {
    "crop_fraction_per_axis_max": 0.10,
    "predicted_aspect_relative_error_max": 0.012,
    "normalized_128_mean_absolute_error_max": 10.0,
    "normalized_128_root_mean_square_error_max": 18.0,
    "normalized_128_channel_fraction_abs_diff_le_16_min": 0.84,
    "low_frequency_32_mean_absolute_error_max": 4.0,
    "low_frequency_32_root_mean_square_error_max": 6.5,
    "low_frequency_32_channel_fraction_abs_diff_le_8_min": 0.88,
    "low_frequency_32_channel_fraction_abs_diff_le_16_min": 0.985,
}


def _direct_metrics_pass(metrics: dict[str, float]) -> bool:
    return (
        metrics["mean_absolute_error"]
        <= _DIRECT_PIXEL_THRESHOLDS["mean_absolute_error_max"]
        and metrics["root_mean_square_error"]
        <= _DIRECT_PIXEL_THRESHOLDS["root_mean_square_error_max"]
        and metrics["channel_fraction_abs_diff_le_8"]
        >= _DIRECT_PIXEL_THRESHOLDS["channel_fraction_abs_diff_le_8_min"]
    )


def _robust_metrics_pass(
    normalized_128: dict[str, float],
    low_frequency_32: dict[str, float],
    thresholds: dict[str, float],
) -> bool:
    """Require both broad full-resolution and strict low-frequency agreement."""

    return (
        normalized_128["mean_absolute_error"]
        <= thresholds["normalized_128_mean_absolute_error_max"]
        and normalized_128["root_mean_square_error"]
        <= thresholds["normalized_128_root_mean_square_error_max"]
        and normalized_128["channel_fraction_abs_diff_le_16"]
        >= thresholds["normalized_128_channel_fraction_abs_diff_le_16_min"]
        and low_frequency_32["mean_absolute_error"]
        <= thresholds["low_frequency_32_mean_absolute_error_max"]
        and low_frequency_32["root_mean_square_error"]
        <= thresholds["low_frequency_32_root_mean_square_error_max"]
        and low_frequency_32["channel_fraction_abs_diff_le_8"]
        >= thresholds["low_frequency_32_channel_fraction_abs_diff_le_8_min"]
        and low_frequency_32["channel_fraction_abs_diff_le_16"]
        >= thresholds["low_frequency_32_channel_fraction_abs_diff_le_16_min"]
    )


def _low_frequency_metrics_pass(
    metrics: dict[str, float], thresholds: dict[str, float]
) -> bool:
    return (
        metrics["mean_absolute_error"]
        <= thresholds["low_frequency_32_mean_absolute_error_max"]
        and metrics["root_mean_square_error"]
        <= thresholds["low_frequency_32_root_mean_square_error_max"]
        and metrics["channel_fraction_abs_diff_le_8"]
        >= thresholds["low_frequency_32_channel_fraction_abs_diff_le_8_min"]
        and metrics["channel_fraction_abs_diff_le_16"]
        >= thresholds["low_frequency_32_channel_fraction_abs_diff_le_16_min"]
    )


def _low_frequency_pixel_metrics(
    normalized_a: Image.Image, normalized_b: Image.Image
) -> dict[str, float]:
    low_a = normalized_a.resize((32, 32), Image.Resampling.LANCZOS)
    low_b = normalized_b.resize((32, 32), Image.Resampling.LANCZOS)
    return _pixel_difference_metrics(low_a, low_b)


def _multiscale_pixel_metrics(
    normalized_a: Image.Image, normalized_b: Image.Image
) -> tuple[dict[str, float], dict[str, float]]:
    normalized_128 = _pixel_difference_metrics(normalized_a, normalized_b)
    return normalized_128, _low_frequency_pixel_metrics(normalized_a, normalized_b)


def _axis_offsets(window_size: int) -> tuple[int, ...]:
    """Return distinct start offsets for leading, centred, and trailing crops."""

    maximum = 128 - window_size
    return tuple(sorted({0, maximum // 2, maximum}))


def _crop_alignment_candidates(
    source: _PixelComparisonRepresentation,
    target: _PixelComparisonRepresentation,
    source_label: str,
) -> Iterable[tuple[Image.Image, Image.Image, dict[str, Any]]]:
    """Yield small geometrically valid crop hypotheses in one direction.

    One retained-axis fraction is sampled and the other is derived from the
    original aspect ratios.  This permits centre, one-edge, and corner crops
    without opening the door to unconstrained anisotropic warping.
    """

    assert source.size is not None and source.normalized_128 is not None
    assert target.size is not None and target.normalized_128 is not None
    source_ratio = source.size[0] / source.size[1]
    target_ratio = target.size[0] / target.size[1]
    # Pixel-aligned fractions at the cached resolution.  At most ~9.4% of an
    # axis may be discarded, deliberately keeping this a *mild* crop test.
    retained_height_samples = range(128, 115, -1)
    emitted: set[tuple[int, int, int, int]] = set()
    for retained_height in retained_height_samples:
        retained_height_fraction = retained_height / 128
        retained_width_fraction = (
            target_ratio / source_ratio * retained_height_fraction
        )
        retained_width = round(128 * retained_width_fraction)
        if not 116 <= retained_width <= 128:
            continue
        predicted_ratio = source_ratio * retained_width / retained_height
        aspect_error = abs(predicted_ratio - target_ratio) / max(
            predicted_ratio, target_ratio
        )
        if (
            aspect_error
            > _ALIGNED_CROP_PIXEL_THRESHOLDS[
                "predicted_aspect_relative_error_max"
            ]
        ):
            continue
        for left in _axis_offsets(retained_width):
            for top in _axis_offsets(retained_height):
                box = (left, top, left + retained_width, top + retained_height)
                if box == (0, 0, 128, 128) or box in emitted:
                    continue
                emitted.add(box)
                # Yield the window itself.  The caller first screens it at 32px
                # and performs the costlier 128px resize only for plausible
                # alignments (or the single best failed alignment for reporting).
                source_window = source.normalized_128.crop(box)
                yield source_window, target.normalized_128, {
                    "mode": f"crop_{source_label}_to_other",
                    "crop_box_on_normalized_128": list(box),
                    "retained_width_fraction": retained_width / 128,
                    "retained_height_fraction": retained_height / 128,
                    "predicted_aspect_relative_error": aspect_error,
                }


def _pixel_verify(
    path_a: Path,
    path_b: Path,
    cache: _PixelComparisonCache | None = None,
) -> dict[str, Any]:
    """Conservatively confirm re-encodes, resizes, or mild crops of one image."""

    comparison_cache = cache if cache is not None else _PixelComparisonCache()
    representation_a = comparison_cache.get(path_a)
    representation_b = comparison_cache.get(path_b)
    if representation_a.error is not None or representation_b.error is not None:
        errors = [
            error
            for error in (representation_a.error, representation_b.error)
            if error is not None
        ]
        return {
            "verification_status": "error",
            "error": "; ".join(errors),
            "high_confidence": False,
        }
    assert representation_a.size is not None
    assert representation_b.size is not None
    assert representation_a.normalized_128 is not None
    assert representation_b.normalized_128 is not None
    ratio_a = representation_a.size[0] / representation_a.size[1]
    ratio_b = representation_b.size[0] / representation_b.size[1]
    aspect_relative_difference = abs(ratio_a - ratio_b) / max(ratio_a, ratio_b)

    direct_metrics, direct_low_frequency_metrics = _multiscale_pixel_metrics(
        representation_a.normalized_128, representation_b.normalized_128
    )
    direct_pass = (
        aspect_relative_difference
        <= _DIRECT_PIXEL_THRESHOLDS["aspect_relative_difference_max"]
        and _direct_metrics_pass(direct_metrics)
    )
    reencode_resize_pass = (
        aspect_relative_difference
        <= _REENCODE_RESIZE_PIXEL_THRESHOLDS[
            "aspect_relative_difference_max"
        ]
        and _robust_metrics_pass(
            direct_metrics,
            direct_low_frequency_metrics,
            _REENCODE_RESIZE_PIXEL_THRESHOLDS,
        )
    )
    best_metrics = direct_metrics
    best_low_frequency_metrics = direct_low_frequency_metrics
    best_alignment: dict[str, Any] = {"mode": "direct_resize"}
    best_score = (
        direct_low_frequency_metrics["mean_absolute_error"],
        direct_low_frequency_metrics["root_mean_square_error"],
        direct_metrics["mean_absolute_error"],
    )
    best_failed_crop: tuple[
        Image.Image,
        Image.Image,
        dict[str, float],
        dict[str, Any],
        tuple[float, float, float],
    ] | None = None
    passing_alignment: tuple[
        dict[str, float],
        dict[str, float],
        dict[str, Any],
        tuple[float, float, float],
    ] | None = None
    hypotheses_tested = 1
    found_passing_alignment = False
    # A confirmed direct re-encode/resize needs no geometric search.  Also
    # reject only extremely dissimilar 32px pairs up front: removing at most
    # 10% of an axis cannot plausibly repair a 64-level average RGB error.
    search_crop_alignment = (
        not reencode_resize_pass
        and direct_low_frequency_metrics["mean_absolute_error"] <= 64.0
    )
    directions = (
        (
            (representation_a, representation_b, "a"),
            (representation_b, representation_a, "b"),
        )
        if search_crop_alignment
        else ()
    )
    for source, target, label in directions:
        assert target.normalized_128 is not None
        target_low_frequency = target.normalized_128.resize(
            (32, 32), Image.Resampling.LANCZOS
        )
        for aligned_source, target_image, alignment in _crop_alignment_candidates(
            source, target, label
        ):
            hypotheses_tested += 1
            aligned_low_frequency = aligned_source.resize(
                (32, 32), Image.Resampling.LANCZOS
            )
            low_frequency_metrics = _pixel_difference_metrics(
                aligned_low_frequency, target_low_frequency
            )
            score = (
                low_frequency_metrics["mean_absolute_error"],
                low_frequency_metrics["root_mean_square_error"],
                0.0,
            )
            if best_failed_crop is None or score < best_failed_crop[4]:
                best_failed_crop = (
                    aligned_source,
                    target_image,
                    low_frequency_metrics,
                    alignment,
                    score,
                )
            if not _low_frequency_metrics_pass(
                low_frequency_metrics, _ALIGNED_CROP_PIXEL_THRESHOLDS
            ):
                continue
            normalized_aligned_source = aligned_source.resize(
                (128, 128), Image.Resampling.LANCZOS
            )
            metrics = _pixel_difference_metrics(
                normalized_aligned_source, target_image
            )
            score = (
                low_frequency_metrics["mean_absolute_error"],
                low_frequency_metrics["root_mean_square_error"],
                metrics["mean_absolute_error"],
            )
            if _robust_metrics_pass(
                metrics, low_frequency_metrics, _ALIGNED_CROP_PIXEL_THRESHOLDS
            ):
                passing_alignment = (
                    metrics,
                    low_frequency_metrics,
                    alignment,
                    score,
                )
                found_passing_alignment = True
                break
        if found_passing_alignment:
            break

    if passing_alignment is None and best_failed_crop is not None:
        (
            failed_source,
            failed_target,
            failed_low_frequency_metrics,
            failed_alignment,
            failed_score,
        ) = best_failed_crop
        if failed_score[:2] < best_score[:2]:
            failed_source_128 = failed_source.resize(
                (128, 128), Image.Resampling.LANCZOS
            )
            failed_metrics = _pixel_difference_metrics(
                failed_source_128, failed_target
            )
            best_metrics = failed_metrics
            best_low_frequency_metrics = failed_low_frequency_metrics
            best_alignment = failed_alignment
            best_score = (
                failed_score[0], failed_score[1], failed_metrics["mean_absolute_error"]
            )

    if direct_pass:
        decision_threshold_set = "direct"
        decision_metrics = direct_metrics
        decision_low_frequency_metrics = direct_low_frequency_metrics
        decision_alignment = {"mode": "direct_resize"}
    elif reencode_resize_pass:
        decision_threshold_set = "reencode_or_resize"
        decision_metrics = direct_metrics
        decision_low_frequency_metrics = direct_low_frequency_metrics
        decision_alignment = {"mode": "direct_resize"}
    elif passing_alignment is not None:
        decision_threshold_set = "aligned_crop"
        (
            decision_metrics,
            decision_low_frequency_metrics,
            decision_alignment,
            _,
        ) = passing_alignment
    else:
        decision_threshold_set = "none"
        decision_metrics, decision_alignment = best_metrics, best_alignment
        decision_low_frequency_metrics = best_low_frequency_metrics

    high_confidence = direct_pass or reencode_resize_pass or passing_alignment is not None
    return {
        "verification_status": "completed",
        "size_a": list(representation_a.size),
        "size_b": list(representation_b.size),
        "aspect_relative_difference": aspect_relative_difference,
        "normalized_128_mean_absolute_error": decision_metrics[
            "mean_absolute_error"
        ],
        "normalized_128_root_mean_square_error": decision_metrics[
            "root_mean_square_error"
        ],
        "normalized_128_channel_fraction_abs_diff_le_8": decision_metrics[
            "channel_fraction_abs_diff_le_8"
        ],
        "normalized_128_channel_fraction_abs_diff_le_16": decision_metrics[
            "channel_fraction_abs_diff_le_16"
        ],
        "low_frequency_32_mean_absolute_error": decision_low_frequency_metrics[
            "mean_absolute_error"
        ],
        "low_frequency_32_root_mean_square_error": decision_low_frequency_metrics[
            "root_mean_square_error"
        ],
        "low_frequency_32_channel_fraction_abs_diff_le_8": decision_low_frequency_metrics[
            "channel_fraction_abs_diff_le_8"
        ],
        "low_frequency_32_channel_fraction_abs_diff_le_16": decision_low_frequency_metrics[
            "channel_fraction_abs_diff_le_16"
        ],
        "thresholds": {
            **_DIRECT_PIXEL_THRESHOLDS,
            "direct": _DIRECT_PIXEL_THRESHOLDS,
            "reencode_or_resize": _REENCODE_RESIZE_PIXEL_THRESHOLDS,
            "aligned_crop": _ALIGNED_CROP_PIXEL_THRESHOLDS,
        },
        "decision_threshold_set": decision_threshold_set,
        "alignment": decision_alignment,
        "alignment_hypotheses_tested": hypotheses_tested,
        "high_confidence": high_confidence,
    }


def _near_duplicate_audit(
    records: Sequence[ImageRecord], distance_threshold: int
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    pixel_cache = _PixelComparisonCache()
    high_confidence_edges: list[tuple[str, str]] = []
    pair_counts: Counter[tuple[str, str]] = Counter()
    high_pair_counts: Counter[tuple[str, str]] = Counter()
    for left_index, left in enumerate(records):
        for right in records[left_index + 1 :]:
            if left.canonical_rgb_sha256 == right.canonical_rgb_sha256:
                continue
            distance = hamming_distance(left.perceptual_hash, right.perceptual_hash)
            if distance > distance_threshold:
                continue
            pair = tuple(sorted((left.dataset, right.dataset)))
            pair_counts[pair] += 1
            pixel = _pixel_verify(left.path, right.path, pixel_cache)
            high_confidence = bool(pixel.get("high_confidence"))
            if high_confidence:
                high_pair_counts[pair] += 1
                high_confidence_edges.append((left.key, right.key))
            candidates.append(
                {
                    "record_a": {
                        "key": left.key,
                        "dataset": left.dataset,
                        "mapping_index": left.mapping_index,
                        "image_filename": left.image_filename,
                        "canonical_rgb_sha256": left.canonical_rgb_sha256,
                        "perceptual_hash": left.perceptual_hash,
                    },
                    "record_b": {
                        "key": right.key,
                        "dataset": right.dataset,
                        "mapping_index": right.mapping_index,
                        "image_filename": right.image_filename,
                        "canonical_rgb_sha256": right.canonical_rgb_sha256,
                        "perceptual_hash": right.perceptual_hash,
                    },
                    "scope": "within_dataset" if left.dataset == right.dataset else "cross_dataset",
                    "dhash_hamming_distance": distance,
                    "pixel_verification": pixel,
                }
            )

    parent: dict[str, str] = {}

    def find(value: str) -> str:
        parent.setdefault(value, value)
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left, right in high_confidence_edges:
        union(left, right)
    grouped: dict[str, list[str]] = defaultdict(list)
    for value in parent:
        grouped[find(value)].append(value)
    groups = [sorted(values) for values in grouped.values() if len(values) > 1]
    groups.sort()
    return {
        "algorithm": "64-bit dHash candidates followed by conservative normalized-pixel verification",
        "dhash_hamming_distance_threshold": distance_threshold,
        "candidate_pair_count": len(candidates),
        "high_confidence_pair_count": len(high_confidence_edges),
        "unresolved_high_confidence_group_count": len(groups),
        "unresolved_high_confidence_groups": groups,
        "candidate_pairs": candidates,
        "pair_candidate_counts": {
            "|".join(pair): count for pair, count in sorted(pair_counts.items())
        },
        "pair_high_confidence_counts": {
            "|".join(pair): count for pair, count in sorted(high_pair_counts.items())
        },
        "pixel_comparison_cache": pixel_cache.report(),
    }


def _overlap_audit(records: Sequence[ImageRecord]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_dataset = {name: [record for record in records if record.dataset == name] for name in DATASET_NAMES}
    pairwise: list[dict[str, Any]] = []
    for left_index, left_name in enumerate(DATASET_NAMES):
        left = by_dataset[left_name]
        for right_name in DATASET_NAMES[left_index + 1 :]:
            right = by_dataset[right_name]
            left_canonical = {record.canonical_rgb_sha256 for record in left}
            right_canonical = {record.canonical_rgb_sha256 for record in right}
            left_source_hash = {
                record.source_canonical_rgb_sha256
                for record in left
                if isinstance(record.source_canonical_rgb_sha256, str)
                and HEX64_RE.fullmatch(record.source_canonical_rgb_sha256)
            }
            right_source_hash = {
                record.source_canonical_rgb_sha256
                for record in right
                if isinstance(record.source_canonical_rgb_sha256, str)
                and HEX64_RE.fullmatch(record.source_canonical_rgb_sha256)
            }
            left_ids = {
                (EXPECTED_SOURCE_DATASET[left_name], str(record.source_image_id))
                for record in left
                if record.source_image_id is not None and str(record.source_image_id).strip()
            }
            right_ids = {
                (EXPECTED_SOURCE_DATASET[right_name], str(record.source_image_id))
                for record in right
                if record.source_image_id is not None and str(record.source_image_id).strip()
            }
            shared_canonical = sorted(left_canonical.intersection(right_canonical))
            shared_source_hash = sorted(left_source_hash.intersection(right_source_hash))
            shared_ids = sorted(left_ids.intersection(right_ids))
            pairwise.append(
                {
                    "dataset_a": left_name,
                    "dataset_b": right_name,
                    "shared_canonical_rgb_sha256_count": len(shared_canonical),
                    "shared_source_canonical_rgb_sha256_count": len(shared_source_hash),
                    "shared_source_image_id_count": len(shared_ids),
                    "shared_canonical_rgb_sha256": shared_canonical,
                    "shared_source_canonical_rgb_sha256": shared_source_hash,
                    "shared_source_image_ids": [
                        {"source_dataset": source_dataset, "source_image_id": image_id}
                        for source_dataset, image_id in shared_ids
                    ],
                }
            )
    exact_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        exact_groups[record.canonical_rgb_sha256].append(
            {
                "dataset": record.dataset,
                "mapping_index": record.mapping_index,
                "image_filename": record.image_filename,
            }
        )
    cross_groups = [
        {"canonical_rgb_sha256": image_hash, "records": members}
        for image_hash, members in sorted(exact_groups.items())
        if len({member["dataset"] for member in members}) > 1
    ]
    return {
        "pairwise": pairwise,
        "cross_dataset_exact_rgb_group_count": len(cross_groups),
        "cross_dataset_exact_rgb_groups": cross_groups,
    }, pairwise


def _markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Final independent dataset validation",
        "",
        f"- Dataset root: `{report['dataset_root']}`",
        f"- Expected rows/images per dataset: {report['expected_count']}",
        f"- Overall result: **{'PASS' if report['overall_pass'] else 'FAIL'}**",
        f"- Generated (UTC): {report['generated_at_utc']}",
        "",
        "The construction control called `blind_black_image_control` still sends a "
        "336×336 black image through the visual encoder; it is not a literal zero-token path.",
        "The 0% blind / 100% Full figures below are construction-filter invariants, not "
        "accuracy from the frozen paper evaluation harness.",
        "",
        "## Per-dataset hard checks",
        "",
        "| Dataset | Result | Mapping | Existing | Unique names | Unique source IDs | Unique RGB | Missing | Orphan | Corrupt | Blind correct | Full correct | Issues |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in DATASET_NAMES:
        result = report["datasets"][name]
        counts = result["counts"]
        lines.append(
            "| {name} | {status} | {mapping} | {existing} | {filenames} | {source} | "
            "{rgb} | {missing} | {orphan} | {corrupt} | {blind} | {full} | {issues} |".format(
                name=name,
                status="PASS" if result["pass"] else "FAIL",
                mapping=counts["mapping_rows"],
                existing=counts["existing_referenced_images"],
                filenames=counts["unique_referenced_filenames"],
                source=counts["unique_source_image_identities"],
                rgb=counts["unique_canonical_rgb_sha256"],
                missing=counts["missing_referenced_images"],
                orphan=counts["orphan_image_files"],
                corrupt=counts["unreadable_images"],
                blind=counts["blind_control_correct_retained"],
                full=counts["full_visual_correct_retained"],
                issues=result["issues"]["total"],
            )
        )
    lines.extend(
        [
            "",
            "## Cross-dataset exact overlap",
            "",
            "| Dataset A | Dataset B | Saved RGB hashes | Source RGB hashes | Source image IDs |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in report["overlap"]["pairwise"]:
        lines.append(
            f"| {row['dataset_a']} | {row['dataset_b']} | "
            f"{row['shared_canonical_rgb_sha256_count']} | "
            f"{row['shared_source_canonical_rgb_sha256_count']} | "
            f"{row['shared_source_image_id_count']} |"
        )
    near = report["near_duplicate_audit"]
    lines.extend(
        [
            "",
            "## Near-duplicate review",
            "",
            f"- dHash candidate pairs (distance ≤ {near['dhash_hamming_distance_threshold']}): "
            f"{near['candidate_pair_count']}",
            f"- Pixel-verified high-confidence pairs: {near['high_confidence_pair_count']}",
            f"- Unresolved high-confidence groups: {near['unresolved_high_confidence_group_count']}",
            "",
        ]
    )
    failures = report["hard_check_failures"]
    if failures:
        lines.extend(["## Failed hard checks", ""])
        lines.extend(f"- {failure}" for failure in failures)
        lines.append("")
    for name in DATASET_NAMES:
        issues = report["datasets"][name]["issues"]
        if issues["total"]:
            lines.extend([f"## {name} issue counts", ""])
            lines.extend(f"- `{code}`: {count}" for code, count in issues["counts"].items())
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def validate_dataset_root(
    dataset_root: Path | str,
    expected_count: int = 1000,
    near_duplicate_distance: int = 4,
    reports_dir: Path | str | None = None,
    max_issue_examples: int = 100,
    write_outputs: bool = True,
) -> dict[str, Any]:
    """Run the complete independent validation and optionally write reports."""

    if expected_count < 1:
        raise ValueError("expected_count must be positive")
    if not 0 <= near_duplicate_distance <= 64:
        raise ValueError("near_duplicate_distance must be between 0 and 64")
    root = Path(dataset_root).resolve()
    if write_outputs and reports_dir is None:
        raise ValueError("reports_dir is required when write_outputs=True")
    output_dir = (
        assert_output_separate(reports_dir, {"dataset root": root})
        if reports_dir
        else None
    )
    datasets: dict[str, Any] = {}
    records: list[ImageRecord] = []
    for name in DATASET_NAMES:
        result, dataset_records = validate_dataset(
            root, name, expected_count, max_issue_examples
        )
        datasets[name] = result
        records.extend(dataset_records)

    overlap, pairwise = _overlap_audit(records)
    near = _near_duplicate_audit(records, near_duplicate_distance)
    pair_near = near["pair_candidate_counts"]
    pair_high = near["pair_high_confidence_counts"]
    csv_rows: list[dict[str, Any]] = []
    for row in pairwise:
        pair_key = "|".join(sorted((row["dataset_a"], row["dataset_b"])))
        csv_rows.append(
            {
                "dataset_a": row["dataset_a"],
                "dataset_b": row["dataset_b"],
                "shared_canonical_rgb_sha256_count": row[
                    "shared_canonical_rgb_sha256_count"
                ],
                "shared_source_canonical_rgb_sha256_count": row[
                    "shared_source_canonical_rgb_sha256_count"
                ],
                "shared_source_image_id_count": row["shared_source_image_id_count"],
                "dhash_candidate_pair_count": pair_near.get(pair_key, 0),
                "pixel_verified_high_confidence_pair_count": pair_high.get(pair_key, 0),
            }
        )

    failures: list[str] = []
    for name, result in datasets.items():
        if not result["pass"]:
            failures.append(f"{name}: one or more per-dataset hard checks failed")
    for row in pairwise:
        pair_label = f"{row['dataset_a']}/{row['dataset_b']}"
        if row["shared_canonical_rgb_sha256_count"]:
            failures.append(f"{pair_label} share saved canonical RGB images")
        if row["shared_source_canonical_rgb_sha256_count"]:
            failures.append(f"{pair_label} share source canonical RGB images")
    vqa_pair = next(
        row
        for row in pairwise
        if {row["dataset_a"], row["dataset_b"]} == {"VQAv2_Open", "VQAv2_MC"}
    )
    if vqa_pair["shared_source_image_id_count"]:
        failures.append("VQAv2_Open/VQAv2_MC share source image IDs")
    if near["unresolved_high_confidence_group_count"]:
        failures.append("unresolved pixel-verified high-confidence near-duplicate groups remain")

    report = {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(root),
        "expected_count": expected_count,
        "canonical_rgb_hash_specification": (
            "SHA256(b'RGB\\0' + width_uint64_be + height_uint64_be + "
            "EXIF-transposed RGB pixel bytes)"
        ),
        "overall_pass": not failures,
        "hard_check_failures": failures,
        "datasets": datasets,
        "overlap": overlap,
        "near_duplicate_audit": near,
    }
    if write_outputs:
        assert output_dir is not None
        image_hash_manifest = output_dir / "image_hash_manifest.jsonl"
        _atomic_write_jsonl(
            image_hash_manifest,
            (record.manifest_row() for record in records),
        )
        report["output_artifact_sha256"] = {
            "image_hash_manifest": saved_file_sha256(image_hash_manifest),
        }
        _atomic_write_json(output_dir / "final_validation.json", report)
        _atomic_write_text(output_dir / "final_validation.md", _markdown_report(report))
        columns = (
            "dataset_a",
            "dataset_b",
            "shared_canonical_rgb_sha256_count",
            "shared_source_canonical_rgb_sha256_count",
            "shared_source_image_id_count",
            "dhash_candidate_pair_count",
            "pixel_verified_high_confidence_pair_count",
        )
        _atomic_write_csv(output_dir / "cross_dataset_overlap.csv", csv_rows, columns)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root_positional", nargs="?", type=Path)
    parser.add_argument("--dataset-root", type=Path, help="root containing the four datasets")
    parser.add_argument(
        "--reports-dir",
        type=Path,
        required=True,
        help="explicit output directory for validation reports",
    )
    parser.add_argument("--expected-count", type=int, default=1000)
    parser.add_argument(
        "--near-duplicate-distance",
        type=int,
        default=4,
        help="inclusive 64-bit dHash Hamming-distance candidate threshold",
    )
    parser.add_argument("--max-issue-examples", type=int, default=100)
    parser.add_argument(
        "--no-fail-exit",
        action="store_true",
        help="write validation failures but return process status zero",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    dataset_root = args.dataset_root or args.dataset_root_positional
    if dataset_root is None:
        _parser().error("a dataset root is required (positional or --dataset-root)")
    report = validate_dataset_root(
        dataset_root=dataset_root,
        expected_count=args.expected_count,
        near_duplicate_distance=args.near_duplicate_distance,
        reports_dir=args.reports_dir,
        max_issue_examples=args.max_issue_examples,
        write_outputs=True,
    )
    print(
        json.dumps(
            {
                "dataset_root": report["dataset_root"],
                "overall_pass": report["overall_pass"],
                "hard_check_failures": report["hard_check_failures"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["overall_pass"] or args.no_fail_exit else 1


if __name__ == "__main__":
    sys.exit(main())
