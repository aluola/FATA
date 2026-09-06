#!/usr/bin/env python3
"""Generate the fail-closed reproducibility manifest for a completed build."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image

from .build_core import (
    CHECKPOINT_FORMAT_VERSION,
    CONSTRUCTION_PROTOCOL_VERSION,
    BuildConfig,
    BuildExclusions,
    _reconstruct_committed_state,
    _stable_json_digest as _builder_stable_json_digest,
    _stable_source_metadata,
)
from .common import (
    CANONICAL_HASH_DESCRIPTION,
    DATASET_NAMES,
    JPEG_SETTINGS,
    SEED,
    atomic_write_json,
    canonical_rgb_sha256_path,
    dhash_path,
    resolve_builder_source_cache,
    resolve_source_cache_child,
    sha256_file,
)
from .sources import (
    VQA_ANNOTATION_MEMBER,
    VQA_QUESTION_MEMBER,
    Candidate,
    SourceBundle,
    validate_coco_identity_cache,
)
from fata.utils.paths import assert_output_separate, resolve_dataset_relative_path


RELEASE_ROOT = Path(__file__).resolve().parents[4]
BUILDER_CODE_ROOT = Path(__file__).resolve().parent
OLD_DATASET_ROOT = (
    Path(os.environ["FATA_LEGACY_DATA_ROOT"])
    if os.environ.get("FATA_LEGACY_DATA_ROOT")
    else None
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_MODEL_MANIFEST_ALGORITHM = "sha256-per-file-and-canonical-json-aggregate"


def _validate_formal_validation_gate(validation: dict[str, Any]) -> None:
    """Require the exact validator settings and zero-overlap formal outcome."""

    if validation.get("hard_check_failures") != []:
        raise RuntimeError("formal validation hard_check_failures must be empty")
    near = validation.get("near_duplicate_audit")
    if not isinstance(near, dict):
        raise RuntimeError("formal validation is missing near_duplicate_audit")
    if (
        type(near.get("dhash_hamming_distance_threshold")) is not int
        or near["dhash_hamming_distance_threshold"] != 4
    ):
        raise RuntimeError("formal validation requires dHash distance threshold 4")
    if (
        type(near.get("unresolved_high_confidence_group_count")) is not int
        or near["unresolved_high_confidence_group_count"] != 0
    ):
        raise RuntimeError(
            "formal validation has unresolved high-confidence near duplicates"
        )

    overlap = validation.get("overlap")
    if not isinstance(overlap, dict):
        raise RuntimeError("formal validation is missing overlap audit")
    if (
        type(overlap.get("cross_dataset_exact_rgb_group_count")) is not int
        or overlap["cross_dataset_exact_rgb_group_count"] != 0
        or overlap.get("cross_dataset_exact_rgb_groups") != []
    ):
        raise RuntimeError("formal validation has cross-dataset exact RGB groups")
    pairwise = overlap.get("pairwise")
    expected_pairs = [
        (left, right)
        for index, left in enumerate(DATASET_NAMES)
        for right in DATASET_NAMES[index + 1 :]
    ]
    if not isinstance(pairwise, list) or len(pairwise) != len(expected_pairs):
        raise RuntimeError("formal validation must contain all six dataset pairs")
    seen_pairs: set[tuple[str, str]] = set()
    for row in pairwise:
        if not isinstance(row, dict):
            raise RuntimeError("formal validation has an invalid pairwise overlap row")
        pair = (row.get("dataset_a"), row.get("dataset_b"))
        if pair not in expected_pairs or pair in seen_pairs:
            raise RuntimeError("formal validation has invalid dataset pair coverage")
        seen_pairs.add(pair)
        for count_key, values_key in (
            (
                "shared_canonical_rgb_sha256_count",
                "shared_canonical_rgb_sha256",
            ),
            (
                "shared_source_canonical_rgb_sha256_count",
                "shared_source_canonical_rgb_sha256",
            ),
            ("shared_source_image_id_count", "shared_source_image_ids"),
        ):
            if (
                type(row.get(count_key)) is not int
                or row[count_key] != 0
                or row.get(values_key) != []
            ):
                raise RuntimeError(
                    f"formal validation pair {pair!r} has nonzero exact overlap"
                )
    if seen_pairs != set(expected_pairs):
        raise RuntimeError("formal validation does not cover all six dataset pairs")


def _validate_formal_exclusion_lineage(
    dataset: str,
    exclusions_raw: Any,
    *,
    prior_source_hashes: set[str],
    prior_saved_hashes: set[str],
    vqav2_open_source_ids: set[str],
) -> None:
    """Bind each dataset to the exact cumulative exclusion contract."""

    if dataset not in DATASET_NAMES:
        raise RuntimeError(f"unexpected formal dataset {dataset!r}")
    expected = {
        "source_image_ids": (
            sorted(vqav2_open_source_ids) if dataset == "VQAv2_MC" else []
        ),
        "source_canonical_rgb_sha256": sorted(prior_source_hashes),
        "saved_canonical_rgb_sha256": sorted(prior_saved_hashes),
    }
    if _canonical_json(exclusions_raw) != _canonical_json(expected):
        raise RuntimeError(f"formal cumulative exclusion lineage mismatch for {dataset}")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_jsonl_strict(path: Path, *, label: str) -> list[Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise RuntimeError(f"{label} is missing its final newline: {path}")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"{label} is not UTF-8: {path}") from exc
    rows: list[Any] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise RuntimeError(f"{label} contains a blank line {line_number}: {path}")
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"{label} contains invalid JSON at line {line_number}: {path}"
            ) from exc
    return rows


def _validate_image_hash_manifest(
    dataset_root: Path,
    reports_dir: Path,
    *,
    expected_count: int,
    dataset_names: tuple[str, ...] = DATASET_NAMES,
) -> dict[str, Any]:
    manifest_path = reports_dir / "image_hash_manifest.jsonl"
    rows = _read_jsonl_strict(manifest_path, label="image hash manifest")
    if len(rows) != expected_count * len(dataset_names):
        raise RuntimeError(
            "image hash manifest row count does not match the exact formal cohort"
        )
    expected_fields = {
        "dataset",
        "mapping_index",
        "image_filename",
        "width",
        "height",
        "file_size",
        "saved_image_sha256",
        "canonical_rgb_sha256",
        "perceptual_hash",
        "source_dataset",
        "source_split",
        "source_index",
        "source_question_id",
        "source_image_id",
        "source_canonical_rgb_sha256",
    }
    grouped: dict[str, list[dict[str, Any]]] = {name: [] for name in dataset_names}
    for row_number, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != expected_fields:
            raise RuntimeError(
                f"invalid image hash manifest schema at row {row_number}"
            )
        dataset = row["dataset"]
        if dataset not in grouped:
            raise RuntimeError(
                f"unexpected image hash manifest dataset at row {row_number}"
            )
        grouped[dataset].append(row)

    for dataset in dataset_names:
        dataset_rows = grouped[dataset]
        if len(dataset_rows) != expected_count:
            raise RuntimeError(
                f"image hash manifest cohort count mismatch for {dataset}"
            )
        for expected_index, row in enumerate(dataset_rows):
            if type(row["mapping_index"]) is not int or row["mapping_index"] != expected_index:
                raise RuntimeError(
                    f"image hash manifest indices are not contiguous for {dataset}"
                )
            expected_filename = f"images/{dataset}_{expected_index:04d}.jpg"
            if row["image_filename"] != expected_filename:
                raise RuntimeError(
                    f"image hash manifest filename mismatch for {dataset}:{expected_index}"
                )
            try:
                image_path = resolve_dataset_relative_path(
                    dataset_root / dataset, expected_filename
                )
            except ValueError as exc:
                raise RuntimeError(
                    f"unsafe image hash manifest path for {dataset}:{expected_index}"
                ) from exc
            if not image_path.is_file():
                raise RuntimeError(f"validated image is missing: {image_path}")
            if type(row["file_size"]) is not int or row["file_size"] < 0:
                raise RuntimeError(
                    f"invalid declared image size for {dataset}:{expected_index}"
                )
            if image_path.stat().st_size != row["file_size"]:
                raise RuntimeError(
                    f"validated image size changed for {dataset}:{expected_index}"
                )
            declared_bytes = _require_sha256(
                row["saved_image_sha256"],
                label=f"saved image digest for {dataset}:{expected_index}",
            )
            declared_rgb = _require_sha256(
                row["canonical_rgb_sha256"],
                label=f"canonical RGB digest for {dataset}:{expected_index}",
            )
            if sha256_file(image_path) != declared_bytes:
                raise RuntimeError(
                    f"validated image bytes changed for {dataset}:{expected_index}"
                )
            if canonical_rgb_sha256_path(image_path) != declared_rgb:
                raise RuntimeError(
                    f"validated image RGB changed for {dataset}:{expected_index}"
                )
            if not isinstance(row["perceptual_hash"], str) or dhash_path(
                image_path
            ) != row["perceptual_hash"]:
                raise RuntimeError(
                    f"validated image perceptual hash changed for {dataset}:{expected_index}"
                )
            with Image.open(image_path) as image:
                image.load()
                width, height = image.size
                image_format = image.format
                image_mode = image.mode
            if image_format != "JPEG" or image_mode != "RGB":
                raise RuntimeError(
                    f"validated image is not an RGB JPEG for {dataset}:{expected_index}"
                )
            if width <= 0 or height <= 0:
                raise RuntimeError(
                    f"validated image has non-positive dimensions for {dataset}:{expected_index}"
                )
            if (
                type(row["width"]) is not int
                or type(row["height"]) is not int
                or [row["width"], row["height"]] != [width, height]
            ):
                raise RuntimeError(
                    f"validated image dimensions changed for {dataset}:{expected_index}"
                )
    return _file_manifest(manifest_path)


def _require_sorted_string_set(
    checkpoint: dict[str, Any], key: str, *, sha256: bool = False
) -> set[str]:
    raw = checkpoint.get(key)
    if not isinstance(raw, list):
        raise RuntimeError(f"checkpoint {key} must be a list")
    values: list[str] = []
    for index, value in enumerate(raw):
        if not isinstance(value, str) or not value:
            raise RuntimeError(f"checkpoint {key}[{index}] must be a string")
        if sha256:
            _require_sha256(value, label=f"checkpoint {key}[{index}]")
        values.append(value)
    if values != sorted(set(values)):
        raise RuntimeError(f"checkpoint {key} is not a sorted unique set")
    return set(values)


def _validate_vqa_source_archives(
    cache_root: Path, source_metadata: dict[str, Any], *, dataset: str
) -> None:
    pairs = (
        ("question_zip", "question_member", "questions_json", VQA_QUESTION_MEMBER),
        (
            "annotation_zip",
            "annotation_member",
            "annotations_json",
            VQA_ANNOTATION_MEMBER,
        ),
    )
    for archive_key, member_key, path_key, expected_member in pairs:
        archive = source_metadata.get(archive_key)
        member = source_metadata.get(member_key)
        if not isinstance(archive, dict) or not isinstance(member, dict):
            raise RuntimeError(f"missing VQA archive provenance for {dataset}")
        if set(member) != {"member", "path", "size_bytes", "sha256"}:
            raise RuntimeError(f"invalid VQA member provenance for {dataset}")
        raw_archive_path = Path(str(archive.get("path")))
        raw_extracted_path = Path(str(member.get("path")))
        if not raw_archive_path.is_absolute() or not raw_extracted_path.is_absolute():
            raise RuntimeError(f"VQA provenance paths must be absolute for {dataset}")
        try:
            archive_relative = raw_archive_path.relative_to(cache_root).as_posix()
            extracted_relative = raw_extracted_path.relative_to(cache_root).as_posix()
            archive_path = resolve_source_cache_child(
                cache_root, archive_relative, require_file=True
            )
            extracted_path = resolve_source_cache_child(
                cache_root, extracted_relative, require_file=True
            )
        except (ValueError, FileNotFoundError) as exc:
            raise RuntimeError(
                f"VQA provenance path escapes or symlinks source_cache for {dataset}"
            ) from exc
        if Path(str(source_metadata.get(path_key))) != raw_extracted_path:
            raise RuntimeError(f"VQA extracted path disagreement for {dataset}")
        if member["member"] != expected_member:
            raise RuntimeError(f"VQA ZIP member name mismatch for {dataset}")
        if not archive_path.is_file() or not extracted_path.is_file():
            raise RuntimeError(f"VQA source archive/member is missing for {dataset}")
        if type(archive.get("size")) is not int or archive["size"] != archive_path.stat().st_size:
            raise RuntimeError(f"VQA archive size changed for {dataset}")
        archive_digest = _require_sha256(
            archive.get("sha256"), label=f"VQA archive digest for {dataset}"
        )
        member_digest = _require_sha256(
            member.get("sha256"), label=f"VQA member digest for {dataset}"
        )
        if sha256_file(archive_path) != archive_digest:
            raise RuntimeError(f"VQA archive bytes changed for {dataset}")
        extracted_bytes = extracted_path.read_bytes()
        if (
            type(member.get("size_bytes")) is not int
            or member["size_bytes"] != len(extracted_bytes)
            or hashlib.sha256(extracted_bytes).hexdigest() != member_digest
        ):
            raise RuntimeError(f"VQA extracted member bytes changed for {dataset}")
        with zipfile.ZipFile(archive_path) as handle:
            if handle.testzip() is not None:
                raise RuntimeError(f"VQA archive CRC failure for {dataset}")
            try:
                archived_bytes = handle.read(expected_member)
            except KeyError as exc:
                raise RuntimeError(f"VQA archive member is missing for {dataset}") from exc
        if archived_bytes != extracted_bytes:
            raise RuntimeError(
                f"VQA extracted member differs from current ZIP for {dataset}"
            )


def _validate_completed_build_artifacts(
    root: Path,
    report_root: Path,
    dataset: str,
    source: dict[str, Any],
    checkpoint: dict[str, Any],
    stats: dict[str, Any],
    *,
    expected_count: int,
) -> list[dict[str, Any]]:
    manifests = root / "manifests"
    if checkpoint.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise RuntimeError(f"invalid checkpoint format for {dataset}")
    initial_stats = source.get("initial_selection_stats")
    if not isinstance(initial_stats, dict):
        raise RuntimeError(f"missing initial_selection_stats for {dataset}")
    initial_stats_fingerprint = _require_sha256(
        source.get("initial_selection_stats_fingerprint"),
        label=f"initial selection stats fingerprint for {dataset}",
    )
    if _stable_json_digest(initial_stats) != initial_stats_fingerprint:
        raise RuntimeError(
            f"initial_selection_stats body/fingerprint mismatch for {dataset}"
        )
    exclusions_raw = source.get("build_exclusions")
    expected_exclusion_fields = {
        "source_image_ids",
        "source_canonical_rgb_sha256",
        "saved_canonical_rgb_sha256",
    }
    if not isinstance(exclusions_raw, dict) or set(exclusions_raw) != expected_exclusion_fields:
        raise RuntimeError(f"invalid build_exclusions for {dataset}")
    for key in expected_exclusion_fields:
        values = exclusions_raw[key]
        if (
            not isinstance(values, list)
            or any(not isinstance(value, str) or not value for value in values)
            or values != sorted(set(values))
        ):
            raise RuntimeError(
                f"build_exclusions {key} is not a sorted unique string list for {dataset}"
            )
    source_exclusions_fingerprint = _require_sha256(
        source.get("exclusions_fingerprint"),
        label=f"source exclusions fingerprint for {dataset}",
    )
    checkpoint_exclusions_fingerprint = _require_sha256(
        checkpoint.get("exclusions_fingerprint"),
        label=f"checkpoint exclusions fingerprint for {dataset}",
    )
    if (
        _stable_json_digest(exclusions_raw) != source_exclusions_fingerprint
        or source_exclusions_fingerprint != checkpoint_exclusions_fingerprint
    ):
        raise RuntimeError(f"build exclusions body/fingerprint mismatch for {dataset}")
    try:
        exclusions = BuildExclusions(
            source_image_ids=frozenset(exclusions_raw["source_image_ids"]),
            source_canonical_rgb_sha256=frozenset(
                exclusions_raw["source_canonical_rgb_sha256"]
            ),
            saved_canonical_rgb_sha256=frozenset(
                exclusions_raw["saved_canonical_rgb_sha256"]
            ),
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"invalid build exclusions for {dataset}") from exc
    candidates = _read_jsonl_strict(
        manifests / f"{dataset}_candidate_pool.jsonl",
        label=f"{dataset} candidate-pool manifest",
    )
    screening = _read_jsonl_strict(
        manifests / f"{dataset}_screening.jsonl",
        label=f"{dataset} screening manifest",
    )
    samples = _read_jsonl_strict(
        manifests / f"{dataset}_samples.jsonl",
        label=f"{dataset} sample sidecar",
    )
    mappings = _read_jsonl_strict(
        root / dataset / f"{dataset}_mapping.jsonl",
        label=f"{dataset} mapping",
    )
    checkpoint_samples = checkpoint.get("samples")
    if _canonical_json(checkpoint_samples) != _canonical_json(samples):
        raise RuntimeError(f"checkpoint/public samples mismatch for {dataset}")
    if len(samples) != expected_count or len(mappings) != expected_count:
        raise RuntimeError(f"completed cohort size mismatch for {dataset}")
    if stats.get("retained_count") != expected_count:
        raise RuntimeError(f"selection_stats retained_count mismatch for {dataset}")

    next_position = checkpoint.get("next_candidate_position")
    if (
        type(next_position) is not int
        or not 0 <= next_position <= len(candidates)
        or next_position != len(screening)
    ):
        raise RuntimeError(f"invalid completed checkpoint position for {dataset}")
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
    pool_fingerprint_rows: list[dict[str, Any]] = []
    reconstructed_candidates: list[Candidate] = []
    for position, candidate in enumerate(candidates):
        if not isinstance(candidate, dict) or set(candidate) != candidate_fields:
            raise RuntimeError(f"invalid candidate schema for {dataset}:{position}")
        if type(candidate["candidate_position"]) is not int or candidate[
            "candidate_position"
        ] != position:
            raise RuntimeError(f"non-contiguous candidate positions for {dataset}")
        pool_fingerprint_rows.append(
            {
                "source_dataset": candidate["source_dataset"],
                "source_split": candidate["source_split"],
                "source_index": candidate["source_index"],
                "source_split_index": candidate["source_split_index"],
                "source_question_id": candidate["source_question_id"],
                "source_image_id": candidate["source_image_id"],
                "source_canonical_rgb_sha256": candidate[
                    "source_canonical_rgb_sha256"
                ],
                "task_type": candidate["task_type"],
                "question": candidate["question"],
                "answers": candidate["answers"],
                "options": candidate["options"],
                "ground_truth_letter": candidate["ground_truth_letter"],
                "ground_truth_text": candidate["ground_truth_text"],
                "prompt": candidate["prompt"],
                "mapping_payload": candidate["mapping_template"],
                "legacy_temperature_zero": candidate[
                    "legacy_temperature_zero"
                ],
                "provenance": candidate["provenance"],
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
                    image_loader=lambda: None,  # Never invoked by state reconstruction.
                    source_canonical_rgb_sha256=candidate[
                        "source_canonical_rgb_sha256"
                    ],
                    options=tuple(candidate["options"]),
                    ground_truth_letter=candidate["ground_truth_letter"],
                    ground_truth_text=candidate["ground_truth_text"],
                    legacy_temperature_zero=candidate[
                        "legacy_temperature_zero"
                    ],
                    provenance=candidate["provenance"],
                )
            )
        except (KeyError, TypeError) as exc:
            raise RuntimeError(
                f"invalid candidate values for {dataset}:{position}"
            ) from exc
    recomputed_pool_fingerprint = _stable_json_digest(pool_fingerprint_rows)
    if recomputed_pool_fingerprint != source["candidate_pool_fingerprint"]:
        raise RuntimeError(f"candidate-pool body/fingerprint mismatch for {dataset}")

    committed: dict[int, dict[str, Any]] = {}
    accepted: dict[int, tuple[int, dict[str, Any]]] = {}
    seen_source_ids: set[str] = set()
    seen_source_hashes: set[str] = set()
    for row_number, row in enumerate(screening):
        if not isinstance(row, dict):
            raise RuntimeError(f"invalid screening row for {dataset}:{row_number}")
        position = row.get("candidate_position")
        if type(position) is not int or position != row_number:
            raise RuntimeError(f"non-contiguous screening positions for {dataset}")
        committed[position] = row
        candidate = candidates[position]
        for key in (
            "source_dataset",
            "source_split",
            "source_index",
            "source_split_index",
            "source_question_id",
            "source_image_id",
        ):
            if _canonical_json(row.get(key)) != _canonical_json(candidate[key]):
                raise RuntimeError(
                    f"screening/candidate lineage mismatch for {dataset}:{position}"
                )
        source_id = row.get("source_image_id")
        if source_id is not None:
            seen_source_ids.add(str(source_id))
        source_hash = row.get("source_canonical_rgb_sha256")
        if source_hash is not None:
            seen_source_hashes.add(
                _require_sha256(
                    source_hash,
                    label=f"screening source digest for {dataset}:{position}",
                )
            )
        row_accepted = row.get("accepted")
        reason = row.get("reason")
        if type(row_accepted) is not bool or row_accepted != (reason == "accepted"):
            raise RuntimeError(f"invalid screening decision for {dataset}:{position}")
        if row_accepted:
            retained_index = row.get("retained_index")
            if type(retained_index) is not int or retained_index in accepted:
                raise RuntimeError(
                    f"invalid screening retained index for {dataset}:{position}"
                )
            accepted[retained_index] = (position, row)
    if set(accepted) != set(range(expected_count)):
        raise RuntimeError(f"accepted screening cohort mismatch for {dataset}")
    if accepted[expected_count - 1][0] != next_position - 1:
        raise RuntimeError(f"screening continued after target for {dataset}")

    reconstructed_bundle = SourceBundle(
        dataset_name=dataset,
        candidates=reconstructed_candidates,
        selection_stats=initial_stats,
        source_metadata=source.get("source_metadata", {}),
    )
    reconstructed_config = BuildConfig(
        output_root=root,
        target=expected_count,
        seed=checkpoint["seed"],
        resume=True,
        exclusions=exclusions,
    )
    (
        reconstructed_stats,
        reconstructed_accepted,
        reconstructed_source_ids,
        reconstructed_source_hashes,
        reconstructed_saved_hashes,
    ) = _reconstruct_committed_state(
        reconstructed_bundle,
        reconstructed_config,
        committed,
        next_position,
    )
    if _canonical_json(reconstructed_stats) != _canonical_json(stats):
        raise RuntimeError(f"selection_stats reconstruction mismatch for {dataset}")
    if set(reconstructed_accepted) != set(accepted):
        raise RuntimeError(f"accepted state reconstruction mismatch for {dataset}")

    sample_fields = {
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
        "mapping",
        "options",
        "ground_truth_letter",
        "ground_truth_text",
        "legacy_temperature_zero",
        "jpeg_settings",
        "provenance",
    }
    image_rows = {
        row["mapping_index"]: row
        for row in _read_jsonl_strict(
            report_root / "image_hash_manifest.jsonl",
            label="image hash manifest",
        )
        if isinstance(row, dict) and row.get("dataset") == dataset
    }
    if set(image_rows) != set(range(expected_count)):
        raise RuntimeError(f"image hash cohort mismatch for {dataset}")
    expected_saved_hashes: set[str] = set()
    for retained_index, sample in enumerate(samples):
        if not isinstance(sample, dict) or set(sample) != sample_fields:
            raise RuntimeError(f"invalid sample schema for {dataset}:{retained_index}")
        if sample["dataset"] != dataset or sample["retained_index"] != retained_index:
            raise RuntimeError(f"invalid sample identity for {dataset}:{retained_index}")
        expected_filename = f"images/{dataset}_{retained_index:04d}.jpg"
        if sample["image_filename"] != expected_filename:
            raise RuntimeError(f"invalid sample filename for {dataset}:{retained_index}")
        if _canonical_json(sample["mapping"]) != _canonical_json(mappings[retained_index]):
            raise RuntimeError(f"sample/mapping mismatch for {dataset}:{retained_index}")
        position, row = accepted[retained_index]
        candidate = candidates[position]
        for key in (
            "source_dataset",
            "source_split",
            "source_index",
            "source_split_index",
            "source_question_id",
            "source_image_id",
            "source_canonical_rgb_sha256",
        ):
            if _canonical_json(sample[key]) != _canonical_json(row.get(key)):
                raise RuntimeError(
                    f"sample/screening lineage mismatch for {dataset}:{retained_index}"
                )
        for sample_key, candidate_key in (
            ("source_dataset", "source_dataset"),
            ("source_split", "source_split"),
            ("source_index", "source_index"),
            ("source_split_index", "source_split_index"),
            ("source_question_id", "source_question_id"),
            ("source_image_id", "source_image_id"),
            ("prompt", "prompt"),
            ("options", "options"),
            ("ground_truth_letter", "ground_truth_letter"),
            ("ground_truth_text", "ground_truth_text"),
            ("legacy_temperature_zero", "legacy_temperature_zero"),
            ("provenance", "provenance"),
        ):
            if _canonical_json(sample[sample_key]) != _canonical_json(
                candidate[candidate_key]
            ):
                raise RuntimeError(
                    f"sample/candidate mismatch for {dataset}:{retained_index}"
                )
        for key in (
            "image_filename",
            "saved_image_sha256",
            "saved_canonical_rgb_sha256",
            "perceptual_hash",
        ):
            if sample[key] != row.get(key):
                raise RuntimeError(
                    f"sample/screening artifact mismatch for {dataset}:{retained_index}"
                )
        if sample["canonical_rgb_sha256"] != sample[
            "saved_canonical_rgb_sha256"
        ]:
            raise RuntimeError(f"sample canonical hash alias mismatch for {dataset}")
        if sample["blind_prediction"] != sample[
            "blind_black_image_control_prediction"
        ] or sample["blind_correct"] is not False or sample[
            "blind_black_image_control_correct"
        ] is not False:
            raise RuntimeError(f"sample blind aliases mismatch for {dataset}")
        if sample["full_prediction"] != sample["full_visual_prediction"] or sample[
            "full_correct"
        ] is not True or sample["full_visual_correct"] is not True:
            raise RuntimeError(f"sample Full aliases mismatch for {dataset}")
        if _canonical_json(sample["jpeg_settings"]) != _canonical_json(JPEG_SETTINGS):
            raise RuntimeError(f"sample JPEG settings mismatch for {dataset}")
        expected_mapping = dict(candidate["mapping_template"])
        expected_mapping["image_filename"] = expected_filename
        if _canonical_json(sample["mapping"]) != _canonical_json(expected_mapping):
            raise RuntimeError(f"sample mapping template mismatch for {dataset}")
        image_row = image_rows[retained_index]
        for sample_key, image_key in (
            ("image_filename", "image_filename"),
            ("saved_image_sha256", "saved_image_sha256"),
            ("saved_canonical_rgb_sha256", "canonical_rgb_sha256"),
            ("perceptual_hash", "perceptual_hash"),
            ("source_dataset", "source_dataset"),
            ("source_split", "source_split"),
            ("source_index", "source_index"),
            ("source_question_id", "source_question_id"),
            ("source_image_id", "source_image_id"),
            ("source_canonical_rgb_sha256", "source_canonical_rgb_sha256"),
        ):
            if _canonical_json(sample[sample_key]) != _canonical_json(
                image_row[image_key]
            ):
                raise RuntimeError(
                    f"sample/image-hash manifest mismatch for {dataset}:{retained_index}"
                )
        expected_saved_hashes.add(
            _require_sha256(
                sample["saved_canonical_rgb_sha256"],
                label=f"sample saved digest for {dataset}:{retained_index}",
            )
        )

    accepted_source_ids = [
        str(sample["source_image_id"])
        for sample in samples
        if sample["source_image_id"] is not None
    ]
    accepted_source_hashes = [
        sample["source_canonical_rgb_sha256"] for sample in samples
    ]
    if len(set(accepted_source_ids)) != len(accepted_source_ids):
        raise RuntimeError(f"accepted source IDs are not unique for {dataset}")
    if len(set(accepted_source_hashes)) != len(accepted_source_hashes):
        raise RuntimeError(f"accepted source hashes are not unique for {dataset}")
    if len(expected_saved_hashes) != expected_count:
        raise RuntimeError(f"accepted saved hashes are not unique for {dataset}")

    if reconstructed_source_ids != seen_source_ids:
        raise RuntimeError(f"internal reconstructed source-ID mismatch for {dataset}")
    if reconstructed_source_hashes != seen_source_hashes:
        raise RuntimeError(f"internal reconstructed source-hash mismatch for {dataset}")
    if reconstructed_saved_hashes != expected_saved_hashes:
        raise RuntimeError(f"internal reconstructed saved-hash mismatch for {dataset}")
    if _require_sorted_string_set(
        checkpoint, "seen_source_image_ids"
    ) != reconstructed_source_ids:
        raise RuntimeError(f"checkpoint source-ID set mismatch for {dataset}")
    if _require_sorted_string_set(
        checkpoint, "seen_source_canonical_rgb_sha256", sha256=True
    ) != reconstructed_source_hashes:
        raise RuntimeError(f"checkpoint source-hash set mismatch for {dataset}")
    if _require_sorted_string_set(
        checkpoint, "seen_saved_canonical_rgb_sha256", sha256=True
    ) != reconstructed_saved_hashes:
        raise RuntimeError(f"checkpoint saved-hash set mismatch for {dataset}")
    return samples


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _stable_json_digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise RuntimeError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _validate_model_manifest(model: Any, *, dataset: str) -> str:
    if not isinstance(model, dict):
        raise RuntimeError(f"missing model fingerprint for {dataset}")
    expected_keys = {"path", "algorithm", "aggregate_sha256", "files"}
    if set(model) != expected_keys:
        raise RuntimeError(
            f"invalid model manifest fields for {dataset}: {sorted(model)!r}"
        )
    model_path = model["path"]
    if not isinstance(model_path, str) or not model_path or not Path(model_path).is_absolute():
        raise RuntimeError(f"invalid model path in manifest for {dataset}")
    if model["algorithm"] != _MODEL_MANIFEST_ALGORITHM:
        raise RuntimeError(f"invalid model manifest algorithm for {dataset}")
    files = model["files"]
    if not isinstance(files, list) or not files:
        raise RuntimeError(f"model manifest files must be non-empty for {dataset}")
    paths: list[str] = []
    for index, entry in enumerate(files):
        if not isinstance(entry, dict) or set(entry) != {
            "path",
            "size_bytes",
            "sha256",
        }:
            raise RuntimeError(
                f"invalid model file manifest entry {index} for {dataset}"
            )
        relative = entry["path"]
        if not isinstance(relative, str) or not relative or "\\" in relative:
            raise RuntimeError(
                f"invalid model file path at entry {index} for {dataset}"
            )
        pure = PurePosixPath(relative)
        if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
            raise RuntimeError(
                f"unsafe model file path at entry {index} for {dataset}"
            )
        if type(entry["size_bytes"]) is not int or entry["size_bytes"] < 0:
            raise RuntimeError(
                f"invalid model file size at entry {index} for {dataset}"
            )
        _require_sha256(
            entry["sha256"],
            label=f"model file digest at entry {index} for {dataset}",
        )
        paths.append(relative)
    if paths != sorted(paths) or len(set(paths)) != len(paths):
        raise RuntimeError(
            f"model file manifest paths are not unique canonical order for {dataset}"
        )
    aggregate = _require_sha256(
        model["aggregate_sha256"],
        label=f"model aggregate digest for {dataset}",
    )
    if aggregate != _stable_json_digest(files):
        raise RuntimeError(
            f"model aggregate digest does not match file manifest for {dataset}"
        )
    return aggregate


def _validate_formal_protocol(protocol: dict[str, Any], *, dataset: str) -> None:
    expected_top_level = {
        "protocol_version",
        "scorers",
        "canonical_hash",
        "jpeg_settings",
        "runner",
    }
    if set(protocol) != expected_top_level:
        raise RuntimeError(f"invalid formal protocol fields for {dataset}")
    if protocol["protocol_version"] != CONSTRUCTION_PROTOCOL_VERSION:
        raise RuntimeError(f"invalid formal protocol version for {dataset}")
    expected_scorers = {
        "open": "legacy-official-vqa-process-plus-truth-substring-v1",
        "multiple_choice": (
            "legacy-letter-regex-then-ground-truth-substring-v1;"
            "VQAv2=A-D;ScienceQA=A-F"
        ),
    }
    if _canonical_json(protocol["scorers"]) != _canonical_json(expected_scorers):
        raise RuntimeError(f"invalid formal scorer contract for {dataset}")
    if protocol["canonical_hash"] != CANONICAL_HASH_DESCRIPTION:
        raise RuntimeError(f"invalid formal canonical hash contract for {dataset}")
    if _canonical_json(protocol["jpeg_settings"]) != _canonical_json(JPEG_SETTINGS):
        raise RuntimeError(f"invalid formal JPEG contract for {dataset}")

    runner = protocol["runner"]
    if not isinstance(runner, dict) or set(runner) != {
        "runner",
        "model",
        "runtime",
        "visual_patch_tokens",
        "blind_control",
        "generation",
    }:
        raise RuntimeError(f"invalid formal runner fields for {dataset}")
    if runner["runner"] != "LlavaConstructionRunner":
        raise RuntimeError(f"invalid formal runner identity for {dataset}")
    if type(runner["visual_patch_tokens"]) is not int or runner["visual_patch_tokens"] != 576:
        raise RuntimeError(f"invalid formal visual patch count for {dataset}")
    expected_control = {
        "name": "blind_black_image_control",
        "image_mode": "RGB",
        "image_size": [336, 336],
        "color": [0, 0, 0],
        "passes_through_visual_encoder": True,
    }
    if _canonical_json(runner["blind_control"]) != _canonical_json(expected_control):
        raise RuntimeError(f"invalid formal blind-control contract for {dataset}")
    expected_generation = {
        "max_new_tokens": 32,
        "do_sample": False,
        "num_beams": 1,
        "legacy_textvqa_temperature_zero": True,
    }
    if _canonical_json(runner["generation"]) != _canonical_json(expected_generation):
        raise RuntimeError(f"invalid formal generation contract for {dataset}")

    runtime = runner["runtime"]
    if not isinstance(runtime, dict) or set(runtime) != {
        "python",
        "platform",
        "torch",
        "transformers",
        "datasets",
        "numpy",
        "pillow",
        "libjpeg",
        "cuda_runtime",
        "cudnn",
        "gpu",
    }:
        raise RuntimeError(f"invalid formal runtime identity for {dataset}")
    for key in (
        "python",
        "platform",
        "torch",
        "transformers",
        "datasets",
        "numpy",
        "pillow",
    ):
        if not isinstance(runtime[key], str) or not runtime[key]:
            raise RuntimeError(f"invalid runtime {key} for {dataset}")
    if runtime["libjpeg"] is not None and not isinstance(runtime["libjpeg"], str):
        raise RuntimeError(f"invalid runtime libjpeg for {dataset}")
    if runtime["cuda_runtime"] is not None and not isinstance(
        runtime["cuda_runtime"], str
    ):
        raise RuntimeError(f"invalid runtime cuda_runtime for {dataset}")
    if runtime["cudnn"] is not None and type(runtime["cudnn"]) is not int:
        raise RuntimeError(f"invalid runtime cudnn for {dataset}")
    gpu = runtime["gpu"]
    if not isinstance(gpu, dict) or set(gpu) != {
        "name",
        "total_memory_bytes",
        "compute_capability",
    }:
        raise RuntimeError(f"invalid formal GPU identity for {dataset}")
    if not isinstance(gpu["name"], str) or not gpu["name"]:
        raise RuntimeError(f"invalid GPU name for {dataset}")
    if type(gpu["total_memory_bytes"]) is not int or gpu["total_memory_bytes"] <= 0:
        raise RuntimeError(f"invalid GPU memory identity for {dataset}")
    capability = gpu["compute_capability"]
    if (
        not isinstance(capability, list)
        or len(capability) != 2
        or any(type(value) is not int or value < 0 for value in capability)
    ):
        raise RuntimeError(f"invalid GPU compute capability for {dataset}")


def _validate_protocol_and_seed(
    source: dict[str, Any], checkpoint: dict[str, Any], *, dataset: str
) -> tuple[dict[str, Any], str, dict[str, Any], str, int]:
    protocol = source.get("construction_protocol")
    if not isinstance(protocol, dict):
        raise RuntimeError(f"missing construction protocol for {dataset}")
    fingerprint = _require_sha256(
        source.get("construction_protocol_fingerprint"),
        label=f"construction protocol fingerprint for {dataset}",
    )
    checkpoint_fingerprint = _require_sha256(
        checkpoint.get("construction_protocol_fingerprint"),
        label=f"checkpoint construction protocol fingerprint for {dataset}",
    )
    if fingerprint != checkpoint_fingerprint:
        raise RuntimeError(f"protocol fingerprint mismatch for {dataset}")
    if _stable_json_digest(protocol) != fingerprint:
        raise RuntimeError(
            f"construction protocol body does not match fingerprint for {dataset}"
        )
    _validate_formal_protocol(protocol, dataset=dataset)
    runner = protocol.get("runner")
    if not isinstance(runner, dict):
        raise RuntimeError(f"missing protocol runner for {dataset}")
    model = runner.get("model")
    aggregate = _validate_model_manifest(model, dataset=dataset)
    seed = checkpoint.get("seed")
    if type(seed) is not int or seed < 0:
        raise RuntimeError(f"invalid seed for {dataset}")
    if seed != SEED:
        raise RuntimeError(
            f"formal checkpoint seed must equal the frozen seed {SEED} for {dataset}"
        )
    return protocol, fingerprint, model, aggregate, seed


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _file_manifest(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    return {
        "path": (
            path.relative_to(relative_to).as_posix()
            if relative_to is not None
            else str(path.resolve())
        ),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _gpu_manifest() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,uuid,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(
        command, check=True, capture_output=True, text=True, timeout=30
    )
    rows: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",", 4)]
        if len(fields) != 5:
            raise RuntimeError(f"unexpected nvidia-smi output row: {line!r}")
        rows.append(
            {
                "physical_index": int(fields[0]),
                "name": fields[1],
                "uuid": fields[2],
                "driver_version": fields[3],
                "memory_mib": int(fields[4]),
            }
        )
    return rows


def generate(dataset_root: Path | str, *, reports_dir: Path | str) -> Path:
    root = Path(dataset_root).resolve()
    source_cache_root = resolve_builder_source_cache(
        root, require_existing=True
    )
    validate_coco_identity_cache(source_cache_root, allow_create=False)
    report_root = assert_output_separate(
        reports_dir, {"dataset root": root}
    )
    report_path = report_root / "final_validation.json"
    validation = _load_json(report_path)
    if validation.get("overall_pass") is not True:
        raise RuntimeError("refusing reproducibility manifest for failed validation")
    if validation.get("expected_count") != 1000:
        raise RuntimeError("formal reproducibility manifest requires expected_count=1000")
    if Path(str(validation.get("dataset_root"))).resolve() != root:
        raise RuntimeError("validation report belongs to a different dataset root")
    _validate_formal_validation_gate(validation)
    validation_datasets = validation.get("datasets")
    if not isinstance(validation_datasets, dict) or set(validation_datasets) != set(
        DATASET_NAMES
    ):
        raise RuntimeError("validation report does not cover the exact four datasets")
    for dataset in DATASET_NAMES:
        result = validation_datasets[dataset]
        if not isinstance(result, dict) or result.get("pass") is not True:
            raise RuntimeError(f"validation report does not pass for {dataset}")
        counts = result.get("counts")
        if not isinstance(counts, dict):
            raise RuntimeError(f"validation report is missing counts for {dataset}")
        for key in (
            "mapping_rows",
            "sample_sidecar_rows",
            "actual_files_in_images_directory",
            "existing_referenced_images",
        ):
            if type(counts.get(key)) is not int or counts[key] != 1000:
                raise RuntimeError(
                    f"validation report {key} is not the formal cohort for {dataset}"
                )
    image_hash_manifest = _validate_image_hash_manifest(
        root, report_root, expected_count=1000
    )
    validation_output_paths = {
        "final_validation_json": report_path,
        "final_validation_markdown": report_root / "final_validation.md",
        "image_hash_manifest": report_root / "image_hash_manifest.jsonl",
        "cross_dataset_overlap": report_root / "cross_dataset_overlap.csv",
    }
    validation_outputs = {
        key: _file_manifest(path) for key, path in validation_output_paths.items()
    }

    source_manifests: dict[str, Any] = {}
    checkpoint_files: dict[str, Any] = {}
    protocol_fingerprints: set[str] = set()
    model_fingerprints: set[str] = set()
    model_manifest: dict[str, Any] | None = None
    construction_protocol: dict[str, Any] | None = None
    seeds: set[int] = set()
    prior_source_hashes: set[str] = set()
    prior_saved_hashes: set[str] = set()
    vqav2_open_source_ids: set[str] = set()
    for dataset in DATASET_NAMES:
        source_path = root / "manifests" / f"{dataset}_source.json"
        checkpoint_path = root / "manifests" / f"{dataset}_checkpoint.json"
        stats_path = root / "manifests" / f"{dataset}_selection_stats.json"
        source = _load_json(source_path)
        checkpoint = _load_json(checkpoint_path)
        stats = _load_json(stats_path)
        if (
            source.get("dataset") != dataset
            or checkpoint.get("dataset") != dataset
            or stats.get("dataset") != dataset
        ):
            raise RuntimeError(f"manifest dataset mismatch for {dataset}")
        if _canonical_json(checkpoint.get("selection_stats")) != _canonical_json(stats):
            raise RuntimeError(f"public stats differ from checkpoint for {dataset}")
        if type(checkpoint.get("target")) is not int or checkpoint["target"] != 1000:
            raise RuntimeError(f"formal checkpoint target must be 1000 for {dataset}")
        if type(stats.get("target")) is not int or stats["target"] != 1000:
            raise RuntimeError(f"formal selection_stats target must be 1000 for {dataset}")
        candidate_pool_fingerprint = _require_sha256(
            source.get("candidate_pool_fingerprint"),
            label=f"source candidate-pool fingerprint for {dataset}",
        )
        checkpoint_pool_fingerprint = _require_sha256(
            checkpoint.get("candidate_pool_fingerprint"),
            label=f"checkpoint candidate-pool fingerprint for {dataset}",
        )
        if candidate_pool_fingerprint != checkpoint_pool_fingerprint:
            raise RuntimeError(f"candidate-pool fingerprint mismatch for {dataset}")
        source_metadata = source.get("source_metadata")
        if not isinstance(source_metadata, dict):
            raise RuntimeError(f"missing source_metadata for {dataset}")
        metadata_fingerprint = _require_sha256(
            source.get("source_metadata_stable_fingerprint"),
            label=f"stable source_metadata fingerprint for {dataset}",
        )
        if metadata_fingerprint != _builder_stable_json_digest(
            _stable_source_metadata(source_metadata)
        ):
            raise RuntimeError(
                f"stable source_metadata body does not match fingerprint for {dataset}"
            )
        if dataset in {"VQAv2_Open", "VQAv2_MC"}:
            _validate_vqa_source_archives(
                source_cache_root, source_metadata, dataset=dataset
            )
        (
            protocol,
            fingerprint,
            current_model,
            aggregate,
            seed,
        ) = _validate_protocol_and_seed(source, checkpoint, dataset=dataset)
        if type(stats.get("seed")) is not int or stats["seed"] != seed:
            raise RuntimeError(f"checkpoint/selection_stats seed mismatch for {dataset}")
        _validate_formal_exclusion_lineage(
            dataset,
            source.get("build_exclusions"),
            prior_source_hashes=prior_source_hashes,
            prior_saved_hashes=prior_saved_hashes,
            vqav2_open_source_ids=vqav2_open_source_ids,
        )
        validated_samples = _validate_completed_build_artifacts(
            root,
            report_root,
            dataset,
            source,
            checkpoint,
            stats,
            expected_count=1000,
        )
        if not isinstance(validated_samples, list):
            raise RuntimeError(
                f"completed artifact validation returned invalid samples for {dataset}"
            )
        if dataset == "VQAv2_Open":
            vqav2_open_source_ids = {
                str(sample["source_image_id"])
                for sample in validated_samples
                if sample.get("source_image_id") is not None
            }
        prior_source_hashes.update(
            str(sample["source_canonical_rgb_sha256"])
            for sample in validated_samples
        )
        prior_saved_hashes.update(
            str(sample["saved_canonical_rgb_sha256"])
            for sample in validated_samples
        )
        protocol_fingerprints.add(fingerprint)
        if construction_protocol is None:
            construction_protocol = protocol
        elif _canonical_json(construction_protocol) != _canonical_json(protocol):
            raise RuntimeError(
                "datasets have non-identical construction protocol bodies"
            )
        model_fingerprints.add(aggregate)
        if model_manifest is None:
            model_manifest = current_model
        seeds.add(seed)
        source_manifests[dataset] = {
            "path": str(source_path),
            "sha256": sha256_file(source_path),
            "candidate_pool_fingerprint": source.get("candidate_pool_fingerprint"),
            "source_metadata": source.get("source_metadata"),
        }
        checkpoint_files[dataset] = {
            "checkpoint": _file_manifest(checkpoint_path),
            "selection_stats": _file_manifest(stats_path),
            "candidate_pool": _file_manifest(
                root / "manifests" / f"{dataset}_candidate_pool.jsonl"
            ),
            "screening": _file_manifest(
                root / "manifests" / f"{dataset}_screening.jsonl"
            ),
            "samples": _file_manifest(
                root / "manifests" / f"{dataset}_samples.jsonl"
            ),
            "mapping": _file_manifest(
                root / dataset / f"{dataset}_mapping.jsonl"
            ),
        }
    if len(protocol_fingerprints) != 1:
        raise RuntimeError("datasets were built under different construction protocols")
    if len(model_fingerprints) != 1:
        raise RuntimeError("datasets were built with different model artifacts")
    if len(seeds) != 1:
        raise RuntimeError("datasets were built with different global seeds")

    source_cache_paths = sorted(source_cache_root.rglob("*"))
    source_cache_files: list[dict[str, Any]] = []
    for path in source_cache_paths:
        relative = path.relative_to(source_cache_root)
        if "build_env" in relative.parts:
            continue
        if path.is_symlink():
            raise RuntimeError(f"source_cache contains a symlink: {path}")
        resolved_path = resolve_source_cache_child(
            source_cache_root, relative.as_posix()
        )
        if resolved_path.is_file():
            source_cache_files.append(
                _file_manifest(resolved_path, relative_to=source_cache_root)
            )
    code_files = [
        _file_manifest(path, relative_to=RELEASE_ROOT)
        for path in sorted(BUILDER_CODE_ROOT.rglob("*"))
        if path.is_file()
        and path.suffix in {".py", ".md", ".txt"}
        and "__pycache__" not in path.parts
    ]
    output = report_root / "reproducibility_manifest.json"
    payload = {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(root),
        "old_dataset_root_read_only": (
            str(OLD_DATASET_ROOT.resolve()) if OLD_DATASET_ROOT is not None else None
        ),
        "validation": _file_manifest(report_path),
        "validation_outputs": validation_outputs,
        "validated_image_hash_manifest": image_hash_manifest,
        "global_seed": next(iter(seeds)),
        "determinism": {
            "pythonhashseed": str(next(iter(seeds))),
            "python_random_seeded": True,
            "numpy_seeded": True,
            "torch_cpu_and_cuda_seeded": True,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "vqav2_mc_rng": "local SHA-256(global_seed, question_id)-derived RNG",
            "generation": {
                "do_sample": False,
                "num_beams": 1,
                "max_new_tokens": 32,
            },
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": {
                name: _package_version(name)
                for name in (
                    "torch",
                    "transformers",
                    "datasets",
                    "Pillow",
                    "numpy",
                    "requests",
                    "accelerate",
                )
            },
            "gpus": _gpu_manifest(),
        },
        "construction_protocol_fingerprint": next(iter(protocol_fingerprints)),
        "construction_protocol": construction_protocol,
        "model": model_manifest,
        "source_manifests": source_manifests,
        "local_source_cache_files": source_cache_files,
        "build_artifacts": checkpoint_files,
        "builder_code_files": code_files,
    }
    atomic_write_json(output, payload)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--reports-dir",
        type=Path,
        required=True,
        help="explicit report directory containing final_validation.json",
    )
    args = parser.parse_args(argv)
    print(generate(args.dataset_root, reports_dir=args.reports_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
