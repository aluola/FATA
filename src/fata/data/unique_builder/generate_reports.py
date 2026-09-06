#!/usr/bin/env python3
"""Generate selection, old/new quality, and paper-ready LaTeX reports.

The generator is deliberately inference-free.  It summarizes facts already
recorded by the builders, the independent validator, and the read-only audit of
the legacy datasets.  Unknown values are emitted as ``N/A``; a hard-check result
or a target size is never used as a substitute for an unrecorded measurement.

The public entry point is :func:`generate_reports`.  Run ``python
generate_reports.py --help`` for the command-line interface.
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
import tempfile
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps
from fata.utils.paths import (
    assert_no_output_file_collision,
    assert_output_separate,
    resolve_owned_output_path,
)


DATASET_NAMES = ("TextVQA_Open", "VQAv2_Open", "VQAv2_MC", "ScienceQA_MC")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
NA = "N/A"
_MISSING = object()
DEFAULT_OLD_DATASET_ROOT = (
    Path(os.environ["FATA_LEGACY_DATA_ROOT"])
    if os.environ.get("FATA_LEGACY_DATA_ROOT")
    else None
)

FUNNEL_COLUMNS = (
    "dataset",
    "source_splits",
    "source_qa_records",
    "source_unique_images",
    "eligible_unique_images",
    "eligible_unique_images_basis",
    "invalid_metadata",
    "yes_no_removed",
    "missing_or_corrupt_images",
    "duplicate_source_ids",
    "exact_image_hash_duplicates",
    "unique_images_evaluated",
    "blind_wrong",
    "full_correct_among_blind_wrong",
    "target_reached_source_index",
    "retained",
    "acceptance_rate",
    "retained_train",
    "retained_validation",
    "retained_test",
)

FUNNEL_LABELS = {
    "dataset": "Dataset",
    "source_splits": "Source split(s)",
    "source_qa_records": "Source QA records",
    "source_unique_images": "Source unique images",
    "eligible_unique_images": "Eligible unique images",
    "eligible_unique_images_basis": "Eligible-image basis",
    "invalid_metadata": "Invalid metadata",
    "yes_no_removed": "Yes/no removed",
    "missing_or_corrupt_images": "Missing/corrupt images",
    "duplicate_source_ids": "Duplicate source IDs",
    "exact_image_hash_duplicates": "Exact image-hash duplicates",
    "unique_images_evaluated": "Unique images evaluated",
    "blind_wrong": "Blind-wrong",
    "full_correct_among_blind_wrong": "Full-correct",
    "target_reached_source_index": "Target source index",
    "retained": "Retained",
    "acceptance_rate": "Acceptance rate",
    "retained_train": "Retained train",
    "retained_validation": "Retained validation",
    "retained_test": "Retained test",
}

QUALITY_COLUMNS = (
    "dataset",
    "old_rows",
    "old_unique_rgb_images",
    "old_duplicate_rows",
    "old_duplicate_rate",
    "new_rows",
    "new_unique_rgb_images",
    "new_duplicate_rows",
    "new_duplicate_rate",
    "old_new_image_overlap",
    "old_new_sample_overlap",
    "old_full_construction_accuracy",
    "old_blind_control_construction_accuracy",
    "new_full_construction_accuracy",
    "new_blind_control_construction_accuracy",
)

QUALITY_LABELS = {
    "dataset": "Dataset",
    "old_rows": "Old rows",
    "old_unique_rgb_images": "Old unique RGB images",
    "old_duplicate_rows": "Old duplicate rows",
    "old_duplicate_rate": "Old duplicate rate",
    "new_rows": "New rows",
    "new_unique_rgb_images": "New unique RGB images",
    "new_duplicate_rows": "New duplicate rows",
    "new_duplicate_rate": "New duplicate rate",
    "old_new_image_overlap": "Old--new image overlap",
    "old_new_sample_overlap": "Old--new sample overlap",
    "old_full_construction_accuracy": "Old Full construction accuracy",
    "old_blind_control_construction_accuracy": (
        "Old blind-control construction accuracy"
    ),
    "new_full_construction_accuracy": "New Full construction accuracy",
    "new_blind_control_construction_accuracy": (
        "New blind-control construction accuracy"
    ),
}


def _normalize_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _is_missing(value: Any) -> bool:
    if value is _MISSING or value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "n/a", "na", "unknown", "null", "none"}
    return False


def _public(value: Any) -> Any:
    """Convert missing/non-finite values to the report's explicit marker."""

    if _is_missing(value):
        return NA
    if isinstance(value, float) and not math.isfinite(value):
        return NA
    return value


def _as_number(value: Any) -> int | float | None:
    if _is_missing(value) or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value if math.isfinite(float(value)) else None
    if isinstance(value, str):
        candidate = value.strip().replace(",", "")
        is_percent = candidate.endswith("%")
        if is_percent:
            candidate = candidate[:-1].strip()
        try:
            number = float(candidate)
        except ValueError:
            return None
        if not math.isfinite(number):
            return None
        if is_percent:
            number /= 100.0
        return int(number) if number.is_integer() else number
    return None


def _json_load(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in {path}: {error}") from error


def _saved_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_validation_artifact_binding(
    root: Path, dataset: str, validation_result: Mapping[str, Any]
) -> None:
    """Reject a validation JSON that predates any builder-artifact change."""

    manifests = root / "manifests"
    artifact_paths = {
        "mapping": root / dataset / f"{dataset}_mapping.jsonl",
        "samples": manifests / f"{dataset}_samples.jsonl",
        "candidate_pool": manifests / f"{dataset}_candidate_pool.jsonl",
        "screening": manifests / f"{dataset}_screening.jsonl",
        "selection_stats": manifests / f"{dataset}_selection_stats.json",
        "checkpoint": manifests / f"{dataset}_checkpoint.json",
        "source": manifests / f"{dataset}_source.json",
    }
    recorded = validation_result.get("input_artifact_sha256")
    if not isinstance(recorded, Mapping) or set(recorded) != set(artifact_paths):
        raise RuntimeError(
            f"final validation lacks exact input-artifact bindings for {dataset}"
        )
    for name, path in artifact_paths.items():
        expected = recorded.get(name)
        if (
            not isinstance(expected, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected) is None
        ):
            raise RuntimeError(
                f"final validation has invalid {dataset}/{name} digest"
            )
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(
                f"validated {dataset}/{name} artifact is missing or symlinked: {path}"
            )
        if _saved_file_sha256(path) != expected:
            raise RuntimeError(
                f"{dataset}/{name} changed after final validation; rerun validation"
            )


def _require_validation_image_manifest_binding(
    validation_path: Path, validation_payload: Mapping[str, Any]
) -> Path:
    """Require the validator's exact image cohort before generating reports."""

    manifest_path = validation_path.parent / "image_hash_manifest.jsonl"
    recorded = validation_payload.get("output_artifact_sha256")
    expected = (
        recorded.get("image_hash_manifest") if isinstance(recorded, Mapping) else None
    )
    if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
        raise RuntimeError(
            "final validation lacks an exact image-hash-manifest binding"
        )
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise RuntimeError(
            f"validated image hash manifest is missing or symlinked: {manifest_path}"
        )
    if _saved_file_sha256(manifest_path) != expected:
        raise RuntimeError(
            "image hash manifest changed after final validation; rerun validation"
        )
    return manifest_path


def _legacy_input_artifacts(root: Path, dataset: str) -> dict[str, Any]:
    """Recompute the exact mapping/image byte manifest audited for legacy data."""

    directory = root / dataset
    mapping = directory / f"{dataset}_mapping.jsonl"
    if directory.is_symlink() or not directory.is_dir():
        raise RuntimeError(f"legacy dataset directory is missing/symlinked: {directory}")
    if mapping.is_symlink() or not mapping.is_file():
        raise RuntimeError(f"legacy mapping is missing/symlinked: {mapping}")
    images = sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if any(path.is_symlink() for path in images):
        raise RuntimeError(f"legacy image input is symlinked under {directory}")
    return {
        "mapping": {
            "relative_path": mapping.relative_to(root).as_posix(),
            "size_bytes": mapping.stat().st_size,
            "sha256": _saved_file_sha256(mapping),
        },
        "images": [
            {
                "relative_path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _saved_file_sha256(path),
            }
            for path in images
        ],
    }


def _canonical_rgb_sha256(path: Path) -> str:
    """Independently apply the normative canonical-RGB hash to one image file."""

    try:
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            image.load()
    except (OSError, ValueError, Image.DecompressionBombError) as error:
        raise ValueError(f"cannot decode image required for overlap: {path}: {error}") from error
    width, height = image.size
    digest = hashlib.sha256()
    digest.update(b"RGB\0")
    digest.update(struct.pack(">QQ", width, height))
    digest.update(image.tobytes())
    return digest.hexdigest()


def _mapping_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"mapping required for old/new overlap does not exist: {path}"
        )
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"blank JSONL line in {path}:{line_number}")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON in {path}:{line_number}: {error}") from error
            if not isinstance(value, dict):
                raise ValueError(f"mapping row is not an object in {path}:{line_number}")
            rows.append(value)
    return rows


def _mapping_image_path(dataset_directory: Path, filename: Any) -> Path:
    if not isinstance(filename, str) or not filename:
        raise ValueError(
            f"invalid image_filename {filename!r} in {dataset_directory} mapping"
        )
    candidate = (dataset_directory / filename).resolve(strict=False)
    try:
        candidate.relative_to(dataset_directory.resolve(strict=False))
    except ValueError as error:
        raise ValueError(
            f"image_filename escapes dataset directory: {filename!r}"
        ) from error
    if not candidate.is_file():
        raise FileNotFoundError(f"mapping-referenced image does not exist: {candidate}")
    return candidate


def _dataset_overlap_identities(
    dataset_root: Path,
    dataset: str,
    *,
    known_hashes: Mapping[tuple[str, str], str] | None = None,
    path_cache: dict[Path, str] | None = None,
    hash_workers: int = 8,
) -> tuple[set[str], set[tuple[str, str]]]:
    """Return unique image hashes and exact (hash, question) sample identities."""

    dataset_directory = dataset_root / dataset
    rows = _mapping_rows(dataset_directory / f"{dataset}_mapping.jsonl")
    path_cache = {} if path_cache is None else path_cache
    resolved_rows: list[tuple[Path, str, str | None]] = []
    for row_index, row in enumerate(rows):
        path = _mapping_image_path(dataset_directory, row.get("image_filename"))
        filename = str(row.get("image_filename"))
        question = row.get("question")
        if not isinstance(question, str):
            raise ValueError(
                f"question must be a string for sample overlap: "
                f"{dataset_directory} row {row_index}"
            )
        known = known_hashes.get((dataset, filename)) if known_hashes is not None else None
        if known is not None and not re.fullmatch(r"[0-9a-f]{64}", known):
            raise ValueError(
                f"invalid canonical hash in independent image manifest for "
                f"{dataset}/{filename}: {known!r}"
            )
        resolved_rows.append((path, question, known))

    missing_paths = sorted(
        {
            path
            for path, _, _ in resolved_rows
            if path not in path_cache
        },
        key=str,
    )
    if missing_paths:
        workers = max(1, min(int(hash_workers), len(missing_paths)))
        if workers == 1:
            computed = map(_canonical_rgb_sha256, missing_paths)
            path_cache.update(zip(missing_paths, computed))
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                path_cache.update(
                    zip(missing_paths, executor.map(_canonical_rgb_sha256, missing_paths))
                )

    image_hashes: set[str] = set()
    sample_identities: set[tuple[str, str]] = set()
    for path, question, known in resolved_rows:
        digest = path_cache[path]
        if known is not None and known != digest:
            raise ValueError(
                "independent image-hash manifest disagrees with recomputed pixels for "
                f"{dataset}/{path.relative_to(dataset_directory).as_posix()}: "
                f"manifest={known}, recomputed={digest}"
            )
        image_hashes.add(digest)
        sample_identities.add((digest, question))
    return image_hashes, sample_identities


def _load_independent_hash_manifest(path: Path) -> dict[tuple[str, str], str]:
    if not path.is_file():
        return {}
    hashes: dict[tuple[str, str], str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"blank JSONL line in {path}:{line_number}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON in {path}:{line_number}: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"hash manifest row is not an object in {path}:{line_number}")
            dataset = row.get("dataset")
            filename = row.get("image_filename")
            digest = row.get("canonical_rgb_sha256")
            if (
                dataset not in DATASET_NAMES
                or not isinstance(filename, str)
                or not isinstance(digest, str)
                or not re.fullmatch(r"[0-9a-f]{64}", digest)
            ):
                raise ValueError(
                    f"invalid independent hash-manifest identity in {path}:{line_number}"
                )
            key = (dataset, filename)
            if key in hashes and hashes[key] != digest:
                raise ValueError(f"conflicting hash-manifest entries for {key!r}")
            hashes[key] = digest
    return hashes


def compute_old_new_overlaps(
    old_dataset_root: str | Path,
    new_dataset_root: str | Path,
    *,
    new_hash_manifest_path: str | Path | None = None,
    hash_workers: int = 8,
) -> dict[str, dict[str, int]]:
    """Read both roots without mutation and calculate actual exact intersections.

    ``image_overlap`` is the number of shared unique canonical RGB hashes.
    ``sample_overlap`` is the number of shared unique ``(RGB hash, exact question)``
    pairs.  Exact question text is intentionally not normalized or guessed.
    """

    old_root = Path(old_dataset_root).expanduser().resolve()
    new_root = Path(new_dataset_root).expanduser().resolve()
    if not old_root.is_dir():
        raise FileNotFoundError(f"old dataset root does not exist: {old_root}")
    if not new_root.is_dir():
        raise FileNotFoundError(f"new dataset root does not exist: {new_root}")
    manifest_path = (
        Path(new_hash_manifest_path).expanduser().resolve()
        if new_hash_manifest_path is not None
        else new_root / "reports" / "image_hash_manifest.jsonl"
    )
    known_new_hashes = _load_independent_hash_manifest(manifest_path)
    path_cache: dict[Path, str] = {}
    result: dict[str, dict[str, int]] = {}
    for dataset in DATASET_NAMES:
        old_images, old_samples = _dataset_overlap_identities(
            old_root, dataset, path_cache=path_cache, hash_workers=hash_workers
        )
        new_images, new_samples = _dataset_overlap_identities(
            new_root,
            dataset,
            known_hashes=known_new_hashes,
            path_cache=path_cache,
            hash_workers=hash_workers,
        )
        result[dataset] = {
            "old_new_image_overlap": len(old_images.intersection(new_images)),
            "old_new_sample_overlap": len(old_samples.intersection(new_samples)),
        }
    return result


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


def _atomic_write_csv(
    path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]
) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(columns), extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: _flat_value(row.get(column, NA)) for column in columns})
    _atomic_write_text(path, buffer.getvalue())


def _flat_value(value: Any) -> Any:
    value = _public(value)
    if isinstance(value, (list, tuple, set)):
        return "; ".join(str(item) for item in value)
    if isinstance(value, Mapping):
        return "; ".join(f"{key}={item}" for key, item in value.items())
    return value


def _dataset_sections(payload: Any, dataset: str) -> list[Mapping[str, Any]]:
    """Return every dataset-specific object from common legacy/new schemas."""

    wanted = _normalize_key(dataset)
    sections: list[Mapping[str, Any]] = []
    seen_ids: set[int] = set()
    queue: deque[Any] = deque([payload])
    while queue:
        node = queue.popleft()
        if isinstance(node, Mapping):
            identity = id(node)
            if identity in seen_ids:
                continue
            seen_ids.add(identity)
            dataset_value = next(
                (
                    value
                    for key, value in node.items()
                    if _normalize_key(key) in {"dataset", "datasetname", "name"}
                ),
                _MISSING,
            )
            if not _is_missing(dataset_value) and _normalize_key(dataset_value) == wanted:
                sections.append(node)
            for key, value in node.items():
                if _normalize_key(key) == wanted and isinstance(value, Mapping):
                    sections.append(value)
                if isinstance(value, (Mapping, list, tuple)):
                    queue.append(value)
        elif isinstance(node, (list, tuple)):
            queue.extend(node)

    unique: list[Mapping[str, Any]] = []
    seen_ids.clear()
    for section in sections:
        if id(section) not in seen_ids:
            seen_ids.add(id(section))
            unique.append(section)
    return unique


def _selection_section(payload: Any, dataset: str) -> Mapping[str, Any]:
    sections = _dataset_sections(payload, dataset)
    if sections:
        return sections[0]
    # Per-dataset selection files are commonly flat and need not repeat a name.
    return payload if isinstance(payload, Mapping) else {}


def _lookup(mapping: Mapping[str, Any], aliases: Iterable[str]) -> Any:
    """Breadth-first alias lookup; shallower, explicit values win."""

    wanted = {_normalize_key(alias) for alias in aliases}
    queue: deque[Any] = deque([mapping])
    seen_ids: set[int] = set()
    while queue:
        node = queue.popleft()
        if not isinstance(node, Mapping):
            continue
        identity = id(node)
        if identity in seen_ids:
            continue
        seen_ids.add(identity)
        for key, value in node.items():
            if _normalize_key(key) in wanted and not _is_missing(value):
                return value
        for value in node.values():
            if isinstance(value, Mapping):
                queue.append(value)
    return _MISSING


def _lookup_sources(sources: Iterable[Mapping[str, Any]], aliases: Iterable[str]) -> Any:
    for source in sources:
        value = _lookup(source, aliases)
        if not _is_missing(value):
            return value
    return _MISSING


def _count_value(value: Any) -> int | float | None:
    number = _as_number(value)
    if number is not None:
        return number
    if isinstance(value, (list, tuple, set)):
        return len(value)
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if _normalize_key(key) in {
                "count",
                "total",
                "value",
                "rows",
                "number",
                "n",
            }:
                number = _as_number(nested)
                if number is not None:
                    return number
    return None


def _lookup_count(mapping: Mapping[str, Any], aliases: Iterable[str]) -> int | float | None:
    wanted = {_normalize_key(alias) for alias in aliases}
    queue: deque[Any] = deque([mapping])
    seen_ids: set[int] = set()
    while queue:
        node = queue.popleft()
        if not isinstance(node, Mapping):
            continue
        identity = id(node)
        if identity in seen_ids:
            continue
        seen_ids.add(identity)
        for key, value in node.items():
            if _normalize_key(key) in wanted:
                number = _count_value(value)
                if number is not None:
                    return number
        for value in node.values():
            if isinstance(value, Mapping):
                queue.append(value)
    return None


def _count(
    sources: Iterable[Mapping[str, Any]], aliases: Iterable[str]
) -> int | float | str:
    for source in sources:
        number = _lookup_count(source, aliases)
        if number is not None:
            return number
    return NA


def _difference(left: Any, right: Any) -> int | float | str:
    left_number, right_number = _as_number(left), _as_number(right)
    if left_number is None or right_number is None:
        return NA
    return max(0, left_number - right_number)


def _ratio(numerator: Any, denominator: Any) -> float | str:
    numerator_number = _as_number(numerator)
    denominator_number = _as_number(denominator)
    if numerator_number is None or denominator_number in (None, 0):
        return NA
    return numerator_number / denominator_number


def _recorded_or_ratio(
    sources: Iterable[Mapping[str, Any]],
    rate_aliases: Iterable[str],
    numerator: Any,
    denominator: Any,
) -> float | int | str:
    recorded = _lookup_sources(sources, rate_aliases)
    number = _as_number(recorded)
    return number if number is not None else _ratio(numerator, denominator)


def _source_splits(section: Mapping[str, Any]) -> Any:
    value = _lookup(
        section,
        (
            "source_splits",
            "source_split",
            "splits",
            "split",
            "source_split_names",
        ),
    )
    if _is_missing(value):
        return NA
    if isinstance(value, Mapping):
        return "+".join(str(key) for key in value)
    if isinstance(value, (list, tuple, set)):
        return "+".join(str(item) for item in value)
    return str(value)


def _split_retained(section: Mapping[str, Any], split: str) -> int | float | str:
    direct = _count(
        (section,),
        (
            f"retained_{split}",
            f"{split}_retained",
            f"selected_{split}",
            f"{split}_selected",
            f"final_{split}",
        ),
    )
    if direct != NA:
        return direct
    composition = _lookup(
        section,
        (
            "scienceqa_split_composition",
            "retained_split_composition",
            "final_split_composition",
            "split_composition",
            "retained_by_split",
            "selected_by_split",
        ),
    )
    if isinstance(composition, Mapping):
        for key, value in composition.items():
            if _normalize_key(key) == _normalize_key(split):
                number = _as_number(value)
                return number if number is not None else _public(value)
    return NA


def build_funnel_rows(
    selection_payloads: Mapping[str, Any], validation_payload: Any
) -> list[dict[str, Any]]:
    """Normalize builder statistics into one row per benchmark."""

    rows: list[dict[str, Any]] = []
    for dataset in DATASET_NAMES:
        section = _selection_section(selection_payloads.get(dataset, {}), dataset)
        validation_sections = _dataset_sections(validation_payload, dataset)
        sources = (section, *validation_sections)
        retained = _count(
            sources,
            (
                "retained",
                "retained_count",
                "final_retained",
                "selected_count",
                "mapping_rows",
            ),
        )
        evaluated = _count(
            sources,
            (
                "unique_images_evaluated",
                "model_evaluated_unique_images",
                "model_evaluated",
                "evaluated_unique_images",
                "evaluated_count",
                "model_evaluations",
                "scanned_until_target",
                "scanned_unique_candidates",
            ),
        )
        row = {
            "dataset": dataset,
            "source_splits": _source_splits(section),
            "source_qa_records": _count(
                (section,),
                (
                    "source_qa_records",
                    "source_qa_records_total",
                    "source_records",
                    "source_record_count",
                    "raw_qa_records",
                    "total_qa_records",
                    "qa_records_total",
                ),
            ),
            "source_unique_images": _count(
                (section,),
                (
                    "source_unique_images",
                    "source_unique_images_total",
                    "source_unique_image_count",
                    "raw_unique_images",
                    "unique_images_in_source_pool",
                    "source_pool_unique_images",
                ),
            ),
            "eligible_unique_images": _count(
                (section,),
                (
                    "eligible_unique_images",
                    "prefiltered_unique_images",
                    "post_prefilter_unique_images",
                    "unique_candidates_after_prefilter",
                    "task_prefilter_unique_images",
                    "candidate_unique_images",
                ),
            ),
            "eligible_unique_images_basis": _public(
                _lookup(
                    section,
                    (
                        "task_prefilter_unique_images_basis",
                        "eligible_unique_images_basis",
                        "candidate_identity_basis",
                    ),
                )
            ),
            "invalid_metadata": _count(
                (section,),
                (
                    "invalid_metadata",
                    "invalid_metadata_count",
                    "metadata_invalid",
                    "invalid_records",
                ),
            ),
            "yes_no_removed": _count(
                (section,),
                (
                    "yes_no_removed",
                    "yes_no_removed_count",
                    "removed_yes_no",
                    "yesno_removed",
                    "yes_no_filtered",
                ),
            ),
            "missing_or_corrupt_images": _count(
                (section,),
                (
                    "missing_or_corrupt_images",
                    "missing_or_corrupted_images",
                    "missing_corrupt_images",
                    "missing_or_unreadable_images",
                    "image_load_failures",
                    "image_errors",
                ),
            ),
            "duplicate_source_ids": _count(
                (section,),
                (
                    "duplicate_source_ids",
                    "duplicate_source_image_ids",
                    "source_id_duplicates",
                    "duplicate_image_ids",
                    "source_image_id_duplicates",
                    "source_id_duplicate_count",
                    "source_id_duplicate_rows",
                    "duplicate_source_id_rows",
                ),
            ),
            "exact_image_hash_duplicates": _count(
                (section,),
                (
                    "exact_image_hash_duplicates",
                    "exact_hash_duplicates",
                    "duplicate_rgb_hashes",
                    "canonical_hash_duplicates",
                    "exact_rgb_duplicates",
                    "duplicate_image_hashes",
                    "exact_hash_duplicate_count",
                    "exact_hash_duplicate_rows",
                    "exact_image_hash_duplicate_count",
                ),
            ),
            "unique_images_evaluated": evaluated,
            "blind_wrong": _count(
                (section,),
                (
                    "blind_wrong",
                    "blind_wrong_count",
                    "blind_control_wrong",
                    "blind_black_image_control_wrong",
                    "blind_incorrect",
                    "blind_rejected_as_wrong",
                ),
            ),
            "full_correct_among_blind_wrong": _count(
                (section,),
                (
                    "full_correct_among_blind_wrong",
                    "full_correct_given_blind_wrong",
                    "blind_wrong_full_correct",
                    "full_correct_after_blind_wrong",
                    "full_correct",
                    "full_correct_count",
                    "qualified",
                    "accepted_count",
                ),
            ),
            "target_reached_source_index": _count(
                (section,),
                (
                    "target_reached_source_index",
                    "source_index_at_target",
                    "target_source_index",
                    "thousandth_source_index",
                    "last_source_index",
                ),
            ),
            "retained": retained,
            "acceptance_rate": _recorded_or_ratio(
                (section,),
                ("acceptance_rate", "selection_rate", "retained_rate"),
                retained,
                evaluated,
            ),
            "retained_train": _split_retained(section, "train"),
            "retained_validation": _split_retained(section, "validation"),
            "retained_test": _split_retained(section, "test"),
        }
        rows.append({key: _public(row.get(key, NA)) for key in FUNNEL_COLUMNS})
    return rows


def _accuracy(
    sources: Iterable[Mapping[str, Any]],
    accuracy_aliases: Iterable[str],
    correct_aliases: Iterable[str],
    denominator: Any,
) -> float | int | str:
    sources = tuple(sources)
    recorded = _lookup_sources(sources, accuracy_aliases)
    recorded_number = _as_number(recorded)
    if recorded_number is not None:
        return recorded_number
    correct = _count(sources, correct_aliases)
    return _ratio(correct, denominator)


def build_quality_rows(
    old_audit_payload: Any,
    validation_payload: Any,
    selection_payloads: Mapping[str, Any],
    measured_overlaps: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build old/new quality rows without inferring unavailable overlaps."""

    rows: list[dict[str, Any]] = []
    for dataset in DATASET_NAMES:
        old_sections = _dataset_sections(old_audit_payload, dataset)
        validation_sections = _dataset_sections(validation_payload, dataset)
        selection_section = _selection_section(selection_payloads.get(dataset, {}), dataset)
        new_sources = (*validation_sections, selection_section)

        old_rows = _count(
            old_sections, ("mapping_rows", "old_rows", "row_count", "rows")
        )
        old_unique = _count(
            old_sections,
            (
                "unique_canonical_rgb_images",
                "old_unique_rgb_images",
                "unique_rgb_images",
                "unique_image_content",
            ),
        )
        old_duplicates = _count(
            old_sections,
            (
                "duplicate_image_rows",
                "old_duplicate_rows",
                "duplicate_rows",
                "exact_duplicate_rows",
            ),
        )
        if old_duplicates == NA:
            old_duplicates = _difference(old_rows, old_unique)

        new_rows = _count(
            new_sources,
            (
                "mapping_rows",
                "new_rows",
                "row_count",
                "retained",
                "retained_count",
            ),
        )
        new_unique = _count(
            new_sources,
            (
                "unique_canonical_rgb_sha256",
                "unique_canonical_rgb_sha256_count",
                "unique_canonical_rgb_images",
                "new_unique_rgb_images",
                "unique_rgb_images",
                "unique_image_content",
            ),
        )
        new_duplicates = _count(
            new_sources,
            (
                "duplicate_image_rows",
                "new_duplicate_rows",
                "duplicate_rgb_rows",
                "canonical_rgb_duplicate_rows",
                "duplicate_canonical_rgb_sha256",
                "duplicate_rgb_sha256_rows",
                "duplicate_canonical_rgb_images",
            ),
        )
        if new_duplicates == NA:
            new_duplicates = _difference(new_rows, new_unique)

        measured = (measured_overlaps or {}).get(dataset, {})
        image_overlap = _count(
            (measured,),
            ("old_new_image_overlap",),
        )
        sample_overlap = _count(
            (measured,),
            ("old_new_sample_overlap",),
        )

        old_full_accuracy = _accuracy(
            old_sections,
            ("old_full_construction_accuracy", "full_construction_accuracy"),
            (
                "old_full_construction_correct_count",
                "full_construction_correct_count",
                "full_control_correct_count",
            ),
            old_rows,
        )
        old_blind_accuracy = _accuracy(
            old_sections,
            (
                "old_blind_control_construction_accuracy",
                "blind_control_construction_accuracy",
                "blind_black_image_control_accuracy",
            ),
            (
                "old_blind_control_correct_count",
                "blind_control_correct_count",
                "blind_black_image_control_correct_count",
            ),
            old_rows,
        )
        new_full_accuracy = _accuracy(
            new_sources,
            ("new_full_construction_accuracy", "full_construction_accuracy"),
            (
                "full_control_correct_count_among_retained",
                "full_construction_correct_count_among_retained",
                "retained_full_correct_count",
                "full_control_correct_count",
                "full_control_correct",
                "full_correct_count",
                "full_visual_correct_retained",
            ),
            new_rows,
        )
        new_blind_accuracy = _accuracy(
            new_sources,
            (
                "new_blind_control_construction_accuracy",
                "blind_control_construction_accuracy",
                "blind_black_image_control_accuracy",
            ),
            (
                "blind_control_correct_count_among_retained",
                "blind_black_image_control_correct_count_among_retained",
                "retained_blind_correct_count",
                "blind_control_correct_count",
                "blind_control_correct",
                "blind_black_image_control_correct",
                "blind_control_correct_retained",
            ),
            new_rows,
        )

        row = {
            "dataset": dataset,
            "old_rows": old_rows,
            "old_unique_rgb_images": old_unique,
            "old_duplicate_rows": old_duplicates,
            "old_duplicate_rate": _recorded_or_ratio(
                old_sections,
                ("old_duplicate_rate", "duplicate_image_rate", "duplicate_rate"),
                old_duplicates,
                old_rows,
            ),
            "new_rows": new_rows,
            "new_unique_rgb_images": new_unique,
            "new_duplicate_rows": new_duplicates,
            "new_duplicate_rate": _recorded_or_ratio(
                new_sources,
                ("new_duplicate_rate", "duplicate_image_rate", "duplicate_rate"),
                new_duplicates,
                new_rows,
            ),
            "old_new_image_overlap": image_overlap,
            "old_new_sample_overlap": sample_overlap,
            "old_full_construction_accuracy": old_full_accuracy,
            "old_blind_control_construction_accuracy": old_blind_accuracy,
            "new_full_construction_accuracy": new_full_accuracy,
            "new_blind_control_construction_accuracy": new_blind_accuracy,
        }
        rows.append({key: _public(row.get(key, NA)) for key in QUALITY_COLUMNS})
    return rows


def _display(value: Any, *, rate: bool = False) -> str:
    value = _public(value)
    if value == NA:
        return NA
    if rate:
        number = _as_number(value)
        if number is None:
            return str(value)
        # A recorded 19.9 means 19.9%; a recorded 0.199 means 19.9%.
        percent = number if abs(number) > 1 else number * 100
        return f"{percent:.2f}%"
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else f"{value:.6g}"
    return str(_flat_value(value))


def _markdown_table(
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str],
    labels: Mapping[str, str],
    rate_columns: Iterable[str] = (),
) -> str:
    rates = set(rate_columns)
    header = "| " + " | ".join(labels[column] for column in columns) + " |"
    rule = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        values = [
            _display(row.get(column, NA), rate=column in rates)
            .replace("|", "\\|")
            .replace("\n", " ")
            for column in columns
        ]
        body.append("| " + " | ".join(values) + " |")
    return "\n".join((header, rule, *body))


def _tex_escape(value: Any) -> str:
    text = _display(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in text)


def _tex_display(value: Any, *, rate: bool = False) -> str:
    return _tex_escape(_display(value, rate=rate))


def _paired(left: Any, right: Any) -> str:
    return f"{_display(left)} / {_display(right)}"


def _selection_table_tex(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        (
            r"Dataset & Source split(s) & Source QA / unique images & "
            r"Eligible unique images & Evaluated & Blind-wrong & Full-correct & Retained \\"
        ),
        r"\midrule",
    ]
    for row in rows:
        cells = (
            _tex_display(row["dataset"]),
            _tex_display(row["source_splits"]),
            _tex_escape(
                _paired(row["source_qa_records"], row["source_unique_images"])
            ),
            _tex_display(row["eligible_unique_images"]),
            _tex_display(row["unique_images_evaluated"]),
            _tex_display(row["blind_wrong"]),
            _tex_display(row["full_correct_among_blind_wrong"]),
            _tex_display(row["retained"]),
        )
        lines.append(" & ".join(cells) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\caption{Dataset construction funnel. Source QA is the total number "
                r"of source question--answer records. Source and eligible image counts "
                r"use the identity basis recorded in the machine-readable funnel. For "
                r"lazily fetched VQAv2, the eligible-pool count is source-ID unique; "
                r"canonical RGB uniqueness is enforced on every scanned candidate before "
                r"inference but is not misreported as a complete full-pool RGB census. "
                r"Evaluated is the number of unique candidates actually run through the "
                r"model before the target was reached, not the full source-pool size. "
                r"Full-correct counts Full checks passed among blind-wrong candidates. "
                r"The retained sets contain the reported number of visually unique samples.}"
            ),
            r"\label{tab:dataset-selection}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def _quality_table_tex(rows: Sequence[Mapping[str, Any]]) -> str:
    columns = QUALITY_COLUMNS
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrrrrrrrrrrr}",
        r"\toprule",
        (
            r"Dataset & Old rows & Old unique RGB & Old duplicate rows & Old duplicate "
            r"rate & New rows & New unique RGB & New duplicate rows & New duplicate rate "
            r"& Old--new image overlap & Old--new sample overlap & Old Full acc. & Old blind "
            r"acc. & New Full acc. & New blind acc. \\"
        ),
        r"\midrule",
    ]
    rates = {
        "old_duplicate_rate",
        "new_duplicate_rate",
        "old_full_construction_accuracy",
        "old_blind_control_construction_accuracy",
        "new_full_construction_accuracy",
        "new_blind_control_construction_accuracy",
    }
    for row in rows:
        lines.append(
            " & ".join(
                _tex_display(row[column], rate=column in rates) for column in columns
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            (
                r"\caption{Legacy and rebuilt dataset quality. Exact image identity uses "
                r"canonical RGB SHA-256. Old--new image overlap counts shared unique "
                r"hashes; sample overlap counts shared unique (hash, exact question) "
                r"pairs. N/A denotes a quantity not recorded or not measured; it is "
                r"not estimated.}"
            ),
            r"\label{tab:old-new-dataset-quality}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def _appendix_tex(rows: Sequence[Mapping[str, Any]]) -> str:
    retained = ", ".join(
        f"{_tex_display(row['dataset'])}: {_tex_display(row['retained'])}" for row in rows
    )
    return "\n".join(
        [
            r"\paragraph{Unique-image benchmark construction.}",
            (
                r"For each source image $x$, we formed its question group before any model "
                r"inference and selected a single representative $q^{\star}(x)$ "
                r"deterministically (the first question in native source order, or "
                r"equivalently the minimum question ID where that convention was used). "
                r"If the representative failed, no later question for the same image was "
                r"tried. Thus images with more annotated questions received no additional "
                r"opportunities to enter the benchmark."
            ),
            r"\begin{equation}",
            (
                r"\mathcal{E}_j = \operatorname{First}_{1000}\!\left\{" 
                r"(x,q^{\star}(x),y)\in\mathcal{D}_j : "
                r"\neg C_{\mathrm{blind\_black}}(x,q^{\star},y) \wedge "
                r"C_{\mathrm{Full}}(x,q^{\star},y),\ "
                r"\operatorname{id}(x)\notin I_j,\ "
                r"h_{\mathrm{RGB}}(x)\notin H_j\right\}."
            ),
            r"\end{equation}",
            (
                r"Here $I_j$ and $H_j$ are, respectively, the already-retained reliable "
                r"source image IDs and canonical RGB SHA-256 hashes. The latter hashes the "
                r"EXIF-transposed RGB width, height, and pixel bytes. ``Source pool'' in "
                r"Table~\ref{tab:dataset-selection} denotes the complete available pool, "
                r"whereas ``Evaluated'' denotes only the deterministic prefix processed "
                r"before the retention target was reached."
            ),
            (
                r"The blind condition is the \texttt{blind\_black\_image\_control}: a "
                r"$336\!\times\!336$ black RGB image still passes through the ordinary "
                r"visual encoder. It is therefore a content-free visual control, not a "
                r"strict zero-visual-token ($K=0$) condition. The construction invariants "
                r"must not be interpreted as accuracy from the paper's unified evaluation "
                r"harness."
            ),
            f"The generated retained counts are {retained}.",
            "",
        ]
    )


def _funnel_markdown(rows: Sequence[Mapping[str, Any]]) -> str:
    visible = (
        "dataset",
        "source_splits",
        "source_qa_records",
        "source_unique_images",
        "eligible_unique_images",
        "unique_images_evaluated",
        "blind_wrong",
        "full_correct_among_blind_wrong",
        "retained",
        "acceptance_rate",
    )
    detail = (
        "dataset",
        "invalid_metadata",
        "yes_no_removed",
        "missing_or_corrupt_images",
        "duplicate_source_ids",
        "exact_image_hash_duplicates",
        "eligible_unique_images_basis",
        "target_reached_source_index",
        "retained_train",
        "retained_validation",
        "retained_test",
    )
    return (
        "# Dataset selection funnel\n\n"
        + _markdown_table(
            rows, visible, FUNNEL_LABELS, rate_columns=("acceptance_rate",)
        )
        + "\n\n## Prefilter and provenance details\n\n"
        + _markdown_table(rows, detail, FUNNEL_LABELS)
        + "\n\n"
        + "`Source QA records` is the complete source-pool size. `Eligible unique "
        "images` uses the per-dataset identity basis shown in the detail table. For "
        "lazily fetched VQAv2 it is the reliable-source-ID-unique pool; canonical RGB "
        "uniqueness is enforced before every model call, but this count is not a "
        "complete full-pool RGB census. `Unique "
        "images evaluated` is only the prefix actually run through the model before "
        "the target was reached. `Full-correct` is counted within blind-wrong "
        "candidates. N/A means the input artifacts did not record that measurement.\n\n"
        + "The formal blind condition is `blind_black_image_control`: its 336x336 "
        "black RGB image still passes through the visual encoder and is not a strict "
        "zero-visual-token condition.\n"
    )


def _quality_markdown(rows: Sequence[Mapping[str, Any]]) -> str:
    return (
        "# Old versus new dataset quality\n\n"
        + _markdown_table(
            rows,
            QUALITY_COLUMNS,
            QUALITY_LABELS,
            rate_columns=(
                "old_duplicate_rate",
                "new_duplicate_rate",
                "old_full_construction_accuracy",
                "old_blind_control_construction_accuracy",
                "new_full_construction_accuracy",
                "new_blind_control_construction_accuracy",
            ),
        )
        + "\n\n"
        + "Duplicate rates use mapping rows as the denominator and exact canonical "
        "RGB SHA-256 identity. Old--new image overlap is the unique hash-set "
        "intersection; sample overlap is the unique `(hash, exact question)` "
        "intersection. Construction accuracies are included only when saved "
        "predictions/counts or an actual re-evaluation support them. N/A means the "
        "quantity was not recorded or measured; no value was guessed. Construction "
        "controls are not a substitute for evaluation with the frozen paper harness.\n"
    )


def generate_reports(
    dataset_root: str | Path,
    *,
    reports_dir: str | Path,
    old_dataset_root: str | Path = DEFAULT_OLD_DATASET_ROOT,
    manifests_dir: str | Path | None = None,
    validation_path: str | Path | None = None,
    old_audit_path: str | Path | None = None,
) -> dict[str, Path]:
    """Generate all required reports and return their paths.

    Missing per-dataset selection fields are represented by N/A.  The old and new
    mappings/images are required because their exact image/sample intersections are
    measured directly rather than inferred from aggregate audit counters.
    """

    root = Path(dataset_root).expanduser().resolve()
    manifest_root = (
        Path(manifests_dir).expanduser().resolve()
        if manifests_dir is not None
        else root / "manifests"
    )
    report_root = Path(reports_dir).expanduser().resolve()
    old_root = Path(old_dataset_root).expanduser().resolve()
    report_root = assert_output_separate(
        report_root,
        {
            "new dataset root": root,
            "legacy dataset root": old_root,
            "selection manifest root": manifest_root,
        },
    )
    validation = (
        Path(validation_path).expanduser().resolve()
        if validation_path is not None
        else report_root / "final_validation.json"
    )
    old_audit = (
        Path(old_audit_path).expanduser().resolve()
        if old_audit_path is not None
        else report_root / "old_dataset_duplicate_audit.json"
    )
    paths = {
        "dataset_selection_funnel_csv": report_root / "dataset_selection_funnel.csv",
        "dataset_selection_funnel_json": report_root / "dataset_selection_funnel.json",
        "dataset_selection_funnel_md": report_root / "dataset_selection_funnel.md",
        "old_vs_new_dataset_quality_csv": (
            report_root / "old_vs_new_dataset_quality.csv"
        ),
        "old_vs_new_dataset_quality_md": report_root / "old_vs_new_dataset_quality.md",
        "old_vs_new_dataset_quality_tex": (
            report_root / "old_vs_new_dataset_quality.tex"
        ),
        "table_dataset_selection_tex": report_root / "table_dataset_selection.tex",
        "dataset_selection_appendix_snippet_tex": (
            report_root / "dataset_selection_appendix_snippet.tex"
        ),
    }
    assert_no_output_file_collision(
        paths,
        {
            "final validation input": validation,
            "legacy dataset audit input": old_audit,
        },
    )
    if not validation.is_file():
        raise FileNotFoundError(f"validation JSON does not exist: {validation}")
    if not old_audit.is_file():
        raise FileNotFoundError(f"old-audit JSON does not exist: {old_audit}")

    validation_payload = _json_load(validation)
    if not isinstance(validation_payload, Mapping):
        raise ValueError(f"validation report must be a JSON object: {validation}")
    if validation_payload.get("overall_pass") is not True:
        raise RuntimeError(
            "refusing to generate paper-ready reports from a failed final validation"
        )
    if _as_number(validation_payload.get("expected_count")) != 1000:
        raise RuntimeError(
            "paper-ready reports require a passing final validation with expected_count=1000"
        )
    recorded_root = validation_payload.get("dataset_root")
    if not isinstance(recorded_root, str) or Path(recorded_root).resolve() != root:
        raise RuntimeError(
            "final validation dataset_root does not match the requested report root"
        )
    validation_datasets = validation_payload.get("datasets")
    if not isinstance(validation_datasets, Mapping) or any(
        not isinstance(validation_datasets.get(dataset), Mapping)
        or validation_datasets[dataset].get("pass") is not True
        for dataset in DATASET_NAMES
    ):
        raise RuntimeError(
            "final validation does not contain a passing result for all four datasets"
        )
    for dataset in DATASET_NAMES:
        _require_validation_artifact_binding(
            root, dataset, validation_datasets[dataset]
        )
    validated_image_hash_manifest = _require_validation_image_manifest_binding(
        validation, validation_payload
    )

    old_audit_payload = _json_load(old_audit)
    if not isinstance(old_audit_payload, Mapping):
        raise RuntimeError("legacy dataset audit must be a JSON object")
    audited_root = old_audit_payload.get("audited_root")
    if not isinstance(audited_root, str) or Path(audited_root).resolve() != old_root:
        raise RuntimeError(
            "legacy dataset audit root does not match --old-dataset-root"
        )
    recorded_old_inputs = old_audit_payload.get("input_artifacts")
    if not isinstance(recorded_old_inputs, Mapping) or set(recorded_old_inputs) != set(
        DATASET_NAMES
    ):
        raise RuntimeError("legacy dataset audit lacks exact input-artifact bindings")
    for dataset in DATASET_NAMES:
        if recorded_old_inputs[dataset] != _legacy_input_artifacts(old_root, dataset):
            raise RuntimeError(
                f"legacy {dataset} inputs changed after duplicate audit; rerun audit"
            )
    selection_payloads: dict[str, Any] = {}
    input_paths: dict[str, str | None] = {}
    for dataset in DATASET_NAMES:
        path = manifest_root / f"{dataset}_selection_stats.json"
        if not path.is_file():
            raise FileNotFoundError(f"selection statistics do not exist: {path}")
        input_paths[dataset] = str(path)
        selection_payloads[dataset] = _json_load(path)

    funnel_rows = build_funnel_rows(selection_payloads, validation_payload)
    measured_overlaps = compute_old_new_overlaps(
        old_root,
        root,
        new_hash_manifest_path=validated_image_hash_manifest,
    )
    quality_rows = build_quality_rows(
        old_audit_payload,
        validation_payload,
        selection_payloads,
        measured_overlaps=measured_overlaps,
    )
    report_root.mkdir(parents=True, exist_ok=True)

    _atomic_write_csv(paths["dataset_selection_funnel_csv"], funnel_rows, FUNNEL_COLUMNS)
    _atomic_write_json(
        paths["dataset_selection_funnel_json"],
        {
            "schema_version": "fata-dataset-selection-funnel-v1",
            "missing_value": NA,
            "inputs": {
                "selection_stats": input_paths,
                "selection_stats_sha256": {
                    dataset: _saved_file_sha256(
                        manifest_root / f"{dataset}_selection_stats.json"
                    )
                    for dataset in DATASET_NAMES
                },
                "final_validation": str(validation) if validation.is_file() else None,
                "final_validation_sha256": _saved_file_sha256(validation),
                "old_dataset_duplicate_audit": (
                    str(old_audit) if old_audit.is_file() else None
                ),
                "old_dataset_duplicate_audit_sha256": _saved_file_sha256(old_audit),
                "old_dataset_root_for_overlap": str(
                    old_root
                ),
                "old_new_overlap_definition": (
                    "unique canonical RGB hashes; samples are unique "
                    "(canonical RGB hash, exact question) pairs"
                ),
            },
            "datasets": funnel_rows,
        },
    )
    _atomic_write_text(
        paths["dataset_selection_funnel_md"], _funnel_markdown(funnel_rows)
    )
    _atomic_write_csv(
        paths["old_vs_new_dataset_quality_csv"], quality_rows, QUALITY_COLUMNS
    )
    _atomic_write_text(
        paths["old_vs_new_dataset_quality_md"], _quality_markdown(quality_rows)
    )
    _atomic_write_text(
        paths["old_vs_new_dataset_quality_tex"], _quality_table_tex(quality_rows)
    )
    _atomic_write_text(
        paths["table_dataset_selection_tex"], _selection_table_tex(funnel_rows)
    )
    _atomic_write_text(
        paths["dataset_selection_appendix_snippet_tex"], _appendix_tex(funnel_rows)
    )
    return paths


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate FATA unique-dataset selection, old/new quality, and LaTeX "
            "reports from recorded JSON artifacts."
        )
    )
    parser.add_argument(
        "dataset_root_positional",
        nargs="?",
        type=Path,
        help="dataset root (positional alternative to --dataset-root)",
    )
    parser.add_argument(
        "--dataset-root",
        "--root",
        "--new-root",
        dest="dataset_root",
        type=Path,
        help=(
            "new dataset root (or set FATA_DATA_ROOT)"
        ),
    )
    parser.add_argument(
        "--manifests-dir",
        type=Path,
        help="selection-stats directory (default: DATASET_ROOT/manifests)",
    )
    parser.add_argument(
        "--old-dataset-root",
        type=Path,
        default=DEFAULT_OLD_DATASET_ROOT,
        help=(
            "read-only legacy dataset root used to compute actual old/new overlap "
            "(or set FATA_LEGACY_DATA_ROOT)"
        ),
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        required=True,
        help="explicit output directory for generated reports",
    )
    parser.add_argument(
        "--validation",
        "--validation-json",
        dest="validation_path",
        type=Path,
        help="independent final_validation.json path",
    )
    parser.add_argument(
        "--old-audit",
        "--old-audit-json",
        dest="old_audit_path",
        type=Path,
        help="old_dataset_duplicate_audit.json path",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.dataset_root is not None and args.dataset_root_positional is not None:
        _parser().error("use either positional DATASET_ROOT or --dataset-root, not both")
    dataset_root = (
        args.dataset_root
        or args.dataset_root_positional
        or (Path(os.environ["FATA_DATA_ROOT"]) if os.environ.get("FATA_DATA_ROOT") else None)
    )
    if dataset_root is None:
        _parser().error(
            "provide DATASET_ROOT/--dataset-root or set FATA_DATA_ROOT"
        )
    if args.old_dataset_root is None:
        _parser().error(
            "provide --old-dataset-root or set FATA_LEGACY_DATA_ROOT"
        )
    paths = generate_reports(
        dataset_root,
        reports_dir=args.reports_dir,
        old_dataset_root=args.old_dataset_root,
        manifests_dir=args.manifests_dir,
        validation_path=args.validation_path,
        old_audit_path=args.old_audit_path,
    )
    print(f"Generated {len(paths)} reports:")
    for path in paths.values():
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
