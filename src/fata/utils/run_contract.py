"""Small fail-closed helpers for resumable experiment outputs."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence


_ARTIFACT_IDENTITY_CACHE: dict[str, dict[str, Any]] = {}
_DATASET_IMAGE_IDENTITY_CACHE: dict[tuple[str, str], dict[str, Any]] = {}


def is_sha256_digest(value: Any) -> bool:
    """Return whether *value* is a canonical lowercase SHA-256 digest."""

    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def dataset_identity_valid(value: Any) -> bool:
    """Validate the portable structure emitted by :func:`dataset_image_identity`."""

    return (
        isinstance(value, dict)
        and value.get("algorithm") == "ordered-path-size-file-sha256-v1"
        and isinstance(value.get("count"), int)
        and not isinstance(value.get("count"), bool)
        and value["count"] > 0
        and isinstance(value.get("total_bytes"), int)
        and not isinstance(value.get("total_bytes"), bool)
        and value["total_bytes"] > 0
        and is_sha256_digest(value.get("aggregate_sha256"))
    )


def artifact_identity_valid(value: Any) -> bool:
    """Validate a file/directory identity without requiring the source to exist."""

    if (
        not isinstance(value, dict)
        or not isinstance(value.get("resolved_path"), str)
        or not value["resolved_path"]
        or not Path(value["resolved_path"]).is_absolute()
    ):
        return False
    if (
        value.get("algorithm") == "single-file-sha256-v1"
        and is_sha256_digest(value.get("file_sha256"))
    ):
        return True
    if value.get("algorithm") != "recursive-model-artifact-sha256-v2":
        return False
    metadata = value.get("metadata_sha256")
    weights = value.get("weight_files")
    indexes = value.get("index_sha256")
    if not isinstance(metadata, dict) or not isinstance(weights, list) or not isinstance(indexes, dict):
        return False
    if not all(
        isinstance(name, str) and name and is_sha256_digest(digest)
        for name, digest in metadata.items()
    ):
        return False
    if not all(
        isinstance(name, str) and name and is_sha256_digest(digest)
        for name, digest in indexes.items()
    ):
        return False
    if not weights:
        return False
    relative_paths: list[str] = []
    for item in weights:
        if not isinstance(item, dict):
            return False
        relative_path = item.get("relative_path")
        if (
            not isinstance(relative_path, str)
            or not relative_path
            or Path(relative_path).is_absolute()
            or ".." in Path(relative_path).parts
            or not isinstance(item.get("size"), int)
            or isinstance(item.get("size"), bool)
            or item["size"] <= 0
            or not is_sha256_digest(item.get("sha256"))
        ):
            return False
        relative_paths.append(relative_path)
    return len(relative_paths) == len(set(relative_paths))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_stat(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino)


def _stable_artifact_sha256(
    path: Path, expected_stat: tuple[int, int, int, int]
) -> str:
    """Hash one artifact only if its filesystem identity stays unchanged."""

    before = _artifact_stat(path)
    if before != expected_stat:
        raise RuntimeError(f"artifact changed before hashing: {path}")
    digest = sha256_file(path)
    after = _artifact_stat(path)
    if after != before:
        raise RuntimeError(f"artifact changed while hashing: {path}")
    return digest


def artifact_identity(path: str | Path) -> dict[str, Any]:
    """Identify all model-affecting bytes, including weights and tokenization.

    Full weight hashes cost a short sequential read when a run starts, but avoid
    silently resuming against replaced weights that happen to share a config.
    Every other regular file is hashed as metadata so tokenizer, vocabulary,
    processor, chat-template, and remote-code changes also invalidate resume.
    """

    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"artifact path does not exist: {resolved}")
    if resolved.is_file():
        file_stat = _artifact_stat(resolved)
        snapshot = (str(resolved), *file_stat)
        cached = _ARTIFACT_IDENTITY_CACHE.get(str(resolved))
        if cached is not None and cached.get("snapshot") == snapshot:
            return json.loads(json.dumps(cached["identity"]))
        identity: dict[str, Any] = {
            "algorithm": "single-file-sha256-v1",
            "resolved_path": str(resolved),
        }
        identity["file_sha256"] = _stable_artifact_sha256(resolved, file_stat)
        _ARTIFACT_IDENTITY_CACHE[str(resolved)] = {
            "snapshot": snapshot,
            "identity": identity,
        }
        return json.loads(json.dumps(identity))

    files = sorted(candidate for candidate in resolved.rglob("*") if candidate.is_file())
    file_stats = {candidate: _artifact_stat(candidate) for candidate in files}
    snapshot = [
        (candidate.relative_to(resolved).as_posix(), *file_stats[candidate])
        for candidate in files
    ]
    cached = _ARTIFACT_IDENTITY_CACHE.get(str(resolved))
    if cached is not None and cached.get("snapshot") == snapshot:
        return json.loads(json.dumps(cached["identity"]))

    identity = {
        "algorithm": "recursive-model-artifact-sha256-v2",
        "resolved_path": str(resolved),
    }
    weight_suffixes = (
        ".safetensors", ".bin", ".pt", ".pth", ".h5", ".msgpack", ".onnx"
    )
    weights = [
        candidate
        for candidate in files
        if candidate.name.lower().endswith(weight_suffixes)
    ]
    index_files = [candidate for candidate in files if candidate.name.endswith(".index.json")]
    metadata_files = [
        candidate for candidate in files
        if candidate not in weights and candidate not in index_files
    ]
    identity["metadata_sha256"] = {
        candidate.relative_to(resolved).as_posix(): _stable_artifact_sha256(
            candidate, file_stats[candidate]
        )
        for candidate in metadata_files
    }
    identity["weight_files"] = [
        {
            "relative_path": candidate.relative_to(resolved).as_posix(),
            "size": candidate.stat().st_size,
            "sha256": _stable_artifact_sha256(candidate, file_stats[candidate]),
        }
        for candidate in weights
    ]
    identity["index_sha256"] = {
        candidate.relative_to(resolved).as_posix(): _stable_artifact_sha256(
            candidate, file_stats[candidate]
        )
        for candidate in index_files
        if candidate.is_file()
    }
    final_files = sorted(candidate for candidate in resolved.rglob("*") if candidate.is_file())
    final_snapshot = [
        (candidate.relative_to(resolved).as_posix(), *_artifact_stat(candidate))
        for candidate in final_files
    ]
    if final_snapshot != snapshot:
        raise RuntimeError(f"artifact directory changed while hashing: {resolved}")
    _ARTIFACT_IDENTITY_CACHE[str(resolved)] = {
        "snapshot": snapshot,
        "identity": identity,
    }
    return json.loads(json.dumps(identity))


def dataset_image_identity(
    mapping_path: str | Path,
    dataset_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Hash the ordered relative paths and exact bytes of every mapped image."""

    from fata.utils.paths import resolve_dataset_relative_path

    mapping = Path(mapping_path).expanduser().resolve()
    root = mapping.parent if dataset_dir is None else Path(dataset_dir).expanduser().resolve()
    cache_key = (str(mapping), str(root))
    rows: list[dict[str, Any]] = []
    with mapping.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not row.get("image_filename"):
                raise ValueError(f"invalid image mapping row at {mapping}:{line_number}")
            rows.append(row)
    image_names = expected_unique_ids(rows, "image_filename")
    if not image_names:
        raise ValueError(f"mapping contains no images: {mapping}")
    image_paths: list[Path] = []
    image_stats: list[tuple[str, int, int, int, int]] = []
    for image_name in image_names:
        image_path = resolve_dataset_relative_path(root, image_name)
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        stat = image_path.stat()
        image_paths.append(image_path)
        image_stats.append(
            (image_name, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino)
        )
    mapping_stat = mapping.stat()
    snapshot = {
        "mapping": (
            mapping_stat.st_size,
            mapping_stat.st_mtime_ns,
            mapping_stat.st_ctime_ns,
            mapping_stat.st_ino,
        ),
        "images": image_stats,
    }
    cached = _DATASET_IMAGE_IDENTITY_CACHE.get(cache_key)
    if cached is not None and cached.get("snapshot") == snapshot:
        return json.loads(json.dumps(cached["identity"]))
    aggregate = hashlib.sha256()
    total_bytes = 0
    for image_name, image_path in zip(image_names, image_paths):
        image_digest = sha256_file(image_path)
        size = image_path.stat().st_size
        total_bytes += size
        aggregate.update(image_name.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(str(size).encode("ascii"))
        aggregate.update(b"\0")
        aggregate.update(image_digest.encode("ascii"))
        aggregate.update(b"\n")
    identity = {
        "algorithm": "ordered-path-size-file-sha256-v1",
        "count": len(image_names),
        "total_bytes": total_bytes,
        "aggregate_sha256": aggregate.hexdigest(),
    }
    _DATASET_IMAGE_IDENTITY_CACHE[cache_key] = {
        "snapshot": snapshot,
        "identity": identity,
    }
    return json.loads(json.dumps(identity))


def explicit_image_identity(
    rows: Iterable[dict[str, Any]],
    *,
    id_key: str = "image_id",
    path_key: str = "image_path",
) -> dict[str, Any]:
    """Hash image bytes referenced by an explicit-path manifest."""

    records = list(rows)
    identifiers = expected_unique_ids(records, id_key)
    if not identifiers:
        raise ValueError("image manifest is empty")
    aggregate = hashlib.sha256()
    total_bytes = 0
    for row, identifier in zip(records, identifiers):
        raw_path = row.get(path_key)
        if not raw_path:
            raise ValueError(f"image manifest row {identifier!r} lacks {path_key}")
        image_path = Path(str(raw_path)).expanduser().resolve()
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        size = image_path.stat().st_size
        total_bytes += size
        aggregate.update(identifier.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(str(size).encode("ascii"))
        aggregate.update(b"\0")
        aggregate.update(sha256_file(image_path).encode("ascii"))
        aggregate.update(b"\n")
    return {
        "algorithm": "ordered-id-size-file-sha256-v1",
        "count": len(records),
        "total_bytes": total_bytes,
        "aggregate_sha256": aggregate.hexdigest(),
    }


def ensure_run_contract(
    path: str | Path,
    contract: dict[str, Any],
    *,
    result_path: str | Path | Sequence[str | Path] | None = None,
) -> None:
    """Create an immutable JSON contract, or require an exact resume match."""

    destination = Path(path)
    normalized = json.loads(json.dumps(contract, sort_keys=True))
    if destination.is_symlink():
        raise RuntimeError(f"run contract must not be a symbolic link: {destination}")
    if destination.exists():
        if not destination.is_file():
            raise RuntimeError(f"run contract is not a regular file: {destination}")
        stored = json.loads(destination.read_text(encoding="utf-8"))
        if stored != normalized:
            raise RuntimeError(
                f"resume contract mismatch for {destination}; choose a new output"
            )
        return
    if result_path is not None:
        results = (
            [result_path]
            if isinstance(result_path, (str, Path))
            else list(result_path)
        )
        for item in results:
            result = Path(item)
            if result.is_file() and result.stat().st_size > 0:
                raise RuntimeError(
                    f"existing result has no immutable run contract: {result}"
                )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.parent.is_symlink() or not destination.parent.is_dir():
        raise RuntimeError(
            f"run contract parent must be a real directory: {destination.parent}"
        )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.tmp.", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(normalized, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        if destination.is_symlink():
            raise RuntimeError(
                f"run contract became a symbolic link: {destination}"
            )
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def ensure_cache_contract(path: str | Path, contract: dict[str, Any]) -> None:
    """Create a cache contract only for a genuinely empty cache namespace.

    A missing sidecar next to pre-existing artifacts is not evidence that those
    artifacts satisfy the requested configuration.  Refuse to "bless" such an
    orphaned cache; callers must regenerate it in a new/empty namespace.
    """

    destination = Path(path)
    if not destination.exists() and destination.parent.exists():
        existing = [
            candidate
            for candidate in destination.parent.rglob("*")
            if candidate.is_file() and candidate != destination
        ]
        if existing:
            examples = ", ".join(str(item) for item in sorted(existing)[:3])
            raise RuntimeError(
                "cache artifacts exist without an immutable contract at "
                f"{destination}; use a new empty cache namespace or regenerate it "
                f"(examples: {examples})"
            )
    ensure_run_contract(destination, contract)


def require_run_contract(path: str | Path, contract: dict[str, Any]) -> None:
    """Require an existing immutable contract to match exactly."""

    source = Path(path)
    if source.is_symlink():
        raise RuntimeError(f"run contract must not be a symbolic link: {source}")
    if not source.is_file():
        raise RuntimeError(
            f"required cache/run contract is missing: {source}; regenerate the artifact"
        )
    stored = json.loads(source.read_text(encoding="utf-8"))
    normalized = json.loads(json.dumps(contract, sort_keys=True))
    if stored != normalized:
        raise RuntimeError(
            f"cache/run contract mismatch for {source}; regenerate or use a separate root"
        )


def expected_unique_ids(rows: Iterable[dict[str, Any]], key: str) -> list[str]:
    values: list[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or key not in row:
            raise ValueError(f"mapping row {index} lacks required key {key!r}")
        value = row[key]
        if not isinstance(value, str) or not value:
            raise ValueError(f"mapping row {index} has invalid {key!r}")
        values.append(value)
    if len(values) != len(set(values)):
        raise RuntimeError(f"mapping contains duplicate {key} values")
    return values


def read_completed_csv_ids(
    path: str | Path,
    *,
    expected_header: Sequence[str],
    id_column: int = 0,
    numeric_columns: Iterable[str] = (),
    numeric_bounds: dict[str, tuple[float, float]] | None = None,
    expected_values: dict[str, dict[str, str]] | None = None,
) -> set[str]:
    """Validate a row-oriented CSV and return unique completed identifiers."""

    source = Path(path)
    if not source.exists():
        return set()
    completed: set[str] = set()
    numeric = set(numeric_columns)
    unknown_numeric = numeric.difference(expected_header)
    if unknown_numeric:
        raise ValueError(f"numeric columns are absent from header: {sorted(unknown_numeric)}")
    numeric_indices = [
        index for index, name in enumerate(expected_header) if name in numeric
    ]
    bounds = {} if numeric_bounds is None else numeric_bounds
    unknown_bounds = set(bounds).difference(expected_header)
    if unknown_bounds:
        raise ValueError(f"bounded columns are absent from header: {sorted(unknown_bounds)}")
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header != list(expected_header):
            raise RuntimeError(f"unexpected CSV header in {source}")
        for line_number, row in enumerate(reader, start=2):
            if len(row) != len(expected_header):
                raise RuntimeError(
                    f"incomplete CSV row at {source}:{line_number}: "
                    f"expected {len(expected_header)} columns, got {len(row)}"
                )
            identifier = row[id_column]
            if not identifier:
                raise RuntimeError(f"empty result identifier at {source}:{line_number}")
            if identifier in completed:
                raise RuntimeError(f"duplicate result identifier in {source}: {identifier}")
            for index in numeric_indices:
                try:
                    value = float(row[index])
                except (TypeError, ValueError) as error:
                    raise RuntimeError(
                        f"non-numeric {expected_header[index]} at {source}:{line_number}"
                    ) from error
                if not math.isfinite(value):
                    raise RuntimeError(
                        f"non-finite {expected_header[index]} at {source}:{line_number}"
                    )
                if expected_header[index] in bounds:
                    lower, upper = bounds[expected_header[index]]
                    if value < lower or value > upper:
                        raise RuntimeError(
                            f"out-of-range {expected_header[index]} at "
                            f"{source}:{line_number}: {value} not in [{lower}, {upper}]"
                        )
            if expected_values is not None:
                if identifier not in expected_values:
                    raise RuntimeError(
                        f"unexpected result identifier in {source}: {identifier}"
                    )
                for column, expected in expected_values[identifier].items():
                    try:
                        index = list(expected_header).index(column)
                    except ValueError as error:
                        raise ValueError(
                            f"expected-value column is absent from header: {column}"
                        ) from error
                    if row[index] != expected:
                        raise RuntimeError(
                            f"{column} mismatch for {identifier} at {source}:{line_number}"
                        )
            completed.add(identifier)
    return completed


def assert_exact_completion(expected: Iterable[str], completed: Iterable[str], label: str) -> None:
    expected_set = set(expected)
    completed_set = set(completed)
    missing = sorted(expected_set - completed_set)
    unexpected = sorted(completed_set - expected_set)
    if missing or unexpected:
        raise RuntimeError(
            f"incomplete {label}: missing={len(missing)} unexpected={len(unexpected)} "
            f"missing_examples={missing[:5]} unexpected_examples={unexpected[:5]}"
        )
