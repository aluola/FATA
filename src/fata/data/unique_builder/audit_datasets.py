#!/usr/bin/env python3
"""Read-only exact/perceptual image audit for FATA mapping datasets."""

from __future__ import annotations

import argparse
import base64
import csv
import html
import io
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps
from fata.utils.paths import assert_output_separate, safe_filename_component

try:
    from .common import (
        CANONICAL_HASH_DESCRIPTION,
        DATASET_NAMES,
        atomic_write_json,
        atomic_write_jsonl,
        atomic_write_text,
        canonical_rgb_sha256,
        dhash,
        hamming_distance,
        sha256_file,
        write_csv_atomic,
    )
except ImportError:  # Direct script execution.
    from common import (  # type: ignore
        CANONICAL_HASH_DESCRIPTION,
        DATASET_NAMES,
        atomic_write_json,
        atomic_write_jsonl,
        atomic_write_text,
        canonical_rgb_sha256,
        dhash,
        hamming_distance,
        sha256_file,
        write_csv_atomic,
    )


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def image_record(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        normalized = ImageOps.exif_transpose(image).convert("RGB")
        normalized.load()
        width, height = normalized.size
        return {
            "canonical_rgb_sha256": canonical_rgb_sha256(normalized),
            "saved_image_sha256": sha256_file(path),
            "dhash": dhash(normalized),
            "width": width,
            "height": height,
        }


def _thumbnail_data_url(path: Path) -> str:
    try:
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image.thumbnail((220, 160), Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=70, optimize=True)
            return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()
    except (OSError, ValueError):
        return ""


def _resolve_reference(dataset_directory: Path, filename: Any) -> Path | None:
    if not isinstance(filename, str) or not filename:
        return None
    candidate = (dataset_directory / filename).resolve()
    try:
        candidate.relative_to(dataset_directory.resolve())
    except ValueError:
        return None
    return candidate


def audit_dataset_root(
    dataset_root: Path,
    reports_directory: Path,
    near_distance: int = 4,
    prefix: str = "old_dataset",
) -> dict[str, Any]:
    """Audit without mutating anything below ``dataset_root``."""

    dataset_root = dataset_root.resolve()
    prefix = safe_filename_component(prefix, label="report prefix")
    reports_directory = assert_output_separate(
        reports_directory, {"dataset root": dataset_root}
    )
    reports_directory.mkdir(parents=True, exist_ok=True)
    all_entries: list[dict[str, Any]] = []
    result: dict[str, Any] = {
        "audit_version": "dataset-duplicate-audit-v1",
        "audited_root": str(dataset_root),
        "canonical_rgb_sha256_definition": CANONICAL_HASH_DESCRIPTION,
        "perceptual_hash": (
            f"64-bit dHash; Hamming <= {near_distance} creates review-only candidates "
            "and is never automatic proof of equivalence"
        ),
        "input_artifacts": {},
        "datasets": {},
        "cross_dataset_exact_overlap": [],
    }

    for dataset in DATASET_NAMES:
        directory = dataset_root / dataset
        mapping_path = directory / f"{dataset}_mapping.jsonl"
        if not mapping_path.is_file():
            raise FileNotFoundError(mapping_path)
        raw_lines = mapping_path.read_text(encoding="utf-8").splitlines()
        rows: list[dict[str, Any]] = []
        parse_errors: list[dict[str, Any]] = []
        for line_number, line in enumerate(raw_lines, 1):
            try:
                parsed = json.loads(line)
                if not isinstance(parsed, dict):
                    raise TypeError("mapping row is not an object")
                rows.append(parsed)
            except (json.JSONDecodeError, TypeError) as error:
                parse_errors.append({"line": line_number, "error": repr(error)})

        actual_images = sorted(
            path
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if mapping_path.is_symlink() or any(path.is_symlink() for path in actual_images):
            raise ValueError(
                f"legacy audit inputs must not be symbolic links: {directory}"
            )
        result["input_artifacts"][dataset] = {
            "mapping": {
                "relative_path": mapping_path.relative_to(dataset_root).as_posix(),
                "size_bytes": mapping_path.stat().st_size,
                "sha256": sha256_file(mapping_path),
            },
            "images": [
                {
                    "relative_path": path.relative_to(dataset_root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in actual_images
            ],
        }
        missing: list[dict[str, Any]] = []
        invalid_filenames: list[int] = []
        corrupt: list[dict[str, Any]] = []
        entries: list[dict[str, Any]] = []
        existing_references: set[Path] = set()
        path_cache: dict[Path, dict[str, Any]] = {}
        for row_index, row in enumerate(rows):
            path = _resolve_reference(directory, row.get("image_filename"))
            if path is None:
                invalid_filenames.append(row_index)
                continue
            if not path.is_file():
                missing.append(
                    {"row_index": row_index, "image_filename": row.get("image_filename")}
                )
                continue
            existing_references.add(path)
            try:
                details = path_cache.get(path)
                if details is None:
                    details = image_record(path)
                    path_cache[path] = details
                entry = {
                    "dataset": dataset,
                    "mapping_row_index": row_index,
                    "image_filename": row.get("image_filename"),
                    "question": row.get("question", ""),
                    "path": str(path),
                    **details,
                }
                entries.append(entry)
                all_entries.append(entry)
            except (OSError, ValueError) as error:
                corrupt.append(
                    {
                        "row_index": row_index,
                        "image_filename": row.get("image_filename"),
                        "error": repr(error),
                    }
                )
        orphan_images = [
            str(path.relative_to(directory))
            for path in actual_images
            if path.resolve() not in existing_references
        ]
        canonical_counts = Counter(
            entry["canonical_rgb_sha256"] for entry in entries
        )
        file_counts = Counter(entry["saved_image_sha256"] for entry in entries)
        row_counts = Counter(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            for row in rows
        )
        question_image_counts = Counter(
            (entry["question"], entry["canonical_rgb_sha256"]) for entry in entries
        )
        hash_questions: dict[str, set[str]] = defaultdict(set)
        for entry in entries:
            hash_questions[entry["canonical_rgb_sha256"]].add(entry["question"])
        duplicate_rows = sum(count - 1 for count in canonical_counts.values() if count > 1)
        metrics = {
            "mapping_path": str(mapping_path),
            "mapping_rows": len(rows),
            "json_parse_errors": len(parse_errors),
            "actual_image_files": len(actual_images),
            "referenced_existing_images": len(entries),
            "unique_referenced_filenames": len(
                {entry["image_filename"] for entry in entries}
            ),
            "missing_images": len(missing),
            "orphan_images": len(orphan_images),
            "unreadable_or_corrupted_images": len(corrupt),
            "unique_canonical_rgb_images": len(canonical_counts),
            "duplicate_image_rows": duplicate_rows,
            "duplicate_image_rate": duplicate_rows / len(entries) if entries else None,
            "exact_canonical_duplicate_groups": sum(
                count > 1 for count in canonical_counts.values()
            ),
            "exact_saved_file_duplicate_groups": sum(
                count > 1 for count in file_counts.values()
            ),
            "max_duplicate_cluster_size": max(canonical_counts.values(), default=0),
            "max_distinct_questions_for_one_image": max(
                (len(questions) for questions in hash_questions.values()), default=0
            ),
            "duplicate_mapping_rows": sum(
                count - 1 for count in row_counts.values() if count > 1
            ),
            "duplicate_question_image_rows": sum(
                count - 1 for count in question_image_counts.values() if count > 1
            ),
            "invalid_image_filename_rows": len(invalid_filenames),
            "missing_details": missing,
            "orphan_details": orphan_images,
            "corrupt_details": corrupt,
            "parse_error_details": parse_errors,
        }
        result["datasets"][dataset] = metrics

    by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in all_entries:
        by_hash[entry["canonical_rgb_sha256"]].append(entry)
    exact_groups: list[dict[str, Any]] = []
    for group_number, (digest, entries) in enumerate(
        sorted((digest, entries) for digest, entries in by_hash.items() if len(entries) > 1),
        1,
    ):
        exact_groups.append(
            {
                "group_id": f"exact-{group_number:04d}",
                "canonical_rgb_sha256": digest,
                "row_count": len(entries),
                "datasets": sorted({entry["dataset"] for entry in entries}),
                "distinct_questions": len({entry["question"] for entry in entries}),
                "entries": [
                    {
                        key: entry[key]
                        for key in (
                            "dataset",
                            "mapping_row_index",
                            "image_filename",
                            "question",
                            "saved_image_sha256",
                            "dhash",
                            "width",
                            "height",
                        )
                    }
                    for entry in entries
                ],
            }
        )
    atomic_write_jsonl(
        reports_directory / f"{prefix}_duplicate_groups.jsonl", exact_groups
    )

    per_dataset_hashes = {
        dataset: {
            entry["canonical_rgb_sha256"]
            for entry in all_entries
            if entry["dataset"] == dataset
        }
        for dataset in DATASET_NAMES
    }
    for index, left in enumerate(DATASET_NAMES):
        for right in DATASET_NAMES[index + 1 :]:
            overlap = per_dataset_hashes[left] & per_dataset_hashes[right]
            result["cross_dataset_exact_overlap"].append(
                {
                    "dataset_a": left,
                    "dataset_b": right,
                    "unique_canonical_rgb_sha256_overlap": len(overlap),
                    "hashes": sorted(overlap),
                }
            )

    # Connected components among distinct exact images based only on dHash.
    nodes = [
        {"canonical": digest, "dhash": entries[0]["dhash"], "entries": entries}
        for digest, entries in sorted(by_hash.items())
    ]
    parents = list(range(len(nodes)))
    sizes = [1] * len(nodes)
    nearest = [65] * len(nodes)

    def find(node: int) -> int:
        while parents[node] != node:
            parents[node] = parents[parents[node]]
            node = parents[node]
        return node

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return
        if sizes[left_root] < sizes[right_root]:
            left_root, right_root = right_root, left_root
        parents[right_root] = left_root
        sizes[left_root] += sizes[right_root]

    candidate_edges = 0
    for left in range(len(nodes)):
        for right in range(left + 1, len(nodes)):
            distance = hamming_distance(nodes[left]["dhash"], nodes[right]["dhash"])
            if distance <= near_distance:
                union(left, right)
                nearest[left] = min(nearest[left], distance)
                nearest[right] = min(nearest[right], distance)
                candidate_edges += 1
    components: dict[int, list[int]] = defaultdict(list)
    for node in range(len(nodes)):
        components[find(node)].append(node)
    near_components = [members for members in components.values() if len(members) > 1]
    near_components.sort(key=lambda members: (-len(members), nodes[members[0]]["canonical"]))
    result["near_duplicate_review"] = {
        "threshold": near_distance,
        "unique_exact_image_nodes": len(nodes),
        "candidate_edges": candidate_edges,
        "candidate_components": len(near_components),
        "candidate_unique_images": sum(len(component) for component in near_components),
        "classification": "review-only; no automatic equivalence decision",
    }

    page = [
        '<!doctype html><html><head><meta charset="utf-8">',
        "<title>Dataset near-duplicate review</title>",
        "<style>body{font-family:sans-serif;margin:24px;background:#f6f6f6}"
        ".note{background:#fff4cc;padding:12px}.group{background:#fff;margin:20px 0;"
        "padding:14px;border:1px solid #ccc}.grid{display:grid;grid-template-columns:"
        "repeat(auto-fill,minmax(250px,1fr));gap:12px}.card{border:1px solid #ddd;"
        "padding:8px;overflow-wrap:anywhere}.card img{display:block;max-width:220px;"
        "max-height:160px;margin:auto}.q{font-size:12px}.meta{font-size:11px;color:#444}</style>",
        "</head><body><h1>Dataset near-duplicate review</h1>",
        f'<p class="note">dHash distance &le; {near_distance} generates review-only '
        "candidates. No pair is automatically declared identical.</p>",
    ]
    for group_number, members in enumerate(near_components, 1):
        representative_hash = nodes[members[0]]["dhash"]
        page.append(
            f'<section class="group"><h2>Candidate cluster {group_number} '
            f'({len(members)} distinct exact images)</h2><div class="grid">'
        )
        for member in members:
            distance = hamming_distance(representative_hash, nodes[member]["dhash"])
            for entry in nodes[member]["entries"]:
                thumbnail = _thumbnail_data_url(Path(entry["path"]))
                image_tag = f'<img src="{thumbnail}" alt="thumbnail">' if thumbnail else ""
                page.append(
                    '<div class="card">'
                    + image_tag
                    + '<div class="meta">'
                    + f'dataset={html.escape(entry["dataset"])}<br>'
                    + f'file={html.escape(entry["image_filename"])}<br>'
                    + f'dHash={entry["dhash"]}<br>'
                    + f'distance_to_cluster_representative={distance}<br>'
                    + f'nearest_candidate_distance={nearest[member]}<br>'
                    + f'canonical={entry["canonical_rgb_sha256"]}</div>'
                    + f'<div class="q">question={html.escape(str(entry["question"]))}</div>'
                    + "</div>"
                )
        page.append("</div></section>")
    page.append("</body></html>")
    atomic_write_text(
        reports_directory / f"{prefix}_near_duplicate_review.html", "".join(page)
    )

    result["global"] = {
        "mapping_rows": sum(
            metrics["mapping_rows"] for metrics in result["datasets"].values()
        ),
        "referenced_existing_images": len(all_entries),
        "unique_canonical_rgb_images": len(by_hash),
        "duplicate_image_rows_global": len(all_entries) - len(by_hash),
        "exact_duplicate_groups_global": len(exact_groups),
    }
    atomic_write_json(reports_directory / f"{prefix}_duplicate_audit.json", result)
    columns = [
        "dataset",
        "mapping_rows",
        "actual_image_files",
        "referenced_existing_images",
        "unique_referenced_filenames",
        "missing_images",
        "orphan_images",
        "unreadable_or_corrupted_images",
        "unique_canonical_rgb_images",
        "duplicate_image_rows",
        "duplicate_image_rate",
        "exact_canonical_duplicate_groups",
        "exact_saved_file_duplicate_groups",
        "max_duplicate_cluster_size",
        "max_distinct_questions_for_one_image",
        "duplicate_mapping_rows",
        "duplicate_question_image_rows",
        "invalid_image_filename_rows",
    ]
    csv_rows = [
        {"dataset": dataset, **result["datasets"][dataset]}
        for dataset in DATASET_NAMES
    ]
    write_csv_atomic(
        reports_directory / f"{prefix}_duplicate_audit.csv", csv_rows, columns
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--reports-dir", type=Path, required=True)
    parser.add_argument("--near-distance", type=int, default=4)
    parser.add_argument("--prefix", default="old_dataset")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = audit_dataset_root(
        args.dataset_root, args.reports_dir, args.near_distance, args.prefix
    )
    print(json.dumps(summary["global"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
