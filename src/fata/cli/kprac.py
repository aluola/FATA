"""Recompute the 16 LLaVA Clean-threshold K_prac selections."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import math
import tempfile
from pathlib import Path
from statistics import fmean
from typing import Any

from fata.constants import COMPRESSORS, DATASETS, LLAVA_BUDGETS
from fata.attacks.linf_image import IMAGE_SERIALIZATION_CONTRACT
from fata.data.records import validate_record
from fata.evaluation.kprac import select_llava_clean_threshold_kprac
from fata.runtimes.llava.result_schema import (
    read_completed_result_ids,
    result_header,
    result_schema,
)
from fata.utils.paths import assert_output_separate, resolve_owned_output_path
from fata.utils.run_contract import (
    artifact_identity_valid,
    dataset_image_identity,
    dataset_identity_valid,
    expected_unique_ids,
    is_sha256_digest,
    read_completed_csv_ids,
)


SOURCE_SCORE_COLUMNS = ["Lang_Prior_K0"] + [
    f"{mode}_K{budget}"
    for mode in ("Clean", "Base", "FATA")
    for budget in LLAVA_BUDGETS
]
LEGACY_SOURCE_HEADER = ["Image_ID", "Question", *SOURCE_SCORE_COLUMNS]
V2_SOURCE_HEADER = result_header(["Image_ID", "Question"], SOURCE_SCORE_COLUMNS)
V2_RESULT_SCHEMA = result_schema(SOURCE_SCORE_COLUMNS)
# Backward-compatible public name for callers that explicitly mean the archived
# paper score-only layout.
EXPECTED_SOURCE_HEADER = LEGACY_SOURCE_HEADER


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clean_trajectory(path: Path) -> tuple[dict[int, float], list[str], list[str]]:
    required = {"Image_ID", *(f"Clean_K{k}" for k in LLAVA_BUDGETS)}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        header = list(reader.fieldnames or ())
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: no samples")
    ids = [row["Image_ID"] for row in rows]
    if any(not isinstance(value, str) or not value.strip() for value in ids):
        raise ValueError(f"{path}: empty Image_ID record")
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate Image_ID records")
    trajectory = {}
    for k in LLAVA_BUDGETS:
        try:
            values = [float(row[f"Clean_K{k}"]) for row in rows]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path}: non-numeric Clean_K{k}") from exc
        if any(not math.isfinite(value) or value < 0 or value > 1 for value in values):
            raise ValueError(f"{path}: Clean_K{k} values must be finite and in [0, 1]")
        trajectory[k] = fmean(values)
    return trajectory, ids, header


def _load_dataset_mapping(
    dataset_root: Path, dataset: str
) -> tuple[Path, Path, list[dict[str, Any]], list[str]]:
    """Strictly load one canonical mapping below an explicit read-only root."""

    root = dataset_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"dataset root is not a directory: {root}")
    dataset_dir = root / dataset
    mapping_path = dataset_dir / f"{dataset}_mapping.jsonl"
    if mapping_path.is_symlink() or not mapping_path.is_file():
        raise FileNotFoundError(
            f"required real dataset mapping is missing: {mapping_path}"
        )
    resolved_mapping = mapping_path.resolve()
    if not resolved_mapping.is_relative_to(root):
        raise ValueError(f"dataset mapping escapes --dataset-root: {mapping_path}")
    rows: list[dict[str, Any]] = []
    with mapping_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"blank mapping row at {mapping_path}:{line_number}")
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid mapping JSON at {mapping_path}:{line_number}"
                ) from error
            if not isinstance(raw, dict):
                raise ValueError(
                    f"mapping row is not an object at {mapping_path}:{line_number}"
                )
            row = validate_record(raw, source=f"{mapping_path}:{line_number}")
            if not isinstance(row.get("question"), str):
                raise ValueError(
                    f"{mapping_path}:{line_number}: question must be a string"
                )
            rows.append(row)
    ids = expected_unique_ids(rows, "image_filename")
    if not ids:
        raise ValueError(f"dataset mapping is empty: {mapping_path}")
    return dataset_dir, mapping_path, rows, ids


def _validate_v2_source_against_mapping(
    path: Path,
    *,
    contract: dict[str, Any],
    csv_ids: list[str],
    dataset_dir: Path,
    mapping_path: Path,
    mapping_rows: list[dict[str, Any]],
    mapping_ids: list[str],
) -> None:
    """Bind and independently rescore every answer-score-v2 result cell."""

    mapping_digest = _sha256(mapping_path)
    if contract.get("dataset_mapping_sha256") != mapping_digest:
        raise ValueError(f"{path}: dataset mapping SHA-256 does not match current mapping")
    current_images = dataset_image_identity(mapping_path, dataset_dir)
    if contract.get("dataset_images") != current_images:
        raise ValueError(f"{path}: dataset image identity does not match current dataset")
    if csv_ids != mapping_ids:
        raise ValueError(f"{path}: Image_ID order/cohort does not match current mapping")
    expected_values = {
        row["image_filename"]: {"Question": f'"{row["question"]}"'}
        for row in mapping_rows
    }
    completed = read_completed_result_ids(
        path,
        prefix_columns=["Image_ID", "Question"],
        score_columns=SOURCE_SCORE_COLUMNS,
        ground_truth_rows=mapping_rows,
        expected_values=expected_values,
    )
    if completed != set(mapping_ids):
        raise ValueError(f"{path}: v2 completed cohort does not match current mapping")


def _source_contract(
    path: Path,
    *,
    dataset: str,
    compressor: str,
    lam: float,
    seed: int,
    header: list[str],
    image_ids: list[str],
    allow_uncontracted_legacy: bool = False,
) -> tuple[dict | None, str]:
    meta_path = path.with_suffix(".meta.json")
    if meta_path.is_symlink():
        raise ValueError(f"source contract must not be a symbolic link: {meta_path}")
    if not meta_path.is_file():
        if meta_path.exists():
            raise ValueError(f"source contract is not a regular file: {meta_path}")
        if allow_uncontracted_legacy:
            if header != LEGACY_SOURCE_HEADER:
                raise ValueError(
                    f"{path}: only the exact 21-column legacy score-only header "
                    "may omit a source contract"
                )
            return None, "legacy_uncontracted_score_only_v0"
        raise FileNotFoundError(f"missing immutable source contract: {meta_path}")
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1,
        "image_serialization": IMAGE_SERIALIZATION_CONTRACT,
        "runtime": "llava_fata",
        "dataset": dataset,
        "method": compressor,
        "lambda": lam,
        "seed": seed,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"{meta_path}: {key} does not match requested protocol")
    if not is_sha256_digest(payload.get("dataset_mapping_sha256")):
        raise ValueError(f"{meta_path}: invalid dataset_mapping_sha256")
    if not dataset_identity_valid(payload.get("dataset_images")):
        raise ValueError(f"{meta_path}: invalid dataset_images")
    for key in ("model", "clip_model"):
        if not artifact_identity_valid(payload.get(key)):
            raise ValueError(f"{meta_path}: invalid {key}")
    if header == LEGACY_SOURCE_HEADER:
        if "result_schema" in payload:
            raise ValueError(
                f"{meta_path}: legacy score-only header must not claim a result_schema"
            )
        source_schema = "legacy_score_only_v1"
    elif header == V2_SOURCE_HEADER:
        if payload.get("result_schema") != V2_RESULT_SCHEMA:
            raise ValueError(
                f"{meta_path}: v2 answer columns require the exact result_schema contract"
            )
        source_schema = "llava_answer_score_v2"
    else:
        raise ValueError(
            f"{path}: does not have an accepted legacy-v1 or answer-score-v2 "
            "LLaVA main-run schema"
        )
    if payload.get("header") != header:
        raise ValueError(f"{meta_path}: header does not match source CSV")
    if payload.get("expected_image_ids") != image_ids:
        raise ValueError(f"{meta_path}: expected_image_ids do not match source CSV")
    return payload, source_schema


def _validate_uncontracted_legacy_source(path: Path, image_ids: list[str]) -> None:
    """Validate every scalar cell in the narrow uncontracted audit format."""

    completed = read_completed_csv_ids(
        path,
        expected_header=LEGACY_SOURCE_HEADER,
        numeric_columns=SOURCE_SCORE_COLUMNS,
        numeric_bounds={column: (0.0, 1.0) for column in SOURCE_SCORE_COLUMNS},
    )
    if completed != set(image_ids):
        raise ValueError(f"{path}: validated legacy cohort does not match CSV IDs")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True,
                        help="directory containing the 16 LLaVA main-run CSV files")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help=(
            "explicit read-only root containing <dataset>/<dataset>_mapping.jsonl; "
            "required when any input uses answer-score-v2"
        ),
    )
    parser.add_argument(
        "--allow-uncontracted-legacy",
        action="store_true",
        help=(
            "audit-only: allow exact 21-column legacy inputs without .meta.json; "
            "requires 1000 aligned rows in all 16 uniquely named files and cannot "
            "validate answers, dataset/model identity, seed, or attack protocol"
        ),
    )
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--input-pattern",
        default="ultimate_benchmark_lam{lam:g}_seed{seed}_{compressor}_{dataset}.csv",
        help="one filename pattern with lam, seed, compressor, and dataset placeholders",
    )
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--threshold", type=float, default=0.80)
    parser.add_argument(
        "--expected-count",
        type=_positive_int,
        default=1000,
        help="required aligned sample count in every source CSV (default: 1000)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.seed < 0:
        raise ValueError("--seed must be non-negative")
    if not math.isfinite(args.lam) or args.lam < 0:
        raise ValueError("--lam must be finite and non-negative")
    if not math.isfinite(args.threshold) or not 0 < args.threshold <= 1:
        raise ValueError("--threshold must be finite and in (0, 1]")
    if args.allow_uncontracted_legacy and args.expected_count != 1000:
        raise ValueError(
            "--allow-uncontracted-legacy requires --expected-count 1000"
        )
    try:
        probe = args.input_pattern.format(
            lam=args.lam,
            seed=args.seed,
            compressor=COMPRESSORS[0],
            dataset=DATASETS[0],
        )
    except (KeyError, ValueError) as error:
        raise ValueError(f"invalid --input-pattern: {error}") from error
    if Path(probe).name != probe or probe in {"", ".", ".."}:
        raise ValueError("--input-pattern must render to one safe filename component")
    if args.output_csv.resolve() == args.output_json.resolve():
        raise ValueError("--output-csv and --output-json must be different paths")
    # Selection outputs are summaries, never in-place transformations.  Reject
    # the entire source tree so neither a CSV nor a companion contract can be
    # silently replaced by a user-supplied output path.
    protected_inputs = {"input root": args.input_root}
    if args.dataset_root is not None:
        protected_inputs["dataset root"] = args.dataset_root
    assert_output_separate(args.output_csv, protected_inputs)
    assert_output_separate(args.output_json, protected_inputs)
    selections = []
    source_hashes: dict[tuple[str, str], str] = {}
    source_names: dict[tuple[str, str], str] = {}
    source_contract_hashes: dict[tuple[str, str], str] = {}
    source_result_schemas: dict[tuple[str, str], str] = {}
    resolved_source_paths: set[Path] = set()
    dataset_ids: dict[str, list[str]] = {}
    dataset_contracts: dict[str, dict] = {}
    mapping_cache: dict[
        str, tuple[Path, Path, list[dict[str, Any]], list[str]]
    ] = {}
    global_model_contract: dict | None = None
    for dataset in DATASETS:
        for compressor in COMPRESSORS:
            filename = args.input_pattern.format(
                lam=args.lam,
                seed=args.seed,
                compressor=compressor,
                dataset=dataset,
            )
            if Path(filename).name != filename or filename in {"", ".", ".."}:
                raise ValueError("--input-pattern rendered an unsafe filename")
            path = args.input_root / filename
            resolved_source = path.expanduser().resolve()
            if resolved_source in resolved_source_paths:
                raise ValueError(
                    "--input-pattern must resolve to 16 distinct source files"
                )
            resolved_source_paths.add(resolved_source)
            trajectory, ids, header = _clean_trajectory(path)
            count = len(ids)
            if count != args.expected_count:
                raise ValueError(
                    f"{path}: found {count} samples; expected {args.expected_count}"
                )
            if dataset in dataset_ids and ids != dataset_ids[dataset]:
                raise ValueError(
                    f"{path}: Image_ID order/set differs across compressors for {dataset}"
                )
            dataset_ids.setdefault(dataset, ids)
            contract, source_result_schema = _source_contract(
                path,
                dataset=dataset,
                compressor=compressor,
                lam=args.lam,
                seed=args.seed,
                header=header,
                image_ids=ids,
                allow_uncontracted_legacy=args.allow_uncontracted_legacy,
            )
            source_result_schemas[(dataset, compressor)] = source_result_schema
            if source_result_schema == "legacy_uncontracted_score_only_v0":
                _validate_uncontracted_legacy_source(path, ids)
            if source_result_schema == "llava_answer_score_v2":
                if args.dataset_root is None:
                    raise ValueError(
                        "answer-score-v2 inputs require explicit --dataset-root "
                        "for independent mapping and score validation"
                    )
                if dataset not in mapping_cache:
                    mapping_cache[dataset] = _load_dataset_mapping(
                        args.dataset_root, dataset
                    )
                dataset_dir, mapping_path, mapping_rows, mapping_ids = mapping_cache[
                    dataset
                ]
                _validate_v2_source_against_mapping(
                    path,
                    contract=contract,
                    csv_ids=ids,
                    dataset_dir=dataset_dir,
                    mapping_path=mapping_path,
                    mapping_rows=mapping_rows,
                    mapping_ids=mapping_ids,
                )
            if contract is not None:
                shared_contract = {
                    key: contract[key]
                    for key in (
                        "dataset_mapping_sha256", "dataset_images", "model", "clip_model",
                        "seed", "lambda",
                    )
                }
                if dataset in dataset_contracts and shared_contract != dataset_contracts[dataset]:
                    raise ValueError(f"{path}: source contracts disagree across compressors")
                dataset_contracts.setdefault(dataset, shared_contract)
                model_contract = {
                    "model": contract["model"],
                    "clip_model": contract["clip_model"],
                }
                if global_model_contract is not None and model_contract != global_model_contract:
                    raise ValueError(f"{path}: model identities disagree across the 16 inputs")
                if global_model_contract is None:
                    global_model_contract = model_contract
            source_hashes[(dataset, compressor)] = _sha256(path)
            source_names[(dataset, compressor)] = path.name
            if contract is not None:
                source_contract_hashes[(dataset, compressor)] = _sha256(
                    path.with_suffix(".meta.json")
                )
            selections.append(select_llava_clean_threshold_kprac(
                dataset=dataset, compressor=compressor,
                clean_accuracy_by_k=trajectory, sample_count=count,
                threshold=args.threshold,
            ))
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv = resolve_owned_output_path(
        args.output_csv.parent, args.output_csv.name
    )
    args.output_json = resolve_owned_output_path(
        args.output_json.parent, args.output_json.name
    )
    if len(resolved_source_paths) != len(DATASETS) * len(COMPRESSORS):
        raise RuntimeError("K-practical audit did not consume 16 distinct source files")
    contains_uncontracted = any(
        schema == "legacy_uncontracted_score_only_v0"
        for schema in source_result_schemas.values()
    )
    fields = [
        "dataset_version", "dataset", "compressor", "sample_count", "threshold",
        *(f"clean_acc_k{k}" for k in LLAVA_BUDGETS),
        "full_accuracy", "selected_k", "selected_accuracy", "retention",
        "source_file", "source_sha256", "source_result_schema",
    ]
    csv_fd, csv_tmp_name = tempfile.mkstemp(
        prefix=f".{args.output_csv.name}.tmp.", dir=args.output_csv.parent
    )
    json_fd, json_tmp_name = tempfile.mkstemp(
        prefix=f".{args.output_json.name}.tmp.", dir=args.output_json.parent
    )
    csv_tmp = Path(csv_tmp_name)
    json_tmp = Path(json_tmp_name)
    payload = {
        "protocol": "llava_clean_threshold_kprac",
        "dataset_version": args.dataset_version,
        "threshold": args.threshold,
        "lambda": None if contains_uncontracted else args.lam,
        "seed": None if contains_uncontracted else args.seed,
        "input_pattern": args.input_pattern,
        "candidate_k_ascending": sorted(k for k in LLAVA_BUDGETS if k != 576),
        "selection_count": len(selections),
        "expected_sample_count": args.expected_count,
        "source_sha256": {
            f"{dataset}/{compressor}": digest
            for (dataset, compressor), digest in sorted(source_hashes.items())
        },
        "source_contract_sha256": {
            f"{dataset}/{compressor}": digest
            for (dataset, compressor), digest in sorted(source_contract_hashes.items())
        },
        "source_result_schema": {
            f"{dataset}/{compressor}": schema
            for (dataset, compressor), schema in sorted(source_result_schemas.items())
        },
        "contains_uncontracted_legacy": contains_uncontracted,
        "provenance_limitations": (
            [
                "UNCONTRACTED LEGACY AUDIT ONLY: source CSVs have no immutable run sidecars",
                "decoded answers are absent, so stored scores cannot be independently rescored",
                "dataset mapping/images, model/CLIP weights, seed, lambda, and attack protocol are unverified",
            ]
            if contains_uncontracted
            else []
        ),
        "legacy_score_only_provenance_limitation": (
            "score-only sources do not persist decoded answers and cannot be "
            "independently rescored; uncontracted v0 sources additionally lack "
            "dataset/model/run sidecars"
        ),
        "selections": [selection.to_dict() for selection in selections],
    }
    try:
        with os.fdopen(csv_fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for selection in selections:
                row = {
                    "dataset_version": args.dataset_version,
                    "dataset": selection.dataset, "compressor": selection.compressor,
                    "sample_count": selection.sample_count, "threshold": selection.threshold,
                    "full_accuracy": selection.full_accuracy, "selected_k": selection.selected_k,
                    "selected_accuracy": selection.selected_accuracy, "retention": selection.retention,
                    "source_file": source_names[(selection.dataset, selection.compressor)],
                    "source_sha256": source_hashes[(selection.dataset, selection.compressor)],
                    "source_result_schema": source_result_schemas[
                        (selection.dataset, selection.compressor)
                    ],
                }
                row.update({f"clean_acc_k{k}": selection.trajectory[k] for k in LLAVA_BUDGETS})
                writer.writerow(row)
            handle.flush()
            os.fsync(handle.fileno())
        csv_fd = -1
        with os.fdopen(json_fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        json_fd = -1
        args.output_csv = resolve_owned_output_path(
            args.output_csv.parent, args.output_csv.name
        )
        args.output_json = resolve_owned_output_path(
            args.output_json.parent, args.output_json.name
        )
        os.replace(csv_tmp, args.output_csv)
        os.replace(json_tmp, args.output_json)
    finally:
        if csv_fd >= 0:
            os.close(csv_fd)
        if json_fd >= 0:
            os.close(json_fd)
        if csv_tmp.exists():
            csv_tmp.unlink()
        if json_tmp.exists():
            json_tmp.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
