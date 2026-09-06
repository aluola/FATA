"""Fail-closed structural checker for the four-compressor ML-ATD full grid."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Sequence
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any

import numpy as np

from fata.attacks.linf_image import IMAGE_SERIALIZATION_CONTRACT
from fata.runtimes.llava.attack_cache_io import attack_image_path, attack_namespace
from fata.utils.paths import resolve_owned_output_path
from fata.utils.run_contract import (
    artifact_identity_valid as _artifact_identity_valid,
    dataset_identity_valid as _dataset_identity_valid,
    is_sha256_digest as _is_sha256,
)
from .mlat_core import FORMAL_LLM_LAYERS, FORMAL_VISION_LAYERS


METHODS = ("VisionZIP", "VisPruner", "PruMerge", "FlowCut")
DATASETS = ("TextVQA_Open", "VQAv2_Open", "ScienceQA_MC", "VQAv2_MC")
ATTACKS = ("clean_clip", "random_clip", "base", "fata")
PRACTICAL_K = {
    "TextVQA_Open": 64,
    "VQAv2_Open": 64,
    "ScienceQA_MC": 32,
    "VQAv2_MC": 32,
}
VISION_LAYERS = FORMAL_VISION_LAYERS
LLM_LAYERS = FORMAL_LLM_LAYERS
REQUIRED_METADATA_ARRAYS = {
    "sample_idx",
    "label",
    "image_span_valid",
    "image_token_count",
    "llm_sequence_length",
    "vision_layers",
    "llm_layers",
    "image_id",
}
DIRECT_CSV_COLUMNS = (
    "sample_idx",
    "Image_ID",
    "Question",
    "dataset",
    "method",
    "attack_for_detection",
    "label",
    "seed",
    "sample_seed",
    "token_budget",
    "token_budget_mode",
    "eps_255",
    "alpha_255",
    "steps",
    "lam",
    "target_k",
    "vision_layers",
    "llm_layers",
    "image_span_valid",
    "image_token_count",
    "llm_sequence_length",
)
CACHED_CSV_COLUMNS = DIRECT_CSV_COLUMNS + ("cache_path",)
REQUIRED_CSV_COLUMNS = set(DIRECT_CSV_COLUMNS)
REQUIRED_CONTRACT_KEYS = {
    "schema_version",
    "image_serialization",
    "runtime",
    "feature_schema",
    "dataset",
    "dataset_mapping_sha256",
    "dataset_images",
    "method",
    "attack",
    "llava_model",
    "clip_model",
    "seed",
    "eps_255",
    "alpha_255",
    "steps",
    "lam",
    "target_k",
    "token_budget_mode",
    "token_budget",
    "vision_layers",
    "llm_layers",
    "expected_sample_indices",
    "expected_image_ids",
    "source_cache_contract",
}
FEATURE_NORM_MIN = 0.95
FEATURE_NORM_MAX = 1.05


def practical_token_budget(method: str, dataset: str) -> int:
    """Return the actual practical-K rule used by the historical full grid."""

    if method == "FlowCut" and dataset == "VQAv2_MC":
        return 64
    return PRACTICAL_K[dataset]


def _selection(value: str, allowed: Sequence[str]) -> list[str]:
    if value.strip().lower() in {"all", "*"}:
        return list(allowed)
    selected = [part.strip() for part in value.split(",") if part.strip()]
    unknown = [item for item in selected if item not in allowed]
    if not selected or unknown:
        raise ValueError(
            f"invalid selection {value!r}; allowed={','.join(allowed)}"
        )
    return selected


def _feature_names() -> set[str]:
    return {
        *(f"vis_l{layer:02d}" for layer in VISION_LAYERS),
        "proj_in",
        "proj_out",
        *(f"llm_last_l{layer:02d}" for layer in LLM_LAYERS),
        *(f"llm_image_l{layer:02d}" for layer in LLM_LAYERS),
    }


def _finite_numeric(array: np.ndarray) -> bool:
    return np.issubdtype(array.dtype, np.number) and bool(np.isfinite(array).all())


def _integer_column(rows: list[dict[str, str]], name: str) -> list[int]:
    return [int(row[name]) for row in rows]


def _float_column(rows: list[dict[str, str]], name: str) -> list[float]:
    values = [float(row[name]) for row in rows]
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"non-finite {name}")
    return values


def validate_triplet(
    npz_path: Path,
    *,
    method: str,
    dataset: str,
    attack: str,
    token_budget: int,
    start: int,
    limit: int,
    seed: int,
    token_budget_mode: str = "practical",
    eps_255: float = 2.0,
    alpha_255: float = 0.5,
    steps: int = 100,
    lam: float = 1.0,
    target_k: int = 64,
    require_valid_spans: bool = True,
    expected_cache_root: Path | None = None,
) -> list[str]:
    """Validate one NPZ/CSV/meta triple without importing or loading a model."""

    csv_path = npz_path.with_suffix(".csv")
    meta_path = npz_path.with_suffix(".meta.json")
    issues: list[str] = []
    for path, label in (
        (npz_path, "NPZ"),
        (csv_path, "CSV"),
        (meta_path, "META"),
    ):
        if path.is_symlink():
            issues.append(f"SYMLINK_{label} {path}")
        elif not path.is_file():
            issues.append(f"MISSING_{label} {path}")
    if issues:
        return issues

    expected_indices = list(range(start, start + limit))
    expected_label = 0 if attack in {"clean_clip", "random_clip"} else 1
    npz_indices: list[int] | None = None
    npz_image_ids: list[str] | None = None
    npz_row_metadata: dict[str, list[int]] = {}
    try:
        with np.load(npz_path, allow_pickle=False) as archive:
            expected_arrays = REQUIRED_METADATA_ARRAYS | _feature_names()
            actual_arrays = set(archive.files)
            for name in sorted(actual_arrays - expected_arrays):
                issues.append(f"UNEXPECTED_NPZ_ARRAY {npz_path} {name}")
            missing_metadata = sorted(REQUIRED_METADATA_ARRAYS.difference(archive.files))
            for name in missing_metadata:
                issues.append(f"MISSING_METADATA_ARRAY {npz_path} {name}")

            if "sample_idx" in archive.files:
                sample_idx = np.asarray(archive["sample_idx"])
                if sample_idx.ndim != 1 or sample_idx.dtype != np.dtype(np.int64):
                    issues.append(f"BAD_SHAPE_OR_DTYPE {npz_path} sample_idx")
                else:
                    npz_indices = [int(value) for value in sample_idx.tolist()]
                    if len(npz_indices) != len(set(npz_indices)):
                        issues.append(f"DUPLICATE_SAMPLE_IDX {npz_path}")
                    if npz_indices != expected_indices:
                        issues.append(
                            f"NONCONTIGUOUS_SAMPLE_IDX {npz_path} "
                            f"found={npz_indices[:10]} expected={expected_indices[:10]}"
                        )

            if "image_id" in archive.files:
                image_ids = np.asarray(archive["image_id"])
                if (
                    image_ids.ndim != 1
                    or len(image_ids) != limit
                    or image_ids.dtype.kind != "U"
                ):
                    issues.append(f"BAD_SHAPE_OR_DTYPE {npz_path} image_id")
                else:
                    npz_image_ids = [str(value) for value in image_ids.tolist()]
                    if any(not value for value in npz_image_ids):
                        issues.append(f"EMPTY_IMAGE_ID {npz_path}")
                    if len(npz_image_ids) != len(set(npz_image_ids)):
                        issues.append(f"DUPLICATE_IMAGE_ID {npz_path}")

            for name in (
                "label",
                "image_span_valid",
                "image_token_count",
                "llm_sequence_length",
            ):
                if name not in archive.files:
                    continue
                array = np.asarray(archive[name])
                expected_dtype = {
                    "label": np.dtype(np.int8),
                    "image_span_valid": np.dtype(np.int8),
                    "image_token_count": np.dtype(np.int32),
                    "llm_sequence_length": np.dtype(np.int32),
                }[name]
                if (
                    array.ndim != 1
                    or len(array) != limit
                    or array.dtype != expected_dtype
                    or not _finite_numeric(array)
                ):
                    issues.append(f"BAD_SHAPE_DTYPE_OR_VALUE {npz_path} {name}")
                    continue
                npz_row_metadata[name] = [int(value) for value in array.tolist()]
                if name == "label" and np.any(array != expected_label):
                    issues.append(f"BAD_LABEL {npz_path}")
                elif name == "image_span_valid":
                    if np.any((array != 0) & (array != 1)):
                        issues.append(f"NONBINARY_IMAGE_SPAN {npz_path}")
                    elif require_valid_spans and np.any(array != 1):
                        issues.append(f"INVALID_IMAGE_SPAN {npz_path}")
                elif name == "llm_sequence_length" and np.any(
                    array <= (576 if require_valid_spans else 0)
                ):
                    issues.append(f"INVALID_LLM_SEQUENCE_LENGTH {npz_path}")
                elif name == "image_token_count" and (
                    np.any(array != 576)
                    if require_valid_spans
                    else np.any(array < 0)
                ):
                    issues.append(f"INVALID_IMAGE_TOKEN_COUNT {npz_path}")

            for name, expected in (
                ("vision_layers", VISION_LAYERS),
                ("llm_layers", LLM_LAYERS),
            ):
                if name not in archive.files:
                    continue
                array = np.asarray(archive[name])
                if (
                    array.ndim != 1
                    or array.dtype != np.dtype(np.int16)
                    or tuple(int(value) for value in array.tolist()) != expected
                ):
                    issues.append(f"BAD_LAYER_METADATA {npz_path} {name}")

            expected_features = _feature_names()
            missing_features = sorted(expected_features.difference(archive.files))
            for name in missing_features:
                issues.append(f"MISSING_FEATURE_ARRAY {npz_path} {name}")
            discovered_features = [
                name
                for name in archive.files
                if name.startswith(("vis_l", "proj_", "llm_last_l", "llm_image_l"))
            ]
            feature_has_cross_row_variation = False
            for name in sorted(set(discovered_features).difference(expected_features)):
                issues.append(f"UNEXPECTED_FEATURE_ARRAY {npz_path} {name}")
            for name in discovered_features:
                array = np.asarray(archive[name])
                if (
                    array.ndim != 2
                    or array.shape[0] != limit
                    or array.shape[1] <= 0
                    or array.dtype != np.dtype(np.float16)
                    or not _finite_numeric(array)
                ):
                    issues.append(f"BAD_FEATURE_ARRAY {npz_path} {name}")
                    continue
                # Every producer path applies torch.nn.functional.normalize
                # before float16 serialization. Shape/dtype/finite checks alone
                # would accept all-zero or arbitrarily scaled fabricated
                # features and make CPU resume skip real extraction.
                row_norms = np.linalg.norm(array.astype(np.float32), axis=1)
                if np.any(row_norms < FEATURE_NORM_MIN) or np.any(
                    row_norms > FEATURE_NORM_MAX
                ):
                    issues.append(
                        f"BAD_FEATURE_NORM {npz_path} {name} "
                        f"range=[{float(row_norms.min()):.6g},"
                        f"{float(row_norms.max()):.6g}]"
                    )
                if limit > 1 and np.unique(array, axis=0).shape[0] > 1:
                    feature_has_cross_row_variation = True

            # Byte-identical legacy images can legitimately make visual and
            # projector rows equal, while question-dependent LLM features still
            # vary. Reject only a wholly constant feature artifact—the forged
            # case in which every stage/layer repeats one unit vector.
            if limit > 1 and not feature_has_cross_row_variation:
                issues.append(f"ALL_FEATURES_CONSTANT {npz_path}")

            exempt = {"vision_layers", "llm_layers"}
            for name in archive.files:
                array = np.asarray(archive[name])
                if name not in exempt and (
                    array.ndim == 0 or array.shape[0] != limit
                ):
                    issues.append(f"MISALIGNED_ARRAY {npz_path} {name}")
    except Exception as exc:
        issues.append(f"NPZ_ERROR {npz_path} {type(exc).__name__}: {exc}")

    csv_indices: list[int] | None = None
    csv_image_ids: list[str] | None = None
    csv_row_metadata: dict[str, list[int]] = {}
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            ordered_fieldnames = tuple(reader.fieldnames or ())
            fieldnames = set(ordered_fieldnames)
            rows = list(reader)
        expected_csv_columns = (
            CACHED_CSV_COLUMNS
            if attack in {"cage", "caa"}
            else DIRECT_CSV_COLUMNS
        )
        if ordered_fieldnames != expected_csv_columns:
            issues.append(
                f"BAD_CSV_HEADER {csv_path} found={ordered_fieldnames!r} "
                f"expected={expected_csv_columns!r}"
            )
        for name in sorted(REQUIRED_CSV_COLUMNS.difference(fieldnames)):
            issues.append(f"MISSING_CSV_COLUMN {csv_path} {name}")
        if len(rows) != limit:
            issues.append(
                f"BAD_CSV_ROWS {csv_path} found={len(rows)} expected={limit}"
            )
        if {"sample_idx", "Image_ID"}.issubset(fieldnames):
            csv_indices = _integer_column(rows, "sample_idx")
            csv_image_ids = [str(row["Image_ID"]) for row in rows]
            if len(csv_indices) != len(set(csv_indices)):
                issues.append(f"DUPLICATE_SAMPLE_IDX {csv_path}")
            if csv_indices != expected_indices:
                issues.append(f"NONCONTIGUOUS_SAMPLE_IDX {csv_path}")
            if any(not value for value in csv_image_ids):
                issues.append(f"EMPTY_IMAGE_ID {csv_path}")
            if len(csv_image_ids) != len(set(csv_image_ids)):
                issues.append(f"DUPLICATE_IMAGE_ID {csv_path}")
        if REQUIRED_CSV_COLUMNS.issubset(fieldnames):
            expected_text = {
                "dataset": dataset,
                "method": method,
                "attack_for_detection": attack,
                "token_budget_mode": token_budget_mode,
            }
            for name, expected in expected_text.items():
                if any(row[name] != expected for row in rows):
                    issues.append(f"BAD_CSV_METADATA {csv_path} {name}")
            expected_int = {
                "label": expected_label,
                "seed": seed,
                "token_budget": token_budget,
            }
            for name, expected in expected_int.items():
                values = _integer_column(rows, name)
                if name == "label":
                    csv_row_metadata[name] = values
                if any(value != expected for value in values):
                    issues.append(f"BAD_CSV_METADATA {csv_path} {name}")
            span_values = _integer_column(rows, "image_span_valid")
            csv_row_metadata["image_span_valid"] = span_values
            if any(value not in {0, 1} for value in span_values) or (
                require_valid_spans and any(value != 1 for value in span_values)
            ):
                issues.append(f"BAD_CSV_METADATA {csv_path} image_span_valid")
            expected_sample_seeds = [seed * 1_000_003 + index for index in expected_indices]
            if _integer_column(rows, "sample_seed") != expected_sample_seeds:
                issues.append(f"BAD_CSV_METADATA {csv_path} sample_seed")
            for name in (
                "sample_seed",
                "image_token_count",
                "llm_sequence_length",
                "steps",
                "target_k",
            ):
                values = _integer_column(rows, name)
                if name in {"image_token_count", "llm_sequence_length"}:
                    csv_row_metadata[name] = values
                if name in {"llm_sequence_length", "steps", "target_k"} and any(
                    value <= (576 if name == "llm_sequence_length" and require_valid_spans else 0)
                    for value in values
                ):
                    issues.append(f"NONPOSITIVE_CSV_METADATA {csv_path} {name}")
                if name == "image_token_count" and any(
                    value != 576 if require_valid_spans else value < 0
                    for value in values
                ):
                    issues.append(f"INVALID_CSV_METADATA {csv_path} {name}")
            expected_float = {"eps_255": eps_255, "alpha_255": alpha_255, "lam": lam}
            for name, expected in expected_float.items():
                values = _float_column(rows, name)
                if any(value != expected for value in values):
                    issues.append(f"BAD_CSV_METADATA {csv_path} {name}")
            if any(value != steps for value in _integer_column(rows, "steps")):
                issues.append(f"BAD_CSV_METADATA {csv_path} steps")
            if any(value != target_k for value in _integer_column(rows, "target_k")):
                issues.append(f"BAD_CSV_METADATA {csv_path} target_k")
            expected_vision = ",".join(map(str, VISION_LAYERS))
            expected_llm = ",".join(map(str, LLM_LAYERS))
            if any(row["vision_layers"] != expected_vision for row in rows):
                issues.append(f"BAD_CSV_METADATA {csv_path} vision_layers")
            if any(row["llm_layers"] != expected_llm for row in rows):
                issues.append(f"BAD_CSV_METADATA {csv_path} llm_layers")
            if attack in {"cage", "caa"} and "cache_path" in fieldnames:
                namespace = attack_namespace(
                    attack,
                    seed=seed,
                    eps_255=eps_255,
                    alpha_255=alpha_255,
                    steps=steps,
                )
                for row in rows:
                    raw_image_id = str(row["Image_ID"])
                    relative = PurePosixPath(raw_image_id)
                    valid_relative = (
                        raw_image_id
                        and "\\" not in raw_image_id
                        and not relative.is_absolute()
                        and relative.as_posix() == raw_image_id
                        and all(part not in {"", ".", ".."} for part in relative.parts)
                    )
                    raw_cache_path = str(row["cache_path"])
                    cache_path = Path(raw_cache_path)
                    expected_path = None
                    if expected_cache_root is not None and valid_relative:
                        try:
                            expected_path = attack_image_path(
                                cache_root=expected_cache_root,
                                attack=namespace,
                                dataset=dataset,
                                method="shared",
                                image_filename=raw_image_id,
                            )
                        except ValueError:
                            expected_path = None
                    if (
                        not valid_relative
                        or not cache_path.is_absolute()
                        or expected_path is None
                        or cache_path.expanduser().resolve() != expected_path
                    ):
                        issues.append(
                            f"BAD_CSV_CACHE_PATH {csv_path} "
                            f"image_id={raw_image_id!r} path={raw_cache_path!r}"
                        )
    except Exception as exc:
        issues.append(f"CSV_ERROR {csv_path} {type(exc).__name__}: {exc}")

    if (
        npz_indices is not None
        and csv_indices is not None
        and npz_indices != csv_indices
    ):
        issues.append(f"NPZ_CSV_SAMPLE_IDX_MISMATCH {npz_path} {csv_path}")
    if (
        npz_image_ids is not None
        and csv_image_ids is not None
        and npz_image_ids != csv_image_ids
    ):
        issues.append(f"NPZ_CSV_IMAGE_ID_MISMATCH {npz_path} {csv_path}")
    for name in ("label", "image_span_valid", "image_token_count", "llm_sequence_length"):
        if (
            name in npz_row_metadata
            and name in csv_row_metadata
            and npz_row_metadata[name] != csv_row_metadata[name]
        ):
            issues.append(f"NPZ_CSV_METADATA_MISMATCH {npz_path} {csv_path} {name}")

    try:
        contract = json.loads(meta_path.read_text(encoding="utf-8"))
        if not isinstance(contract, dict):
            raise ValueError("expected JSON object")
        for name in sorted(REQUIRED_CONTRACT_KEYS.difference(contract)):
            issues.append(f"MISSING_META_KEY {meta_path} {name}")
        expected_contract: dict[str, Any] = {
            "schema_version": 1,
            "image_serialization": IMAGE_SERIALIZATION_CONTRACT,
            "runtime": "mlatd_feature_extraction_v1",
            "feature_schema": "quarter_layers_projector_llm_last_image_v1",
            "dataset": dataset,
            "method": method,
            "attack": attack,
            "seed": seed,
            "eps_255": eps_255,
            "alpha_255": alpha_255,
            "steps": steps,
            "lam": lam,
            "target_k": target_k,
            "token_budget_mode": token_budget_mode,
            "token_budget": token_budget,
            "vision_layers": list(VISION_LAYERS),
            "llm_layers": list(LLM_LAYERS),
            "expected_sample_indices": expected_indices,
        }
        if npz_image_ids is not None:
            expected_contract["expected_image_ids"] = npz_image_ids
        for name, expected in expected_contract.items():
            if contract.get(name) != expected:
                issues.append(
                    f"BAD_META_VALUE {meta_path} {name} "
                    f"found={contract.get(name)!r} expected={expected!r}"
                )
        if not _is_sha256(contract.get("dataset_mapping_sha256")):
            issues.append(f"BAD_META_IDENTITY {meta_path} dataset_mapping_sha256")
        if not _dataset_identity_valid(contract.get("dataset_images")):
            issues.append(f"BAD_META_IDENTITY {meta_path} dataset_images")
        for name in ("llava_model", "clip_model"):
            if not _artifact_identity_valid(contract.get(name)):
                issues.append(f"BAD_META_IDENTITY {meta_path} {name}")
    except Exception as exc:
        issues.append(f"META_ERROR {meta_path} {type(exc).__name__}: {exc}")

    return issues


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-dir",
        "--result_dir",
        dest="result_dir",
        type=Path,
        required=True,
    )
    parser.add_argument("--methods", default="all")
    parser.add_argument("--datasets", default="all")
    parser.add_argument("--attacks", default="all")
    parser.add_argument("--total_limit", type=int, default=1000)
    parser.add_argument("--chunk_size", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eps-255", type=float, default=2.0)
    parser.add_argument("--alpha-255", type=float, default=0.5)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--target-k", type=int, default=64)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.total_limit <= 0 or args.chunk_size <= 0 or args.seed < 0:
        parser.error("grid sizes must be positive and --seed non-negative")
    if (
        not all(math.isfinite(value) for value in (args.eps_255, args.alpha_255, args.lam))
        or args.eps_255 <= 0
        or args.alpha_255 <= 0
        or args.steps <= 0
        or args.lam < 0
        or args.target_k <= 0
    ):
        parser.error("invalid attack parameters")
    if (
        args.eps_255 != 2.0
        or args.alpha_255 != 0.5
        or args.steps != 100
        or args.lam != 1.0
        or args.target_k != 64
    ):
        parser.error(
            "formal ML-ATD grid is locked to eps=2, alpha=0.5, "
            "steps=100, lambda=1, target_k=64"
        )
    try:
        methods = _selection(args.methods, METHODS)
        datasets = _selection(args.datasets, DATASETS)
        attacks = _selection(args.attacks, ATTACKS)
    except ValueError as exc:
        parser.error(str(exc))

    result_root = args.result_dir.expanduser().resolve()
    if not result_root.is_dir():
        parser.error(f"--result-dir is not a directory: {result_root}")
    issues: list[str] = []
    checked = 0
    global_models: dict[str, Any] | None = None
    dataset_identities: dict[str, dict[str, Any]] = {}
    chunk_image_ids: dict[tuple[str, int, int], list[str]] = {}
    for method in methods:
        for dataset in datasets:
            token_budget = practical_token_budget(method, dataset)
            for attack in attacks:
                for start in range(0, args.total_limit, args.chunk_size):
                    limit = min(args.chunk_size, args.total_limit - start)
                    prefix = (
                        f"mlat_feat_{method}_{dataset}_{attack}_k{token_budget}_"
                        f"start{start}_limit{limit}_seed{args.seed}"
                    )
                    checked += 1
                    try:
                        npz_path = resolve_owned_output_path(
                            result_root, f"{prefix}.npz"
                        )
                    except ValueError as exc:
                        issues.append(f"UNSAFE_RESULT_PATH {prefix}: {exc}")
                        continue
                    triplet_issues = validate_triplet(
                            npz_path,
                            method=method,
                            dataset=dataset,
                            attack=attack,
                            token_budget=token_budget,
                            start=start,
                            limit=limit,
                            seed=args.seed,
                            eps_255=args.eps_255,
                            alpha_255=args.alpha_255,
                            steps=args.steps,
                            lam=args.lam,
                            target_k=args.target_k,
                        )
                    issues.extend(triplet_issues)
                    if triplet_issues:
                        continue
                    contract = json.loads(
                        npz_path.with_suffix(".meta.json").read_text(encoding="utf-8")
                    )
                    models = {
                        "llava_model": contract["llava_model"],
                        "clip_model": contract["clip_model"],
                    }
                    if global_models is not None and models != global_models:
                        issues.append(f"CROSS_GRID_MODEL_IDENTITY_MISMATCH {npz_path}")
                    if global_models is None:
                        global_models = models
                    identities = {
                        "dataset_mapping_sha256": contract["dataset_mapping_sha256"],
                        "dataset_images": contract["dataset_images"],
                    }
                    if dataset in dataset_identities and identities != dataset_identities[dataset]:
                        issues.append(f"CROSS_GRID_DATASET_IDENTITY_MISMATCH {npz_path}")
                    dataset_identities.setdefault(dataset, identities)
                    image_ids = [str(value) for value in contract["expected_image_ids"]]
                    chunk_key = (dataset, start, limit)
                    if chunk_key in chunk_image_ids and image_ids != chunk_image_ids[chunk_key]:
                        issues.append(f"CROSS_GRID_IMAGE_ID_MISMATCH {npz_path}")
                    chunk_image_ids.setdefault(chunk_key, image_ids)
                    if contract.get("source_cache_contract") is not None:
                        issues.append(f"UNEXPECTED_SOURCE_CACHE_CONTRACT {npz_path}")

    # Per-chunk equality is not enough: two different chunks could both carry
    # the same otherwise-valid IDs.  Build one canonical ordered cohort per
    # dataset and require the formal grid to cover exactly ``total_limit``
    # globally unique examples.  All method/attack variants have already been
    # required to equal these chunk-level IDs above.
    for dataset in datasets:
        cohort: list[str] = []
        complete = True
        for start in range(0, args.total_limit, args.chunk_size):
            limit = min(args.chunk_size, args.total_limit - start)
            chunk = chunk_image_ids.get((dataset, start, limit))
            if chunk is None:
                complete = False
                issues.append(
                    "MISSING_VALID_CHUNK_FOR_DATASET_COVERAGE "
                    f"dataset={dataset} start={start} limit={limit}"
                )
                continue
            cohort.extend(chunk)
        if complete and len(cohort) != args.total_limit:
            issues.append(
                "CROSS_CHUNK_COHORT_COUNT_MISMATCH "
                f"dataset={dataset} found={len(cohort)} expected={args.total_limit}"
            )
        if complete and len(set(cohort)) != len(cohort):
            issues.append(
                "CROSS_CHUNK_DUPLICATE_IMAGE_ID "
                f"dataset={dataset} count={len(cohort)} unique={len(set(cohort))}"
            )

    print(f"checked tasks: {checked}")
    print(f"issues: {len(issues)}")
    for message in issues[:300]:
        print(message)
    if len(issues) > 300:
        print(f"... {len(issues) - 300} additional issues omitted")
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
