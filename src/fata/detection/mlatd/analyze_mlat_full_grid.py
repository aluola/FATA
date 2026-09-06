from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from fata.utils.paths import assert_output_separate, resolve_owned_output_path
from fata.utils.run_contract import ensure_run_contract, sha256_file
from .check_mlat_full_grid import (
    ATTACKS as FORMAL_ATTACKS,
    DATASETS as FORMAL_DATASETS,
    METHODS as FORMAL_METHODS,
    _feature_names,
    practical_token_budget,
    validate_triplet,
)


METHOD_PATTERN = r"VisionZIP|VisPruner|PruMerge|FlowCut"
DATASET_PATTERN = r"TextVQA_Open|VQAv2_Open|ScienceQA_MC|VQAv2_MC"
ATTACK_PATTERN = (
    r"clean_clip|random_clip|base|fata|cage|caa"
)
ATTACK_CONTRACT_PATTERN = r"(?:_eps[^_]+_a[^_]+_s\d+_seed-?\d+)?"


@dataclass
class DirectionModel:
    mu_neg: np.ndarray
    direction: np.ndarray
    train_neg_raw_mean: float
    train_neg_raw_std: float


def parse_key(path: str) -> dict[str, str]:
    name = os.path.basename(path)
    pattern = (
        rf"mlat_feat_(?P<method>{METHOD_PATTERN})_"
        rf"(?P<dataset>{DATASET_PATTERN})_"
        rf"(?P<attack>{ATTACK_PATTERN})(?P<attack_contract>{ATTACK_CONTRACT_PATTERN})_"
        r"k(?P<token_budget>\d+)_"
        r"start(?P<start>\d+)_limit(?P<limit>\d+)_seed(?P<seed>-?\d+)\.npz"
    )
    match = re.fullmatch(pattern, name)
    if match is None:
        raise ValueError(f"Cannot parse filename: {name}")
    return match.groupdict()


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


def fit_direction(x_neg: np.ndarray, x_pos: np.ndarray) -> DirectionModel:
    x_neg = l2_normalize(x_neg)
    x_pos = l2_normalize(x_pos)
    mu_neg = x_neg.mean(axis=0)
    mu_pos = x_pos.mean(axis=0)
    direction = mu_pos - mu_neg
    direction_norm = float(np.linalg.norm(direction))
    if not np.isfinite(direction_norm) or direction_norm <= 1e-8:
        raise RuntimeError("degenerate attack direction")
    direction = direction / direction_norm
    train_neg_raw = (x_neg - mu_neg) @ direction
    train_neg_raw_std = float(train_neg_raw.std())
    if not np.isfinite(train_neg_raw_std) or train_neg_raw_std <= 1e-8:
        raise RuntimeError("degenerate negative calibration variance")
    return DirectionModel(
        mu_neg=mu_neg,
        direction=direction,
        train_neg_raw_mean=float(train_neg_raw.mean()),
        train_neg_raw_std=train_neg_raw_std,
    )


def score_direction(x: np.ndarray, model: DirectionModel) -> tuple[np.ndarray, np.ndarray]:
    x = l2_normalize(x)
    raw = (x - model.mu_neg) @ model.direction
    z = (raw - model.train_neg_raw_mean) / model.train_neg_raw_std
    return raw.astype(np.float64), z.astype(np.float64)


def standardize_by_train_neg(
    train_neg_score: np.ndarray,
    *other_scores: np.ndarray,
) -> tuple[np.ndarray, ...]:
    mean = float(np.mean(train_neg_score))
    std = float(np.std(train_neg_score))
    if not np.isfinite(std) or std <= 1e-8:
        raise RuntimeError("degenerate negative stage calibration variance")
    return tuple((score - mean) / std for score in (train_neg_score, *other_scores))


def trajectory_mean(score_by_layer: dict[str, np.ndarray], ordered_names: list[str]) -> np.ndarray:
    matrix = np.stack([score_by_layer[name] for name in ordered_names], axis=1)
    if matrix.shape[1] == 1:
        return matrix[:, 0]
    # Equivalent to normalized trapezoidal AUC on uniformly spaced fixed layers.
    x = np.linspace(0.0, 1.0, matrix.shape[1])
    return np.trapz(matrix, x=x, axis=1)


def safe_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y_true, score))


def safe_aupr(y_true: np.ndarray, score: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(y_true, score))


def tpr_at_fpr(y_true: np.ndarray, score: np.ndarray, target_fpr: float = 0.05) -> float:
    from sklearn.metrics import roc_curve

    fpr, tpr, _thresholds = roc_curve(y_true, score)
    valid = fpr <= target_fpr
    if not np.any(valid):
        return 0.0
    return float(np.max(tpr[valid]))


def metrics(y_true: np.ndarray, score: np.ndarray) -> dict[str, float]:
    return {
        "AUROC": safe_auc(y_true, score),
        "AUPR": safe_aupr(y_true, score),
        "TPR@5FPR": tpr_at_fpr(y_true, score, 0.05),
    }


def load_all_features(
    input_glob: str, *, expected_seed: int = 0
) -> dict[tuple[str, str, str, int], dict[str, Any]]:
    files = sorted(glob.glob(input_glob))
    if not files:
        raise FileNotFoundError(f"No files matched: {input_glob}")

    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = {}
    contracts: dict[tuple[str, str, str, int], str] = {}
    run_contracts: dict[tuple[str, str, str, int], str] = {}
    for path in files:
        meta = parse_key(path)
        start = int(meta["start"])
        limit = int(meta["limit"])
        seed = int(meta["seed"])
        if start < 0 or limit <= 0 or seed != expected_seed or meta.get("attack_contract"):
            raise ValueError(f"invalid non-negative slice/seed in {path}")
        key = (
            meta["method"],
            meta["dataset"],
            meta["attack"],
            int(meta["token_budget"]),
        )
        contract = meta.get("attack_contract") or ""
        if key in contracts and contracts[key] != contract:
            raise ValueError(
                f"mixed attack cache contracts for {key}: "
                f"{contracts[key]!r} vs {contract!r}"
            )
        contracts[key] = contract
        triplet_issues = validate_triplet(
            Path(path), method=meta["method"], dataset=meta["dataset"],
            attack=meta["attack"], token_budget=int(meta["token_budget"]),
            start=start, limit=limit, seed=seed, require_valid_spans=True,
        )
        if triplet_issues:
            raise RuntimeError("invalid ML-ATD artifact:\n" + "\n".join(triplet_issues))
        sidecar = json.loads(Path(path).with_suffix(".meta.json").read_text(encoding="utf-8"))
        if sidecar.get("source_cache_contract") is not None:
            raise ValueError(f"unexpected cached-attack contract in formal full grid: {path}")
        normalized = dict(sidecar)
        normalized.pop("expected_sample_indices", None)
        normalized.pop("expected_image_ids", None)
        fingerprint = hashlib.sha256(
            json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if key in run_contracts and run_contracts[key] != fingerprint:
            raise ValueError(f"mixed immutable run contracts across chunks for {key}")
        run_contracts[key] = fingerprint
        with np.load(path, allow_pickle=False) as arr:
            feature_names = [
                name
                for name in arr.files
                if name.startswith("vis_l")
                or name.startswith("proj_")
                or name.startswith("llm_last_l")
                or name.startswith("llm_image_l")
            ]
            if set(feature_names) != _feature_names():
                raise RuntimeError(f"unexpected feature schema for {path}")
            item: dict[str, Any] = {
                "sample_idx": arr["sample_idx"].astype(np.int64),
                "label": arr["label"].astype(np.int64),
                "image_span_valid": arr["image_span_valid"].astype(np.int8),
                "image_id": arr["image_id"].astype(str),
                "feature_names": feature_names,
                "contract": sidecar,
            }
            for name in feature_names:
                item[name] = arr[name].astype(np.float32)
            grouped.setdefault(key, []).append(item)

    merged: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    for key, items in grouped.items():
        feature_sets = [set(item["feature_names"]) for item in items]
        if any(feature_set != feature_sets[0] for feature_set in feature_sets):
            raise RuntimeError(f"feature schema differs across chunks for {key}")
        common_features = sorted(feature_sets[0])

        result: dict[str, Any] = {}
        result["sample_idx"] = np.concatenate([x["sample_idx"] for x in items])
        result["label"] = np.concatenate([x["label"] for x in items])
        result["image_span_valid"] = np.concatenate(
            [x["image_span_valid"] for x in items]
        )
        result["image_id"] = np.concatenate([x["image_id"] for x in items])
        for name in common_features:
            result[name] = np.concatenate([x[name] for x in items], axis=0)

        order = np.argsort(result["sample_idx"])
        for name in ["sample_idx", "label", "image_span_valid", "image_id", *common_features]:
            result[name] = result[name][order]
        result["feature_names"] = common_features

        unique_idx, counts = np.unique(result["sample_idx"], return_counts=True)
        if np.any(counts != 1):
            duplicated = unique_idx[counts != 1][:10]
            raise RuntimeError(f"Duplicate sample_idx for {key}: {duplicated.tolist()}")
        if len(set(map(str, result["image_id"].tolist()))) != len(result["image_id"]):
            raise RuntimeError(f"Duplicate image_id for {key}")
        result["contract"] = items[0]["contract"]
        merged[key] = result
    return merged


def validate_formal_grid(data: dict[tuple[str, str, str, int], dict[str, Any]], total: int) -> None:
    expected = {
        (method, dataset, attack, practical_token_budget(method, dataset))
        for method in FORMAL_METHODS
        for dataset in FORMAL_DATASETS
        for attack in FORMAL_ATTACKS
    }
    actual = set(data)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise RuntimeError(f"formal ML-ATD grid mismatch; missing={missing[:8]} extra={extra[:8]}")
    indices = np.arange(total, dtype=np.int64)
    global_models = None
    dataset_identities: dict[str, tuple[Any, Any]] = {}
    dataset_ids: dict[str, list[str]] = {}
    for key in sorted(expected):
        item = data[key]
        method, dataset, attack, _budget = key
        if not np.array_equal(item["sample_idx"], indices):
            raise RuntimeError(f"{key}: expected exact sample_idx 0..{total - 1}")
        label = 0 if attack in {"clean_clip", "random_clip"} else 1
        if np.any(item["label"] != label):
            raise RuntimeError(f"{key}: incorrect labels")
        ids = [str(value) for value in item["image_id"].tolist()]
        if dataset in dataset_ids and ids != dataset_ids[dataset]:
            raise RuntimeError(f"{key}: image identities differ across attacks/methods")
        dataset_ids.setdefault(dataset, ids)
        contract = item["contract"]
        models = (contract["llava_model"], contract["clip_model"])
        if global_models is not None and models != global_models:
            raise RuntimeError(f"{key}: model identities differ across grid")
        global_models = models if global_models is None else global_models
        identities = (contract["dataset_mapping_sha256"], contract["dataset_images"])
        if dataset in dataset_identities and identities != dataset_identities[dataset]:
            raise RuntimeError(f"{key}: dataset identities differ across grid")
        dataset_identities.setdefault(dataset, identities)


def _atomic_csv(frame, destination: Path, *, index: bool = False) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        raise RuntimeError(f"analysis output must not be a symbolic link: {destination}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.tmp.", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_csv(temporary, index=index)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def select_range(item: dict[str, Any], name: str, start: int, end: int) -> np.ndarray:
    mask = (item["sample_idx"] >= start) & (item["sample_idx"] < end)
    return item[name][mask]


def select_valid_range(item: dict[str, Any], start: int, end: int) -> np.ndarray:
    mask = (item["sample_idx"] >= start) & (item["sample_idx"] < end)
    return item["image_span_valid"][mask]


def sorted_layer_names(feature_names: list[str], prefix: str) -> list[str]:
    def layer_number(name: str) -> int:
        match = re.search(r"l(\d+)$", name)
        return int(match.group(1)) if match else 0
    return sorted([x for x in feature_names if x.startswith(prefix)], key=layer_number)


def append_metric_row(
    rows: list[dict[str, Any]],
    base: dict[str, Any],
    detector: str,
    scope: str,
    y_true: np.ndarray,
    neg_score: np.ndarray,
    pos_score: np.ndarray,
) -> None:
    score = np.concatenate([neg_score, pos_score])
    row = dict(base)
    row.update(
        {
            "detector": detector,
            "scope": scope,
            "n_test_neg": int(len(neg_score)),
            "n_test_pos": int(len(pos_score)),
            "test_neg_score_mean": float(np.mean(neg_score)),
            "test_pos_score_mean": float(np.mean(pos_score)),
            "test_neg_score_median": float(np.median(neg_score)),
            "test_pos_score_median": float(np.median(pos_score)),
            **metrics(y_true, score),
        }
    )
    rows.append(row)


def main(argv: list[str] | None = None) -> int:
    import pandas as pd

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-glob", "--input_glob", dest="input_glob", required=True,
    )
    parser.add_argument(
        "--output-dir", "--output_dir", dest="output_dir",
        type=Path,
        required=True,
    )
    parser.add_argument("--train_start", type=int, default=0)
    parser.add_argument("--train_end", type=int, default=500)
    parser.add_argument("--test_start", type=int, default=500)
    parser.add_argument("--test_end", type=int, default=1000)
    parser.add_argument("--total-limit", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--llm_pool",
        choices=["image", "last"],
        default="image",
        help="image = mean image-token span; last = final prompt token as in original HiddenDetect.",
    )
    args = parser.parse_args(argv)
    if not (
        args.total_limit > 0
        and args.total_limit % 2 == 0
        and args.seed >= 0
        and args.train_start == 0
        and args.train_end == args.total_limit // 2
        and args.test_start == args.total_limit // 2
        and args.test_end == args.total_limit
    ):
        parser.error(
            "formal analysis requires an even total-limit and exact 50/50 "
            "splits [0, total/2) and [total/2, total)"
        )

    data = load_all_features(args.input_glob, expected_seed=args.seed)
    validate_formal_grid(data, args.total_limit)
    input_files = [Path(path).resolve() for path in glob.glob(args.input_glob)]
    output_root = assert_output_separate(
        args.output_dir,
        {f"input artifact {index}": path for index, path in enumerate(input_files)},
    )
    output_root.mkdir(parents=True, exist_ok=True)
    if not output_root.is_dir():
        raise RuntimeError(f"analysis output root is not a directory: {output_root}")
    combinations = sorted({(k[0], k[1], k[3]) for k in data})
    protocols = [
        # detector sanity
        (
            "base_sanity",
            "base",
            "base",
        ),

        # PRIMARY cross-attack comparison
        (
            "fata_unseen_base_dir",
            "base",
            "fata",
        ),
        # attack-aware upper bound
        (
            "fata_aware",
            "fata",
            "fata",
        ),
    ]

    rows: list[dict[str, Any]] = []
    score_dump: list[pd.DataFrame] = []

    for method, dataset, token_budget in combinations:
        clean_key = (method, dataset, "clean_clip", token_budget)
        random_key = (method, dataset, "random_clip", token_budget)
        if clean_key not in data or random_key not in data:
            raise RuntimeError(f"missing formal negatives for {(method, dataset, token_budget)}")
        common_features = sorted(data[clean_key]["feature_names"])
        visual_names = sorted_layer_names(common_features, "vis_l")
        projector_names = [x for x in ["proj_in", "proj_out"] if x in common_features]
        llm_prefix = "llm_image_l" if args.llm_pool == "image" else "llm_last_l"
        llm_names = sorted_layer_names(common_features, llm_prefix)
        selected_features = visual_names + projector_names + llm_names
        if not visual_names or not projector_names or not llm_names:
            raise RuntimeError(
                f"Incomplete layer groups for {(method, dataset, token_budget)}: "
                f"visual={visual_names}, projector={projector_names}, llm={llm_names}"
            )

        for protocol, train_pos_name, test_pos_name in protocols:
            train_pos_key = (method, dataset, train_pos_name, token_budget)
            test_pos_key = (method, dataset, test_pos_name, token_budget)
            if train_pos_key not in data or test_pos_key not in data:
                raise RuntimeError(f"missing formal protocol input for {protocol}")

            if args.llm_pool == "image":
                validity_parts = [
                    select_valid_range(data[key], args.train_start, args.train_end)
                    for key in [clean_key, random_key, train_pos_key]
                ] + [
                    select_valid_range(data[key], args.test_start, args.test_end)
                    for key in [clean_key, random_key, test_pos_key]
                ]
                invalid = int(sum(np.sum(v == 0) for v in validity_parts))
                if invalid:
                    raise RuntimeError(
                        f"Found {invalid} invalid image spans for "
                        f"{method}/{dataset}/{protocol}. Re-run analysis with --llm_pool last "
                        "or inspect processor/image-token expansion."
                    )

            matrices: dict[str, dict[str, np.ndarray]] = {}
            for feature in selected_features:
                clean_train = select_range(
                    data[clean_key], feature, args.train_start, args.train_end
                )
                random_train = select_range(
                    data[random_key], feature, args.train_start, args.train_end
                )
                pos_train = select_range(
                    data[train_pos_key], feature, args.train_start, args.train_end
                )
                clean_test = select_range(
                    data[clean_key], feature, args.test_start, args.test_end
                )
                random_test = select_range(
                    data[random_key], feature, args.test_start, args.test_end
                )
                pos_test = select_range(
                    data[test_pos_key], feature, args.test_start, args.test_end
                )

                train_neg = np.concatenate([clean_train, random_train], axis=0)
                test_neg = np.concatenate([clean_test, random_test], axis=0)
                model = fit_direction(train_neg, pos_train)
                _train_raw, train_neg_z = score_direction(train_neg, model)
                _pos_train_raw, pos_train_z = score_direction(pos_train, model)
                _test_neg_raw, test_neg_z = score_direction(test_neg, model)
                _test_pos_raw, test_pos_z = score_direction(pos_test, model)
                matrices[feature] = {
                    "train_neg": train_neg_z,
                    "train_pos": pos_train_z,
                    "test_neg": test_neg_z,
                    "test_pos": test_pos_z,
                }

            base_info = {
                "method": method,
                "dataset": dataset,
                "token_budget": token_budget,
                "protocol": protocol,
                "train_pos": train_pos_name,
                "test_pos": test_pos_name,
                "negative_set": "clean_clip_random_clip",
                "llm_pool": args.llm_pool,
                "n_train_neg": int(
                    len(matrices[selected_features[0]]["train_neg"])
                ),
                "n_train_pos": int(
                    len(matrices[selected_features[0]]["train_pos"])
                ),
            }
            y_true = np.concatenate(
                [
                    np.zeros_like(matrices[selected_features[0]]["test_neg"], dtype=np.int64),
                    np.ones_like(matrices[selected_features[0]]["test_pos"], dtype=np.int64),
                ]
            )

            for feature in selected_features:
                if feature.startswith("vis_l"):
                    scope = "visual_layer"
                elif feature.startswith("proj_"):
                    scope = "projector_layer"
                else:
                    scope = "llm_layer"
                append_metric_row(
                    rows,
                    base_info,
                    feature,
                    scope,
                    y_true,
                    matrices[feature]["test_neg"],
                    matrices[feature]["test_pos"],
                )

            stage_groups = {
                "visual_trajectory": visual_names,
                "projector_trajectory": projector_names,
                "llm_trajectory": llm_names,
            }
            stage_scores: dict[str, dict[str, np.ndarray]] = {}
            for stage_name, feature_names in stage_groups.items():
                raw_stage: dict[str, np.ndarray] = {}
                for split in ["train_neg", "train_pos", "test_neg", "test_pos"]:
                    raw_stage[split] = trajectory_mean(
                        {name: matrices[name][split] for name in feature_names},
                        feature_names,
                    )
                (
                    stage_train_neg,
                    stage_train_pos,
                    stage_test_neg,
                    stage_test_pos,
                ) = standardize_by_train_neg(
                    raw_stage["train_neg"],
                    raw_stage["train_pos"],
                    raw_stage["test_neg"],
                    raw_stage["test_pos"],
                )
                stage_scores[stage_name] = {
                    "train_neg": stage_train_neg,
                    "train_pos": stage_train_pos,
                    "test_neg": stage_test_neg,
                    "test_pos": stage_test_pos,
                }
                append_metric_row(
                    rows,
                    base_info,
                    stage_name,
                    "stage_trajectory",
                    y_true,
                    stage_test_neg,
                    stage_test_pos,
                )

            fused_raw: dict[str, np.ndarray] = {}
            for split in ["train_neg", "train_pos", "test_neg", "test_pos"]:
                fused_raw[split] = np.mean(
                    np.stack(
                        [stage_scores[name][split] for name in stage_groups], axis=1
                    ),
                    axis=1,
                )
            fused_train_neg, fused_train_pos, fused_test_neg, fused_test_pos = (
                standardize_by_train_neg(
                    fused_raw["train_neg"],
                    fused_raw["train_pos"],
                    fused_raw["test_neg"],
                    fused_raw["test_pos"],
                )
            )
            append_metric_row(
                rows,
                base_info,
                "all_level_fused",
                "fused",
                y_true,
                fused_test_neg,
                fused_test_pos,
            )

            dump = pd.DataFrame(
                {
                    "method": method,
                    "dataset": dataset,
                    "token_budget": token_budget,
                    "protocol": protocol,
                    "split": np.concatenate(
                        [
                            np.repeat("test_neg", len(fused_test_neg)),
                            np.repeat("test_pos", len(fused_test_pos)),
                        ]
                    ),
                    "label": y_true,
                    "all_level_fused": np.concatenate(
                        [fused_test_neg, fused_test_pos]
                    ),
                }
            )
            score_dump.append(dump)

    result = pd.DataFrame(rows)
    if result.empty:
        raise RuntimeError("No analysis rows were produced")
    result_path = resolve_owned_output_path(output_root, "mlat_detection_auc.csv")

    global_summary = result.groupby(
        ["protocol", "scope", "detector", "llm_pool"]
    )[["AUROC", "AUPR", "TPR@5FPR"]].agg(
        ["mean", "std", "median", "min", "max", "count"]
    )
    global_path = resolve_owned_output_path(
        output_root, "mlat_detection_global_summary.csv"
    )

    main = result[result["detector"].isin(
        ["visual_trajectory", "projector_trajectory", "llm_trajectory", "all_level_fused"]
    )].copy()
    main_path = resolve_owned_output_path(output_root, "mlat_main_stage_results.csv")

    strict_summary = main.groupby(["protocol", "detector"])[
        ["AUROC", "AUPR", "TPR@5FPR"]
    ].agg(["mean", "std", "median", "min", "max", "count"])
    strict_path = resolve_owned_output_path(output_root, "mlat_main_stage_summary.csv")

    score_path = resolve_owned_output_path(output_root, "mlat_fused_test_scores.csv")
    contract_path = resolve_owned_output_path(output_root, "ANALYSIS_CONTRACT.json")
    input_sha256: dict[str, str] = {}
    for npz_path in input_files:
        for path in (
            npz_path,
            npz_path.with_suffix(".csv"),
            npz_path.with_suffix(".meta.json"),
        ):
            if not path.is_file() or path.is_symlink():
                raise RuntimeError(f"analysis input is missing or a symlink: {path}")
            input_sha256[str(path)] = sha256_file(path)
    ensure_run_contract(
        contract_path,
        {
            "schema_version": 1,
            "runtime": "mlatd_full_grid_analysis_v1",
            "analysis_implementation_sha256": sha256_file(Path(__file__)),
            "input_sha256": input_sha256,
            "total_limit": args.total_limit,
            "train_range": [args.train_start, args.train_end],
            "test_range": [args.test_start, args.test_end],
            "seed": args.seed,
            "llm_pool": args.llm_pool,
        },
        result_path=[result_path, global_path, main_path, strict_path, score_path],
    )
    _atomic_csv(result, result_path)
    _atomic_csv(global_summary, global_path, index=True)
    _atomic_csv(main, main_path)
    _atomic_csv(strict_summary, strict_path, index=True)
    _atomic_csv(pd.concat(score_dump, ignore_index=True), score_path)

    print(f"[Done] {result_path}")
    print(f"[Done] {global_path}")
    print(f"[Done] {main_path}")
    print(f"[Done] {strict_path}")
    print(f"[Done] {score_path}")
    print("\n=== Main stage means across 16 method-dataset combinations ===")
    print(
        main.groupby(["protocol", "detector"])[
            ["AUROC", "AUPR", "TPR@5FPR"]
        ].mean().to_string()
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
