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

from fata.attacks.linf_image import IMAGE_SERIALIZATION_CONTRACT

from fata.utils.paths import assert_output_separate, resolve_owned_output_path
from fata.utils.run_contract import ensure_run_contract, sha256_file
from fata.runtimes.llava.attack_cache_io import (
    attack_namespace,
    baseline_attack_contract_extra,
)
from .check_mlat_full_grid import (
    DATASETS as FORMAL_DATASETS,
    METHODS as FORMAL_METHODS,
    _feature_names,
    practical_token_budget,
    validate_triplet,
)


METHOD_PATTERN = (
    r"VisionZIP|VisPruner|PruMerge|FlowCut"
)
DATASET_PATTERN = (
    r"TextVQA_Open|VQAv2_Open|ScienceQA_MC|VQAv2_MC"
)
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
        r"start(?P<start>\d+)_"
        r"limit(?P<limit>\d+)_"
        r"seed(?P<seed>-?\d+)\.npz"
    )
    match = re.fullmatch(
        pattern,
        name,
    )
    if match is None:
        raise ValueError(
            f"Cannot parse filename: {name}"
        )
    return match.groupdict()


def expand_globs(patterns: list[str]) -> list[str]:
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern))
    return sorted(set(files))


def load_all_features(
    patterns: list[str],
    *,
    expected_seed: int = 0,
    expected_cache_root: Path,
) -> dict[
    tuple[str, str, str, int],
    dict[str, Any],
]:
    files = expand_globs(patterns)
    if not files:
        raise FileNotFoundError(
            f"No files matched: {patterns}"
        )

    grouped: dict[
        tuple[str, str, str, int],
        list[dict[str, Any]],
    ] = {}
    contracts: dict[tuple[str, str, str, int], str] = {}
    run_contracts: dict[tuple[str, str, str, int], str] = {}

    for path in files:
        meta = parse_key(path)
        start = int(meta["start"])
        limit = int(meta["limit"])
        seed = int(meta["seed"])
        if start < 0 or limit <= 0 or seed != expected_seed:
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
        expected_alpha = 1.0 if meta["attack"] == "caa" else 0.5
        triplet_issues = validate_triplet(
            Path(path), method=meta["method"], dataset=meta["dataset"],
            attack=meta["attack"], token_budget=int(meta["token_budget"]),
            start=start, limit=limit, seed=seed, alpha_255=expected_alpha,
            require_valid_spans=True, expected_cache_root=expected_cache_root,
        )
        if triplet_issues:
            raise RuntimeError("invalid ML-ATD artifact:\n" + "\n".join(triplet_issues))
        sidecar = json.loads(Path(path).with_suffix(".meta.json").read_text(encoding="utf-8"))
        source = sidecar.get("source_cache_contract")
        if meta["attack"] in {"cage", "caa"}:
            if not isinstance(source, dict):
                raise ValueError(f"cached attack lacks source cache contract: {path}")
            if (
                source.get("schema_version") != 1
                or source.get("image_serialization") != IMAGE_SERIALIZATION_CONTRACT
                or source.get("definition") != f"mlatd_{meta['attack']}_generator_v1"
                or source.get("dataset") != meta["dataset"]
                or source.get("method") != "shared"
                or source.get("extra")
                != baseline_attack_contract_extra(
                    meta["attack"],
                    max_input_tokens=0,
                )
            ):
                raise ValueError(f"invalid cached attack producer contract: {path}")
            expected_namespace = attack_namespace(
                meta["attack"], seed=int(source["seed"]),
                eps_255=float(source["eps_255"]),
                alpha_255=float(source["alpha_255"]), steps=int(source["steps"]),
            )
            filename_namespace = f"{meta['attack']}{meta.get('attack_contract') or ''}"
            if (
                filename_namespace != expected_namespace
                or sidecar.get("cache_namespace") != expected_namespace
            ):
                raise ValueError(f"cached attack namespace mismatch: {path}")
            cross = {
                "image_serialization": "image_serialization",
                "dataset_mapping_sha256": "dataset_mapping_sha256",
                "dataset_images": "dataset_images",
                "llava_model": "model",
                "clip_model": "clip_model",
                "seed": "seed",
                "eps_255": "eps_255",
                "alpha_255": "alpha_255",
                "steps": "steps",
            }
            if any(sidecar.get(left) != source.get(right) for left, right in cross.items()):
                raise ValueError(f"cached feature/source contract mismatch: {path}")
        elif source is not None:
            raise ValueError(f"unexpected source cache contract: {path}")
        elif meta.get("attack_contract"):
            raise ValueError(f"unexpected attack namespace suffix: {path}")
        normalized = dict(sidecar)
        normalized.pop("expected_sample_indices", None)
        normalized.pop("expected_image_ids", None)
        fingerprint = hashlib.sha256(
            json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if key in run_contracts and run_contracts[key] != fingerprint:
            raise ValueError(f"mixed immutable run contracts across chunks for {key}")
        run_contracts[key] = fingerprint
        with np.load(
            path,
            allow_pickle=False,
        ) as arr:
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
                "sample_idx": arr[
                    "sample_idx"
                ].astype(np.int64),
                "label": arr[
                    "label"
                ].astype(np.int64),
                "image_span_valid": arr[
                    "image_span_valid"
                ].astype(np.int8),
                "image_id": arr["image_id"].astype(str),
                "feature_names": feature_names,
                "contract": sidecar,
            }
            for name in feature_names:
                item[name] = arr[
                    name
                ].astype(np.float32)
            grouped.setdefault(
                key,
                [],
            ).append(item)

    merged = {}
    for key, items in grouped.items():
        feature_sets = [set(x["feature_names"]) for x in items]
        if any(value != feature_sets[0] for value in feature_sets):
            raise RuntimeError(f"feature schema differs across chunks for {key}")
        common = sorted(feature_sets[0])

        result: dict[str, Any] = {
            "sample_idx": np.concatenate(
                [
                    x["sample_idx"]
                    for x in items
                ]
            ),
            "label": np.concatenate(
                [
                    x["label"]
                    for x in items
                ]
            ),
            "image_span_valid": np.concatenate(
                [
                    x["image_span_valid"]
                    for x in items
                ]
            ),
            "image_id": np.concatenate([x["image_id"] for x in items]),
            "feature_names": common,
        }
        for name in common:
            result[name] = np.concatenate(
                [
                    x[name]
                    for x in items
                ],
                axis=0,
            )

        order = np.argsort(
            result["sample_idx"]
        )
        for name in [
            "sample_idx",
            "label",
            "image_span_valid",
            "image_id",
            *common,
        ]:
            result[name] = result[name][order]

        unique, counts = np.unique(
            result["sample_idx"],
            return_counts=True,
        )
        if np.any(counts != 1):
            bad = unique[counts != 1][:10]
            raise RuntimeError(
                f"Duplicate sample_idx for {key}: "
                f"{bad.tolist()}"
            )
        if len(set(map(str, result["image_id"].tolist()))) != len(result["image_id"]):
            raise RuntimeError(f"Duplicate image_id for {key}")
        result["contract"] = items[0]["contract"]
        merged[key] = result

    return merged


def validate_formal_grid(data: dict[tuple[str, str, str, int], dict[str, Any]], total: int) -> None:
    formal_attacks = ("clean_clip", "random_clip", "base", "fata", "cage", "caa")
    expected = {
        (method, dataset, attack, practical_token_budget(method, dataset))
        for method in FORMAL_METHODS
        for dataset in FORMAL_DATASETS
        for attack in formal_attacks
    }
    actual = set(data)
    if actual != expected:
        raise RuntimeError(
            "formal cross-attack grid mismatch; "
            f"missing={sorted(expected - actual)[:8]} extra={sorted(actual - expected)[:8]}"
        )
    expected_indices = np.arange(total, dtype=np.int64)
    global_models = None
    dataset_identities: dict[str, tuple[Any, Any]] = {}
    dataset_ids: dict[str, list[str]] = {}
    for key in sorted(expected):
        item = data[key]
        _method, dataset, attack, _budget = key
        if not np.array_equal(item["sample_idx"], expected_indices):
            raise RuntimeError(f"{key}: expected exact sample_idx 0..{total - 1}")
        wanted_label = 0 if attack in {"clean_clip", "random_clip"} else 1
        if np.any(item["label"] != wanted_label):
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


def l2_normalize(
    x: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    x = np.asarray(
        x,
        dtype=np.float32,
    )
    return (
        x
        / (
            np.linalg.norm(
                x,
                axis=1,
                keepdims=True,
            )
            + eps
        )
    )


def fit_direction(
    x_neg: np.ndarray,
    x_pos: np.ndarray,
) -> DirectionModel:
    x_neg = l2_normalize(x_neg)
    x_pos = l2_normalize(x_pos)
    mu_neg = x_neg.mean(axis=0)
    mu_pos = x_pos.mean(axis=0)
    direction = mu_pos - mu_neg
    direction_norm = float(np.linalg.norm(direction))
    if not np.isfinite(direction_norm) or direction_norm <= 1e-8:
        raise RuntimeError("degenerate attack direction")
    direction = direction / direction_norm
    raw_neg = (
        x_neg - mu_neg
    ) @ direction
    train_neg_raw_std = float(raw_neg.std())
    if not np.isfinite(train_neg_raw_std) or train_neg_raw_std <= 1e-8:
        raise RuntimeError("degenerate negative calibration variance")
    return DirectionModel(
        mu_neg=mu_neg,
        direction=direction,
        train_neg_raw_mean=float(
            raw_neg.mean()
        ),
        train_neg_raw_std=train_neg_raw_std,
    )


def score_direction(
    x: np.ndarray,
    model: DirectionModel,
) -> np.ndarray:
    x = l2_normalize(x)
    raw = (
        x - model.mu_neg
    ) @ model.direction
    return (
        (
            raw
            - model.train_neg_raw_mean
        )
        / model.train_neg_raw_std
    ).astype(np.float64)


def standardize(
    train_neg: np.ndarray,
    *others: np.ndarray,
) -> tuple[np.ndarray, ...]:
    mean = float(
        np.mean(train_neg)
    )
    std = float(np.std(train_neg))
    if not np.isfinite(std) or std <= 1e-8:
        raise RuntimeError("degenerate negative stage calibration variance")
    return tuple(
        (x - mean) / std
        for x in (
            train_neg,
            *others,
        )
    )


def trajectory(
    score_by_layer: dict[
        str,
        np.ndarray,
    ],
    ordered_names: list[str],
) -> np.ndarray:
    matrix = np.stack(
        [
            score_by_layer[name]
            for name in ordered_names
        ],
        axis=1,
    )
    if matrix.shape[1] == 1:
        return matrix[:, 0]
    x = np.linspace(
        0.0,
        1.0,
        matrix.shape[1],
    )
    return np.trapz(
        matrix,
        x=x,
        axis=1,
    )


def layer_names(
    names: list[str],
    prefix: str,
) -> list[str]:
    def number(name: str) -> int:
        m = re.search(
            r"l(\d+)$",
            name,
        )
        return int(m.group(1)) if m else 0
    return sorted(
        [
            x
            for x in names
            if x.startswith(prefix)
        ],
        key=number,
    )


def get_indices(
    item: dict[str, Any],
    start: int,
    end: int,
) -> np.ndarray:
    idx = item["sample_idx"]
    return idx[
        (idx >= start)
        & (idx < end)
    ]


def select_by_indices(
    item: dict[str, Any],
    feature: str,
    indices: np.ndarray,
) -> np.ndarray:
    pos = {
        int(idx): i
        for i, idx in enumerate(
            item["sample_idx"]
        )
    }
    missing = [
        int(i)
        for i in indices
        if int(i) not in pos
    ]
    if missing:
        raise RuntimeError(
            f"Missing {len(missing)} indices "
            f"for feature {feature}; "
            f"first={missing[:10]}"
        )
    return np.stack(
        [
            item[feature][pos[int(i)]]
            for i in indices
        ],
        axis=0,
    )


def intersection_indices(
    items: list[dict[str, Any]],
    start: int,
    end: int,
) -> np.ndarray:
    sets = [
        set(
            map(
                int,
                get_indices(
                    item,
                    start,
                    end,
                ),
            )
        )
        for item in items
    ]
    common = set.intersection(*sets)
    return np.asarray(
        sorted(common),
        dtype=np.int64,
    )


def safe_auc(
    y: np.ndarray,
    score: np.ndarray,
) -> float:
    from sklearn.metrics import roc_auc_score

    return float(
        roc_auc_score(
            y,
            score,
        )
    )


def safe_aupr(
    y: np.ndarray,
    score: np.ndarray,
) -> float:
    from sklearn.metrics import average_precision_score

    return float(
        average_precision_score(
            y,
            score,
        )
    )


def tpr5(
    y: np.ndarray,
    score: np.ndarray,
) -> float:
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(
        y,
        score,
    )
    valid = fpr <= 0.05
    return (
        float(tpr[valid].max())
        if valid.any()
        else 0.0
    )


def metrics(
    neg: np.ndarray,
    pos: np.ndarray,
) -> dict[str, float]:
    y = np.concatenate(
        [
            np.zeros(
                len(neg),
                dtype=np.int64,
            ),
            np.ones(
                len(pos),
                dtype=np.int64,
            ),
        ]
    )
    score = np.concatenate(
        [neg, pos]
    )
    return {
        "AUROC": safe_auc(y, score),
        "AUPR": safe_aupr(y, score),
        "TPR@5FPR": tpr5(
            y,
            score,
        ),
    }


def append_row(
    rows: list[dict[str, Any]],
    base: dict[str, Any],
    detector: str,
    scope: str,
    neg: np.ndarray,
    pos: np.ndarray,
) -> None:
    row = dict(base)
    row.update(
        {
            "detector": detector,
            "scope": scope,
            "n_test_neg": len(neg),
            "n_test_pos": len(pos),
            "test_neg_score_mean": float(
                neg.mean()
            ),
            "test_pos_score_mean": float(
                pos.mean()
            ),
            **metrics(neg, pos),
        }
    )
    rows.append(row)


def main(argv: list[str] | None = None) -> int:
    import pandas as pd

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--original-glob", "--original_glob", dest="original_glob", required=True,
    )
    parser.add_argument(
        "--ablation-glob", "--ablation_glob", dest="ablation_glob", required=True,
    )
    parser.add_argument(
        "--output-dir", "--output_dir", dest="output_dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--cache-root", "--cache_root", dest="cache_root",
        type=Path,
        required=True,
        help="Exact root used to generate cached CAGE/CAA image paths.",
    )
    parser.add_argument(
        "--train_start",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--train_end",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--test_start",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--test_end",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--llm_pool",
        choices=["image", "last"],
        default="image",
    )
    parser.add_argument("--total-limit", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
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
    output_root = assert_output_separate(
        args.output_dir,
        {"attack cache root": args.cache_root},
    )

    data = load_all_features(
        [
            args.original_glob,
            args.ablation_glob,
        ],
        expected_seed=args.seed,
        expected_cache_root=args.cache_root,
    )
    validate_formal_grid(data, args.total_limit)
    if args.llm_pool == "image":
        invalid = [
            key
            for key, item in data.items()
            if np.any(item["image_span_valid"] != 1)
        ]
        if invalid:
            raise RuntimeError(
                "--llm_pool image requires a valid image-token span for every "
                f"formal sample; invalid groups={invalid[:8]}"
            )
    input_files = expand_globs([args.original_glob, args.ablation_glob])
    output_root = assert_output_separate(
        output_root,
        {f"input artifact {index}": path for index, path in enumerate(input_files)},
    )
    output_root.mkdir(parents=True, exist_ok=True)
    if not output_root.is_dir():
        raise RuntimeError(f"analysis output root is not a directory: {output_root}")

    combos = sorted(
        {
            (
                k[0],
                k[1],
                k[3],
            )
            for k in data
        }
    )

    # Main comparison: one Base-trained detector, four held-out attacks.
    main_protocols = [
        (
            "base_sanity",
            "base",
            "base",
        ),
        (
            "cage_unseen_base_dir",
            "base",
            "cage",
        ),
        (
            "caa_unseen_base_dir",
            "base",
            "caa",
        ),
        (
            "fata_unseen_base_dir",
            "base",
            "fata",
        ),
    ]
    aware_protocols = [
        (
            "cage_aware",
            "cage",
            "cage",
        ),
        (
            "caa_aware",
            "caa",
            "caa",
        ),
        (
            "fata_aware",
            "fata",
            "fata",
        ),
    ]

    rows = []

    for (
        method,
        dataset,
        token_budget,
    ) in combos:
        clean_key = (
            method,
            dataset,
            "clean_clip",
            token_budget,
        )
        random_key = (
            method,
            dataset,
            "random_clip",
            token_budget,
        )
        base_key = (
            method,
            dataset,
            "base",
            token_budget,
        )
        if (
            clean_key not in data
            or random_key not in data
            or base_key not in data
        ):
            raise RuntimeError(f"missing formal negatives/Base for {method}/{dataset}/K{token_budget}")

        clean = data[clean_key]
        random = data[random_key]
        common_features = sorted(clean["feature_names"])

        vis_names = layer_names(
            common_features,
            "vis_l",
        )
        proj_names = [
            x
            for x in [
                "proj_in",
                "proj_out",
            ]
            if x in common_features
        ]
        llm_prefix = (
            "llm_image_l"
            if args.llm_pool == "image"
            else "llm_last_l"
        )
        llm_names = layer_names(
            common_features,
            llm_prefix,
        )
        selected = (
            vis_names
            + proj_names
            + llm_names
        )
        if (
            not vis_names
            or not proj_names
            or not llm_names
        ):
            raise RuntimeError(
                f"Incomplete stages: "
                f"{method}/{dataset}/K{token_budget}"
            )

        for (
            protocol,
            train_attack,
            test_attack,
        ) in (
            main_protocols
            + aware_protocols
        ):
            train_key = (
                method,
                dataset,
                train_attack,
                token_budget,
            )
            test_key = (
                method,
                dataset,
                test_attack,
                token_budget,
            )
            if (
                train_key not in data
                or test_key not in data
            ):
                raise RuntimeError(f"missing formal protocol input for {protocol}")

            train_pos_item = data[train_key]
            test_pos_item = data[test_key]

            # Training uses matched indices across clean/random/train attack.
            train_idx = intersection_indices(
                [
                    clean,
                    random,
                    train_pos_item,
                ],
                args.train_start,
                args.train_end,
            )
            # Testing uses exactly the same sample identities for negatives
            # and the tested attack. This is especially important because
            # CAA may skip a few long ScienceQA prompts.
            test_idx = intersection_indices(
                [
                    clean,
                    random,
                    test_pos_item,
                ],
                args.test_start,
                args.test_end,
            )

            if (
                len(train_idx) < 10
                or len(test_idx) < 10
            ):
                raise RuntimeError(
                    f"Too few matched samples for "
                    f"{protocol} {method}/{dataset}: "
                    f"train={len(train_idx)}, test={len(test_idx)}"
                )

            layer_train_neg = {}
            layer_train_pos = {}
            layer_test_neg = {}
            layer_test_pos = {}

            for feature in selected:
                clean_train = select_by_indices(
                    clean,
                    feature,
                    train_idx,
                )
                random_train = select_by_indices(
                    random,
                    feature,
                    train_idx,
                )
                pos_train = select_by_indices(
                    train_pos_item,
                    feature,
                    train_idx,
                )

                clean_test = select_by_indices(
                    clean,
                    feature,
                    test_idx,
                )
                random_test = select_by_indices(
                    random,
                    feature,
                    test_idx,
                )
                pos_test = select_by_indices(
                    test_pos_item,
                    feature,
                    test_idx,
                )

                x_train_neg = np.concatenate(
                    [
                        clean_train,
                        random_train,
                    ],
                    axis=0,
                )
                x_test_neg = np.concatenate(
                    [
                        clean_test,
                        random_test,
                    ],
                    axis=0,
                )

                model = fit_direction(
                    x_train_neg,
                    pos_train,
                )
                layer_train_neg[
                    feature
                ] = score_direction(
                    x_train_neg,
                    model,
                )
                layer_train_pos[
                    feature
                ] = score_direction(
                    pos_train,
                    model,
                )
                layer_test_neg[
                    feature
                ] = score_direction(
                    x_test_neg,
                    model,
                )
                layer_test_pos[
                    feature
                ] = score_direction(
                    pos_test,
                    model,
                )

                append_row(
                    rows,
                    {
                        "method": method,
                        "dataset": dataset,
                        "token_budget": token_budget,
                        "protocol": protocol,
                        "train_attack": train_attack,
                        "test_attack": test_attack,
                        "llm_pool": args.llm_pool,
                        "n_train_attack": len(train_idx),
                        "n_test_attack": len(test_idx),
                    },
                    feature,
                    "layer",
                    layer_test_neg[
                        feature
                    ],
                    layer_test_pos[
                        feature
                    ],
                )

            stage_specs = [
                (
                    "visual_trajectory",
                    vis_names,
                ),
                (
                    "projector_trajectory",
                    proj_names,
                ),
                (
                    "llm_trajectory",
                    llm_names,
                ),
            ]

            stage_train_neg = {}
            stage_train_pos = {}
            stage_test_neg = {}
            stage_test_pos = {}

            for (
                stage_name,
                names,
            ) in stage_specs:
                train_neg = trajectory(
                    layer_train_neg,
                    names,
                )
                train_pos = trajectory(
                    layer_train_pos,
                    names,
                )
                test_neg = trajectory(
                    layer_test_neg,
                    names,
                )
                test_pos = trajectory(
                    layer_test_pos,
                    names,
                )

                (
                    train_neg_z,
                    train_pos_z,
                    test_neg_z,
                    test_pos_z,
                ) = standardize(
                    train_neg,
                    train_pos,
                    test_neg,
                    test_pos,
                )
                stage_train_neg[
                    stage_name
                ] = train_neg_z
                stage_train_pos[
                    stage_name
                ] = train_pos_z
                stage_test_neg[
                    stage_name
                ] = test_neg_z
                stage_test_pos[
                    stage_name
                ] = test_pos_z

                append_row(
                    rows,
                    {
                        "method": method,
                        "dataset": dataset,
                        "token_budget": token_budget,
                        "protocol": protocol,
                        "train_attack": train_attack,
                        "test_attack": test_attack,
                        "llm_pool": args.llm_pool,
                        "n_train_attack": len(train_idx),
                        "n_test_attack": len(test_idx),
                    },
                    stage_name,
                    "stage",
                    test_neg_z,
                    test_pos_z,
                )

            fused_neg = np.mean(
                np.stack(
                    [
                        stage_test_neg[
                            "visual_trajectory"
                        ],
                        stage_test_neg[
                            "projector_trajectory"
                        ],
                        stage_test_neg[
                            "llm_trajectory"
                        ],
                    ],
                    axis=1,
                ),
                axis=1,
            )
            fused_pos = np.mean(
                np.stack(
                    [
                        stage_test_pos[
                            "visual_trajectory"
                        ],
                        stage_test_pos[
                            "projector_trajectory"
                        ],
                        stage_test_pos[
                            "llm_trajectory"
                        ],
                    ],
                    axis=1,
                ),
                axis=1,
            )

            append_row(
                rows,
                {
                    "method": method,
                    "dataset": dataset,
                    "token_budget": token_budget,
                    "protocol": protocol,
                    "train_attack": train_attack,
                    "test_attack": test_attack,
                    "llm_pool": args.llm_pool,
                    "n_train_attack": len(train_idx),
                    "n_test_attack": len(test_idx),
                },
                "all_level_fused",
                "fused",
                fused_neg,
                fused_pos,
            )

    result = pd.DataFrame(rows)
    if result.empty:
        raise RuntimeError(
            "No analysis rows produced"
        )

    all_path = resolve_owned_output_path(
        output_root, "mlat_attack_ablation_all_results.csv"
    )

    main_detectors = [
        "visual_trajectory",
        "projector_trajectory",
        "llm_trajectory",
        "all_level_fused",
    ]
    main = result[
        result["detector"].isin(
            main_detectors
        )
    ].copy()
    main_path = resolve_owned_output_path(
        output_root, "mlat_attack_ablation_stage_results.csv"
    )

    summary = (
        main.groupby(
            [
                "protocol",
                "train_attack",
                "test_attack",
                "detector",
            ]
        )[
            [
                "AUROC",
                "AUPR",
                "TPR@5FPR",
                "n_test_attack",
            ]
        ]
        .agg(
            [
                "mean",
                "std",
                "median",
                "min",
                "max",
                "count",
            ]
        )
    )
    summary_path = resolve_owned_output_path(
        output_root, "mlat_attack_ablation_stage_summary.csv"
    )

    # Canonical table for the paper: exactly one Base-trained detector
    # compared across Base/CAGE/CAA/FATA.
    primary_protocols = [
        "base_sanity",
        "cage_unseen_base_dir",
        "caa_unseen_base_dir",
        "fata_unseen_base_dir",
    ]
    primary = main[
        main["protocol"].isin(
            primary_protocols
        )
    ].copy()
    primary_path = resolve_owned_output_path(
        output_root, "mlat_cross_attack_base_direction_results.csv"
    )

    primary_summary = (
        primary.groupby(
            [
                "test_attack",
                "detector",
            ]
        )[
            [
                "AUROC",
                "AUPR",
                "TPR@5FPR",
                "n_test_attack",
            ]
        ]
        .agg(
            [
                "mean",
                "std",
                "median",
                "min",
                "max",
                "count",
            ]
        )
    )
    primary_summary_path = resolve_owned_output_path(
        output_root, "mlat_cross_attack_base_direction_summary.csv"
    )
    contract_path = resolve_owned_output_path(output_root, "ANALYSIS_CONTRACT.json")
    input_sha256: dict[str, str] = {}
    for raw_npz_path in input_files:
        npz_path = Path(raw_npz_path)
        for path in (
            npz_path,
            npz_path.with_suffix(".csv"),
            npz_path.with_suffix(".meta.json"),
        ):
            if not path.is_file() or path.is_symlink():
                raise RuntimeError(f"analysis input is missing or a symlink: {path}")
            input_sha256[str(path.resolve())] = sha256_file(path)
    ensure_run_contract(
        contract_path,
        {
            "schema_version": 1,
            "runtime": "mlatd_attack_ablation_analysis_v1",
            "analysis_implementation_sha256": sha256_file(Path(__file__)),
            "input_sha256": input_sha256,
            "cache_root": str(args.cache_root.expanduser().resolve()),
            "total_limit": args.total_limit,
            "train_range": [args.train_start, args.train_end],
            "test_range": [args.test_start, args.test_end],
            "seed": args.seed,
            "llm_pool": args.llm_pool,
        },
        result_path=[all_path, main_path, summary_path, primary_path, primary_summary_path],
    )
    _atomic_csv(result, all_path)
    _atomic_csv(main, main_path)
    _atomic_csv(summary, summary_path, index=True)
    _atomic_csv(primary, primary_path)
    _atomic_csv(primary_summary, primary_summary_path, index=True)

    print(
        f"[Done] {all_path}"
    )
    print(
        f"[Done] {main_path}"
    )
    print(
        f"[Done] {summary_path}"
    )
    print(
        f"[Done] {primary_path}"
    )
    print(
        f"[Done] {primary_summary_path}"
    )

    print(
        "\n=== Base-trained cross-attack means ==="
    )
    print(
        primary.groupby(
            [
                "test_attack",
                "detector",
            ]
        )[
            [
                "AUROC",
                "AUPR",
                "TPR@5FPR",
                "n_test_attack",
            ]
        ]
        .mean()
        .to_string()
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
