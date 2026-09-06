"""Fail-closed Feature Squeezing metric aggregation."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

from fata.attacks.linf_image import IMAGE_SERIALIZATION_CONTRACT
from fata.runtimes.llava.attack_cache_io import (
    validate_embedded_baseline_cache_contract,
)
from fata.utils.paths import assert_output_separate, resolve_owned_output_path
from fata.utils.run_contract import (
    artifact_identity_valid,
    dataset_identity_valid,
    is_sha256_digest,
)


NEGATIVE_ATTACKS = {"clean", "random"}
POSITIVE_ATTACKS = {"base", "fata", "cage", "caa"}
SQUEEZERS = ("bit5", "bit4", "median3", "jpeg75", "jpeg50")
BASE_COLUMNS = (
    "Image_ID", "Question", "dataset", "method", "detect_k",
    "attack_for_detection", "label", "seed", "sample_seed", "limit",
    "eps_255", "alpha_255", "steps", "lam", "orig_answer",
    "orig_score", "max_fs_score_diff", "max_answer_changed",
)
EXPECTED_HEADER = [*BASE_COLUMNS]
for _squeezer in SQUEEZERS:
    EXPECTED_HEADER.extend(
        (
            f"{_squeezer}_answer",
            f"{_squeezer}_score",
            f"{_squeezer}_score_diff",
            f"{_squeezer}_answer_changed",
        )
    )


def _dependencies():
    """Delay numerical imports so importing this CLI has no global side effects."""

    import numpy as np
    import pandas as pd
    from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

    return np, pd, average_precision_score, roc_auc_score, roc_curve


def _fingerprint(contract: dict) -> str:
    shared_keys = (
        "image_serialization", "dataset", "dataset_mapping_sha256", "dataset_images",
        "expected_image_ids", "method", "detect_k", "model", "clip_model",
        "seed", "eps_255", "squeezers",
    )
    payload = {key: contract.get(key) for key in shared_keys}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _load_one(path: Path):
    np, pd, *_ = _dependencies()
    meta_path = path.with_suffix(".meta.json")
    if path.is_symlink() or meta_path.is_symlink():
        raise ValueError(f"symbolic-link analysis input is not allowed: {path}")
    if not meta_path.is_file():
        raise FileNotFoundError(f"missing immutable run contract: {meta_path}")
    contract = json.loads(meta_path.read_text(encoding="utf-8"))
    if contract.get("schema_version") != 1 or contract.get("runtime") != "feature_squeezing_detection_v1":
        raise ValueError(f"unexpected Feature Squeezing contract: {meta_path}")
    if contract.get("image_serialization") != IMAGE_SERIALIZATION_CONTRACT:
        raise ValueError(f"unsupported attack-image serialization: {meta_path}")
    if not is_sha256_digest(contract.get("dataset_mapping_sha256")):
        raise ValueError(f"invalid dataset mapping identity: {meta_path}")
    if not dataset_identity_valid(contract.get("dataset_images")):
        raise ValueError(f"invalid dataset image identity: {meta_path}")
    if not artifact_identity_valid(contract.get("model")) or not artifact_identity_valid(
        contract.get("clip_model")
    ):
        raise ValueError(f"invalid model identity: {meta_path}")
    if "source_cache_contract" not in contract or "cache_namespace" not in contract:
        raise ValueError(f"missing attack-source lineage: {meta_path}")
    table = pd.read_csv(path)
    if list(table.columns) != contract.get("header"):
        raise ValueError(f"{path}: CSV header disagrees with immutable contract")
    if list(table.columns) != EXPECTED_HEADER:
        raise ValueError(f"{path}: incomplete or unexpected Feature Squeezing schema")
    required = {
        "Image_ID", "dataset", "method", "detect_k", "attack_for_detection",
        "label", "seed", "sample_seed", "limit", "eps_255", "alpha_255",
        "steps", "lam", "max_fs_score_diff", "max_answer_changed",
    }
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if table.empty or table["Image_ID"].isna().any() or table["Image_ID"].duplicated().any():
        raise ValueError(f"{path}: empty, missing, or duplicate Image_ID")
    expected_ids = [str(value) for value in contract.get("expected_image_ids", [])]
    selected_count = int(contract.get("selected_count", -1))
    requested_limit = int(contract.get("limit", -1))
    if (
        selected_count != len(expected_ids)
        or selected_count <= 0
        or requested_limit < 0
        or (requested_limit > 0 and selected_count != requested_limit)
    ):
        raise ValueError(f"{meta_path}: invalid requested/effective cohort count")
    if table["Image_ID"].astype(str).tolist() != expected_ids:
        raise ValueError(f"{path}: row identities disagree with its run contract")
    attacks = set(table["attack_for_detection"].astype(str))
    if len(attacks) != 1:
        raise ValueError(f"{path}: mixed attacks")
    attack = next(iter(attacks))
    expected_label = 0 if attack in NEGATIVE_ATTACKS else 1 if attack in POSITIVE_ATTACKS else None
    labels = pd.to_numeric(table["label"], errors="raise")
    if (
        expected_label is None
        or not np.isfinite(labels).all()
        or not np.equal(labels, expected_label).all()
    ):
        raise ValueError(f"{path}: attack/label contract mismatch")
    for column, expected in {
        "dataset": contract["dataset"], "method": contract["method"],
        "detect_k": contract["detect_k"], "seed": contract["seed"],
        "attack_for_detection": contract["attack_for_detection"],
        "eps_255": contract["eps_255"], "alpha_255": contract["alpha_255"],
        "steps": contract["steps"], "lam": contract["lambda"],
        "limit": contract["limit"],
    }.items():
        if set(table[column].astype(str)) != {str(expected)}:
            raise ValueError(f"{path}: {column} disagrees with its run contract")
    if int(contract["seed"]) < 0 or int(contract["detect_k"]) <= 0:
        raise ValueError(f"{meta_path}: invalid seed/detect_k")
    attack_values = (
        float(contract["eps_255"]),
        float(contract["alpha_255"]),
        float(contract["lambda"]),
    )
    if (
        not all(math.isfinite(value) for value in attack_values)
        or attack_values[0] <= 0
        or attack_values[1] <= 0
        or int(contract["steps"]) <= 0
        or attack_values[2] < 0
    ):
        raise ValueError(f"{meta_path}: invalid attack parameters")
    if attack in {"caa", "cage"}:
        try:
            validate_embedded_baseline_cache_contract(
                contract["source_cache_contract"],
                cache_namespace=contract["cache_namespace"],
                attack=attack,
                dataset=contract["dataset"],
                method=contract["method"],
                seed=contract["seed"],
                eps_255=contract["eps_255"],
                alpha_255=contract["alpha_255"],
                steps=contract["steps"],
                result_dataset_mapping_sha256=contract["dataset_mapping_sha256"],
                result_dataset_images=contract["dataset_images"],
                result_model=contract["model"],
                result_clip_model=contract["clip_model"],
            )
        except ValueError as error:
            raise ValueError(f"invalid attack-source lineage: {meta_path}: {error}") from error
    elif contract["source_cache_contract"] is not None or contract["cache_namespace"] is not None:
        raise ValueError(f"unexpected attack-source lineage: {meta_path}")
    for column in ("eps_255", "alpha_255", "lam"):
        if not np.isfinite(pd.to_numeric(table[column], errors="raise")).all():
            raise ValueError(f"{path}: non-finite {column}")
    squeezers = contract.get("squeezers")
    if squeezers != list(SQUEEZERS):
        raise ValueError(f"{meta_path}: invalid squeezer contract")
    expected_sample_seeds = [int(contract["seed"]) * 1_000_003 + i for i in range(len(table))]
    if pd.to_numeric(table["sample_seed"], errors="raise").tolist() != expected_sample_seeds:
        raise ValueError(f"{path}: sample_seed does not follow the deterministic cohort")
    table["contract_group"] = _fingerprint(contract)
    table["source_file"] = str(path)
    return table


def _atomic_csv(frame, destination: Path, *, index: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination = resolve_owned_output_path(destination.parent, destination.name)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            frame.to_csv(handle, index=index)
            handle.flush()
            os.fsync(handle.fileno())
        fd = -1
        resolve_owned_output_path(destination.parent, destination.name)
        os.replace(temporary, destination)
    finally:
        if fd >= 0:
            os.close(fd)
        if temporary.exists():
            temporary.unlink()


def main(argv: list[str] | None = None) -> int:
    np, pd, average_precision_score, roc_auc_score, roc_curve = _dependencies()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-glob", "--input_glob", dest="input_glob", required=True)
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", type=Path, required=True)
    args = parser.parse_args(argv)
    files = [Path(path) for path in sorted(glob.glob(args.input_glob))]
    if not files:
        raise FileNotFoundError(f"No files matched: {args.input_glob}")
    args.output_dir = args.output_dir.expanduser().resolve()
    assert_output_separate(
        args.output_dir,
        {f"input CSV {index}": path for index, path in enumerate(files)},
    )
    table = pd.concat([_load_one(path) for path in files], ignore_index=True)
    score_cols = ["max_fs_score_diff", "max_answer_changed"] + sorted(
        column for column in table.columns
        if (column.endswith("_score_diff") or column.endswith("_answer_changed"))
        and column not in {"max_fs_score_diff", "max_answer_changed"}
    )
    probability_cols = sorted(
        column for column in table.columns
        if column == "orig_score" or column.endswith("_score")
    )
    for column in probability_cols:
        numeric = pd.to_numeric(table[column], errors="raise")
        if not np.isfinite(numeric).all() or (numeric < 0).any() or (numeric > 1).any():
            raise ValueError(f"invalid model score column: {column}")
    for column in score_cols:
        numeric = pd.to_numeric(table[column], errors="raise")
        if (
            not np.isfinite(numeric).all()
            or (numeric < 0).any()
            or (numeric > 1).any()
            or (column.endswith("_answer_changed") and not numeric.isin([0, 1]).all())
        ):
            raise ValueError(f"invalid detector score column: {column}")
    orig_scores = pd.to_numeric(table["orig_score"], errors="raise").to_numpy()
    per_squeezer_diffs = []
    per_squeezer_changes = []
    for squeezer in SQUEEZERS:
        squeezed_scores = pd.to_numeric(
            table[f"{squeezer}_score"], errors="raise"
        ).to_numpy()
        recorded_diffs = pd.to_numeric(
            table[f"{squeezer}_score_diff"], errors="raise"
        ).to_numpy()
        if not np.allclose(
            recorded_diffs,
            np.abs(orig_scores - squeezed_scores),
            rtol=0.0,
            atol=1.1e-4,
        ):
            raise ValueError(f"inconsistent derived score diff for {squeezer}")
        expected_changes = np.asarray(
            [
                int(
                    ("" if pd.isna(original) else str(original).strip().lower())
                    != ("" if pd.isna(squeezed) else str(squeezed).strip().lower())
                )
                for original, squeezed in zip(
                    table["orig_answer"], table[f"{squeezer}_answer"]
                )
            ],
            dtype=int,
        )
        recorded_changes = pd.to_numeric(
            table[f"{squeezer}_answer_changed"], errors="raise"
        ).to_numpy()
        if not np.array_equal(recorded_changes, expected_changes):
            raise ValueError(f"inconsistent derived answer change for {squeezer}")
        per_squeezer_diffs.append(recorded_diffs)
        per_squeezer_changes.append(recorded_changes)
    if not np.allclose(
        pd.to_numeric(table["max_fs_score_diff"], errors="raise").to_numpy(),
        np.max(np.stack(per_squeezer_diffs, axis=1), axis=1),
        rtol=0.0,
        atol=5e-5,
    ):
        raise ValueError("inconsistent derived max_fs_score_diff")
    if not np.array_equal(
        pd.to_numeric(table["max_answer_changed"], errors="raise").to_numpy(),
        np.max(np.stack(per_squeezer_changes, axis=1), axis=1),
    ):
        raise ValueError("inconsistent derived max_answer_changed")

    metric_rows = []
    for contract_group, group in table.groupby("contract_group", sort=True):
        by_attack = {str(name): part.copy() for name, part in group.groupby("attack_for_detection")}
        for name, frame in by_attack.items():
            if frame["source_file"].nunique() != 1:
                raise RuntimeError(f"multiple {name} runs share one analysis contract group")
        negative_names = sorted(set(by_attack).intersection(NEGATIVE_ATTACKS))
        positive_names = sorted(set(by_attack).intersection(POSITIVE_ATTACKS))
        if not negative_names or not positive_names:
            raise RuntimeError(f"contract group {contract_group} lacks both labels")
        reference_ids = by_attack[negative_names[0]]["Image_ID"].astype(str).tolist()
        for name, frame in by_attack.items():
            if frame["Image_ID"].astype(str).tolist() != reference_ids:
                raise RuntimeError(f"unaligned Image_ID rows for attack {name}")
        negatives = pd.concat([by_attack[name] for name in negative_names], ignore_index=True)
        for positive_name in positive_names:
            positive = by_attack[positive_name]
            combined = pd.concat([negatives, positive], ignore_index=True)
            y = combined["label"].astype(int).to_numpy()
            base = positive.iloc[0]
            for score_col in score_cols:
                scores = combined[score_col].astype(float).to_numpy()
                fpr, tpr, _ = roc_curve(y, scores)
                metric_rows.append({
                    "dataset": base["dataset"], "method": base["method"],
                    "detect_k": int(base["detect_k"]), "seed": int(base["seed"]),
                    "negative_attacks": "+".join(negative_names),
                    "positive_attack": positive_name,
                    "eps_255": float(base["eps_255"]),
                    "alpha_255": float(base["alpha_255"]),
                    "steps": int(base["steps"]), "lambda": float(base["lam"]),
                    "score": score_col, "n_clean_label": int((y == 0).sum()),
                    "n_attack_label": int((y == 1).sum()),
                    "AUROC": float(roc_auc_score(y, scores)),
                    "AUPR": float(average_precision_score(y, scores)),
                    "TPR@5FPR": float(tpr[fpr <= 0.05].max()) if (fpr <= 0.05).any() else 0.0,
                    "contract_group": contract_group,
                })
    metrics = pd.DataFrame(metric_rows)
    if metrics.empty:
        raise RuntimeError("No complete Feature Squeezing protocol was produced")
    summary = table.groupby(
        ["contract_group", "dataset", "method", "detect_k", "attack_for_detection"]
    )[score_cols].agg(["mean", "std", "median", "min", "max"])
    _atomic_csv(
        metrics,
        resolve_owned_output_path(args.output_dir, "fs_detection_auc.csv"),
        index=False,
    )
    _atomic_csv(
        summary,
        resolve_owned_output_path(args.output_dir, "fs_detection_group_summary.csv"),
        index=True,
    )
    print(metrics.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
