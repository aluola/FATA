from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from fata.runtimes.llava.attack_cache_io import (
    attack_cache_contract_path,
    attack_image_exists,
    attack_image_path,
    attack_namespace,
    baseline_attack_contract_extra,
    build_attack_cache_contract,
    load_attack_image,
)
from fata.utils.run_contract import ensure_run_contract, require_run_contract
from fata.utils.run_contract import artifact_identity
from fata.utils.paths import (
    assert_output_separate,
    resolve_owned_output_path,
)


from .mlat_core import (
    ALL_DATASETS,
    ALL_METHODS,
    MLATFeatureEngine,
    RunConfig,
    build_prompt,
    expand2square,
    feature_artifacts_complete,
    feature_chunk_complete_without_model,
    load_dataset_without_model,
    set_sample_seed,
)


ATTACKS = ["cage", "caa"]


def cache_path(
    cache_root: Path,
    attack: str,
    dataset: str,
    filename: str,
) -> Path:
    """Resolve the canonical method-independent ML-ATD attack cache."""

    return attack_image_path(
        cache_root=cache_root,
        method="shared",
        dataset=dataset,
        attack=attack,
        image_filename=filename,
    )


def csv_logical_rows(path: Path) -> int:
    if not path.exists():
        return -1
    try:
        with path.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as f:
            return sum(1 for _ in csv.DictReader(f))
    except Exception:
        return -1


def npz_rows(path: Path) -> int:
    if not path.exists():
        return -1
    try:
        with np.load(path, allow_pickle=False) as arr:
            return len(arr["sample_idx"])
    except Exception:
        return -1


def outputs_complete(
    npz_path: Path,
    csv_path: Path,
    expected_rows: int,
    expected_indices: list[int] | None = None,
    expected_image_ids: list[str] | None = None,
) -> bool:
    if expected_indices is not None and expected_image_ids is not None:
        return feature_artifacts_complete(
            npz_path, csv_path, expected_indices, expected_image_ids
        )
    a = npz_rows(npz_path)
    b = csv_logical_rows(csv_path)
    return expected_rows > 0 and a == b == expected_rows


def write_atomic(
    npz_path: Path,
    csv_path: Path,
    feature_lists: dict[str, list[np.ndarray]],
    rows: list[dict[str, Any]],
    vision_layers: list[int],
    llm_layers: list[int],
) -> None:
    if not rows:
        raise RuntimeError(
            f"No cached attack images available for {npz_path.stem}"
        )

    npz_path.parent.mkdir(parents=True, exist_ok=True)
    npz_path = resolve_owned_output_path(npz_path.parent, npz_path.name)
    csv_path = resolve_owned_output_path(csv_path.parent, csv_path.name)
    fd_npz, tmp_npz_name = tempfile.mkstemp(
        prefix=f".{npz_path.name}.", suffix=".tmp.npz", dir=npz_path.parent
    )
    os.close(fd_npz)
    tmp_npz = Path(tmp_npz_name)
    fd_csv, tmp_csv_name = tempfile.mkstemp(
        prefix=f".{csv_path.name}.", suffix=".tmp", dir=csv_path.parent
    )
    tmp_csv = Path(tmp_csv_name)

    payload: dict[str, Any] = {
        name: np.stack(
            values,
            axis=0,
        ).astype(np.float16)
        for name, values in feature_lists.items()
    }
    payload.update(
        {
            "sample_idx": np.asarray(
                [r["sample_idx"] for r in rows],
                dtype=np.int64,
            ),
            "label": np.ones(
                len(rows),
                dtype=np.int8,
            ),
            "image_span_valid": np.asarray(
                [
                    r["image_span_valid"]
                    for r in rows
                ],
                dtype=np.int8,
            ),
            "image_token_count": np.asarray(
                [r["image_token_count"] for r in rows], dtype=np.int32
            ),
            "llm_sequence_length": np.asarray(
                [r["llm_sequence_length"] for r in rows], dtype=np.int32
            ),
            "image_id": np.asarray([str(r["Image_ID"]) for r in rows]),
            "vision_layers": np.asarray(vision_layers, dtype=np.int16),
            "llm_layers": np.asarray(llm_layers, dtype=np.int16),
        }
    )
    try:
        np.savez_compressed(tmp_npz, **payload)
        with tmp_npz.open("rb") as handle:
            os.fsync(handle.fileno())

        fieldnames = list(rows[0].keys())
        with os.fdopen(fd_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            f.flush()
            os.fsync(f.fileno())
        fd_csv = -1

        resolve_owned_output_path(npz_path.parent, npz_path.name)
        resolve_owned_output_path(csv_path.parent, csv_path.name)
        os.replace(tmp_npz, npz_path)
        os.replace(tmp_csv, csv_path)
    finally:
        if fd_csv >= 0:
            os.close(fd_csv)
        for temporary in (tmp_npz, tmp_csv):
            if temporary.exists():
                temporary.unlink()


def run_chunk(
    engine: MLATFeatureEngine,
    dataset: str,
    attack: str,
    cache_attack: str,
    qa: list[dict[str, Any]],
    start: int,
    limit: int,
    cache_root: Path,
    output_dir: Path,
    overwrite: bool,
) -> None:
    token_budget = engine.token_budget_for_dataset(
        dataset
    )
    if start < 0 or limit <= 0 or start + limit > len(qa):
        raise ValueError(
            f"cached ML-ATD chunk exceeds exact mapping cohort: "
            f"start={start}, limit={limit}, rows={len(qa)}"
        )
    end = start + limit
    expected_rows = limit

    npz_path, csv_path = engine.expected_paths(
        dataset, cache_attack, token_budget, start, limit
    )
    stem = npz_path.stem
    meta_path = npz_path.with_suffix(".meta.json")
    expected_indices = list(range(start, end))
    expected_image_ids = [str(qa[index]["image_filename"]) for index in expected_indices]
    _dataset_dir, mapping_path = engine.dataset_paths(dataset)
    cache_contract_path = attack_cache_contract_path(
        cache_root=cache_root,
        method="shared",
        dataset=dataset,
        attack=cache_attack,
    )
    cache_contract = json.loads(cache_contract_path.read_text(encoding="utf-8"))
    contract = engine.chunk_contract(
        dataset=dataset,
        mapping_path=mapping_path,
        attack=attack,
        token_budget=token_budget,
        expected_indices=expected_indices,
        expected_image_ids=expected_image_ids,
        cache_contract=cache_contract,
    )
    # These artifacts describe a pre-generated attack.  Bind their top-level
    # attack fields to the producer contract rather than the extractor's
    # otherwise-unused RunConfig defaults (notably CAA alpha=1 vs CAGE=.5).
    for key in ("eps_255", "alpha_255", "steps"):
        contract[key] = cache_contract[key]
    contract["attack_parameters_source"] = "source_cache_contract"
    contract["lam_role"] = "feature_extractor_fata_reference_only"
    contract["target_k_role"] = "feature_extractor_attention_target_reference_only"
    contract["cache_namespace"] = cache_attack
    ensure_run_contract(
        meta_path,
        contract,
        result_path=[npz_path, csv_path],
    )

    if (
        engine.output_complete(
            npz_path,
            csv_path,
            expected_indices,
            expected_image_ids,
            dataset=dataset,
            attack=attack,
            token_budget=token_budget,
            eps_255=float(cache_contract["eps_255"]),
            alpha_255=float(cache_contract["alpha_255"]),
            steps=int(cache_contract["steps"]),
        )
        and not overwrite
    ):
        print(
            f"[SKIP] complete {npz_path.name}"
        )
        return

    feature_lists: dict[
        str,
        list[np.ndarray],
    ] = {}
    metadata_rows: list[
        dict[str, Any]
    ] = []

    dataset_dir = engine.cfg.project_root / dataset

    missing_cache = 0

    for sample_idx in tqdm(
        range(start, end),
        desc=stem,
        leave=False,
    ):
        gt = qa[sample_idx]
        filename = gt["image_filename"]
        adv_path = cache_path(
            cache_root,
            cache_attack,
            dataset,
            filename,
        )
        if not attack_image_exists(
            cache_root=cache_root,
            method="shared",
            dataset=dataset,
            attack=cache_attack,
            image_filename=filename,
        ):
            missing_cache += 1

            if missing_cache <= 10:
                print(
                    "[MISSING CACHE]",
                    f"sample_idx={sample_idx}",
                    f"attack={attack}",
                    f"expected={adv_path}",
                )

            continue

        try:
            detection_image = load_attack_image(
                cache_root=cache_root,
                method="shared",
                dataset=dataset,
                attack=cache_attack,
                image_filename=filename,
            )
        except (OSError, RuntimeError, ValueError):
            missing_cache += 1
            continue

        # Cache images were already generated from the CLIP-preprocessed
        # square image. Do not expand/re-preprocess geometrically here.
        prompt = build_prompt(gt)
        features = engine.extract_multilevel_features(
            detection_image,
            prompt,
            token_budget,
        )

        for name, vector in features.arrays.items():
            feature_lists.setdefault(
                name,
                [],
            ).append(vector)

        metadata_rows.append(
            {
                "sample_idx": sample_idx,
                "Image_ID": filename,
                "Question": gt.get(
                    "question",
                    "",
                ),
                "dataset": dataset,
                "method": engine.cfg.method,
                "attack_for_detection": attack,
                "label": 1,
                "seed": engine.cfg.seed,
                "sample_seed": set_sample_seed(
                    engine.cfg.seed,
                    sample_idx,
                ),
                "token_budget": token_budget,
                "token_budget_mode": (
                    engine.cfg.token_budget_mode
                ),
                "eps_255": cache_contract["eps_255"],
                "alpha_255": cache_contract["alpha_255"],
                "steps": cache_contract["steps"],
                "lam": engine.cfg.lam,
                "target_k": engine.cfg.target_k,
                "vision_layers": ",".join(map(str, engine.vision_layers)),
                "llm_layers": ",".join(map(str, engine.llm_layers)),
                "image_span_valid": (
                    features.image_span_valid
                ),
                "image_token_count": (
                    features.image_token_count
                ),
                "llm_sequence_length": (
                    features.llm_sequence_length
                ),
                "cache_path": str(adv_path),
            }
        )

        if (
            torch.cuda.is_available()
            and len(metadata_rows) % 20 == 0
        ):
            torch.cuda.empty_cache()

    if len(metadata_rows) != expected_rows:
        raise RuntimeError(
            f"Incomplete cache coverage for {stem}: expected={expected_rows}, "
            f"extracted={len(metadata_rows)}, missing_or_invalid={missing_cache}. "
            "No partial feature artifact was written."
        )

    write_atomic(
        npz_path,
        csv_path,
        feature_lists,
        metadata_rows,
        engine.vision_layers,
        engine.llm_layers,
    )
    if not engine.output_complete(
        npz_path,
        csv_path,
        expected_indices,
        expected_image_ids,
        dataset=dataset,
        attack=attack,
        token_budget=token_budget,
        eps_255=float(cache_contract["eps_255"]),
        alpha_255=float(cache_contract["alpha_255"]),
        steps=int(cache_contract["steps"]),
    ):
        raise RuntimeError(f"Post-write completeness check failed: {npz_path}")
    print(
        f"[DONE] {stem}: "
        f"rows={len(metadata_rows)}, "
        f"missing_cache={missing_cache}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        required=True,
        choices=ALL_METHODS,
    )
    parser.add_argument(
        "--datasets",
        default="all",
    )
    parser.add_argument(
        "--attacks",
        default="cage,caa",
    )
    parser.add_argument(
        "--total_limit",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=250,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    parser.add_argument("--cage_eps_255", type=float, default=2.0)
    parser.add_argument("--cage_alpha_255", type=float, default=0.5)
    parser.add_argument("--cage_steps", type=int, default=100)
    parser.add_argument("--caa_eps_255", type=float, default=2.0)
    parser.add_argument("--caa_alpha_255", type=float, default=1.0)
    parser.add_argument("--caa_steps", type=int, default=100)
    parser.add_argument(
        "--max_input_tokens",
        type=int,
        default=0,
        help="Must match the dedicated CAGE/CAA cache producer contract.",
    )
    parser.add_argument(
        "--data-root",
        dest="project_root",
        type=Path,
        default=os.environ.get("FATA_DATA_ROOT"),
    )
    parser.add_argument(
        "--llava-path",
        dest="llava_path",
        default=os.environ.get("FATA_LLAVA_MODEL"),
    )
    parser.add_argument(
        "--clip-path",
        dest="clip_path",
        default=os.environ.get("FATA_CLIP_MODEL"),
    )
    parser.add_argument(
        "--cache-root",
        dest="cache_root",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        type=Path,
        default=None,
    )
    parser.add_argument("--output-root", default=os.environ.get("FATA_OUTPUT_ROOT"))
    parser.add_argument(
        "--token_budget_mode",
        choices=["practical"],
        default="practical",
        help="Formal release mode; uncompressed/full was not part of the paper grid.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    args = parser.parse_args()

    if not args.project_root or not args.llava_path or not args.clip_path or not args.cache_root or not (args.output_dir or args.output_root):
        parser.error("provide data, model, CLIP, cache, and output roots")
    if args.total_limit <= 0 or args.chunk_size <= 0 or args.seed < 0:
        parser.error("--total_limit/--chunk_size must be positive and --seed non-negative")
    if args.max_input_tokens < 0:
        parser.error("--max_input_tokens must be non-negative")
    numeric_attack_values = (
        args.cage_eps_255,
        args.cage_alpha_255,
        args.caa_eps_255,
        args.caa_alpha_255,
    )
    if (
        not all(math.isfinite(value) and value > 0 for value in numeric_attack_values)
        or args.cage_steps <= 0
        or args.caa_steps <= 0
    ):
        parser.error("cache attack parameters must be finite and positive")
    if (
        args.max_input_tokens != 0
        or args.cage_eps_255 != 2.0
        or args.cage_alpha_255 != 0.5
        or args.cage_steps != 100
        or args.caa_eps_255 != 2.0
        or args.caa_alpha_255 != 1.0
        or args.caa_steps != 100
    ):
        parser.error(
            "formal cached-attack extraction is locked to max_input_tokens=0, "
            "CAGE eps/alpha/steps=2/0.5/100, and CAA=2/1/100"
        )
    if args.output_dir is None:
        args.output_dir = resolve_owned_output_path(
            args.output_root, "detection/mlatd/cached_attacks"
        )
    else:
        args.output_dir = args.output_dir.expanduser().resolve()
    protected_inputs = {
        "dataset_root": args.project_root,
        "LLaVA model": args.llava_path,
        "CLIP model": args.clip_path,
    }
    assert_output_separate(args.output_dir, protected_inputs)
    assert_output_separate(args.cache_root, protected_inputs)
    assert_output_separate(args.output_dir, {"attack cache": args.cache_root})

    if args.datasets.lower() in {
        "all",
        "*",
    }:
        datasets = list(ALL_DATASETS)
    else:
        datasets = [
            x.strip()
            for x in args.datasets.split(",")
            if x.strip()
        ]
    attacks = [
        x.strip()
        for x in args.attacks.split(",")
        if x.strip()
    ]
    if not datasets or not attacks:
        parser.error("datasets and attacks must select at least one value")
    for dataset in datasets:
        if dataset not in ALL_DATASETS:
            raise ValueError(dataset)
    for attack in attacks:
        if attack not in ATTACKS:
            raise ValueError(attack)
    cache_attacks = {
        "cage": attack_namespace(
            "cage", seed=args.seed, eps_255=args.cage_eps_255,
            alpha_255=args.cage_alpha_255, steps=args.cage_steps,
        ),
        "caa": attack_namespace(
            "caa", seed=args.seed, eps_255=args.caa_eps_255,
            alpha_255=args.caa_alpha_255, steps=args.caa_steps,
        ),
    }

    for dataset in datasets:
        mapping_path = (
            Path(args.project_root) / dataset / f"{dataset}_mapping.jsonl"
        )
        for attack in attacks:
            if attack == "cage":
                eps_255 = args.cage_eps_255
                alpha_255 = args.cage_alpha_255
                steps = args.cage_steps
            else:
                eps_255 = args.caa_eps_255
                alpha_255 = args.caa_alpha_255
                steps = args.caa_steps
            require_run_contract(
                attack_cache_contract_path(
                    cache_root=args.cache_root,
                    method="shared",
                    dataset=dataset,
                    attack=cache_attacks[attack],
                ),
                build_attack_cache_contract(
                    definition=f"mlatd_{attack}_generator_v1",
                    dataset=dataset,
                    mapping_path=mapping_path,
                    method="shared",
                    model_path=args.llava_path,
                    clip_path=args.clip_path,
                    seed=args.seed,
                    eps_255=eps_255,
                    alpha_255=alpha_255,
                    steps=steps,
                    extra=baseline_attack_contract_extra(
                        attack,
                        max_input_tokens=args.max_input_tokens,
                    ),
                ),
            )

    # Attack-generation hyperparameters are irrelevant here because
    # images are read from cache. They are only required by RunConfig.
    cfg = RunConfig(
        project_root=args.project_root,
        llava_path=args.llava_path,
        clip_path=args.clip_path,
        method=args.method,
        seed=args.seed,
        eps_255=2.0,
        alpha_255=0.5,
        steps=100,
        lam=1.0,
        target_k=64,
        token_budget_mode=args.token_budget_mode,
        output_dir=args.output_dir,
        attack_cache_root=args.cache_root,
    )

    llava_identity = artifact_identity(args.llava_path)
    clip_identity = artifact_identity(args.clip_path)
    qa_by_dataset = {}
    all_complete = not args.overwrite
    for dataset in datasets:
        qa = load_dataset_without_model(cfg, dataset)
        if len(qa) < args.total_limit:
            raise RuntimeError(
                f"dataset {dataset} has {len(qa)} rows, fewer than "
                f"the requested exact cohort {args.total_limit}"
            )
        qa_by_dataset[dataset] = qa
        for attack in attacks:
            cache_attack = cache_attacks[attack]
            cache_contract_path = attack_cache_contract_path(
                cache_root=args.cache_root,
                method="shared",
                dataset=dataset,
                attack=cache_attack,
            )
            cache_contract = json.loads(cache_contract_path.read_text(encoding="utf-8"))
            contract_overrides = {
                "eps_255": cache_contract["eps_255"],
                "alpha_255": cache_contract["alpha_255"],
                "steps": cache_contract["steps"],
                "attack_parameters_source": "source_cache_contract",
                "lam_role": "feature_extractor_fata_reference_only",
                "target_k_role": "feature_extractor_attention_target_reference_only",
                "cache_namespace": cache_attack,
            }
            for start in range(0, args.total_limit, args.chunk_size):
                limit = min(args.chunk_size, args.total_limit - start)
                if not feature_chunk_complete_without_model(
                    cfg,
                    dataset=dataset,
                    attack=attack,
                    path_attack=cache_attack,
                    qa_database=qa,
                    start=start,
                    limit=limit,
                    llava_identity=llava_identity,
                    clip_identity=clip_identity,
                    cache_contract=cache_contract,
                    contract_overrides=contract_overrides,
                    eps_255=float(cache_contract["eps_255"]),
                    alpha_255=float(cache_contract["alpha_255"]),
                    steps=int(cache_contract["steps"]),
                    expected_cache_root=args.cache_root,
                ):
                    all_complete = False
    if all_complete:
        print("[SKIP] every requested cached-attack ML-ATD triplet is exactly complete; model loading skipped")
        return

    engine = MLATFeatureEngine(cfg)
    try:
        for dataset in datasets:
            qa = qa_by_dataset[dataset]
            usable = args.total_limit
            for attack in attacks:
                for start in range(
                    0,
                    usable,
                    args.chunk_size,
                ):
                    limit = min(
                        args.chunk_size,
                        usable - start,
                    )
                    run_chunk(
                        engine=engine,
                        dataset=dataset,
                        attack=attack,
                        cache_attack=cache_attacks[attack],
                        qa=qa,
                        start=start,
                        limit=limit,
                        cache_root=args.cache_root,
                        output_dir=args.output_dir,
                        overwrite=args.overwrite,
                    )
    finally:
        engine.close()


if __name__ == "__main__":
    main()
