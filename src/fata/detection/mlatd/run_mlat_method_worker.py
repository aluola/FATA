from __future__ import annotations

import argparse
import os
import math
from pathlib import Path

from .mlat_core import (
    ALL_ATTACKS,
    ALL_DATASETS,
    ALL_METHODS,
    MLATFeatureEngine,
    RunConfig,
    feature_chunk_complete_without_model,
    load_dataset_without_model,
    parse_csv_arg,
)
from fata.utils.paths import (
    assert_output_separate,
    resolve_attack_cache_root,
    resolve_owned_output_path,
)
from fata.utils.run_contract import artifact_identity


WORKER_ATTACKS = tuple(
    attack for attack in ALL_ATTACKS if attack not in {"cage", "caa"}
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one persistent ML-HiddenDetect worker for a fixed compression method."
    )
    parser.add_argument("--method", required=True, choices=ALL_METHODS)
    parser.add_argument("--datasets", default="all")
    parser.add_argument(
        "--attacks",
        default="fata,base,clean_clip,random_clip",
        help="Comma-separated. Heavy attacks are intentionally first by default.",
    )
    parser.add_argument("--total_limit", type=int, default=1000)
    parser.add_argument("--chunk_size", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eps_255", type=float, default=2.0)
    parser.add_argument("--alpha_255", type=float, default=0.5)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--target_k", type=int, default=64)
    parser.add_argument(
        "--token_budget_mode",
        choices=["practical"],
        default="practical",
        help="Formal release mode; uncompressed/full was not part of the paper grid.",
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
    parser.add_argument("--clip-path", dest="clip_path", default=os.environ.get("FATA_CLIP_MODEL"))
    parser.add_argument("--output-root", default=os.environ.get("FATA_OUTPUT_ROOT"))
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
    )
    parser.add_argument("--attack-cache-root", type=Path, default=None)
    parser.add_argument("--overwrite_incomplete", action="store_true")
    args = parser.parse_args()

    if not args.project_root or not args.llava_path or not args.clip_path or not (args.output_dir or args.output_root):
        parser.error("provide data, model, CLIP, and output roots")
    if args.total_limit <= 0 or args.chunk_size <= 0 or args.seed < 0:
        parser.error("--total_limit/--chunk_size must be positive and --seed non-negative")
    if (
        args.steps <= 0
        or not all(math.isfinite(value) for value in (args.eps_255, args.alpha_255, args.lam))
        or args.eps_255 <= 0
        or args.alpha_255 <= 0
        or args.lam < 0
        or args.target_k <= 0
    ):
        parser.error("invalid attack hyperparameters")
    if (
        args.eps_255 != 2.0
        or args.alpha_255 != 0.5
        or args.steps != 100
        or args.lam != 1.0
        or args.target_k != 64
    ):
        parser.error(
            "formal ML-ATD release is locked to eps=2, alpha=0.5, "
            "steps=100, lambda=1, target_k=64"
        )
    if args.output_dir is None:
        args.output_dir = resolve_owned_output_path(
            args.output_root, "detection/mlatd/features"
        )
    else:
        args.output_dir = args.output_dir.expanduser().resolve()
    args.attack_cache_root = resolve_attack_cache_root(
        explicit_cache_root=args.attack_cache_root,
        output_root=args.output_root,
        explicit_output_dir=args.output_dir,
    )
    protected_inputs = {
        "dataset_root": args.project_root,
        "LLaVA model": args.llava_path,
        "CLIP model": args.clip_path,
    }
    assert_output_separate(args.output_dir, protected_inputs)
    assert_output_separate(args.attack_cache_root, protected_inputs)
    assert_output_separate(args.output_dir, {"attack cache": args.attack_cache_root})

    datasets = parse_csv_arg(args.datasets, ALL_DATASETS)
    attacks = parse_csv_arg(args.attacks, WORKER_ATTACKS)
    if not datasets or not attacks:
        parser.error("datasets and attacks must select at least one value")
    for dataset in datasets:
        if dataset not in ALL_DATASETS:
            raise ValueError(f"Unknown dataset: {dataset}")
    for attack in attacks:
        if attack not in WORKER_ATTACKS:
            parser.error(
                f"attack {attack!r} requires generate_attack_cache_cage_caa + "
                "extract_cached_attack_mlat so its cache contract is explicit"
            )

    cfg = RunConfig(
        project_root=args.project_root,
        llava_path=args.llava_path,
        clip_path=args.clip_path,
        method=args.method,
        seed=args.seed,
        eps_255=args.eps_255,
        alpha_255=args.alpha_255,
        steps=args.steps,
        lam=args.lam,
        target_k=args.target_k,
        token_budget_mode=args.token_budget_mode,
        output_dir=args.output_dir,
        attack_cache_root=args.attack_cache_root,
    )

    # A complete immutable grid must be resumable on CPU-only hosts.  Compute
    # byte identities and validate every requested triplet before constructing
    # MLATFeatureEngine, whose constructor allocates both LLaVA and CLIP.
    llava_identity = artifact_identity(args.llava_path)
    clip_identity = artifact_identity(args.clip_path)
    qa_by_dataset = {}
    all_complete = not args.overwrite_incomplete
    for dataset in datasets:
        qa_database = load_dataset_without_model(cfg, dataset)
        if len(qa_database) < args.total_limit:
            raise RuntimeError(
                f"dataset {dataset} has {len(qa_database)} rows, fewer than "
                f"the requested exact cohort {args.total_limit}"
            )
        qa_by_dataset[dataset] = qa_database
        for attack in attacks:
            for start in range(0, args.total_limit, args.chunk_size):
                limit = min(args.chunk_size, args.total_limit - start)
                if not feature_chunk_complete_without_model(
                    cfg,
                    dataset=dataset,
                    attack=attack,
                    qa_database=qa_database,
                    start=start,
                    limit=limit,
                    llava_identity=llava_identity,
                    clip_identity=clip_identity,
                ):
                    all_complete = False
    if all_complete:
        print("[SKIP] every requested ML-ATD feature triplet is exactly complete; model loading skipped")
        return

    engine = MLATFeatureEngine(cfg)
    try:
        for dataset in datasets:
            qa_database = qa_by_dataset[dataset]
            usable_n = args.total_limit
            for attack in attacks:
                for start in range(0, usable_n, args.chunk_size):
                    limit = min(args.chunk_size, usable_n - start)
                    engine.process_chunk(
                        dataset=dataset,
                        qa_database=qa_database,
                        attack=attack,
                        start=start,
                        limit=limit,
                        overwrite_incomplete=args.overwrite_incomplete,
                    )
    finally:
        engine.close()


if __name__ == "__main__":
    main()
