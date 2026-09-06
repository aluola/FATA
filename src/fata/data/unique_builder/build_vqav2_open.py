"""CLI entry point for deterministic unique-image VQAv2-Open construction."""

from __future__ import annotations

import argparse
import os
import json
from pathlib import Path

from .build_core import (
    BuildConfig,
    build_dataset,
    collect_completed_sibling_exclusions,
    load_completed_resume,
    load_stored_build_exclusions,
    resolve_output_root,
)
from .common import (
    LlavaConstructionRunner,
    SEED,
    configure_reproducibility,
    resolve_builder_source_cache,
)
from .sources import load_vqav2
from fata.utils.paths import assert_output_separate


DEFAULT_OUTPUT = (Path(os.environ["FATA_OUTPUT_ROOT"]) / "datasets" / "fata_unique_1000_v1" if os.environ.get("FATA_OUTPUT_ROOT") else None)
DEFAULT_MODEL = (Path(os.environ["FATA_LLAVA_MODEL"]) if os.environ.get("FATA_LLAVA_MODEL") else None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.output is None or args.model is None:
        parser.error("pass --output and --model or set FATA_OUTPUT_ROOT and FATA_LLAVA_MODEL")
    if args.target < 1 or args.seed < 0:
        parser.error("--target must be positive and --seed must be non-negative")
    protected = {"model": args.model}
    if args.cache is not None:
        protected["Hugging Face cache"] = args.cache
    assert_output_separate(args.output, protected)
    configure_reproducibility(args.seed)
    output = resolve_output_root(
        args.output, resume=args.resume, dataset_name="VQAv2_Open"
    )
    exclusions = (
        load_stored_build_exclusions(output, dataset_name="VQAv2_Open")
        if args.resume
        else collect_completed_sibling_exclusions(
            output,
            dataset_name="VQAv2_Open",
            target=args.target,
            seed=args.seed,
        )
    )
    source_cache = resolve_builder_source_cache(output)
    # Official ZIPs and COCO image files are always copied/cached under the
    # selected output root; --cache is accepted uniformly for the four CLIs.
    bundle = load_vqav2(
        "VQAv2_Open",
        source_cache,
        args.seed,
        resume_source=args.resume,
    )
    config = BuildConfig(output, args.target, args.seed, args.resume, exclusions)
    result = load_completed_resume(bundle, config, model_path=args.model)
    if result is None:
        runner = LlavaConstructionRunner(args.model, args.device)
        result = build_dataset(bundle, runner, config)
    print(
        json.dumps(
            {
                "dataset": result.dataset_name,
                "output_root": str(result.output_root),
                "retained": result.stats["retained_count"],
                "target_reached": result.target_reached,
            },
            ensure_ascii=False,
        )
    )
    if not result.target_reached:
        raise RuntimeError("VQAv2_Open target not reached; resume the saved checkpoint")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
