"""CLI entry point for deterministic unique-image VQAv2-MC construction."""

from __future__ import annotations

import argparse
import os
import json
from pathlib import Path

from .build_core import (
    BuildConfig,
    BuildExclusions,
    _canonical_json,
    _read_jsonl_strict,
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
    parser.add_argument(
        "--open-dataset-root",
        type=Path,
        default=None,
        help=(
            "root containing manifests/VQAv2_Open_samples.jsonl; defaults to "
            "--output and is required so the Open/MC image partition cannot be bypassed"
        ),
    )
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
        args.output, resume=args.resume, dataset_name="VQAv2_MC"
    )
    open_root = (
        args.open_dataset_root.resolve()
        if args.open_dataset_root is not None
        else output
    )
    frozen_source_cache = resolve_builder_source_cache(
        open_root, require_existing=True
    )
    source_cache = resolve_builder_source_cache(output)
    open_samples_path = (
        open_root / "manifests" / "VQAv2_Open_samples.jsonl"
    )
    if not open_samples_path.is_file():
        raise FileNotFoundError(
            "standalone VQAv2-MC construction requires the frozen VQAv2-Open "
            f"sidecar for hard image partitioning: {open_samples_path}. "
            "Run build_all.py, or build VQAv2-Open first."
        )
    open_bundle = load_vqav2(
        "VQAv2_Open",
        frozen_source_cache,
        args.seed,
        read_only_source=True,
        resume_source=True,
    )
    open_result = load_completed_resume(
        open_bundle,
        BuildConfig(
            open_root,
            args.target,
            args.seed,
            True,
            load_stored_build_exclusions(
                open_root, dataset_name="VQAv2_Open"
            ),
        ),
        repair_public_state=False,
        model_path=args.model,
    )
    if open_result is None:
        raise RuntimeError(
            "VQAv2-Open checkpoint is not complete for the requested target"
        )
    open_samples = _read_jsonl_strict(
        open_samples_path, label="frozen VQAv2-Open sample sidecar"
    )
    if _canonical_json(
        open_samples, label="frozen VQAv2-Open sample sidecar"
    ) != _canonical_json(
        list(open_result.samples), label="VQAv2-Open checkpoint samples"
    ):
        raise RuntimeError(
            "frozen VQAv2-Open sidecar does not match its completed checkpoint"
        )
    open_mapping_path = (
        open_root / "VQAv2_Open" / "VQAv2_Open_mapping.jsonl"
    )
    open_mappings = _read_jsonl_strict(
        open_mapping_path, label="frozen VQAv2-Open mapping"
    )
    expected_mappings = [sample["mapping"] for sample in open_samples]
    if _canonical_json(
        open_mappings, label="frozen VQAv2-Open mapping"
    ) != _canonical_json(
        expected_mappings, label="VQAv2-Open checkpoint mappings"
    ):
        raise RuntimeError(
            "frozen VQAv2-Open mapping does not match its completed checkpoint"
        )
    if args.resume:
        exclusions = load_stored_build_exclusions(
            output, dataset_name="VQAv2_MC"
        )
    else:
        shared_exclusions = collect_completed_sibling_exclusions(
            output,
            dataset_name="VQAv2_MC",
            target=args.target,
            seed=args.seed,
        )
        exclusions = BuildExclusions.from_iterables(
            source_image_ids=(
                set(shared_exclusions.source_image_ids)
                | {
                    str(row["source_image_id"])
                    for row in open_samples
                    if row.get("source_image_id") is not None
                }
            ),
            source_hashes=(
                set(shared_exclusions.source_canonical_rgb_sha256)
                | {
                    str(row["source_canonical_rgb_sha256"])
                    for row in open_samples
                    if row.get("source_canonical_rgb_sha256")
                }
            ),
            saved_hashes=(
                set(shared_exclusions.saved_canonical_rgb_sha256)
                | {
                    str(row["saved_canonical_rgb_sha256"])
                    for row in open_samples
                    if row.get("saved_canonical_rgb_sha256")
                }
            ),
        )
    bundle = load_vqav2(
        "VQAv2_MC",
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
        raise RuntimeError("VQAv2_MC target not reached; resume the saved checkpoint")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
