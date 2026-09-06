"""Build all four unique-image datasets serially with one loaded LLaVA model."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .build_core import (
    BuildConfig,
    BuildExclusions,
    BuildResult,
    build_dataset,
    load_completed_resume,
    resolve_output_root,
)
from .common import (
    LlavaConstructionRunner,
    SEED,
    configure_reproducibility,
    resolve_builder_source_cache,
)
from .sources import load_scienceqa, load_textvqa, load_vqav2
from fata.utils.paths import assert_output_separate


DEFAULT_OUTPUT = (Path(os.environ["FATA_OUTPUT_ROOT"]) / "datasets" / "fata_unique_1000_v1" if os.environ.get("FATA_OUTPUT_ROOT") else None)
DEFAULT_MODEL = (Path(os.environ["FATA_LLAVA_MODEL"]) if os.environ.get("FATA_LLAVA_MODEL") else None)
BUILD_ORDER = ("TextVQA_Open", "VQAv2_Open", "VQAv2_MC", "ScienceQA_MC")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="optional Hugging Face datasets cache directory",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--resume", action="store_true")
    return parser


def _resume_for(output: Path, dataset_name: str, requested: bool) -> bool:
    if not requested:
        return False
    return (
        output / "manifests" / f"{dataset_name}_checkpoint.json"
    ).is_file()


def run_all(
    *,
    output: Path,
    target: int,
    model: Path,
    hf_cache: Path | None,
    device: str,
    seed: int,
    resume: bool,
    runner: Any | None = None,
) -> list[BuildResult]:
    """Run the fixed build order; later sets exclude prior accepted RGB hashes."""

    if output is None or model is None:
        raise ValueError("pass --output and --model or set FATA_OUTPUT_ROOT and FATA_LLAVA_MODEL")
    if type(target) is not int or target < 1:
        raise ValueError("target must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if type(resume) is not bool:
        raise TypeError("resume must be a boolean")
    protected = {"model": model}
    if hf_cache is not None:
        protected["Hugging Face cache"] = hf_cache
    assert_output_separate(output, protected)
    output = Path(output).resolve()
    source_cache = resolve_builder_source_cache(output)
    configure_reproducibility(seed)

    active_runner = runner

    def run_bundle(bundle: Any, config: BuildConfig) -> BuildResult:
        nonlocal active_runner
        completed = load_completed_resume(
            bundle,
            config,
            current_runner=active_runner,
            model_path=model if active_runner is None else None,
        )
        if completed is not None:
            return completed
        if active_runner is None:
            # At most one model construction, delayed until inference is needed.
            print(
                json.dumps(
                    {"phase": "load_model", "model": str(model), "device": device},
                    ensure_ascii=False,
                ),
                flush=True,
            )
            active_runner = LlavaConstructionRunner(model, device)
        return build_dataset(bundle, active_runner, config)

    results: list[BuildResult] = []
    prior_source_hashes: set[str] = set()
    prior_saved_hashes: set[str] = set()

    print(json.dumps({"phase": "prepare_source", "dataset": "TextVQA_Open"}), flush=True)
    text_bundle = load_textvqa(hf_cache)
    print(
        json.dumps(
            {
                "phase": "source_ready",
                "dataset": "TextVQA_Open",
                "candidates": len(text_bundle.candidates),
            }
        ),
        flush=True,
    )
    text_result = run_bundle(
        text_bundle,
        BuildConfig(
            output,
            target,
            seed,
            _resume_for(output, "TextVQA_Open", resume),
        ),
    )
    results.append(text_result)
    prior_source_hashes.update(text_result.accepted_source_hashes)
    prior_saved_hashes.update(text_result.accepted_saved_hashes)

    print(json.dumps({"phase": "prepare_source", "dataset": "VQAv2_Open"}), flush=True)
    open_resume = _resume_for(output, "VQAv2_Open", resume)
    open_bundle = load_vqav2(
        "VQAv2_Open",
        source_cache,
        seed,
        resume_source=open_resume,
    )
    print(
        json.dumps(
            {
                "phase": "source_ready",
                "dataset": "VQAv2_Open",
                "candidates": len(open_bundle.candidates),
            }
        ),
        flush=True,
    )
    open_result = run_bundle(
        open_bundle,
        BuildConfig(
            output,
            target,
            seed,
            open_resume,
            BuildExclusions.from_iterables(
                source_hashes=prior_source_hashes,
                saved_hashes=prior_saved_hashes,
            ),
        ),
    )
    results.append(open_result)
    prior_source_hashes.update(open_result.accepted_source_hashes)
    prior_saved_hashes.update(open_result.accepted_saved_hashes)

    # VQAv2-MC has the additional hard partition by Open source image_id.
    print(json.dumps({"phase": "prepare_source", "dataset": "VQAv2_MC"}), flush=True)
    mc_resume = _resume_for(output, "VQAv2_MC", resume)
    mc_bundle = load_vqav2(
        "VQAv2_MC",
        source_cache,
        seed,
        resume_source=mc_resume,
    )
    print(
        json.dumps(
            {
                "phase": "source_ready",
                "dataset": "VQAv2_MC",
                "candidates": len(mc_bundle.candidates),
            }
        ),
        flush=True,
    )
    mc_result = run_bundle(
        mc_bundle,
        BuildConfig(
            output,
            target,
            seed,
            mc_resume,
            BuildExclusions.from_iterables(
                source_image_ids=open_result.accepted_source_image_ids,
                source_hashes=prior_source_hashes,
                saved_hashes=prior_saved_hashes,
            ),
        ),
    )
    results.append(mc_result)
    prior_source_hashes.update(mc_result.accepted_source_hashes)
    prior_saved_hashes.update(mc_result.accepted_saved_hashes)

    print(json.dumps({"phase": "prepare_source", "dataset": "ScienceQA_MC"}), flush=True)
    science_bundle = load_scienceqa(hf_cache)
    print(
        json.dumps(
            {
                "phase": "source_ready",
                "dataset": "ScienceQA_MC",
                "candidates": len(science_bundle.candidates),
            }
        ),
        flush=True,
    )
    science_result = run_bundle(
        science_bundle,
        BuildConfig(
            output,
            target,
            seed,
            _resume_for(output, "ScienceQA_MC", resume),
            BuildExclusions.from_iterables(
                source_hashes=prior_source_hashes,
                saved_hashes=prior_saved_hashes,
            ),
        ),
    )
    results.append(science_result)
    return results


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
    output = resolve_output_root(args.output, resume=args.resume)
    print(json.dumps({"actual_output_root": str(output)}, ensure_ascii=False), flush=True)
    results = run_all(
        output=output,
        target=args.target,
        model=args.model,
        hf_cache=args.cache,
        device=args.device,
        seed=args.seed,
        resume=args.resume,
    )
    print(
        json.dumps(
            {
                "actual_output_root": str(output),
                "build_order": list(BUILD_ORDER),
                "datasets": {
                    result.dataset_name: {
                        "retained": result.stats["retained_count"],
                        "target_reached": result.target_reached,
                    }
                    for result in results
                },
            },
            ensure_ascii=False,
        )
    )
    incomplete = [result.dataset_name for result in results if not result.target_reached]
    if incomplete:
        raise RuntimeError(f"dataset target not reached: {incomplete}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
