#!/usr/bin/env python
"""
Q-FATA V6 Compression-Specific — frozen four-compressor formal runner.

For each config, for each sample: run the V6 attack (100 PGD steps), then evaluate the
SAME adversarial image across all 4 compressors x 3 budgets (Full / 1/3 / Practical).
Evaluation scored with the CANONICAL STRICT VQA scorer (V6 audit decision).

CSV schema is fixed (single FIELDNAMES list). One shared adv image per sample.
Resume-safe: skips a group only after all 4 compressors x 3 budgets succeed;
partial groups are deterministically retried without duplicating successful cells.

Usage:
  CUDA_VISIBLE_DEVICES=0,1 python -m fata.runtimes.qwen35.run_qwen35_v6 \
    --dataset TextVQA_Open --configs v6_a_fullpres_light_4comp --limit 20 \
    --device-map balanced
"""
import os, json, csv, time, gc, argparse
from dataclasses import asdict, replace
from pathlib import Path

import torch

from .v6_core import (
    V6Model, FORMAL_CONFIG_NAME, FORMAL_CONFIGS, deterministic_seed, answer_token_ids,
    strict_score_open, strict_score_mc, COMPRESSOR_CLS, METHODS,
)
from .resume import completed_groups, read_successful_cells
from fata.utils.run_contract import (
    artifact_identity,
    ensure_run_contract,
    explicit_image_identity,
    sha256_file,
)
from fata.utils.paths import (
    assert_output_separate,
    create_owned_output_directory,
    resolve_owned_output_path,
    safe_filename_component,
)

BUDGETS = ["Full", "1/3", "Practical"]

# Exact public contract mirrored by
# configs/qwen35/QFATA_V6_COMPRESSION_SPECIFIC_4COMP_FROZEN.json.  Keeping the
# expected payload independent of the V6Config instance makes drift fail closed
# at runtime; a unit test also compares it byte-for-value with the public JSON.
FORMAL_FROZEN_CONFIG = {
    "task_mode": "ground_truth",
    "w_task": 1.5,
    "w_rank": 0.25,
    "w_comp": 0.05,
    "w_full": 1.0,
    "w_distill": 1.0,
    "use_projection": False,
    "epsilon": 2.0 / 255.0,
    "alpha": 0.5 / 255.0,
    "steps": 100,
    "max_answer_tokens": 4,
    "rank_margin": 0.05,
    "comp_margin": 0.05,
    "boundary_window_ratio": 0.05,
    "eps_grad": 1e-8,
    "budget_pattern": ["Practical", "Practical", "1/3", "Practical", "1/3"],
    "method_universe": ["VisionZIP", "VisPruner", "FlowCut", "PruMerge"],
}
FIELDNAMES = [
    "dataset", "config", "sample_index", "image_id", "compressor", "retention_label",
    "retention_ratio", "N_full", "K_actual",
    "clean_full_raw", "fata_full_raw", "clean_full_correct", "fata_full_correct",
    "clean_compressed_raw", "fata_compressed_raw", "clean_compressed_correct",
    "fata_compressed_correct",
    "delta_linf", "seed", "initial_delta_sha256", "delta_sha256",
    "epsilon", "alpha", "steps", "task_mode", "w_task", "w_rank", "w_comp",
    "w_full", "w_distill", "use_projection",
    "full_clean_GT_loss", "full_adv_GT_loss", "compressed_adv_GT_loss", "full_logit_KL",
    "mean_grad_cos_comp_full", "projection_trigger_rate",
    "attack_seconds", "status", "error",
]


def _formal_config_payload(config):
    """Serialize only the fields frozen in the public formal-config JSON."""

    values = asdict(config)
    return {
        key: list(values[key]) if key in {"budget_pattern", "method_universe"} else values[key]
        for key in FORMAL_FROZEN_CONFIG
    }


def _validated_formal_config(*, is_mc):
    """Return a task-local config only if code and the frozen contract agree."""

    if set(FORMAL_CONFIGS) != {FORMAL_CONFIG_NAME}:
        raise RuntimeError("formal Qwen config registry must contain exactly one entry")
    source = FORMAL_CONFIGS[FORMAL_CONFIG_NAME]
    observed = _formal_config_payload(source)
    if observed != FORMAL_FROZEN_CONFIG:
        raise RuntimeError(
            "formal Qwen runtime configuration drifted from the frozen public contract"
        )
    # Never mutate the module-level frozen object when switching open/MC tasks.
    return replace(source, is_mc=bool(is_mc))


def _cohort_identity(samples, *, start_index, limit):
    """Return the immutable identity of one resumable manifest slice."""

    return {
        "selection_contract": "fixed_manifest_slice_v1",
        "start_index": int(start_index),
        "limit": int(limit),
        "selected_count": len(samples),
        "selected_image_ids": [str(sample["image_id"]) for sample in samples],
        "selected_sample_indices": [int(sample["sample_index"]) for sample in samples],
    }


def _select_exact_cohort(samples, *, start_index, limit):
    """Select a fixed manifest slice without silently truncating positive limits."""

    if start_index < 0 or limit < 0:
        raise ValueError("start_index and limit must be non-negative")
    total = len(samples)
    if limit > 0 and start_index + limit > total:
        raise ValueError(
            "requested exact Qwen cohort exceeds manifest length: "
            f"start_index={start_index}, limit={limit}, rows={total}"
        )
    stop = None if limit == 0 else start_index + limit
    selected = samples[start_index:stop]
    if not selected:
        raise ValueError(
            "selected manifest range is empty: "
            f"start_index={start_index}, limit={limit}, rows={total}"
        )
    return selected


def _validate_manifest_rows(samples, *, dataset, is_mc):
    """Validate the manifest's identity and list-order contract."""

    if not isinstance(samples, list):
        raise ValueError("Qwen manifest must be a JSON list")
    image_ids = []
    sample_indices = []
    for position, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ValueError(f"Qwen manifest row {position} must be an object")
        raw_image_id = sample.get("image_id")
        image_id = "" if raw_image_id is None else str(raw_image_id)
        if not image_id.strip():
            raise ValueError(f"Qwen manifest row {position} has an empty image_id")
        raw_index = sample.get("sample_index")
        if isinstance(raw_index, bool) or not isinstance(raw_index, int):
            raise ValueError(
                f"Qwen manifest row {position} has invalid sample_index: {raw_index!r}"
            )
        sample_index = raw_index
        image_ids.append(image_id)
        sample_indices.append(sample_index)

    if len(image_ids) != len(set(image_ids)):
        raise ValueError("Qwen manifest contains duplicate image_id values")
    if len(sample_indices) != len(set(sample_indices)):
        raise ValueError("Qwen manifest contains duplicate sample_index values")
    expected_indices = list(range(len(samples)))
    if sample_indices != expected_indices:
        raise ValueError(
            "Qwen manifest sample_index must equal zero-based manifest order; "
            f"first_actual={sample_indices[:10]} first_expected={expected_indices[:10]}"
        )

    for position, sample in enumerate(samples):
        row_dataset = sample.get("dataset")
        if row_dataset != dataset:
            raise ValueError(
                f"Qwen manifest row {position} dataset mismatch: "
                f"expected {dataset!r}, got {row_dataset!r}"
            )
        question = sample.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Qwen manifest row {position} has an empty question")
        references = sample.get("reference_answers")
        if (
            not isinstance(references, list)
            or not references
            or any(not isinstance(value, str) or not value.strip() for value in references)
        ):
            raise ValueError(
                f"Qwen manifest row {position} requires non-empty reference_answers"
            )
        if is_mc:
            options = sample.get("options")
            if (
                not isinstance(options, list)
                or not 2 <= len(options) <= 6
                or any(not isinstance(value, str) or not value.strip() for value in options)
            ):
                raise ValueError(
                    f"Qwen MC manifest row {position} requires 2--6 non-empty options"
                )
            answer = references[0].strip().upper()
            valid_letters = tuple(chr(ord("A") + index) for index in range(len(options)))
            if answer not in valid_letters:
                raise ValueError(
                    f"Qwen MC manifest row {position} reference_answers[0] must be "
                    f"one of {valid_letters}"
                )
    return image_ids, dict(zip(image_ids, sample_indices))


def _assert_results_separate_from_manifest_images(
    results_dir,
    samples,
    *,
    data_root,
    dataset,
):
    """Protect the declared data tree and every resolved manifest image parent."""

    if not data_root:
        raise ValueError("Qwen data root is required")
    expanded_data_root = Path(data_root).expanduser()
    if not expanded_data_root.is_absolute():
        raise ValueError("Qwen data root must be an absolute path")
    resolved_data_root = expanded_data_root.resolve()
    dataset = safe_filename_component(dataset, label="dataset")
    canonical_dataset_root = (resolved_data_root / dataset).resolve()
    protected = {}
    for position, sample in enumerate(samples):
        if not isinstance(sample, dict) or not sample.get("image_path"):
            raise ValueError(
                f"Qwen manifest row {position} lacks a non-empty image_path"
            )
        raw_image_path = sample["image_path"]
        if not isinstance(raw_image_path, str):
            raise ValueError(
                f"Qwen manifest row {position} image_path must be an absolute string"
            )
        expanded_image_path = Path(raw_image_path).expanduser()
        if not expanded_image_path.is_absolute():
            raise ValueError(
                f"Qwen manifest row {position} image_path must be absolute: "
                f"{raw_image_path!r}"
            )
        image_path = expanded_image_path.resolve()
        if image_path == canonical_dataset_root or not image_path.is_relative_to(
            canonical_dataset_root
        ):
            raise ValueError(
                f"Qwen manifest image[{position}] is outside canonical dataset root: "
                f"{image_path} not under {canonical_dataset_root}"
            )
        image_parent = image_path.parent
        protected[f"manifest image directory[{position}]"] = image_parent
    protected["Qwen data root"] = resolved_data_root
    return assert_output_separate(results_dir, protected)


def _require_exact_completed_cohort(successes, completed, expected, *, cells_per_group):
    """Reject both stale extra groups and incomplete selected groups."""

    observed = set(successes)
    extra_groups = sorted(observed - expected)
    if extra_groups:
        raise RuntimeError(
            f"Qwen result contains {len(extra_groups)} group(s) outside the "
            f"immutable selected cohort; examples={extra_groups[:5]}"
        )
    missing_groups = sorted(expected - completed)
    if missing_groups:
        raise RuntimeError(
            f"Qwen run incomplete: {len(missing_groups)} selected sample/config groups "
            f"lack all {cells_per_group} success cells; "
            f"examples={missing_groups[:5]}"
        )


def _is_exact_complete_resume(successes, completed, expected, *, cells_per_group):
    """Validate resume scope and report whether model execution is unnecessary."""

    extra_groups = sorted(set(successes) - expected)
    if extra_groups:
        raise RuntimeError(
            f"Qwen result contains {len(extra_groups)} group(s) outside the "
            f"immutable selected cohort; examples={extra_groups[:5]}"
        )
    if completed != expected:
        return False
    _require_exact_completed_cohort(
        successes,
        completed,
        expected,
        cells_per_group=cells_per_group,
    )
    return True


def main():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["TextVQA_Open", "VQAv2_Open", "ScienceQA_MC", "VQAv2_MC"])
    ap.add_argument(
        "--configs",
        default=FORMAL_CONFIG_NAME,
        help=f"frozen formal configuration; must be exactly {FORMAL_CONFIG_NAME}",
    )
    ap.add_argument("--limit", type=int, default=0, help="sample count; 0 means all rows from --start-index")
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--output-prefix", default="qwen35_v6_4comp")
    ap.add_argument("--manifest-dir", default=os.environ.get("FATA_QWEN_MANIFEST_ROOT"))
    ap.add_argument("--manifest-suffix", default="n1000")
    ap.add_argument("--data-root", default=os.environ.get("FATA_DATA_ROOT"))
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--model-path", default=os.environ.get("FATA_QWEN_MODEL"))
    ap.add_argument("--output-root", default=os.environ.get("FATA_OUTPUT_ROOT"))
    args = ap.parse_args()

    if args.limit < 0 or args.start_index < 0:
        ap.error("--limit and --start-index must be non-negative")
    try:
        args.output_prefix = safe_filename_component(args.output_prefix, label="output prefix")
        args.manifest_suffix = safe_filename_component(args.manifest_suffix, label="manifest suffix")
    except ValueError as error:
        ap.error(str(error))

    config_names = [c.strip() for c in args.configs.split(",") if c.strip()]
    if config_names != [FORMAL_CONFIG_NAME]:
        ap.error(
            "--configs must select exactly the frozen formal configuration: "
            f"{FORMAL_CONFIG_NAME}"
        )

    if (
        not args.model_path
        or not args.manifest_dir
        or not args.data_root
        or not (args.results_dir or args.output_root)
    ):
        ap.error(
            "provide --model-path, --manifest-dir, --data-root, and "
            "--results-dir or --output-root"
        )
    protected_inputs = {
        "Qwen model": args.model_path,
        "manifest directory": args.manifest_dir,
        "Qwen data root": args.data_root,
    }
    if args.results_dir is None:
        derived_output_root = assert_output_separate(args.output_root, protected_inputs)
        results_root = resolve_owned_output_path(derived_output_root, "qwen35")
    else:
        derived_output_root = None
        results_root = assert_output_separate(args.results_dir, protected_inputs)
    args.results_dir = str(results_root)
    assert_output_separate(
        results_root,
        {
            "Qwen model": args.model_path,
            "manifest directory": args.manifest_dir,
            "Qwen data root": args.data_root,
        },
    )

    ds = args.dataset
    is_mc = "MC" in ds
    max_tok = 8 if is_mc else 16
    try:
        formal_config = _validated_formal_config(is_mc=is_mc)
    except RuntimeError as error:
        ap.error(str(error))
    configs = {FORMAL_CONFIG_NAME: formal_config}
    if tuple(formal_config.method_universe) != tuple(METHODS):
        ap.error("only the frozen four-compressor method universe is allowed")
    serialized_configs = {name: asdict(configs[name]) for name in config_names}

    csv_name = f"{args.output_prefix}_{ds}.csv"
    csv_path = resolve_owned_output_path(results_root, csv_name)
    meta_name = str(Path(csv_name).with_suffix(".meta.json"))
    meta_path = resolve_owned_output_path(results_root, meta_name)

    manifest_path = os.path.join(args.manifest_dir, f"{ds}_{args.manifest_suffix}.json")
    with open(manifest_path) as f:
        all_samples = json.load(f)
    try:
        _all_sample_ids, sample_index_by_id = _validate_manifest_rows(
            all_samples,
            dataset=ds,
            is_mc=is_mc,
        )
    except ValueError as error:
        ap.error(str(error))
    _assert_results_separate_from_manifest_images(
        args.results_dir,
        all_samples,
        data_root=args.data_root,
        dataset=ds,
    )
    try:
        samples = _select_exact_cohort(
            all_samples,
            start_index=args.start_index,
            limit=args.limit,
        )
    except ValueError as error:
        ap.error(str(error))
    sample_ids = [str(sample["image_id"]) for sample in samples]
    expected_seed_by_id = {
        str(sample["image_id"]): deterministic_seed(
            ds,
            str(sample["image_id"]),
            sample.get("sample_seed", 0),
        )
        for sample in samples
    }
    scoring_targets_by_id = {
        str(sample["image_id"]): {
            "is_mc": is_mc,
            "reference_answers": [
                str(value) for value in sample.get("reference_answers", [])
            ],
        }
        for sample in samples
    }

    if derived_output_root is not None:
        created_results_root = create_owned_output_directory(
            derived_output_root, "qwen35"
        )
        if created_results_root != results_root:
            raise RuntimeError("derived Qwen results directory changed during validation")
    else:
        results_root.mkdir(parents=True, exist_ok=True)
        if not results_root.is_dir():
            ap.error(f"--results-dir is not a directory: {results_root}")
    csv_path = resolve_owned_output_path(results_root, csv_name)
    ensure_run_contract(
        meta_path,
        {
            "schema_version": 2,
            "runtime": "qwen35_v6_4comp",
            "device_map": args.device_map,
            "dataset": ds,
            "manifest_sha256": sha256_file(manifest_path),
            "manifest_images": explicit_image_identity(all_samples),
            "cohort": _cohort_identity(
                samples,
                start_index=args.start_index,
                limit=args.limit,
            ),
            "model": artifact_identity(args.model_path),
            "configs": serialized_configs,
            "methods": list(METHODS),
            "budgets": list(BUDGETS),
            "fieldnames": FIELDNAMES,
        },
        result_path=csv_path,
    )

    successful_cells = read_successful_cells(
        csv_path,
        methods=METHODS,
        budgets=BUDGETS,
        expected_header=FIELDNAMES,
        expected_dataset=ds,
        allowed_configs=config_names,
        allowed_image_ids=sample_ids,
        sample_index_by_id=sample_index_by_id,
        expected_seed_by_id=expected_seed_by_id,
        config_contracts=serialized_configs,
        scoring_targets_by_id=scoring_targets_by_id,
    )
    done = completed_groups(
        successful_cells,
        methods=METHODS,
        budgets=BUDGETS,
    )
    expected_groups = {
        (str(sample["image_id"]), config_name)
        for sample in samples
        for config_name in config_names
    }
    if _is_exact_complete_resume(
        successful_cells,
        done,
        expected_groups,
        cells_per_group=len(METHODS) * len(BUDGETS),
    ):
        print(f"[{ds}] DONE (validated resume; model load skipped) -> {csv_path}", flush=True)
        return

    print(f"[{ds}] loading model (device_map={args.device_map})...", flush=True)
    v6model = V6Model(model_path=args.model_path, device_map=args.device_map, enable_gradient_checkpointing=True)
    dev = next(v6model.model.parameters()).device
    print(f"[{ds}] model ready on {dev}", flush=True)

    csv_path = resolve_owned_output_path(results_root, csv_name)
    file_exists = csv_path.exists() and csv_path.stat().st_size > 0
    outf = open(csv_path, "a", newline="")
    writer = csv.DictWriter(outf, fieldnames=FIELDNAMES)
    if not file_exists:
        writer.writeheader()

    for si, sample in enumerate(samples):
        iid = str(sample["image_id"])
        print(f"[{ds}] sample {si + 1}/{len(samples)} {iid}", flush=True)

        try:
            inputs = v6model.prepare_sample(sample, is_mc)
            for k in inputs:
                if isinstance(inputs[k], torch.Tensor):
                    inputs[k] = inputs[k].to(dev)
        except Exception as e:
            for cfg_name in configs:
                if (iid, cfg_name) in done:
                    continue
                writer.writerow({"dataset": ds, "config": cfg_name, "sample_index": sample["sample_index"],
                                 "image_id": iid, "status": "prep_error", "error": str(e)[:300]})
            outf.flush()
            continue

        pv = inputs["pixel_values"].float().detach()

        for cfg_name, cfg in configs.items():
            if (iid, cfg_name) in done:
                print(f"  skip {cfg_name} (done)", flush=True)
                continue
            seed = deterministic_seed(ds, iid, sample.get("sample_seed", 0))
            answer_ids = answer_token_ids(v6model.processor, sample, is_mc, cfg.max_answer_tokens)

            torch.cuda.empty_cache(); gc.collect()
            t0 = time.time()
            try:
                from .v6_core import attack_v6
                result = attack_v6(v6model, sample, inputs, cfg, seed, answer_ids=answer_ids)
                attack_s = time.time() - t0
            except Exception as e:
                import traceback
                writer.writerow({"dataset": ds, "config": cfg_name, "sample_index": sample["sample_index"],
                                 "image_id": iid, "status": "attack_error",
                                 "error": str(e)[:300] + " | " + traceback.format_exc()[-200:]})
                outf.flush()
                print(f"  {cfg_name} ATTACK ERROR: {e}", flush=True)
                continue

            if result["status"] != "success":
                writer.writerow({"dataset": ds, "config": cfg_name, "sample_index": sample["sample_index"],
                                 "image_id": iid, "status": result["status"],
                                 "delta_linf": result.get("final_delta_linf", 0),
                                 "seed": seed, "initial_delta_sha256": result.get("initial_delta_sha256", ""),
                                 "attack_seconds": round(attack_s, 1),
                                 "error": result.get("error", "")})
                outf.flush()
                print(f"  {cfg_name} status={result['status']}", flush=True)
                continue

            adv_pv = result["adv_pv"].to(dev)
            try:
                _eval_and_write(
                    v6model,
                    sample,
                    inputs,
                    pv,
                    adv_pv,
                    ds,
                    cfg_name,
                    cfg,
                    seed,
                    result,
                    is_mc,
                    max_tok,
                    attack_s,
                    writer,
                    successful_cells.setdefault((iid, cfg_name), set()),
                )
                outf.flush()
                print(f"  {cfg_name} ok (linf={result['final_delta_linf']:.5f}, {attack_s:.0f}s)", flush=True)
            except Exception as e:
                import traceback
                writer.writerow({"dataset": ds, "config": cfg_name, "sample_index": sample["sample_index"],
                                 "image_id": iid, "status": "eval_error",
                                 "error": str(e)[:300] + " | " + traceback.format_exc()[-200:]})
                outf.flush()
                print(f"  {cfg_name} EVAL ERROR: {e}", flush=True)

    outf.close()
    final_successes = read_successful_cells(
        csv_path,
        methods=METHODS,
        budgets=BUDGETS,
        expected_header=FIELDNAMES,
        expected_dataset=ds,
        allowed_configs=config_names,
        allowed_image_ids=sample_ids,
        sample_index_by_id=sample_index_by_id,
        expected_seed_by_id=expected_seed_by_id,
        config_contracts=serialized_configs,
        scoring_targets_by_id=scoring_targets_by_id,
    )
    final_done = completed_groups(
        final_successes,
        methods=METHODS,
        budgets=BUDGETS,
    )
    _require_exact_completed_cohort(
        final_successes,
        final_done,
        expected_groups,
        cells_per_group=len(METHODS) * len(BUDGETS),
    )
    print(f"[{ds}] DONE -> {csv_path}", flush=True)
    del v6model
    gc.collect()
    torch.cuda.empty_cache()


def _eval_and_write(v6model, sample, inputs, pv, adv_pv, ds, cfg_name, cfg, seed, result,
                    is_mc, max_tok, attack_s, writer, completed_cells):
    refs = sample.get("reference_answers", [])
    gt_letter = refs[0].strip().upper() if (is_mc and refs) else ""

    clean_full_raw = _clean(
        v6model.generate_answer(pv, inputs, None, 1.0, max_tok, is_mc)
    )
    fata_full_raw = _clean(
        v6model.generate_answer(adv_pv, inputs, None, 1.0, max_tok, is_mc)
    )
    clean_full_correct = _score(clean_full_raw, refs, gt_letter, is_mc)
    fata_full_correct = _score(fata_full_raw, refs, gt_letter, is_mc)

    base = {
        "dataset": ds, "config": cfg_name, "sample_index": sample["sample_index"],
        "image_id": str(sample["image_id"]), "N_full": result["N_full"],
        "clean_full_raw": clean_full_raw, "fata_full_raw": fata_full_raw,
        "clean_full_correct": clean_full_correct, "fata_full_correct": fata_full_correct,
        "delta_linf": result["final_delta_linf"], "seed": seed,
        "initial_delta_sha256": result["initial_delta_sha256"], "delta_sha256": result["delta_sha256"],
        "epsilon": cfg.epsilon, "alpha": cfg.alpha, "steps": cfg.steps,
        "task_mode": cfg.task_mode, "w_task": cfg.w_task, "w_rank": cfg.w_rank,
        "w_comp": cfg.w_comp, "w_full": cfg.w_full, "w_distill": cfg.w_distill,
        "use_projection": int(cfg.use_projection),
        "full_clean_GT_loss": result["full_clean_GT_loss"],
        "full_adv_GT_loss": result["full_adv_GT_loss"],
        "compressed_adv_GT_loss": result["compressed_adv_GT_loss"],
        "full_logit_KL": result["full_logit_KL"],
        "mean_grad_cos_comp_full": result["mean_grad_cos_comp_full"],
        "projection_trigger_rate": result["projection_trigger_rate"],
        "attack_seconds": round(attack_s, 1),
        "status": "success",
    }

    for comp_name in cfg.method_universe:
        comp_cls = COMPRESSOR_CLS[comp_name]
        for bl in BUDGETS:
            cell = (comp_name, bl)
            if cell in completed_cells:
                continue
            ratio = 1.0 if bl == "Full" else cfg.ratio_for(bl)
            if bl == "Full":
                row = dict(base)
                row.update({"compressor": comp_name, "retention_label": "Full", "retention_ratio": 1.0,
                            "K_actual": result["N_full"],
                            "clean_compressed_raw": clean_full_raw, "fata_compressed_raw": fata_full_raw,
                            "clean_compressed_correct": clean_full_correct, "fata_compressed_correct": fata_full_correct})
                writer.writerow(row)
                completed_cells.add(cell)
                continue
            cc_raw = _clean(
                v6model.generate_answer(pv, inputs, comp_cls, ratio, max_tok, is_mc)
            )
            fc_raw = _clean(
                v6model.generate_answer(adv_pv, inputs, comp_cls, ratio, max_tok, is_mc)
            )
            K = max(1, round(result["N_full"] * ratio))
            row = dict(base)
            row.update({"compressor": comp_name, "retention_label": bl, "retention_ratio": ratio,
                        "K_actual": K,
                        "clean_compressed_raw": cc_raw, "fata_compressed_raw": fc_raw,
                        "clean_compressed_correct": _score(cc_raw, refs, gt_letter, is_mc),
                        "fata_compressed_correct": _score(fc_raw, refs, gt_letter, is_mc)})
            writer.writerow(row)
            completed_cells.add(cell)


def _score(raw, refs, gt_letter, is_mc):
    if is_mc:
        return strict_score_mc(raw, gt_letter)
    return strict_score_open(raw, refs)


def _clean(raw):
    return " ".join(str(raw).split())[:200]


if __name__ == "__main__":
    main()
