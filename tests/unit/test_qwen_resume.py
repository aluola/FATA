from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest
import torch

from fata.runtimes.qwen35.resume import completed_groups, read_successful_cells
from fata.runtimes.qwen35.v6_core import (
    _final_diagnostic_status,
    delta_sha256,
    deterministic_seed,
    strict_score_mc,
    strict_score_open,
)
from fata.runtimes.qwen35.run_qwen35_v6 import (
    FORMAL_FROZEN_CONFIG,
    _assert_results_separate_from_manifest_images,
    _cohort_identity,
    _formal_config_payload,
    _is_exact_complete_resume,
    _require_exact_completed_cohort,
    _select_exact_cohort,
    _validated_formal_config,
    _validate_manifest_rows,
    main as qwen_main,
)
from fata.runtimes.qwen35.v6_core import FORMAL_CONFIG_NAME, FORMAL_CONFIGS
from fata.utils.run_contract import ensure_run_contract


METHODS = ("VisionZIP", "VisPruner", "PruMerge", "FlowCut")
BUDGETS = ("Full", "1/3", "Practical")
FIELDS = (
    "dataset",
    "config",
    "sample_index",
    "image_id",
    "compressor",
    "retention_label",
    "retention_ratio",
    "N_full",
    "K_actual",
    "clean_full_raw",
    "fata_full_raw",
    "clean_full_correct",
    "fata_full_correct",
    "clean_compressed_raw",
    "fata_compressed_raw",
    "clean_compressed_correct",
    "fata_compressed_correct",
    "delta_linf",
    "seed",
    "initial_delta_sha256",
    "delta_sha256",
    "epsilon",
    "alpha",
    "steps",
    "task_mode",
    "w_task",
    "w_rank",
    "w_comp",
    "w_full",
    "w_distill",
    "use_projection",
    "full_clean_GT_loss",
    "full_adv_GT_loss",
    "compressed_adv_GT_loss",
    "full_logit_KL",
    "mean_grad_cos_comp_full",
    "projection_trigger_rate",
    "attack_seconds",
    "status",
    "error",
)
CONFIG = {
    "epsilon": 2 / 255,
    "alpha": 0.5 / 255,
    "steps": 100,
    "task_mode": "ground_truth",
    "w_task": 1.5,
    "w_rank": 0.25,
    "w_comp": 0.05,
    "w_full": 1.0,
    "w_distill": 1.0,
    "use_projection": True,
    "is_mc": False,
}


def _write(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _row(
    method,
    budget,
    status="success",
    *,
    dataset="TextVQA_Open",
    n_full=576,
):
    is_mc = dataset.endswith("_MC")
    if is_mc:
        clean_raw, fata_raw = "A", "B"
        clean_score = strict_score_mc(clean_raw, "A")
        fata_score = strict_score_mc(fata_raw, "A")
    else:
        clean_raw, fata_raw = "cat", "dog"
        references = ["cat"]
        clean_score = strict_score_open(clean_raw, references)
        fata_score = strict_score_open(fata_raw, references)
    if budget == "Full":
        ratio = 1.0
    elif budget == "1/3":
        ratio = 1 / 3
    else:
        ratio = 1 / 18 if is_mc else 1 / 9
    return {
        "dataset": dataset,
        "config": "formal",
        "sample_index": 7,
        "image_id": "sample-1",
        "compressor": method,
        "retention_label": budget,
        "retention_ratio": ratio,
        "N_full": n_full,
        "K_actual": max(1, round(n_full * ratio)),
        "clean_full_raw": clean_raw,
        "fata_full_raw": fata_raw,
        "clean_full_correct": clean_score,
        "fata_full_correct": fata_score,
        "clean_compressed_raw": clean_raw,
        "fata_compressed_raw": fata_raw,
        "clean_compressed_correct": clean_score,
        "fata_compressed_correct": fata_score,
        "delta_linf": 0.007,
        "seed": deterministic_seed(dataset, "sample-1", 0),
        "initial_delta_sha256": delta_sha256(torch.zeros(1, dtype=torch.float32)),
        "delta_sha256": delta_sha256(torch.ones(1, dtype=torch.float32)),
        "epsilon": CONFIG["epsilon"],
        "alpha": CONFIG["alpha"],
        "steps": CONFIG["steps"],
        "task_mode": CONFIG["task_mode"],
        "w_task": CONFIG["w_task"],
        "w_rank": CONFIG["w_rank"],
        "w_comp": CONFIG["w_comp"],
        "w_full": CONFIG["w_full"],
        "w_distill": CONFIG["w_distill"],
        "use_projection": 1,
        "full_clean_GT_loss": 1.0,
        "full_adv_GT_loss": 1.2,
        "compressed_adv_GT_loss": 1.3,
        "full_logit_KL": 0.1,
        "mean_grad_cos_comp_full": 0.25,
        "projection_trigger_rate": 0.5,
        "attack_seconds": 12.3,
        "status": status,
        "error": "",
    }


def test_partial_group_is_not_complete(tmp_path):
    path = tmp_path / "partial.csv"
    _write(path, [_row("VisionZIP", "Full")])
    successes = read_successful_cells(path, methods=METHODS, budgets=BUDGETS)
    assert completed_groups(successes, methods=METHODS, budgets=BUDGETS) == set()
    assert successes[("sample-1", "formal")] == {("VisionZIP", "Full")}


def test_exact_twelve_cell_group_is_complete(tmp_path):
    path = tmp_path / "complete.csv"
    _write(path, [_row(method, budget) for method in METHODS for budget in BUDGETS])
    successes = read_successful_cells(path, methods=METHODS, budgets=BUDGETS)
    assert completed_groups(successes, methods=METHODS, budgets=BUDGETS) == {
        ("sample-1", "formal")
    }


@pytest.mark.parametrize(
    "rows",
    [
        [_row("VisionZIP", "Full"), _row("VisionZIP", "Full")],
        [_row("DivPrune", "Full")],
        [_row("VisionZIP", "quarter")],
    ],
)
def test_ambiguous_or_out_of_contract_success_fails_closed(tmp_path, rows):
    path = tmp_path / "invalid.csv"
    _write(path, rows)
    with pytest.raises(ValueError):
        read_successful_cells(path, methods=METHODS, budgets=BUDGETS)


def test_failure_row_does_not_complete_or_conflict(tmp_path):
    path = tmp_path / "retry.csv"
    _write(path, [_row("VisionZIP", "Full", "eval_error")])
    assert read_successful_cells(path, methods=METHODS, budgets=BUDGETS) == {}


@pytest.mark.parametrize("status", ("success", "eval_error"))
def test_rows_outside_fixed_cohort_fail_closed_before_resume(tmp_path, status):
    path = tmp_path / "stale-slice.csv"
    row = _row("VisionZIP", "Full", status)
    row["image_id"] = "old-slice-image"
    _write(path, [row])
    with pytest.raises(ValueError, match="outside the immutable selected cohort"):
        read_successful_cells(
            path,
            methods=METHODS,
            budgets=BUDGETS,
            allowed_image_ids={"sample-1"},
        )


def test_strict_resume_binds_header_manifest_and_hyperparameters(tmp_path):
    path = tmp_path / "strict.csv"
    row = _row("VisionZIP", "Full")
    _write(path, [row])
    kwargs = {
        "methods": METHODS,
        "budgets": BUDGETS,
        "expected_header": FIELDS,
        "expected_dataset": "TextVQA_Open",
        "allowed_configs": ("formal",),
        "sample_index_by_id": {"sample-1": 7},
        "expected_seed_by_id": {
            "sample-1": deterministic_seed("TextVQA_Open", "sample-1", 0)
        },
        "config_contracts": {"formal": CONFIG},
        "scoring_targets_by_id": {
            "sample-1": {"is_mc": False, "reference_answers": ["cat"]}
        },
    }
    assert read_successful_cells(path, **kwargs)[("sample-1", "formal")]

    row["steps"] = 99
    _write(path, [row])
    with pytest.raises(ValueError, match="steps"):
        read_successful_cells(path, **kwargs)


def test_strict_resume_rejects_malformed_width(tmp_path):
    path = tmp_path / "wide.csv"
    path.write_text(
        ",".join(FIELDS) + "\n" + ",".join(["x"] * (len(FIELDS) + 1)) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="malformed row width"):
        read_successful_cells(
            path,
            methods=METHODS,
            budgets=BUDGETS,
            expected_header=FIELDS,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("task_mode", "", "empty success fields"),
        ("full_clean_GT_loss", "nan", "non-finite full_clean_GT_loss"),
        ("initial_delta_sha256", "not-a-hash", "initial_delta_sha256"),
        ("seed", -1, "seed"),
        ("delta_linf", 0.1, "delta_linf"),
        ("attack_seconds", 0.0, "attack_seconds"),
        ("projection_trigger_rate", 1.1, "projection_trigger_rate"),
        ("retention_ratio", 0.5, "retention_ratio"),
        ("K_actual", 5, "K_actual"),
    ),
)
def test_invalid_success_evidence_is_rejected(tmp_path, field, value, message):
    path = tmp_path / "bad-success.csv"
    row = _row("VisionZIP", "Practical")
    row[field] = value
    _write(path, [row])
    with pytest.raises(ValueError, match=message):
        read_successful_cells(path, methods=METHODS, budgets=BUDGETS)


def test_mc_practical_budget_uses_one_eighteenth(tmp_path):
    path = tmp_path / "mc.csv"
    row = _row("VisionZIP", "Practical", dataset="ScienceQA_MC")
    _write(path, [row])
    contract = {**CONFIG, "is_mc": True}
    assert read_successful_cells(
        path,
        methods=METHODS,
        budgets=BUDGETS,
        expected_dataset="ScienceQA_MC",
        config_contracts={"formal": contract},
    )[("sample-1", "formal")]


def test_empty_raw_answers_are_valid_immediate_eos_results(tmp_path):
    path = tmp_path / "empty-answers.csv"
    row = _row("VisionZIP", "Full")
    for field in (
        "clean_full_raw",
        "fata_full_raw",
        "clean_compressed_raw",
        "fata_compressed_raw",
    ):
        row[field] = ""
    _write(path, [row])
    assert read_successful_cells(path, methods=METHODS, budgets=BUDGETS) == {
        ("sample-1", "formal"): {("VisionZIP", "Full")}
    }


@pytest.mark.parametrize("value", ("nan", "inf", -0.1, 1.1))
def test_open_correctness_requires_finite_unit_interval(tmp_path, value):
    path = tmp_path / "bad-correctness.csv"
    row = _row("VisionZIP", "Full")
    row["clean_full_correct"] = value
    _write(path, [row])
    with pytest.raises(ValueError, match="clean_full_correct"):
        read_successful_cells(path, methods=METHODS, budgets=BUDGETS)


def test_mc_correctness_is_binary(tmp_path):
    path = tmp_path / "fractional-mc-score.csv"
    row = _row("VisionZIP", "Full", dataset="ScienceQA_MC")
    row["clean_full_correct"] = strict_score_open("cat", ["cat"])
    _write(path, [row])
    with pytest.raises(ValueError, match="non-binary MC score"):
        read_successful_cells(path, methods=METHODS, budgets=BUDGETS)


def test_float32_two_over_255_delta_is_resumable(tmp_path):
    path = tmp_path / "float32-epsilon.csv"
    row = _row("VisionZIP", "Full")
    row["delta_linf"] = float(torch.tensor(2 / 255, dtype=torch.float32))
    _write(path, [row])
    assert read_successful_cells(path, methods=METHODS, budgets=BUDGETS)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("clean_compressed_raw", "forged full prediction"),
        ("fata_compressed_correct", 1.0),
    ),
)
def test_full_cell_must_duplicate_full_predictions_and_scores(
    tmp_path, field, value
):
    path = tmp_path / "forged-full-cell.csv"
    row = _row("VisionZIP", "Full")
    row[field] = value
    _write(path, [row])
    with pytest.raises(ValueError, match="Full .* different"):
        read_successful_cells(path, methods=METHODS, budgets=BUDGETS)


def test_strict_resume_recomputes_correctness_from_raw_and_manifest(tmp_path):
    path = tmp_path / "forged-correctness.csv"
    row = _row("VisionZIP", "Practical")
    row["clean_compressed_correct"] = 0.0
    _write(path, [row])
    with pytest.raises(ValueError, match="inconsistent with .* manifest target"):
        read_successful_cells(
            path,
            methods=METHODS,
            budgets=BUDGETS,
            scoring_targets_by_id={
                "sample-1": {"is_mc": False, "reference_answers": ["cat"]}
            },
        )


def test_success_row_requires_empty_error_field(tmp_path):
    path = tmp_path / "success-with-error.csv"
    row = _row("VisionZIP", "Full")
    row["error"] = "stale exception"
    _write(path, [row])
    with pytest.raises(ValueError, match="non-empty error"):
        read_successful_cells(path, methods=METHODS, budgets=BUDGETS)


def test_projection_trigger_rate_is_zero_when_projection_is_disabled(tmp_path):
    path = tmp_path / "impossible-projection-rate.csv"
    row = _row("VisionZIP", "Full")
    row["use_projection"] = 0
    row["projection_trigger_rate"] = 0.1
    _write(path, [row])
    disabled = {**CONFIG, "use_projection": False}
    with pytest.raises(ValueError, match="projection_trigger_rate must be zero"):
        read_successful_cells(
            path,
            methods=METHODS,
            budgets=BUDGETS,
            config_contracts={"formal": disabled},
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("seed", deterministic_seed("TextVQA_Open", "sample-1", 0) + 1),
        ("initial_delta_sha256", "c" * 16),
        ("delta_sha256", "d" * 16),
        ("clean_full_raw", "different full prediction"),
        ("clean_full_correct", 0.0),
        ("full_clean_GT_loss", 2.0),
    ),
)
def test_mixed_attack_success_cells_are_rejected(tmp_path, field, value):
    path = tmp_path / "mixed.csv"
    first = _row("VisionZIP", "Full")
    second = _row("VisionZIP", "1/3")
    second[field] = value
    _write(path, [first, second])
    with pytest.raises(ValueError, match="inconsistent"):
        read_successful_cells(path, methods=METHODS, budgets=BUDGETS)


def test_mixed_n_full_is_rejected_even_when_cell_k_is_self_consistent(tmp_path):
    path = tmp_path / "mixed-n.csv"
    first = _row("VisionZIP", "Full")
    second = _row("VisionZIP", "1/3", n_full=600)
    _write(path, [first, second])
    with pytest.raises(ValueError, match="inconsistent"):
        read_successful_cells(path, methods=METHODS, budgets=BUDGETS)


def test_attack_seconds_may_differ_across_legitimate_retry_cells(tmp_path):
    path = tmp_path / "retry-time.csv"
    first = _row("VisionZIP", "Full")
    second = _row("VisionZIP", "1/3")
    second["attack_seconds"] = 99.0
    _write(path, [first, second])
    successes = read_successful_cells(path, methods=METHODS, budgets=BUDGETS)
    assert successes[("sample-1", "formal")] == {
        ("VisionZIP", "Full"),
        ("VisionZIP", "1/3"),
    }


def test_final_diagnostic_classifier_never_blesses_nan():
    values = {
        "delta_linf": float(torch.tensor(2 / 255, dtype=torch.float32)),
        "epsilon": 2 / 255,
        "full_clean_gt_loss": 1.0,
        "full_adv_gt_loss": 1.1,
        "compressed_adv_gt_loss": 1.2,
        "full_logit_kl": 0.1,
        "mean_grad_cos_comp_full": 0.25,
        "projection_trigger_rate": 0.5,
    }
    assert _final_diagnostic_status(**values) == ("success", "")
    values["full_adv_gt_loss"] = float("nan")
    status, message = _final_diagnostic_status(**values)
    assert status == "diagnostic_error"
    assert "non-finite" in message


def test_qwen_run_contract_binds_exact_manifest_slice(tmp_path):
    samples = [
        {"image_id": "image-a", "sample_index": 4},
        {"image_id": "image-b", "sample_index": 5},
    ]
    first = _cohort_identity(samples, start_index=4, limit=2)
    assert first["selected_image_ids"] == ["image-a", "image-b"]
    meta = tmp_path / "qwen.meta.json"
    ensure_run_contract(meta, {"cohort": first})
    ensure_run_contract(meta, {"cohort": first})

    changed_slice = _cohort_identity(samples[1:], start_index=5, limit=1)
    with pytest.raises(RuntimeError, match="resume contract mismatch"):
        ensure_run_contract(meta, {"cohort": changed_slice})


def test_qwen_final_cohort_rejects_extra_groups_even_if_expected_is_done():
    expected = {("sample-1", "formal")}
    completed = set(expected)
    successes = {
        ("sample-1", "formal"): {("VisionZIP", "Full")},
        ("old-slice-image", "formal"): {("VisionZIP", "Full")},
    }
    with pytest.raises(RuntimeError, match="outside the immutable selected cohort"):
        _require_exact_completed_cohort(
            successes,
            completed,
            expected,
            cells_per_group=12,
        )


def test_qwen_complete_resume_can_skip_model_load_only_for_exact_cohort():
    expected = {("sample-1", "formal")}
    cells = {(method, budget) for method in METHODS for budget in BUDGETS}
    successes = {("sample-1", "formal"): cells}
    assert _is_exact_complete_resume(
        successes,
        set(expected),
        expected,
        cells_per_group=12,
    )
    assert not _is_exact_complete_resume(
        {("sample-1", "formal"): {("VisionZIP", "Full")}},
        set(),
        expected,
        cells_per_group=12,
    )


def test_qwen_results_path_is_disjoint_from_every_manifest_image(tmp_path):
    data_root = tmp_path / "data"
    image = data_root / "TextVQA_Open" / "images" / "one.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"image")
    samples = [{"image_id": "one", "sample_index": 0, "image_path": str(image)}]

    safe_output = tmp_path / "results"
    assert (
        _assert_results_separate_from_manifest_images(
            safe_output,
            samples,
            data_root=data_root,
            dataset="TextVQA_Open",
        )
        == safe_output
    )
    assert not safe_output.exists()

    for unsafe_output in (
        image,
        image.parent,
        image / "nested-output",
        image.parent / "sibling-output",
        data_root,
        data_root / "results-sibling-of-images",
    ):
        with pytest.raises(
            ValueError,
            match="overlaps protected (manifest image directory|Qwen data root)",
        ):
            _assert_results_separate_from_manifest_images(
                unsafe_output,
                samples,
                data_root=data_root,
                dataset="TextVQA_Open",
            )


def test_qwen_image_path_guard_rejects_incomplete_manifest_row(tmp_path):
    with pytest.raises(ValueError, match="lacks a non-empty image_path"):
        _assert_results_separate_from_manifest_images(
            tmp_path / "results",
            [{"image_id": "missing-path", "sample_index": 0}],
            data_root=tmp_path / "data",
            dataset="TextVQA_Open",
        )


def test_qwen_image_path_guard_rejects_image_outside_declared_data_root(tmp_path):
    with pytest.raises(ValueError, match="outside canonical dataset root"):
        _assert_results_separate_from_manifest_images(
            tmp_path / "results",
            [
                {
                    "image_id": "outside",
                    "sample_index": 0,
                    "image_path": str(tmp_path / "other" / "x.jpg"),
                }
            ],
            data_root=tmp_path / "data",
            dataset="TextVQA_Open",
        )


def test_qwen_exact_cohort_rejects_truncated_positive_limit():
    samples = [{"image_id": "a"}, {"image_id": "b"}, {"image_id": "c"}]
    assert _select_exact_cohort(samples, start_index=1, limit=2) == samples[1:3]
    assert _select_exact_cohort(samples, start_index=1, limit=0) == samples[1:]
    with pytest.raises(ValueError, match="exceeds manifest length"):
        _select_exact_cohort(samples, start_index=2, limit=2)
    with pytest.raises(ValueError, match="selected manifest range is empty"):
        _select_exact_cohort(samples, start_index=3, limit=0)


def test_qwen_image_path_guard_rejects_relative_paths(tmp_path):
    with pytest.raises(ValueError, match="image_path must be absolute"):
        _assert_results_separate_from_manifest_images(
            tmp_path / "results",
            [{"image_id": "relative", "sample_index": 0, "image_path": "images/x.jpg"}],
            data_root=tmp_path / "data",
            dataset="TextVQA_Open",
        )


@pytest.mark.parametrize(
    ("manifest", "message"),
    [
        ({"not": "a list"}, "JSON list"),
        (["not-a-row"], "must be an object"),
        ([{"image_id": "", "sample_index": 0}], "empty image_id"),
        ([{"image_id": "a", "sample_index": "bad"}], "invalid sample_index"),
        (
            [
                {"image_id": "same", "sample_index": 0},
                {"image_id": "same", "sample_index": 1},
            ],
            "duplicate image_id",
        ),
        (
            [
                {"image_id": "a", "sample_index": 0},
                {"image_id": "b", "sample_index": 0},
            ],
            "duplicate sample_index",
        ),
        (
            [
                {"image_id": "a", "sample_index": 1},
                {"image_id": "b", "sample_index": 0},
            ],
            "zero-based manifest order",
        ),
    ],
)
def test_qwen_manifest_identity_rows_fail_closed(manifest, message):
    with pytest.raises(ValueError, match=message):
        _validate_manifest_rows(manifest, dataset="TextVQA_Open", is_mc=False)


@pytest.mark.parametrize("sample_index", ("0", 0.0, True, None))
def test_qwen_manifest_identity_rejects_noninteger_index_types(sample_index):
    row = {
        "image_id": "a",
        "dataset": "TextVQA_Open",
        "sample_index": sample_index,
        "question": "Question A?",
        "reference_answers": ["answer-a"],
    }
    with pytest.raises(ValueError, match="invalid sample_index"):
        _validate_manifest_rows(
            [row],
            dataset="TextVQA_Open",
            is_mc=False,
        )


@pytest.mark.parametrize(
    ("sample", "message"),
    [
        (
            {"image_id": "a", "sample_index": 0, "question": "", "reference_answers": ["x"]},
            "empty question",
        ),
        (
            {"image_id": "a", "sample_index": 0, "question": "q", "reference_answers": []},
            "non-empty reference_answers",
        ),
        (
            {
                "image_id": "a",
                "sample_index": 0,
                "question": "q",
                "reference_answers": ["A"],
                "options": ["only one"],
            },
            "2--6 non-empty options",
        ),
        (
            {
                "image_id": "a",
                "sample_index": 0,
                "question": "q",
                "reference_answers": ["C"],
                "options": ["one", "two"],
            },
            "reference_answers\\[0\\]",
        ),
    ],
)
def test_qwen_manifest_task_fields_fail_closed(sample, message):
    with pytest.raises(ValueError, match=message):
        sample = {"dataset": "ScienceQA_MC", **sample}
        _validate_manifest_rows(
            [sample],
            dataset="ScienceQA_MC",
            is_mc=True,
        )


def test_qwen_manifest_mc_schema_accepts_option_letter_contract():
    image_ids, index_by_id = _validate_manifest_rows(
        [
            {
                "image_id": "mc-1",
                "dataset": "ScienceQA_MC",
                "sample_index": 0,
                "question": "Which?",
                "reference_answers": ["B"],
                "options": ["first", "second", "third"],
            }
        ],
        dataset="ScienceQA_MC",
        is_mc=True,
    )
    assert image_ids == ["mc-1"]
    assert index_by_id == {"mc-1": 0}


def test_qwen_manifest_rejects_dataset_relabeling():
    row = {
        "dataset": "VQAv2_Open",
        "image_id": "vqa-1",
        "sample_index": 0,
        "question": "What?",
        "reference_answers": ["answer"],
    }
    with pytest.raises(ValueError, match="dataset mismatch"):
        _validate_manifest_rows(
            [row],
            dataset="TextVQA_Open",
            is_mc=False,
        )


def test_qwen_cli_requires_explicit_or_environment_data_root(monkeypatch):
    monkeypatch.delenv("FATA_DATA_ROOT", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_qwen35_v6",
            "--dataset",
            "TextVQA_Open",
            "--model-path",
            "/model",
            "--manifest-dir",
            "/manifests",
            "--output-root",
            "/outputs",
        ],
    )
    with pytest.raises(SystemExit) as error:
        qwen_main()
    assert error.value.code == 2


def test_qwen_formal_config_exactly_matches_public_frozen_json():
    frozen_path = (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "qwen35"
        / "QFATA_V6_COMPRESSION_SPECIFIC_4COMP_FROZEN.json"
    )
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    assert frozen["name"] == FORMAL_CONFIG_NAME
    assert set(FORMAL_CONFIGS) == {FORMAL_CONFIG_NAME}
    assert frozen["config"] == FORMAL_FROZEN_CONFIG
    assert _formal_config_payload(FORMAL_CONFIGS[FORMAL_CONFIG_NAME]) == frozen["config"]


def test_qwen_task_mode_copy_does_not_mutate_frozen_config():
    original = FORMAL_CONFIGS[FORMAL_CONFIG_NAME]
    assert original.is_mc is False
    assert _validated_formal_config(is_mc=True).is_mc is True
    assert _validated_formal_config(is_mc=False).is_mc is False
    assert original.is_mc is False


@pytest.mark.parametrize(
    "selection",
    (
        "v5_task_strong_control",
        "v6_a_fullpres_light",
        "v6_a_fullpres_light_4comp,v6_a_fullpres_light_4comp",
        "",
    ),
)
def test_qwen_cli_rejects_every_nonformal_config_selection(
    selection,
    monkeypatch,
    capsys,
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_qwen35_v6",
            "--dataset",
            "TextVQA_Open",
            "--configs",
            selection,
        ],
    )
    with pytest.raises(SystemExit) as error:
        qwen_main()
    assert error.value.code == 2
    assert "must select exactly the frozen formal configuration" in capsys.readouterr().err


def test_qwen_cli_rejects_truncated_cohort_before_creating_output(
    tmp_path,
    monkeypatch,
    capsys,
):
    data_root = tmp_path / "data"
    image = data_root / "TextVQA_Open" / "images" / "one.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"image")
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    (manifest_dir / "TextVQA_Open_n1000.json").write_text(
        json.dumps(
            [
                {
                    "dataset": "TextVQA_Open",
                    "image_id": "one",
                    "sample_index": 0,
                    "image_path": str(image),
                    "question": "What?",
                    "reference_answers": ["answer"],
                }
            ]
        ),
        encoding="utf-8",
    )
    output_root = tmp_path / "outputs"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_qwen35_v6",
            "--dataset",
            "TextVQA_Open",
            "--limit",
            "2",
            "--model-path",
            str(tmp_path / "model"),
            "--manifest-dir",
            str(manifest_dir),
            "--data-root",
            str(data_root),
            "--output-root",
            str(output_root),
        ],
    )
    with pytest.raises(SystemExit) as error:
        qwen_main()
    assert error.value.code == 2
    assert "exceeds manifest length" in capsys.readouterr().err
    assert not output_root.exists()
