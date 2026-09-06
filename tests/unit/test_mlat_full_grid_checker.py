from __future__ import annotations

import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fata.detection.mlatd import check_mlat_full_grid as checker
from fata.detection.mlatd.mlat_core import MLATFeatureEngine


METHOD = "FlowCut"
DATASET = "VQAv2_MC"
ATTACK = "clean_clip"
TOKEN_BUDGET = 64
VISION_LAYERS = (6, 12, 18, 24)
LLM_LAYERS = (8, 16, 24, 32)


def _unit_rows(rows, width):
    result = np.zeros((rows, width), dtype=np.float16)
    for row in range(rows):
        result[row, row % width] = 1.0
    return result


def _valid_grid_item(tmp_path):
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    prefix = (
        "mlat_feat_FlowCut_VQAv2_MC_clean_clip_k64_"
        "start0_limit2_seed0"
    )
    npz_path = result_dir / f"{prefix}.npz"
    image_ids = np.asarray(["images/a.jpg", "images/b.jpg"])
    payload = {
        "sample_idx": np.asarray([0, 1], dtype=np.int64),
        "label": np.asarray([0, 0], dtype=np.int8),
        "image_span_valid": np.asarray([1, 1], dtype=np.int8),
        "image_token_count": np.asarray([576, 576], dtype=np.int32),
        "llm_sequence_length": np.asarray([640, 641], dtype=np.int32),
        "vision_layers": np.asarray(VISION_LAYERS, dtype=np.int16),
        "llm_layers": np.asarray(LLM_LAYERS, dtype=np.int16),
        "image_id": image_ids,
        "proj_in": _unit_rows(2, 3),
        "proj_out": _unit_rows(2, 5),
    }
    for layer in VISION_LAYERS:
        payload[f"vis_l{layer:02d}"] = _unit_rows(2, 3)
    for layer in LLM_LAYERS:
        payload[f"llm_last_l{layer:02d}"] = _unit_rows(2, 5)
        payload[f"llm_image_l{layer:02d}"] = _unit_rows(2, 5)
    np.savez_compressed(npz_path, **payload)

    fieldnames = (
        "sample_idx",
        "Image_ID",
        "Question",
        "dataset",
        "method",
        "attack_for_detection",
        "label",
        "seed",
        "sample_seed",
        "token_budget",
        "token_budget_mode",
        "eps_255",
        "alpha_255",
        "steps",
        "lam",
        "target_k",
        "vision_layers",
        "llm_layers",
        "image_span_valid",
        "image_token_count",
        "llm_sequence_length",
    )
    csv_rows = []
    for index, image_id in enumerate(image_ids.tolist()):
        csv_rows.append(
            {
                "sample_idx": index,
                "Image_ID": image_id,
                "Question": f"question {index}",
                "dataset": DATASET,
                "method": METHOD,
                "attack_for_detection": ATTACK,
                "label": 0,
                "seed": 0,
                "sample_seed": index,
                "token_budget": TOKEN_BUDGET,
                "token_budget_mode": "practical",
                "eps_255": 2.0,
                "alpha_255": 0.5,
                "steps": 100,
                "lam": 1.0,
                "target_k": 64,
                "vision_layers": "6,12,18,24",
                "llm_layers": "8,16,24,32",
                "image_span_valid": 1,
                "image_token_count": 576,
                "llm_sequence_length": 640 + index,
            }
        )
    csv_path = npz_path.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    contract = {
        "schema_version": 1,
        "image_serialization": "png_u8_round_project_linf_v1",
        "runtime": "mlatd_feature_extraction_v1",
        "feature_schema": "quarter_layers_projector_llm_last_image_v1",
        "dataset": DATASET,
        "dataset_mapping_sha256": "0" * 64,
        "dataset_images": {
            "algorithm": "ordered-path-size-file-sha256-v1",
            "count": 2,
            "total_bytes": 20,
            "aggregate_sha256": "1" * 64,
        },
        "method": METHOD,
        "attack": ATTACK,
        "llava_model": {
            "algorithm": "recursive-model-artifact-sha256-v2",
            "resolved_path": "/models/llava",
            "metadata_sha256": {"config.json": "2" * 64},
            "weight_files": [{"relative_path": "model.bin", "size": 1, "sha256": "3" * 64}],
            "index_sha256": {},
        },
        "clip_model": {
            "algorithm": "recursive-model-artifact-sha256-v2",
            "resolved_path": "/models/clip",
            "metadata_sha256": {"config.json": "4" * 64},
            "weight_files": [{"relative_path": "model.bin", "size": 1, "sha256": "5" * 64}],
            "index_sha256": {},
        },
        "seed": 0,
        "eps_255": 2.0,
        "alpha_255": 0.5,
        "steps": 100,
        "lam": 1.0,
        "target_k": 64,
        "token_budget_mode": "practical",
        "token_budget": TOKEN_BUDGET,
        "vision_layers": list(VISION_LAYERS),
        "llm_layers": list(LLM_LAYERS),
        "expected_sample_indices": [0, 1],
        "expected_image_ids": image_ids.tolist(),
        "source_cache_contract": None,
    }
    meta_path = npz_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(contract), encoding="utf-8")

    args = [
        "--result-dir",
        str(result_dir),
        "--methods",
        METHOD,
        "--datasets",
        DATASET,
        "--attacks",
        ATTACK,
        "--total_limit",
        "2",
        "--chunk_size",
        "2",
    ]
    return args, npz_path, csv_path, meta_path, payload, fieldnames, csv_rows


def _resume_gate(npz_path, csv_path):
    engine = object.__new__(MLATFeatureEngine)
    engine.cfg = SimpleNamespace(
        method=METHOD,
        seed=0,
        token_budget_mode="practical",
        eps_255=2.0,
        alpha_255=0.5,
        steps=100,
        lam=1.0,
        target_k=64,
    )
    engine.vision_layers = list(VISION_LAYERS)
    engine.llm_layers = list(LLM_LAYERS)
    return engine.output_complete(
        npz_path,
        csv_path,
        [0, 1],
        ["images/a.jpg", "images/b.jpg"],
        dataset=DATASET,
        attack=ATTACK,
        token_budget=TOKEN_BUDGET,
    )


class _Tokenizer:
    def __init__(self, token_id=32000):
        self.token_id = token_id

    def convert_tokens_to_ids(self, _token):
        return self.token_id


def _span_engine():
    engine = object.__new__(MLATFeatureEngine)
    engine.device = torch.device("cpu")
    engine.llava_model = SimpleNamespace(
        config=SimpleNamespace(image_token_index=32000)
    )
    engine.llava_processor = SimpleNamespace(tokenizer=_Tokenizer())
    return engine


def test_mlat_image_span_requires_exact_single_placeholder_expansion():
    engine = _span_engine()
    input_ids = torch.tensor([[1, 32000, 2, 3]])
    assert engine._infer_image_indices(input_ids, 579, 576).tolist() == list(
        range(1, 577)
    )


@pytest.mark.parametrize(
    "input_ids,output_length",
    (
        (torch.tensor([[1, 2, 3, 4]]), 579),
        (torch.tensor([[1, 32000, 32000, 4]]), 580),
        (torch.tensor([[1, 32000, 2, 3]]), 578),
        (torch.tensor([[1, 2, 3, 32000]]), 579),
    ),
)
def test_mlat_image_span_rejects_ambiguous_or_textless_spans(
    input_ids, output_length
):
    with pytest.raises(RuntimeError):
        _span_engine()._infer_image_indices(input_ids, output_length, 576)


def test_flowcut_vqav2_mc_uses_k64_and_complete_triplet_passes(tmp_path, capsys):
    assert checker.practical_token_budget("FlowCut", "VQAv2_MC") == 64
    assert checker.practical_token_budget("VisionZIP", "VQAv2_MC") == 32
    args, *_rest = _valid_grid_item(tmp_path)
    assert checker.main(args) == 0
    assert "issues: 0" in capsys.readouterr().out


def test_grid_checker_rejects_result_artifact_symlink(tmp_path, capsys):
    args, npz_path, *_rest = _valid_grid_item(tmp_path)
    external = tmp_path / "external.npz"
    npz_path.rename(external)
    npz_path.symlink_to(external)

    assert checker.main(args) == 1
    output = capsys.readouterr().out
    assert "UNSAFE_RESULT_PATH" in output or "SYMLINK_NPZ" in output


def test_runtime_resume_gate_accepts_only_full_valid_triplet(tmp_path):
    _args, npz_path, csv_path, _meta, _payload, _fieldnames, _rows = _valid_grid_item(tmp_path)
    assert _resume_gate(npz_path, csv_path)


@pytest.mark.parametrize(
    "corruption",
    ("wrong_label", "invalid_span", "extra_feature", "csv_metadata"),
)
def test_runtime_resume_gate_rejects_semantically_stale_triplet(tmp_path, corruption):
    _args, npz_path, csv_path, _meta, payload, fieldnames, rows = _valid_grid_item(tmp_path)
    if corruption == "wrong_label":
        payload["label"] = np.asarray([1, 1], dtype=np.int8)
        np.savez_compressed(npz_path, **payload)
    elif corruption == "invalid_span":
        payload["image_span_valid"] = np.asarray([0, 0], dtype=np.int8)
        np.savez_compressed(npz_path, **payload)
    elif corruption == "extra_feature":
        payload["vis_l99"] = np.ones((2, 3), dtype=np.float32)
        np.savez_compressed(npz_path, **payload)
    else:
        rows[0]["method"] = "VisionZIP"
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    assert not _resume_gate(npz_path, csv_path)


def test_runtime_resume_gate_and_checker_reject_zero_norm_features(tmp_path, capsys):
    args, npz_path, csv_path, _meta, payload, *_rest = _valid_grid_item(tmp_path)
    payload["proj_in"] = np.zeros((2, 3), dtype=np.float16)
    np.savez_compressed(npz_path, **payload)
    assert not _resume_gate(npz_path, csv_path)
    assert checker.main(args) == 1
    assert "BAD_FEATURE_NORM" in capsys.readouterr().out


def test_runtime_resume_gate_rejects_wrong_reconstructed_token_count(tmp_path):
    _args, npz_path, csv_path, _meta, payload, fieldnames, rows = _valid_grid_item(
        tmp_path
    )
    payload["image_token_count"] = np.asarray([1, 1], dtype=np.int32)
    np.savez_compressed(npz_path, **payload)
    for row in rows:
        row["image_token_count"] = 1
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    assert not _resume_gate(npz_path, csv_path)


def test_runtime_resume_gate_rejects_constant_unit_feature_rows(tmp_path):
    _args, npz_path, csv_path, _meta, payload, *_rest = _valid_grid_item(tmp_path)
    for name, array in payload.items():
        if name.startswith(("vis_l", "proj_", "llm_last_l", "llm_image_l")):
            payload[name] = np.repeat(array[:1], repeats=2, axis=0)
    np.savez_compressed(npz_path, **payload)
    assert not _resume_gate(npz_path, csv_path)


@pytest.mark.parametrize(
    "corruption", ("junk_npz", "float_label", "narrow_sample_idx", "float32_feature")
)
def test_exact_npz_schema_rejects_extra_arrays_and_float_metadata(
    tmp_path,
    capsys,
    corruption,
):
    args, npz_path, _csv_path, _meta, payload, *_rest = _valid_grid_item(tmp_path)
    if corruption == "junk_npz":
        payload["junk"] = np.zeros((2, 1), dtype=np.float32)
    elif corruption == "float_label":
        payload["label"] = np.asarray([0.0, 0.0], dtype=np.float32)
    elif corruption == "narrow_sample_idx":
        payload["sample_idx"] = np.asarray([0, 1], dtype=np.int16)
    else:
        payload["proj_in"] = np.ones((2, 3), dtype=np.float32)
    np.savez_compressed(npz_path, **payload)
    assert checker.main(args) == 1
    output = capsys.readouterr().out
    expected = {
        "junk_npz": "UNEXPECTED_NPZ_ARRAY",
        "float_label": "BAD_SHAPE_DTYPE_OR_VALUE",
        "narrow_sample_idx": "BAD_SHAPE_OR_DTYPE",
        "float32_feature": "BAD_FEATURE_ARRAY",
    }[corruption]
    assert expected in output


@pytest.mark.parametrize("corruption", ("extra", "reordered"))
def test_exact_csv_schema_rejects_extra_or_reordered_columns(
    tmp_path,
    capsys,
    corruption,
):
    args, _npz_path, csv_path, _meta, _payload, fieldnames, rows = _valid_grid_item(
        tmp_path
    )
    fieldnames = list(fieldnames)
    if corruption == "extra":
        fieldnames.append("junk")
        for row in rows:
            row["junk"] = "unexpected"
    else:
        fieldnames[0], fieldnames[1] = fieldnames[1], fieldnames[0]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    assert checker.main(args) == 1
    assert "BAD_CSV_HEADER" in capsys.readouterr().out


@pytest.mark.parametrize(("name", "value"), (("--total_limit", "0"), ("--chunk_size", "0")))
def test_nonpositive_grid_sizes_are_argparse_errors(tmp_path, name, value):
    with pytest.raises(SystemExit) as error:
        checker.main(["--result-dir", str(tmp_path), name, value])
    assert error.value.code == 2


def test_nonformal_hyperparameters_are_argparse_errors(tmp_path):
    with pytest.raises(SystemExit) as error:
        checker.main(["--result-dir", str(tmp_path), "--lam", "0.5"])
    assert error.value.code == 2


def test_missing_meta_is_nonzero(tmp_path, capsys):
    args, _npz, _csv, meta, *_rest = _valid_grid_item(tmp_path)
    meta.unlink()
    assert checker.main(args) == 1
    assert "MISSING_META" in capsys.readouterr().out


def test_duplicate_sample_indices_and_nonfinite_feature_are_nonzero(tmp_path, capsys):
    args, npz_path, _csv, _meta, payload, *_rest = _valid_grid_item(tmp_path)
    payload["sample_idx"] = np.asarray([0, 0], dtype=np.int64)
    payload["proj_in"] = np.asarray([[1.0, np.nan], [1.0, 2.0]])
    np.savez_compressed(npz_path, **payload)
    assert checker.main(args) == 1
    output = capsys.readouterr().out
    assert "DUPLICATE_SAMPLE_IDX" in output
    assert "BAD_FEATURE_ARRAY" in output


def test_npz_csv_image_id_mismatch_is_nonzero(tmp_path, capsys):
    args, _npz, csv_path, _meta, _payload, fieldnames, rows = _valid_grid_item(tmp_path)
    rows[1]["Image_ID"] = "images/different.jpg"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    assert checker.main(args) == 1
    assert "NPZ_CSV_IMAGE_ID_MISMATCH" in capsys.readouterr().out


def test_cross_chunk_duplicate_image_ids_are_nonzero(tmp_path, capsys):
    args, npz_path, csv_path, meta_path, payload, fieldnames, rows = _valid_grid_item(
        tmp_path
    )
    args[args.index("2")] = "4"  # --total_limit
    args[args.index("2", args.index("4") + 1)] = "2"  # --chunk_size

    duplicate_ids = np.asarray(["images/a.jpg", "images/b.jpg"])
    payload["sample_idx"] = np.asarray([2, 3], dtype=np.int64)
    payload["image_id"] = duplicate_ids
    second_npz = npz_path.with_name(
        "mlat_feat_FlowCut_VQAv2_MC_clean_clip_k64_start2_limit2_seed0.npz"
    )
    np.savez_compressed(second_npz, **payload)

    second_rows = [dict(row) for row in rows]
    for offset, row in enumerate(second_rows):
        row["sample_idx"] = 2 + offset
        row["sample_seed"] = 2 + offset
    with second_npz.with_suffix(".csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(second_rows)

    contract = json.loads(meta_path.read_text(encoding="utf-8"))
    contract["expected_sample_indices"] = [2, 3]
    contract["expected_image_ids"] = duplicate_ids.tolist()
    second_npz.with_suffix(".meta.json").write_text(
        json.dumps(contract), encoding="utf-8"
    )

    assert checker.main(args) == 1
    assert "CROSS_CHUNK_DUPLICATE_IMAGE_ID" in capsys.readouterr().out
