from __future__ import annotations

import csv
import json

import numpy as np
import pytest
from PIL import Image

from fata.detection.mlatd import check_mlat_attack_ablation as checker
from fata.runtimes.llava.attack_cache_io import (
    attack_namespace,
    baseline_attack_contract_extra,
)
from fata.utils.run_contract import sha256_file


def _valid_single_item_grid(tmp_path):
    cache_root = tmp_path / "cache"
    result_dir = tmp_path / "results"
    namespace = attack_namespace(
        "cage", seed=0, eps_255=2.0, alpha_255=0.5, steps=100
    )
    shared = cache_root / namespace / "TextVQA_Open" / "shared"
    image_path = shared / "images" / "sample.jpg.png"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (2, 2), (12, 34, 56)).save(image_path, format="PNG")
    image_path.with_name(image_path.name + ".sha256").write_text(
        sha256_file(image_path) + "\n", encoding="ascii"
    )
    (shared / "CACHE_CONTRACT.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "image_serialization": "png_u8_round_project_linf_v1",
                "definition": "mlatd_cage_generator_v1",
                "dataset": "TextVQA_Open",
                "dataset_mapping_sha256": "0" * 64,
                "dataset_images": {
                    "algorithm": "ordered-path-size-file-sha256-v1",
                    "count": 1,
                    "total_bytes": 10,
                    "aggregate_sha256": "1" * 64,
                },
                "method": "shared",
                "model": {
                    "algorithm": "recursive-model-artifact-sha256-v2",
                    "resolved_path": "/models/llava",
                    "metadata_sha256": {},
                    "weight_files": [{"relative_path": "model.bin", "size": 1, "sha256": "2" * 64}],
                    "index_sha256": {},
                },
                "clip_model": {
                    "algorithm": "recursive-model-artifact-sha256-v2",
                    "resolved_path": "/models/clip",
                    "metadata_sha256": {},
                    "weight_files": [{"relative_path": "model.bin", "size": 1, "sha256": "3" * 64}],
                    "index_sha256": {},
                },
                "seed": 0,
                "eps_255": 2.0,
                "alpha_255": 0.5,
                "steps": 100,
                "extra": baseline_attack_contract_extra(
                    "cage",
                    max_input_tokens=0,
                ),
            }
        ),
        encoding="utf-8",
    )

    stem = checker._feature_stem(
        method="VisionZIP",
        dataset="TextVQA_Open",
        namespace=namespace,
        token_budget=64,
        start=0,
        limit=1,
        seed=0,
    )
    result_dir.mkdir()
    npz_path = result_dir / f"{stem}.npz"
    payload = {
        "sample_idx": np.asarray([0], dtype=np.int64),
        "label": np.asarray([1], dtype=np.int8),
        "image_span_valid": np.asarray([1], dtype=np.int8),
        "image_token_count": np.asarray([576], dtype=np.int32),
        "llm_sequence_length": np.asarray([640], dtype=np.int32),
        "vision_layers": np.asarray([6, 12, 18, 24], dtype=np.int16),
        "llm_layers": np.asarray([8, 16, 24, 32], dtype=np.int16),
        "image_id": np.asarray(["images/sample.jpg"]),
        "proj_in": np.full((1, 2), 1.0 / np.sqrt(2), dtype=np.float16),
        "proj_out": np.full((1, 2), 1.0 / np.sqrt(2), dtype=np.float16),
    }
    for layer in (6, 12, 18, 24):
        payload[f"vis_l{layer:02d}"] = np.full(
            (1, 2), 1.0 / np.sqrt(2), dtype=np.float16
        )
    for layer in (8, 16, 24, 32):
        payload[f"llm_last_l{layer:02d}"] = np.full(
            (1, 2), 1.0 / np.sqrt(2), dtype=np.float16
        )
        payload[f"llm_image_l{layer:02d}"] = np.full(
            (1, 2), 1.0 / np.sqrt(2), dtype=np.float16
        )
    np.savez_compressed(npz_path, **payload)
    fieldnames = (
        "sample_idx", "Image_ID", "Question", "dataset", "method",
        "attack_for_detection", "label", "seed", "sample_seed", "token_budget",
        "token_budget_mode", "eps_255", "alpha_255", "steps", "lam", "target_k",
        "vision_layers", "llm_layers", "image_span_valid", "image_token_count",
        "llm_sequence_length",
        "cache_path",
    )
    with npz_path.with_suffix(".csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({
            "sample_idx": 0, "Image_ID": "images/sample.jpg", "Question": "q",
            "dataset": "TextVQA_Open", "method": "VisionZIP",
            "attack_for_detection": "cage", "label": 1, "seed": 0,
            "sample_seed": 0, "token_budget": 64, "token_budget_mode": "practical",
            "eps_255": 2.0, "alpha_255": 0.5, "steps": 100, "lam": 1.0,
            "target_k": 64, "vision_layers": "6,12,18,24",
            "llm_layers": "8,16,24,32", "image_span_valid": 1,
            "image_token_count": 576, "llm_sequence_length": 640,
            "cache_path": str(image_path),
        })
    source_contract = json.loads((shared / "CACHE_CONTRACT.json").read_text(encoding="utf-8"))
    npz_path.with_suffix(".meta.json").write_text(json.dumps({
        "schema_version": 1,
        "image_serialization": "png_u8_round_project_linf_v1",
        "runtime": "mlatd_feature_extraction_v1",
        "feature_schema": "quarter_layers_projector_llm_last_image_v1",
        "dataset": "TextVQA_Open",
        "dataset_mapping_sha256": "0" * 64,
        "dataset_images": source_contract["dataset_images"],
        "method": "VisionZIP", "attack": "cage",
        "llava_model": source_contract["model"], "clip_model": source_contract["clip_model"],
        "seed": 0, "eps_255": 2.0, "alpha_255": 0.5, "steps": 100,
        "lam": 1.0, "target_k": 64, "token_budget_mode": "practical",
        "token_budget": 64, "vision_layers": [6, 12, 18, 24],
        "llm_layers": [8, 16, 24, 32], "expected_sample_indices": [0],
        "expected_image_ids": ["images/sample.jpg"],
        "source_cache_contract": source_contract, "cache_namespace": namespace,
    }) + "\n", encoding="utf-8")

    args = [
        "--cache-root",
        str(cache_root),
        "--result-dir",
        str(result_dir),
        "--datasets",
        "TextVQA_Open",
        "--methods",
        "VisionZIP",
        "--attacks",
        "cage",
        "--total_limit",
        "1",
        "--chunk_size",
        "1",
    ]
    return args, image_path, npz_path


def test_checker_accepts_recursive_namespaced_cache_and_complete_feature_pair(
    tmp_path, capsys
):
    args, _image_path, _npz_path = _valid_single_item_grid(tmp_path)
    assert checker.main(args) == 0
    assert "issues: 0" in capsys.readouterr().out


def test_checker_returns_nonzero_for_missing_cache_image(tmp_path, capsys):
    args, image_path, _npz_path = _valid_single_item_grid(tmp_path)
    image_path.unlink()
    assert checker.main(args) == 1
    assert "cache_count_mismatch" in capsys.readouterr().out


def test_checker_returns_nonzero_for_bad_feature_file(tmp_path, capsys):
    args, _image_path, npz_path = _valid_single_item_grid(tmp_path)
    np.savez_compressed(
        npz_path,
        sample_idx=np.asarray([0], dtype=np.int64),
        image_span_valid=np.asarray([0], dtype=np.int8),
        image_id=np.asarray(["images/sample.jpg"]),
    )
    assert checker.main(args) == 1
    assert "INVALID_IMAGE_SPAN" in capsys.readouterr().out


def test_checker_rejects_cache_namespace_directory_symlink(tmp_path, capsys):
    args, image_path, _npz_path = _valid_single_item_grid(tmp_path)
    shared = image_path.parents[1]
    external = tmp_path / "external-shared"
    shared.rename(external)
    shared.symlink_to(external, target_is_directory=True)

    assert checker.main(args) == 1
    assert "unsafe_cache_directory" in capsys.readouterr().out


def test_checker_rejects_non_rgb_png_even_with_updated_digest(tmp_path, capsys):
    args, image_path, _npz_path = _valid_single_item_grid(tmp_path)
    Image.new("L", (2, 2), 17).save(image_path, format="PNG")
    image_path.with_name(image_path.name + ".sha256").write_text(
        sha256_file(image_path) + "\n", encoding="ascii"
    )
    assert checker.main(args) == 1
    assert "cache_image_is_not_rgb" in capsys.readouterr().out


def test_checker_rejects_nonformal_attack_parameters(tmp_path):
    args, _image_path, _npz_path = _valid_single_item_grid(tmp_path)
    args.extend(("--cage_eps_255", "3"))
    with pytest.raises(SystemExit) as error:
        checker.main(args)
    assert error.value.code == 2


def test_checker_rejects_objective_drift_in_cache_contract(tmp_path, capsys):
    args, image_path, _npz_path = _valid_single_item_grid(tmp_path)
    contract_path = image_path.parents[1] / "CACHE_CONTRACT.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["extra"]["objective"]["lambda_cage"] = 0.006
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    assert checker.main(args) == 1
    assert "cache_contract_mismatch" in capsys.readouterr().out


def test_checker_rejects_cached_feature_path_from_wrong_namespace(tmp_path, capsys):
    args, _image_path, npz_path = _valid_single_item_grid(tmp_path)
    csv_path = npz_path.with_suffix(".csv")
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)
    rows[0]["cache_path"] = "/cache/wrong_namespace/TextVQA_Open/shared/images/sample.jpg.png"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    assert checker.main(args) == 1
    assert "BAD_CSV_CACHE_PATH" in capsys.readouterr().out


def test_checker_rejects_cached_feature_path_from_wrong_root(tmp_path, capsys):
    args, _image_path, npz_path = _valid_single_item_grid(tmp_path)
    csv_path = npz_path.with_suffix(".csv")
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)
    namespace = attack_namespace(
        "cage", seed=0, eps_255=2.0, alpha_255=0.5, steps=100
    )
    rows[0]["cache_path"] = str(
        tmp_path / "untrusted-cache" / namespace / "TextVQA_Open"
        / "shared" / "images" / "sample.jpg.png"
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    assert checker.main(args) == 1
    assert "BAD_CSV_CACHE_PATH" in capsys.readouterr().out


def test_checker_rejects_nonfinite_attack_parameters(tmp_path):
    args, _image_path, _npz_path = _valid_single_item_grid(tmp_path)
    args.extend(("--caa_alpha_255", "nan"))
    with pytest.raises(SystemExit) as error:
        checker.main(args)
    assert error.value.code == 2


def test_checker_uses_flowcut_vqav2_mc_k64():
    assert checker.practical_k("FlowCut", "VQAv2_MC") == 64
    assert checker.practical_k("VisionZIP", "VQAv2_MC") == 32
