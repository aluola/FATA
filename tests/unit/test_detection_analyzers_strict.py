from __future__ import annotations

import csv
import json

import pytest

from fata.detection.feature_squeezing import analyze_feature_squeezing as fs
from fata.detection.mahalanobis import analyze_mahalanobis as maha


def _dataset_identity():
    return {
        "algorithm": "ordered-path-size-file-sha256-v1",
        "count": 2,
        "total_bytes": 20,
        "aggregate_sha256": "1" * 64,
    }


def _artifact_identity(name: str):
    return {
        "algorithm": "recursive-model-artifact-sha256-v2",
        "resolved_path": f"/models/{name}",
        "metadata_sha256": {"config.json": "2" * 64},
        "weight_files": [
            {"relative_path": "model.bin", "size": 1, "sha256": "3" * 64}
        ],
        "index_sha256": {},
    }


def _cached_source_contract(contract: dict, attack: str) -> tuple[dict, str]:
    namespace = f"{attack}_eps2_a0.5_s100_seed0"
    definition = {
        "cage": "llava_cage_efd_rda_lam0.005_k16_192_v1",
        "caa": "llava_caa_layer1_region30_w10_10_2_5_v1",
    }[attack]
    objective = (
        {"lambda_cage": 0.005, "k_min": 16, "k_max": 192}
        if attack == "cage"
        else {
            "target_layer": 1,
            "least_important_region_fraction": 0.30,
            "weights": {
                "bpr_inter": 10.0,
                "bpr_intra": 10.0,
                "semantic": 2.0,
                "question_answer": 5.0,
            },
        }
    )
    return {
        "schema_version": 1,
        "image_serialization": "png_u8_round_project_linf_v1",
        "definition": definition,
        "dataset": contract["dataset"],
        "dataset_mapping_sha256": contract["dataset_mapping_sha256"],
        "dataset_images": contract["dataset_images"],
        "method": contract["method"],
        "model": contract["model"],
        "clip_model": None if attack == "caa" else contract["clip_model"],
        "seed": contract["seed"],
        "eps_255": contract["eps_255"],
        "alpha_255": contract["alpha_255"],
        "steps": contract["steps"],
        "extra": {"max_input_tokens": 0, "objective": objective},
    }, namespace


def _write_pair(root, kind: str, attack: str, scores: list[float]):
    path = root / f"{kind}_{attack}.csv"
    ids = ["images/a.jpg", "images/b.jpg"]
    if kind == "fs":
        header = [
            "Image_ID", "Question", "dataset", "method", "detect_k",
            "attack_for_detection", "label", "seed", "sample_seed", "limit",
            "eps_255", "alpha_255", "steps", "lam", "orig_answer",
            "orig_score", "max_fs_score_diff", "max_answer_changed",
        ]
        squeezers = ["bit5", "bit4", "median3", "jpeg75", "jpeg50"]
        for squeezer in squeezers:
            header.extend(
                [
                    f"{squeezer}_answer", f"{squeezer}_score",
                    f"{squeezer}_score_diff", f"{squeezer}_answer_changed",
                ]
            )
        rows = []
        for index, (image_id, score) in enumerate(zip(ids, scores)):
            row = {
                "Image_ID": image_id, "Question": "q", "dataset": "TextVQA_Open",
                "method": "VisionZIP", "detect_k": 64,
                "attack_for_detection": attack,
                "label": 0 if attack == "clean" else 1, "seed": 0,
                "sample_seed": index, "limit": 2, "eps_255": 2.0,
                "alpha_255": 0.5, "steps": 100, "lam": 1.0,
                "orig_answer": "x", "orig_score": 1.0,
                "max_fs_score_diff": score, "max_answer_changed": int(score > 0.5),
            }
            for squeezer in squeezers:
                row.update(
                    {
                        f"{squeezer}_answer": "y" if score > 0.5 else "x",
                        f"{squeezer}_score": 1.0 - score,
                        f"{squeezer}_score_diff": score,
                        f"{squeezer}_answer_changed": int(score > 0.5),
                    }
                )
            rows.append(row)
        contract = {
            "schema_version": 1,
            "image_serialization": "png_u8_round_project_linf_v1",
            "runtime": "feature_squeezing_detection_v1",
            "dataset": "TextVQA_Open", "dataset_mapping_sha256": "0" * 64,
            "dataset_images": _dataset_identity(),
            "expected_image_ids": ids, "method": "VisionZIP", "detect_k": 64,
            "attack_for_detection": attack, "model": _artifact_identity("m"),
            "clip_model": _artifact_identity("c"), "seed": 0, "limit": 2,
            "selected_count": 2,
            "eps_255": 2.0, "alpha_255": 0.5, "steps": 100,
            "lambda": 1.0, "squeezers": squeezers, "header": header,
            "source_cache_contract": None, "cache_namespace": None,
        }
    else:
        header = [
            "Image_ID", "Question", "dataset", "method", "attack_for_detection",
            "label", "seed", "sample_seed", "eval_start", "eval_limit",
            "eps_255", "alpha_255", "steps", "lam", "stats_sha256",
            "maha_cls", "maha_cls_z", "maha_mean_patch", "maha_mean_patch_z",
            "maha_max_z", "maha_avg_z",
        ]
        rows = []
        for index, (image_id, score) in enumerate(zip(ids, scores)):
            rows.append({
                "Image_ID": image_id, "Question": "q", "dataset": "TextVQA_Open",
                "method": "VisionZIP", "attack_for_detection": attack,
                "label": 0 if attack == "clean" else 1, "seed": 0,
                "sample_seed": index, "eval_start": 0, "eval_limit": 2,
                "eps_255": 2.0, "alpha_255": 0.5, "steps": 100, "lam": 1.0,
                "stats_sha256": "2" * 64, "maha_cls": abs(score),
                "maha_cls_z": score, "maha_mean_patch": abs(score),
                "maha_mean_patch_z": score, "maha_max_z": score,
                "maha_avg_z": score,
            })
        contract = {
            "schema_version": 1,
            "image_serialization": "png_u8_round_project_linf_v1",
            "runtime": "mahalanobis_eval_v1",
            "dataset": "TextVQA_Open", "dataset_mapping_sha256": "0" * 64,
            "dataset_images": _dataset_identity(),
            "expected_image_ids": ids, "method": "VisionZIP",
            "attack_for_detection": attack, "model": _artifact_identity("m"),
            "clip_model": _artifact_identity("c"), "seed": 0, "eval_start": 0,
            "eval_limit": 2, "eps_255": 2.0, "alpha_255": 0.5,
            "steps": 100, "lambda": 1.0, "stats_sha256": "2" * 64,
            "header": header,
            "source_cache_contract": None, "cache_namespace": None,
        }
    if attack in {"cage", "caa"}:
        source_contract, namespace = _cached_source_contract(contract, attack)
        contract["source_cache_contract"] = source_contract
        contract["cache_namespace"] = namespace
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    path.with_suffix(".meta.json").write_text(json.dumps(contract), encoding="utf-8")
    return path


def test_feature_analyzer_complete_protocol_and_bounds(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    _write_pair(inputs, "fs", "clean", [0.1, 0.2])
    bad = _write_pair(inputs, "fs", "fata", [0.8, 0.9])
    out = tmp_path / "out"
    assert fs.main(["--input-glob", str(inputs / "*.csv"), "--output-dir", str(out)]) == 0
    assert (out / "fs_detection_auc.csv").is_file()
    rows = list(csv.DictReader(bad.open(encoding="utf-8")))
    rows[0]["max_fs_score_diff"] = "2"
    with bad.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="invalid detector score"):
        fs.main(["--input-glob", str(inputs / "*.csv"), "--output-dir", str(tmp_path / "bad-out")])


def test_feature_analyzer_accepts_zero_as_full_cohort_request(tmp_path):
    path = _write_pair(tmp_path, "fs", "clean", [0.1, 0.2])
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["limit"] = "0"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    meta_path = path.with_suffix(".meta.json")
    contract = json.loads(meta_path.read_text(encoding="utf-8"))
    contract["limit"] = 0
    meta_path.write_text(json.dumps(contract), encoding="utf-8")
    assert len(fs._load_one(path)) == 2


def test_feature_analyzer_rejects_forged_incomplete_squeezer_schema(tmp_path):
    path = _write_pair(tmp_path, "fs", "clean", [0.1, 0.2])
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    retained = [name for name in rows[0] if not name.startswith("jpeg50_")]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=retained, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    meta_path = path.with_suffix(".meta.json")
    contract = json.loads(meta_path.read_text(encoding="utf-8"))
    contract["header"] = retained
    contract["squeezers"] = contract["squeezers"][:-1]
    meta_path.write_text(json.dumps(contract), encoding="utf-8")
    with pytest.raises(ValueError, match="Feature Squeezing schema"):
        fs._load_one(path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("bit5_score_diff", "0.15", "derived score diff"),
        ("bit5_answer_changed", "1", "derived answer change"),
        ("max_fs_score_diff", "0.15", "derived max_fs_score_diff"),
    ),
)
def test_feature_analyzer_rejects_forged_derived_values(
    tmp_path, field, value, message
):
    path = _write_pair(tmp_path, "fs", "clean", [0.1, 0.2])
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0][field] = value
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    _write_pair(tmp_path, "fs", "fata", [0.8, 0.9])
    with pytest.raises(ValueError, match=message):
        fs.main(
            [
                "--input-glob", str(tmp_path / "fs_*.csv"),
                "--output-dir", str(tmp_path / "derived-out"),
            ]
        )


def test_mahalanobis_analyzer_complete_protocol_and_header_lock(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    _write_pair(inputs, "maha", "clean", [-1.0, -0.5])
    attack = _write_pair(inputs, "maha", "fata", [1.0, 2.0])
    out = tmp_path / "out"
    assert maha.main(["--input-glob", str(inputs / "*.csv"), "--output-dir", str(out)]) == 0
    result = list(csv.DictReader((out / "maha_detection_auc.csv").open(encoding="utf-8")))
    assert result and result[0]["positive_attack"] == "fata"
    meta = json.loads(attack.with_suffix(".meta.json").read_text(encoding="utf-8"))
    meta["header"] = meta["header"][:-1]
    attack.with_suffix(".meta.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="header disagrees"):
        maha.main(["--input-glob", str(inputs / "*.csv"), "--output-dir", str(tmp_path / "bad-out")])


@pytest.mark.parametrize("field", ("maha_max_z", "maha_avg_z"))
def test_mahalanobis_analyzer_rejects_forged_derived_values(tmp_path, field):
    path = _write_pair(tmp_path, "maha", "clean", [-1.0, -0.5])
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0][field] = "0.25"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="inconsistent derived"):
        maha._load_one(path)


@pytest.mark.parametrize("kind,module", (("fs", fs), ("maha", maha)))
def test_detection_analyzer_rejects_pre_quantization_contract(tmp_path, kind, module):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    path = _write_pair(inputs, kind, "clean", [0.1, 0.2])
    contract_path = path.with_suffix(".meta.json")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract.pop("image_serialization")
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported attack-image serialization"):
        module._load_one(path)


@pytest.mark.parametrize("module", (fs, maha))
def test_detection_analyzer_refuses_output_over_input_tree(tmp_path, module):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    kind = "fs" if module is fs else "maha"
    _write_pair(inputs, kind, "clean", [0.1, 0.2])
    _write_pair(inputs, kind, "fata", [0.8, 0.9])
    with pytest.raises(ValueError, match="overlaps protected"):
        module.main(["--input-glob", str(inputs / "*.csv"), "--output-dir", str(inputs)])


@pytest.mark.parametrize("kind,module", (("fs", fs), ("maha", maha)))
def test_detection_analyzer_rejects_symbolic_link_input(tmp_path, kind, module):
    real = tmp_path / "real"
    linked = tmp_path / "linked"
    real.mkdir()
    linked.mkdir()
    source = _write_pair(real, kind, "clean", [0.1, 0.2])
    input_link = linked / source.name
    input_link.symlink_to(source)
    with pytest.raises(ValueError, match="symbolic-link analysis input"):
        module._load_one(input_link)


@pytest.mark.parametrize(
    ("kind", "module", "result_name"),
    (
        ("fs", fs, "fs_detection_auc.csv"),
        ("maha", maha, "maha_detection_auc.csv"),
    ),
)
def test_detection_analyzer_rejects_final_output_symlink(
    tmp_path, kind, module, result_name
):
    inputs = tmp_path / "inputs"
    output = tmp_path / "output"
    inputs.mkdir()
    output.mkdir()
    _write_pair(inputs, kind, "clean", [0.1, 0.2])
    _write_pair(inputs, kind, "fata", [0.8, 0.9])
    sentinel = tmp_path / "sentinel.csv"
    sentinel.write_text("unchanged", encoding="utf-8")
    (output / result_name).symlink_to(sentinel)
    with pytest.raises(ValueError, match="symbolic link"):
        module.main(
            ["--input-glob", str(inputs / "*.csv"), "--output-dir", str(output)]
        )
    assert sentinel.read_text(encoding="utf-8") == "unchanged"


@pytest.mark.parametrize("kind,module", (("fs", fs), ("maha", maha)))
@pytest.mark.parametrize(
    "corruption",
    ("schema", "mapping", "dataset_images", "model", "clip_model"),
)
def test_detection_analyzer_rejects_malformed_identity_contracts(
    tmp_path, kind, module, corruption
):
    path = _write_pair(tmp_path, kind, "clean", [0.1, 0.2])
    meta_path = path.with_suffix(".meta.json")
    contract = json.loads(meta_path.read_text(encoding="utf-8"))
    if corruption == "schema":
        contract["schema_version"] = 0
    elif corruption == "mapping":
        contract["dataset_mapping_sha256"] = "not-a-digest"
    elif corruption == "dataset_images":
        contract["dataset_images"] = {"aggregate_sha256": "1" * 64}
    else:
        contract[corruption] = {"resolved_path": "relative/model"}
    meta_path.write_text(json.dumps(contract), encoding="utf-8")
    with pytest.raises(ValueError):
        module._load_one(path)


@pytest.mark.parametrize("kind,module", (("fs", fs), ("maha", maha)))
@pytest.mark.parametrize("value", (float("nan"), float("inf")))
def test_detection_analyzer_rejects_nonfinite_attack_contract(
    tmp_path, kind, module, value
):
    path = _write_pair(tmp_path, kind, "clean", [0.1, 0.2])
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["eps_255"] = str(value)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    meta_path = path.with_suffix(".meta.json")
    contract = json.loads(meta_path.read_text(encoding="utf-8"))
    contract["eps_255"] = value
    meta_path.write_text(json.dumps(contract), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid attack parameters"):
        module._load_one(path)


@pytest.mark.parametrize("kind,module", (("fs", fs), ("maha", maha)))
def test_detection_analyzer_validates_cached_attack_lineage(tmp_path, kind, module):
    path = _write_pair(tmp_path, kind, "cage", [0.8, 0.9])
    assert len(module._load_one(path)) == 2

    meta_path = path.with_suffix(".meta.json")
    contract = json.loads(meta_path.read_text(encoding="utf-8"))
    contract["source_cache_contract"]["extra"]["objective"]["lambda_cage"] = 0.01
    meta_path.write_text(json.dumps(contract), encoding="utf-8")
    with pytest.raises(ValueError, match="attack-source lineage"):
        module._load_one(path)


@pytest.mark.parametrize("kind,module", (("fs", fs), ("maha", maha)))
def test_detection_analyzer_rejects_missing_source_lineage(tmp_path, kind, module):
    path = _write_pair(tmp_path, kind, "clean", [0.1, 0.2])
    meta_path = path.with_suffix(".meta.json")
    contract = json.loads(meta_path.read_text(encoding="utf-8"))
    contract.pop("source_cache_contract")
    meta_path.write_text(json.dumps(contract), encoding="utf-8")
    with pytest.raises(ValueError, match="missing attack-source lineage"):
        module._load_one(path)
