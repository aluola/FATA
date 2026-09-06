import csv
import json

import pytest

from fata.cli.kprac import main
from fata.constants import COMPRESSORS, DATASETS, LLAVA_BUDGETS
from fata.runtimes.llava.result_schema import (
    raw_answer_column,
    result_header,
    result_schema,
)
from fata.utils.run_contract import dataset_image_identity, sha256_file


def _write_dataset_root(path, rows=2):
    for dataset in DATASETS:
        dataset_dir = path / dataset
        dataset_dir.mkdir(parents=True)
        mapping_rows = []
        for index in range(rows):
            image_id = f"sample-{index}"
            (dataset_dir / image_id).write_bytes(f"{dataset}-{index}".encode())
            mapping_rows.append(
                {
                    "image_filename": image_id,
                    "question": "q",
                    "answers": ["A"],
                    "type": "multiple_choice",
                }
            )
        (dataset_dir / f"{dataset}_mapping.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in mapping_rows),
            encoding="utf-8",
        )
    return path


def _write_input(path, rows=2, *, schema="legacy", dataset_root=None):
    score_fields = ["Lang_Prior_K0"] + [
        f"{mode}_K{k}"
        for mode in ("Clean", "Base", "FATA")
        for k in LLAVA_BUDGETS
    ]
    legacy_fields = ["Image_ID", "Question", *score_fields]
    if schema == "legacy":
        fields = legacy_fields
    elif schema == "v2":
        fields = result_header(["Image_ID", "Question"], score_fields)
    else:
        raise AssertionError(schema)
    try:
        method, dataset = next(
            (method, dataset)
            for dataset in DATASETS
            for method in COMPRESSORS
            if path.stem.endswith(f"_{method}_{dataset}")
        )
    except StopIteration as error:
        raise AssertionError(f"cannot infer method/dataset from {path.name}") from error
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(rows):
            if schema == "legacy":
                row = {
                    "Image_ID": f"sample-{index}",
                    "Question": "q",
                    "Lang_Prior_K0": 0,
                    "Clean_K576": 1,
                    "Clean_K192": 0.95,
                    "Clean_K128": 0.90,
                    "Clean_K64": 0.81,
                    "Clean_K32": 0.79,
                    "Clean_K16": 0.5,
                }
                row.update(
                    {
                        f"{mode}_K{k}": 0
                        for mode in ("Base", "FATA")
                        for k in LLAVA_BUDGETS
                    }
                )
            else:
                raw_answers = {field: "B" for field in score_fields}
                for k in (576, 192, 128, 64):
                    raw_answers[f"Clean_K{k}"] = "Option A."
                row = {
                    "Image_ID": f"sample-{index}",
                    "Question": '"q"',
                    **{
                        field: "1.00" if raw_answers[field] == "Option A." else "0.00"
                        for field in score_fields
                    },
                    **{
                        raw_answer_column(field): raw_answers[field]
                        for field in score_fields
                    },
                }
            writer.writerow(row)
    if method == "VisionZIP":
        pass
    elif method == "VisPruner":
        pass
    elif method == "PruMerge":
        pass
    elif method == "FlowCut":
        pass
    else:
        raise AssertionError(method)
    if schema == "v2":
        if dataset_root is None:
            raise AssertionError("v2 fixture requires dataset_root")
        mapping_path = dataset_root / dataset / f"{dataset}_mapping.jsonl"
        mapping_sha256 = sha256_file(mapping_path)
        image_identity = dataset_image_identity(mapping_path, mapping_path.parent)
    else:
        mapping_sha256 = "0" * 64
        image_identity = {
            "algorithm": "ordered-path-size-file-sha256-v1",
            "count": rows,
            "total_bytes": rows,
            "aggregate_sha256": "1" * 64,
        }
    contract = {
                "schema_version": 1,
                "image_serialization": "png_u8_round_project_linf_v1",
                "runtime": "llava_fata",
                "dataset": dataset,
                "method": method,
                "lambda": 1.0,
                "seed": 0,
                "dataset_mapping_sha256": mapping_sha256,
                "dataset_images": image_identity,
                "model": {
                    "algorithm": "recursive-model-artifact-sha256-v2",
                    "resolved_path": "/models/llava",
                    "metadata_sha256": {},
                    "weight_files": [
                        {"relative_path": "model.bin", "size": 1, "sha256": "2" * 64}
                    ],
                    "index_sha256": {},
                },
                "clip_model": {
                    "algorithm": "recursive-model-artifact-sha256-v2",
                    "resolved_path": "/models/clip",
                    "metadata_sha256": {},
                    "weight_files": [
                        {"relative_path": "model.bin", "size": 1, "sha256": "3" * 64}
                    ],
                    "index_sha256": {},
                },
                "header": fields,
                "expected_image_ids": [f"sample-{index}" for index in range(rows)],
            }
    if schema == "v2":
        contract["result_schema"] = result_schema(score_fields)
    path.with_suffix(".meta.json").write_text(
        json.dumps(contract),
        encoding="utf-8",
    )


def _write_all_v2_inputs(tmp_path, rows=2):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    dataset_root = _write_dataset_root(tmp_path / "datasets", rows=rows)
    for dataset in DATASETS:
        for compressor in COMPRESSORS:
            _write_input(
                inputs
                / f"ultimate_benchmark_lam1_seed0_{compressor}_{dataset}.csv",
                rows=rows,
                schema="v2",
                dataset_root=dataset_root,
            )
    return inputs, dataset_root


def _write_all_uncontracted_legacy_inputs(tmp_path):
    inputs = tmp_path / "legacy-inputs"
    inputs.mkdir()
    for dataset in DATASETS:
        for compressor in COMPRESSORS:
            source = inputs / f"ultimate_benchmark_{compressor}_{dataset}.csv"
            _write_input(source, rows=1000, schema="legacy")
            source.with_suffix(".meta.json").unlink()
    return inputs


def _uncontracted_legacy_args(tmp_path, inputs):
    return [
        "--input-root", str(inputs),
        "--input-pattern", "ultimate_benchmark_{compressor}_{dataset}.csv",
        "--allow-uncontracted-legacy",
        "--output-csv", str(tmp_path / "legacy-selection.csv"),
        "--output-json", str(tmp_path / "legacy-selection.json"),
        "--dataset-version", "legacy_paper_dataset_version",
        "--expected-count", "1000",
    ]


def _v2_args(tmp_path, inputs, dataset_root):
    return [
        "--input-root", str(inputs),
        "--dataset-root", str(dataset_root),
        "--output-csv", str(tmp_path / "selection.csv"),
        "--output-json", str(tmp_path / "selection.json"),
        "--dataset-version", "fixture_v2",
        "--expected-count", "2",
    ]


def test_16_file_cli_records_hashes_and_enforces_count(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    for dataset in DATASETS:
        for compressor in COMPRESSORS:
            _write_input(
                inputs
                / f"ultimate_benchmark_lam1_seed0_{compressor}_{dataset}.csv"
            )
    output_csv = tmp_path / "selection.csv"
    output_json = tmp_path / "selection.json"
    assert main(
        [
            "--input-root", str(inputs),
            "--output-csv", str(output_csv),
            "--output-json", str(output_json),
            "--dataset-version", "fixture_v1",
            "--expected-count", "2",
        ]
    ) == 0
    rows = list(csv.DictReader(output_csv.open(encoding="utf-8")))
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert len(rows) == len(payload["source_sha256"]) == 16
    assert {int(row["selected_k"]) for row in rows} == {64}
    assert all(len(row["source_sha256"]) == 64 for row in rows)

    with pytest.raises(ValueError, match="expected 3"):
        main(
            [
                "--input-root", str(inputs),
                "--output-csv", str(output_csv),
                "--output-json", str(output_json),
                "--dataset-version", "fixture_v1",
                "--expected-count", "3",
            ]
        )


def test_cli_rejects_output_inside_input_tree(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    with pytest.raises(ValueError, match="overlaps protected input root"):
        main(
            [
                "--input-root", str(inputs),
                "--output-csv", str(inputs / "selection.csv"),
                "--output-json", str(tmp_path / "selection.json"),
                "--dataset-version", "fixture_v1",
                "--expected-count", "2",
            ]
        )


def test_cli_rejects_final_output_symlink(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    for dataset in DATASETS:
        for compressor in COMPRESSORS:
            _write_input(
                inputs / f"ultimate_benchmark_lam1_seed0_{compressor}_{dataset}.csv"
            )
    sentinel = tmp_path / "sentinel.csv"
    sentinel.write_text("unchanged", encoding="utf-8")
    output_csv = tmp_path / "selection.csv"
    output_csv.symlink_to(sentinel)
    with pytest.raises(ValueError, match="symbolic link"):
        main(
            [
                "--input-root", str(inputs),
                "--output-csv", str(output_csv),
                "--output-json", str(tmp_path / "selection.json"),
                "--dataset-version", "fixture_v1",
                "--expected-count", "2",
            ]
        )
    assert sentinel.read_text(encoding="utf-8") == "unchanged"


def test_clean_trajectory_rejects_cancelling_out_of_range_cells(tmp_path):
    from fata.cli.kprac import _clean_trajectory

    path = tmp_path / "bad.csv"
    fields = ["Image_ID", *(f"Clean_K{k}" for k in LLAVA_BUDGETS)]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for image_id, bad in (("a", 2), ("b", -1)):
            row = {"Image_ID": image_id}
            row.update({f"Clean_K{k}": 0.5 for k in LLAVA_BUDGETS})
            row["Clean_K64"] = bad
            writer.writerow(row)
    with pytest.raises(ValueError, match=r"finite and in \[0, 1\]"):
        _clean_trajectory(path)


def test_cli_rejects_malformed_source_identity(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    for dataset in DATASETS:
        for compressor in COMPRESSORS:
            _write_input(
                inputs / f"ultimate_benchmark_lam1_seed0_{compressor}_{dataset}.csv"
            )
    meta_path = inputs / "ultimate_benchmark_lam1_seed0_VisionZIP_TextVQA_Open.meta.json"
    contract = json.loads(meta_path.read_text(encoding="utf-8"))
    contract["model"] = {"sha256": "not-portable-or-complete"}
    meta_path.write_text(json.dumps(contract), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid model"):
        main(
            [
                "--input-root", str(inputs),
                "--output-csv", str(tmp_path / "selection.csv"),
                "--output-json", str(tmp_path / "selection.json"),
                "--dataset-version", "fixture_v1",
                "--expected-count", "2",
            ]
        )


def test_cli_rejects_negative_seed_before_reading_inputs(tmp_path):
    with pytest.raises(ValueError, match="--seed must be non-negative"):
        main(
            [
                "--input-root", str(tmp_path / "missing"),
                "--output-csv", str(tmp_path / "selection.csv"),
                "--output-json", str(tmp_path / "selection.json"),
                "--dataset-version", "fixture_v1",
                "--seed", "-1",
            ]
        )


def test_cli_accepts_contract_marked_v2_and_records_source_schema(tmp_path):
    inputs, dataset_root = _write_all_v2_inputs(tmp_path)
    output_json = tmp_path / "selection.json"
    assert main(_v2_args(tmp_path, inputs, dataset_root)) == 0
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert set(payload["source_result_schema"].values()) == {
        "llava_answer_score_v2"
    }


def test_cli_rejects_v2_header_without_exact_result_schema(tmp_path):
    inputs, dataset_root = _write_all_v2_inputs(tmp_path)
    meta = inputs / "ultimate_benchmark_lam1_seed0_VisionZIP_TextVQA_Open.meta.json"
    contract = json.loads(meta.read_text(encoding="utf-8"))
    contract["result_schema"]["score_format"] = "not-canonical"
    meta.write_text(json.dumps(contract), encoding="utf-8")
    with pytest.raises(ValueError, match="exact result_schema"):
        main(_v2_args(tmp_path, inputs, dataset_root))


def test_cli_requires_explicit_dataset_root_for_any_v2_input(tmp_path):
    inputs, _ = _write_all_v2_inputs(tmp_path)
    with pytest.raises(ValueError, match="require explicit --dataset-root"):
        main(
            [
                "--input-root", str(inputs),
                "--output-csv", str(tmp_path / "selection.csv"),
                "--output-json", str(tmp_path / "selection.json"),
                "--dataset-version", "fixture_v2",
                "--expected-count", "2",
            ]
        )


def test_cli_independently_rejects_v2_score_raw_answer_mismatch(tmp_path):
    inputs, dataset_root = _write_all_v2_inputs(tmp_path)
    source = inputs / "ultimate_benchmark_lam1_seed0_VisionZIP_TextVQA_Open.csv"
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or ())
        rows = list(reader)
    rows[0][raw_answer_column("Clean_K576")] = "B"
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(RuntimeError, match="score/raw-answer mismatch"):
        main(_v2_args(tmp_path, inputs, dataset_root))


@pytest.mark.parametrize("mutation", ["mapping", "image"])
def test_cli_binds_v2_to_current_mapping_and_image_identity(tmp_path, mutation):
    inputs, dataset_root = _write_all_v2_inputs(tmp_path)
    dataset_dir = dataset_root / "TextVQA_Open"
    if mutation == "mapping":
        mapping = dataset_dir / "TextVQA_Open_mapping.jsonl"
        rows = [json.loads(line) for line in mapping.read_text(encoding="utf-8").splitlines()]
        rows[0]["question"] = "changed"
        mapping.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        expected = "mapping SHA-256"
    else:
        (dataset_dir / "sample-0").write_bytes(b"replaced image bytes")
        expected = "image identity"
    with pytest.raises(ValueError, match=expected):
        main(_v2_args(tmp_path, inputs, dataset_root))


def test_cli_rejects_v2_csv_order_outside_mapping_cohort_even_if_hashes_match(tmp_path):
    inputs, dataset_root = _write_all_v2_inputs(tmp_path)
    dataset_dir = dataset_root / "TextVQA_Open"
    mapping = dataset_dir / "TextVQA_Open_mapping.jsonl"
    rows = mapping.read_text(encoding="utf-8").splitlines()
    mapping.write_text("\n".join(reversed(rows)) + "\n", encoding="utf-8")

    meta = inputs / "ultimate_benchmark_lam1_seed0_VisionZIP_TextVQA_Open.meta.json"
    contract = json.loads(meta.read_text(encoding="utf-8"))
    contract["dataset_mapping_sha256"] = sha256_file(mapping)
    contract["dataset_images"] = dataset_image_identity(mapping, dataset_dir)
    meta.write_text(json.dumps(contract), encoding="utf-8")

    with pytest.raises(ValueError, match="order/cohort"):
        main(_v2_args(tmp_path, inputs, dataset_root))


def test_uncontracted_legacy_is_default_rejected_and_explicitly_audited(tmp_path):
    inputs = _write_all_uncontracted_legacy_inputs(tmp_path)
    args = _uncontracted_legacy_args(tmp_path, inputs)
    without_opt_in = [value for value in args if value != "--allow-uncontracted-legacy"]
    with pytest.raises(FileNotFoundError, match="missing immutable source contract"):
        main(without_opt_in)

    assert main(args) == 0
    payload = json.loads(
        (tmp_path / "legacy-selection.json").read_text(encoding="utf-8")
    )
    rows = list(
        csv.DictReader(
            (tmp_path / "legacy-selection.csv").open(newline="", encoding="utf-8")
        )
    )
    assert payload["contains_uncontracted_legacy"] is True
    assert payload["lambda"] is payload["seed"] is None
    assert len(payload["source_sha256"]) == len(rows) == 16
    assert payload["source_contract_sha256"] == {}
    assert set(payload["source_result_schema"].values()) == {
        "legacy_uncontracted_score_only_v0"
    }
    assert {row["source_result_schema"] for row in rows} == {
        "legacy_uncontracted_score_only_v0"
    }
    assert "UNCONTRACTED LEGACY AUDIT ONLY" in payload["provenance_limitations"][0]


@pytest.mark.parametrize("mutation", ["nonfinite_attack_score", "misaligned_ids"])
def test_uncontracted_legacy_audit_rejects_bad_cells_and_misalignment(
    tmp_path, mutation
):
    inputs = _write_all_uncontracted_legacy_inputs(tmp_path)
    source = inputs / "ultimate_benchmark_VisionZIP_TextVQA_Open.csv"
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or ())
        rows = list(reader)
    if mutation == "nonfinite_attack_score":
        rows[0]["Base_K576"] = "nan"
        expected = "non-finite Base_K576"
    else:
        rows[0], rows[1] = rows[1], rows[0]
        expected = "order/set differs across compressors"
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises((RuntimeError, ValueError), match=expected):
        main(_uncontracted_legacy_args(tmp_path, inputs))


def test_uncontracted_legacy_audit_forces_paper_cohort_size(tmp_path):
    with pytest.raises(ValueError, match="requires --expected-count 1000"):
        main(
            [
                "--input-root", str(tmp_path / "unused"),
                "--allow-uncontracted-legacy",
                "--output-csv", str(tmp_path / "selection.csv"),
                "--output-json", str(tmp_path / "selection.json"),
                "--dataset-version", "legacy",
                "--expected-count", "2",
            ]
        )
