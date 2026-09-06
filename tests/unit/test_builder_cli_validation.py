from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from PIL import Image

from fata.data.unique_builder import (
    audit_datasets,
    build_all,
    build_scienceqa,
    build_textvqa,
    build_vqav2_mc,
    build_vqav2_open,
    generate_reports,
    generate_reproducibility,
    loader_smoke_test,
    validate_datasets,
)


@pytest.mark.parametrize(
    "module",
    (build_all, build_textvqa, build_vqav2_open, build_vqav2_mc, build_scienceqa),
)
def test_builder_missing_output_and_model_is_argparse_error(module, monkeypatch):
    monkeypatch.delenv("FATA_OUTPUT_ROOT", raising=False)
    monkeypatch.delenv("FATA_LLAVA_MODEL", raising=False)
    monkeypatch.setattr(module, "DEFAULT_OUTPUT", None)
    monkeypatch.setattr(module, "DEFAULT_MODEL", None)
    with pytest.raises(SystemExit) as error:
        module.main([])
    assert error.value.code == 2


@pytest.mark.parametrize(
    "module",
    (validate_datasets, generate_reports, generate_reproducibility),
)
def test_report_cli_requires_explicit_reports_dir(module, tmp_path):
    with pytest.raises(SystemExit) as error:
        module.main(["--dataset-root", str(tmp_path)])
    assert error.value.code == 2


def test_validator_refuses_implicit_dataset_root_report_output(tmp_path):
    with pytest.raises(ValueError, match="reports_dir is required"):
        validate_datasets.validate_dataset_root(tmp_path)


@pytest.mark.parametrize(
    "writer",
    (
        generate_reports._atomic_write_text,
        loader_smoke_test._atomic_write_text,
        validate_datasets._atomic_write_text,
    ),
)
def test_dataset_report_writers_reject_final_symlink(tmp_path, writer):
    sentinel = tmp_path / "sentinel.txt"
    destination = tmp_path / "report.txt"
    sentinel.write_text("unchanged", encoding="utf-8")
    destination.symlink_to(sentinel)
    with pytest.raises(ValueError, match="symbolic link"):
        writer(destination, "replacement")
    assert sentinel.read_text(encoding="utf-8") == "unchanged"


@pytest.mark.parametrize("prefix", ("../../outside/report", "/tmp/absolute-report", "a/b"))
def test_dataset_audit_rejects_report_prefix_path_escape(tmp_path, prefix):
    dataset_root = tmp_path / "datasets"
    dataset_root.mkdir()
    reports = tmp_path / "reports"
    with pytest.raises(ValueError, match="unsafe report prefix"):
        audit_datasets.audit_dataset_root(
            dataset_root,
            reports,
            prefix=prefix,
        )
    assert not reports.exists()


@pytest.mark.parametrize(
    ("input_argument", "input_name"),
    (
        ("validation_path", "dataset_selection_funnel.json"),
        ("old_audit_path", "old_vs_new_dataset_quality.csv"),
    ),
)
def test_report_generator_rejects_input_output_file_collision(
    tmp_path,
    input_argument,
    input_name,
):
    dataset_root = tmp_path / "new-datasets"
    legacy_root = tmp_path / "legacy-datasets"
    manifests = tmp_path / "manifests"
    reports = tmp_path / "reports"
    for directory in (dataset_root, legacy_root, manifests, reports):
        directory.mkdir()
    colliding_input = reports / input_name
    colliding_input.write_text("{}", encoding="utf-8")
    other_input = tmp_path / "other-input.json"
    other_input.write_text("{}", encoding="utf-8")
    inputs = {"validation_path": other_input, "old_audit_path": other_input}
    inputs[input_argument] = colliding_input
    with pytest.raises(ValueError, match="would overwrite protected"):
        generate_reports.generate_reports(
            dataset_root,
            reports_dir=reports,
            old_dataset_root=legacy_root,
            manifests_dir=manifests,
            **inputs,
        )
    assert colliding_input.read_text(encoding="utf-8") == "{}"


@pytest.mark.parametrize("corruption", ("truncated_tail", "forged_row"))
def test_vqav2_mc_rejects_nonexact_frozen_open_sidecar(
    tmp_path, monkeypatch, corruption
):
    output = tmp_path / "output"
    model = tmp_path / "model"
    manifests = output / "manifests"
    mapping_dir = output / "VQAv2_Open"
    source_cache = output / "source_cache"
    manifests.mkdir(parents=True)
    mapping_dir.mkdir()
    source_cache.mkdir()
    model.mkdir()
    expected_sample = {
        "retained_index": 0,
        "source_image_id": "open-image-0",
        "source_canonical_rgb_sha256": "a" * 64,
        "saved_canonical_rgb_sha256": "b" * 64,
        "mapping": {"image_filename": "images/VQAv2_Open_0000.jpg"},
    }
    sidecar = manifests / "VQAv2_Open_samples.jsonl"
    if corruption == "truncated_tail":
        sidecar.write_bytes(
            (json.dumps(expected_sample) + "\n" + '{"truncated"').encode("utf-8")
        )
    else:
        forged = dict(expected_sample)
        forged["source_image_id"] = "forged-image"
        sidecar.write_text(json.dumps(forged) + "\n", encoding="utf-8")
    (mapping_dir / "VQAv2_Open_mapping.jsonl").write_text(
        json.dumps(expected_sample["mapping"]) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        build_vqav2_mc, "load_vqav2", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(
        build_vqav2_mc,
        "load_stored_build_exclusions",
        lambda *args, **kwargs: build_vqav2_mc.BuildExclusions(),
    )
    monkeypatch.setattr(
        build_vqav2_mc,
        "load_completed_resume",
        lambda *args, **kwargs: SimpleNamespace(samples=(expected_sample,)),
    )

    expected_message = (
        "frozen VQAv2-Open sample sidecar"
        if corruption == "truncated_tail"
        else "does not match its completed checkpoint"
    )
    with pytest.raises(RuntimeError, match=expected_message):
        build_vqav2_mc.main(
            [
                "--output",
                str(output),
                "--model",
                str(model),
                "--target",
                "1",
                "--resume",
            ]
        )


@pytest.mark.parametrize(
    ("encoding", "expected_issue"),
    (
        ("png", "invalid_encoded_image_format"),
        ("grayscale_jpeg", "invalid_encoded_image_mode"),
    ),
)
def test_central_validator_requires_native_rgb_jpeg_bytes(
    tmp_path, encoding, expected_issue
):
    dataset = "TextVQA_Open"
    dataset_dir = tmp_path / dataset
    image_dir = dataset_dir / "images"
    image_dir.mkdir(parents=True)
    image_name = f"images/{dataset}_0000.jpg"
    image_path = dataset_dir / image_name
    if encoding == "png":
        Image.new("RGB", (3, 2), (1, 2, 3)).save(image_path, format="PNG")
    else:
        Image.new("L", (3, 2), 127).save(image_path, format="JPEG")
    mapping = {
        "dataset": dataset,
        "question": "fixture question",
        "answers": ["fixture answer"],
        "image_filename": image_name,
    }
    (dataset_dir / f"{dataset}_mapping.jsonl").write_text(
        json.dumps(mapping) + "\n", encoding="utf-8"
    )

    result, _ = validate_datasets.validate_dataset(
        tmp_path, dataset, expected_count=1
    )

    assert result["issues"]["counts"][expected_issue] == 1
    assert len(result["input_artifact_sha256"]["mapping"]) == 64


def test_report_binding_rejects_selection_stats_changed_after_validation(tmp_path):
    root = tmp_path / "datasets"
    dataset = "TextVQA_Open"
    dataset_dir = root / dataset
    manifests = root / "manifests"
    dataset_dir.mkdir(parents=True)
    manifests.mkdir()
    artifact_paths = {
        "mapping": dataset_dir / f"{dataset}_mapping.jsonl",
        "samples": manifests / f"{dataset}_samples.jsonl",
        "candidate_pool": manifests / f"{dataset}_candidate_pool.jsonl",
        "screening": manifests / f"{dataset}_screening.jsonl",
        "selection_stats": manifests / f"{dataset}_selection_stats.json",
        "checkpoint": manifests / f"{dataset}_checkpoint.json",
        "source": manifests / f"{dataset}_source.json",
    }
    for path in artifact_paths.values():
        path.write_text("{}\n", encoding="utf-8")
    validation_result = {
        "input_artifact_sha256": {
            name: generate_reports._saved_file_sha256(path)
            for name, path in artifact_paths.items()
        }
    }
    generate_reports._require_validation_artifact_binding(
        root, dataset, validation_result
    )

    artifact_paths["selection_stats"].write_text(
        '{"retained_count": 999}\n', encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="changed after final validation"):
        generate_reports._require_validation_artifact_binding(
            root, dataset, validation_result
        )


def test_report_binding_rejects_missing_or_changed_image_hash_manifest(tmp_path):
    validation_path = tmp_path / "reports" / "final_validation.json"
    validation_path.parent.mkdir()
    image_manifest = validation_path.parent / "image_hash_manifest.jsonl"
    image_manifest.write_text('{"dataset":"TextVQA_Open"}\n', encoding="utf-8")
    payload = {
        "output_artifact_sha256": {
            "image_hash_manifest": generate_reports._saved_file_sha256(image_manifest)
        }
    }

    assert (
        generate_reports._require_validation_image_manifest_binding(
            validation_path, payload
        )
        == image_manifest
    )
    image_manifest.write_text('{"dataset":"VQAv2_Open"}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed after final validation"):
        generate_reports._require_validation_image_manifest_binding(
            validation_path, payload
        )
    image_manifest.unlink()
    with pytest.raises(RuntimeError, match="missing or symlinked"):
        generate_reports._require_validation_image_manifest_binding(
            validation_path, payload
        )
    outside = tmp_path / "outside-image-manifest.jsonl"
    outside.write_text('{"dataset":"TextVQA_Open"}\n', encoding="utf-8")
    image_manifest.symlink_to(outside)
    with pytest.raises(RuntimeError, match="missing or symlinked"):
        generate_reports._require_validation_image_manifest_binding(
            validation_path, payload
        )


def test_report_overlap_rejects_image_changed_after_validation(tmp_path):
    dataset = "TextVQA_Open"
    dataset_dir = tmp_path / dataset
    image_dir = dataset_dir / "images"
    image_dir.mkdir(parents=True)
    image = image_dir / "sample.jpg"
    Image.new("RGB", (4, 3), (12, 34, 56)).save(image, format="JPEG")
    mapping = dataset_dir / f"{dataset}_mapping.jsonl"
    mapping.write_text(
        json.dumps(
            {
                "image_filename": "images/sample.jpg",
                "question": "fixture question",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    validated_hash = generate_reports._canonical_rgb_sha256(image)
    Image.new("RGB", (4, 3), (210, 190, 170)).save(image, format="JPEG")

    with pytest.raises(ValueError, match="disagrees with recomputed pixels"):
        generate_reports._dataset_overlap_identities(
            tmp_path,
            dataset,
            known_hashes={(dataset, "images/sample.jpg"): validated_hash},
            hash_workers=1,
        )


def test_legacy_audit_binds_exact_mapping_and_image_bytes(tmp_path):
    legacy_root = tmp_path / "legacy"
    reports = tmp_path / "reports"
    for dataset in validate_datasets.DATASET_NAMES:
        dataset_dir = legacy_root / dataset
        images = dataset_dir / "images"
        images.mkdir(parents=True)
        image_path = images / f"{dataset}_0000.jpg"
        Image.new("RGB", (3, 2), (12, 34, 56)).save(image_path, format="JPEG")
        (dataset_dir / f"{dataset}_mapping.jsonl").write_text(
            json.dumps(
                {
                    "image_filename": f"images/{dataset}_0000.jpg",
                    "question": "fixture question",
                }
            )
            + "\n",
            encoding="utf-8",
        )

    audit = audit_datasets.audit_dataset_root(legacy_root, reports)
    for dataset in validate_datasets.DATASET_NAMES:
        assert audit["input_artifacts"][dataset] == (
            generate_reports._legacy_input_artifacts(legacy_root, dataset)
        )

    changed = legacy_root / "TextVQA_Open" / "images" / "TextVQA_Open_0000.jpg"
    Image.new("RGB", (3, 2), (99, 88, 77)).save(changed, format="JPEG")
    assert audit["input_artifacts"]["TextVQA_Open"] != (
        generate_reports._legacy_input_artifacts(legacy_root, "TextVQA_Open")
    )


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (left, right)
        for index, left in enumerate(validate_datasets.DATASET_NAMES)
        for right in validate_datasets.DATASET_NAMES[index + 1 :]
    ],
)
def test_central_validator_hard_fails_exact_overlap_for_every_dataset_pair(
    tmp_path, monkeypatch, left, right
):
    shared = "a" * 64

    def fake_validate(_root, dataset, _expected_count, _max_examples):
        records = []
        if dataset in {left, right}:
            records.append(
                validate_datasets.ImageRecord(
                    dataset=dataset,
                    mapping_index=0,
                    image_filename=f"images/{dataset}_0000.jpg",
                    path=tmp_path / f"{dataset}.jpg",
                    width=1,
                    height=1,
                    file_size=1,
                    saved_image_sha256="b" * 64,
                    canonical_rgb_sha256=shared,
                    perceptual_hash="0" * 16,
                    source_dataset=dataset,
                    source_canonical_rgb_sha256=shared,
                )
            )
        return {"pass": True}, records

    monkeypatch.setattr(validate_datasets, "validate_dataset", fake_validate)
    monkeypatch.setattr(
        validate_datasets,
        "_near_duplicate_audit",
        lambda *_args: {
            "pair_candidate_counts": {},
            "pair_high_confidence_counts": {},
            "unresolved_high_confidence_group_count": 0,
        },
    )

    report = validate_datasets.validate_dataset_root(
        tmp_path,
        expected_count=1,
        write_outputs=False,
    )

    pair_label = f"{left}/{right}"
    assert report["overall_pass"] is False
    assert f"{pair_label} share saved canonical RGB images" in report[
        "hard_check_failures"
    ]
    assert f"{pair_label} share source canonical RGB images" in report[
        "hard_check_failures"
    ]
