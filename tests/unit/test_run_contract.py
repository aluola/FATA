from __future__ import annotations

import csv
import os

import pytest

from fata.utils.run_contract import (
    artifact_identity,
    artifact_identity_valid,
    assert_exact_completion,
    dataset_image_identity,
    ensure_cache_contract,
    ensure_run_contract,
    read_completed_csv_ids,
    require_run_contract,
)


def test_dataset_identity_cache_invalidates_when_image_changes(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    image = images / "sample.jpg"
    image.write_bytes(b"first-image-bytes")
    mapping = tmp_path / "mapping.jsonl"
    mapping.write_text('{"image_filename":"images/sample.jpg"}\n', encoding="utf-8")

    before = dataset_image_identity(mapping, tmp_path)
    assert before["count"] == 1
    assert dataset_image_identity(mapping, tmp_path) == before

    image.write_bytes(b"second-image-bytes")
    after = dataset_image_identity(mapping, tmp_path)
    assert after != before
    assert after["aggregate_sha256"] != before["aggregate_sha256"]


def test_model_identity_covers_tokenizer_and_invalidates_cached_identity(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"fixture"}\n', encoding="utf-8")
    tokenizer = model / "tokenizer.json"
    tokenizer.write_text('{"vocab":{"a":0}}\n', encoding="utf-8")
    (model / "model.bin").write_bytes(b"fixture-weight")

    before = artifact_identity(model)
    assert before["algorithm"] == "recursive-model-artifact-sha256-v2"
    assert "tokenizer.json" in before["metadata_sha256"]
    assert artifact_identity_valid(before)

    tokenizer.write_text('{"vocab":{"changed-token":0}}\n', encoding="utf-8")
    after = artifact_identity(model)
    assert after != before
    assert after["metadata_sha256"]["tokenizer.json"] != before["metadata_sha256"]["tokenizer.json"]
    assert not artifact_identity_valid({key: value for key, value in after.items() if key != "algorithm"})


@pytest.mark.parametrize("as_directory", [False, True])
def test_artifact_identity_cache_rejects_same_size_replacement_with_restored_mtime(
    tmp_path, as_directory
):
    root = tmp_path / "model"
    root.mkdir()
    artifact = root / "model.bin"
    artifact.write_bytes(b"original")
    target = root if as_directory else artifact
    before = artifact_identity(target)
    old_stat = artifact.stat()

    replacement = root / "replacement.bin"
    replacement.write_bytes(b"modified")
    os.utime(replacement, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))
    replacement.replace(artifact)

    after = artifact_identity(target)
    assert after != before
    digest_key = "weight_files" if as_directory else "file_sha256"
    assert after[digest_key] != before[digest_key]


def test_contract_is_immutable_across_resume(tmp_path):
    path = tmp_path / "run.meta.json"
    ensure_run_contract(path, {"seed": 0, "steps": 100})
    ensure_run_contract(path, {"steps": 100, "seed": 0})
    with pytest.raises(RuntimeError, match="mismatch"):
        ensure_run_contract(path, {"seed": 1, "steps": 100})
    require_run_contract(path, {"steps": 100, "seed": 0})
    with pytest.raises(RuntimeError, match="mismatch"):
        require_run_contract(path, {"steps": 100, "seed": 9})
    with pytest.raises(RuntimeError, match="missing"):
        require_run_contract(tmp_path / "missing.json", {"seed": 0})


def test_existing_result_without_contract_fails_closed(tmp_path):
    result = tmp_path / "result.csv"
    result.write_text("header\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="no immutable run contract"):
        ensure_run_contract(tmp_path / "meta.json", {"seed": 0}, result_path=result)


def test_contract_helpers_reject_final_symlink(tmp_path):
    external = tmp_path / "external.json"
    external.write_text('{"seed": 0}\n', encoding="utf-8")
    link = tmp_path / "run.meta.json"
    link.symlink_to(external)

    with pytest.raises(RuntimeError, match="symbolic link"):
        ensure_run_contract(link, {"seed": 0})
    with pytest.raises(RuntimeError, match="symbolic link"):
        require_run_contract(link, {"seed": 0})
    assert external.read_text(encoding="utf-8") == '{"seed": 0}\n'


def test_cache_contract_creation_requires_empty_namespace(tmp_path):
    empty_contract = tmp_path / "empty" / "CACHE_CONTRACT.json"
    ensure_cache_contract(empty_contract, {"seed": 0})
    require_run_contract(empty_contract, {"seed": 0})

    orphaned = tmp_path / "orphaned"
    orphaned.mkdir()
    (orphaned / "sample.png").write_bytes(b"not relevant")
    with pytest.raises(RuntimeError, match="artifacts exist without"):
        ensure_cache_contract(orphaned / "CACHE_CONTRACT.json", {"seed": 0})


def test_csv_resume_rejects_duplicate_or_truncated_rows(tmp_path):
    path = tmp_path / "result.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "value"])
        writer.writerow(["a", "1"])
        writer.writerow(["a", "2"])
    with pytest.raises(RuntimeError, match="duplicate"):
        read_completed_csv_ids(path, expected_header=("id", "value"))
    path.write_text("id,value\na\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="incomplete CSV row"):
        read_completed_csv_ids(path, expected_header=("id", "value"))


def test_exact_completion_rejects_missing_or_unexpected():
    assert_exact_completion(["a", "b"], ["b", "a"], "fixture")
    with pytest.raises(RuntimeError, match="missing=1"):
        assert_exact_completion(["a", "b"], ["a"], "fixture")
    with pytest.raises(RuntimeError, match="unexpected=1"):
        assert_exact_completion(["a"], ["a", "b"], "fixture")
