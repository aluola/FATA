from __future__ import annotations

import csv

import pytest

from fata.detection.mlatd.generate_attack_cache_cage_caa import (
    MANIFEST_FIELDS,
    append_manifest,
    validate_complete_manifest,
    write_manifest_atomic,
)


def _row(index: int, image_id: str) -> dict[str, object]:
    return {
        "attack": "cage",
        "dataset": "TextVQA_Open",
        "sample_idx": index,
        "image_filename": image_id,
        "status": "success",
        "seed": index,
        "eps_255": 2.0,
        "alpha_255": 0.5,
        "steps": 100,
        "message": "",
    }


def _validate(path) -> None:
    validate_complete_manifest(
        path,
        attack="cage",
        dataset="TextVQA_Open",
        start=0,
        expected_image_ids=["a.jpg", "b.jpg"],
        seed=0,
        eps_255=2.0,
        alpha_255=0.5,
        steps=100,
    )


def test_atomic_rewrite_canonicalizes_interrupted_append_log(tmp_path):
    manifest = tmp_path / "manifest.csv"
    append_manifest(manifest, _row(0, "a.jpg"))
    append_manifest(manifest, dict(_row(0, "a.jpg"), message="retry"))
    write_manifest_atomic(manifest, [_row(0, "a.jpg"), _row(1, "b.jpg")])

    _validate(manifest)
    with manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == MANIFEST_FIELDS
        assert [int(row["sample_idx"]) for row in reader] == [0, 1]


@pytest.mark.parametrize(
    "rows",
    [
        [_row(0, "a.jpg")],
        [_row(1, "b.jpg"), _row(0, "a.jpg")],
        [_row(0, "a.jpg"), dict(_row(1, "b.jpg"), status="error")],
    ],
)
def test_manifest_validation_fails_closed(tmp_path, rows):
    manifest = tmp_path / "manifest.csv"
    write_manifest_atomic(manifest, rows)
    with pytest.raises(RuntimeError):
        _validate(manifest)


@pytest.mark.parametrize("writer", (append_manifest, write_manifest_atomic))
def test_manifest_writers_reject_final_symlink(tmp_path, writer):
    external = tmp_path / "external.csv"
    external.write_text("sentinel\n", encoding="utf-8")
    manifest = tmp_path / "manifest.csv"
    manifest.symlink_to(external)
    rows = _row(0, "a.jpg") if writer is append_manifest else [_row(0, "a.jpg")]
    with pytest.raises((RuntimeError, OSError), match="symbolic link|Too many levels"):
        writer(manifest, rows)
    assert external.read_text(encoding="utf-8") == "sentinel\n"
