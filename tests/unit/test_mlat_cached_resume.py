from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fata.detection.mlatd import extract_cached_attack_mlat as cached_extract
from fata.detection.mlatd.extract_cached_attack_mlat import (
    outputs_complete,
    run_chunk,
    write_atomic,
)
from fata.detection.mlatd.mlat_core import feature_chunk_paths


def _write_pair(tmp_path, rows):
    npz = tmp_path / "features.npz"
    table = tmp_path / "features.csv"
    image_ids = np.asarray([f"image-{index}" for index in range(rows)])
    matrix = np.ones((rows, 2), dtype=np.float32)
    np.savez_compressed(
        npz,
        sample_idx=np.arange(rows),
        image_id=image_ids,
        label=np.zeros(rows, dtype=np.int8),
        image_span_valid=np.ones(rows, dtype=np.int8),
        image_token_count=np.full(rows, 576, dtype=np.int32),
        llm_sequence_length=np.full(rows, 600, dtype=np.int32),
        vis_l01=matrix,
        proj_in=matrix,
        proj_out=matrix,
        llm_last_l01=matrix,
        llm_image_l01=matrix,
    )
    with table.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_idx", "Image_ID"])
        writer.writerows([[index, f"image-{index}"] for index in range(rows)])
    return npz, table


def test_partial_equal_pair_is_not_complete(tmp_path):
    npz, table = _write_pair(tmp_path, 1)
    assert not outputs_complete(npz, table, expected_rows=250)


def test_exact_expected_pair_is_complete(tmp_path):
    npz, table = _write_pair(tmp_path, 3)
    assert outputs_complete(npz, table, expected_rows=3)


def test_exact_identity_pair_is_complete_and_reordered_is_not(tmp_path):
    npz, table = _write_pair(tmp_path, 3)
    expected_ids = ["image-0", "image-1", "image-2"]
    assert outputs_complete(npz, table, 3, [0, 1, 2], expected_ids)
    assert not outputs_complete(npz, table, 3, [1, 0, 2], expected_ids)


def test_feature_chunk_path_rejects_existing_output_symlink(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    cfg = SimpleNamespace(output_dir=output, method="VisionZIP", seed=0)
    npz, _csv = feature_chunk_paths(
        cfg, "TextVQA_Open", "fata", 64, 0, 1
    )
    npz.symlink_to(tmp_path / "outside.npz")
    with pytest.raises(ValueError, match="symbolic link"):
        feature_chunk_paths(cfg, "TextVQA_Open", "fata", 64, 0, 1)


def test_cached_writer_rejects_existing_output_symlink(tmp_path):
    npz = tmp_path / "features.npz"
    table = tmp_path / "features.csv"
    npz.symlink_to(tmp_path / "outside.npz")
    rows = [
        {
            "sample_idx": 0,
            "Image_ID": "image-0",
            "image_span_valid": 1,
            "image_token_count": 576,
            "llm_sequence_length": 600,
        }
    ]
    with pytest.raises(ValueError, match="symbolic link"):
        write_atomic(
            npz,
            table,
            {"vis_l01": [np.ones(2, dtype=np.float32)]},
            rows,
            [1],
            [1],
        )


def test_run_chunk_uses_output_stem_and_writes_complete_pair(tmp_path, monkeypatch):
    """Exercise the non-resume path that previously referenced undefined ``stem``."""

    contract_path = tmp_path / "attack-contract.json"
    contract_path.write_text(
        json.dumps({"eps_255": 2.0, "alpha_255": 0.5, "steps": 100}),
        encoding="utf-8",
    )
    progress_descriptions = []

    class FakeEngine:
        cfg = SimpleNamespace(
            project_root=tmp_path,
            method="VisionZIP",
            seed=7,
            lam=1.0,
            target_k=64,
            token_budget_mode="practical",
        )
        vision_layers = [6]
        llm_layers = [8]

        def __init__(self):
            self.completeness_calls = 0

        def token_budget_for_dataset(self, _dataset):
            return 64

        def expected_paths(self, _dataset, _attack, _budget, _start, _limit):
            return tmp_path / "cached_chunk.npz", tmp_path / "cached_chunk.csv"

        def dataset_paths(self, _dataset):
            return tmp_path, tmp_path / "mapping.jsonl"

        def chunk_contract(self, **_kwargs):
            return {}

        def output_complete(self, *_args, **_kwargs):
            self.completeness_calls += 1
            return self.completeness_calls > 1

        def extract_multilevel_features(self, _image, _prompt, _budget):
            return SimpleNamespace(
                arrays={
                    name: np.ones(2, dtype=np.float32)
                    for name in (
                        "vis_l06",
                        "proj_in",
                        "proj_out",
                        "llm_last_l08",
                        "llm_image_l08",
                    )
                },
                image_span_valid=True,
                image_token_count=576,
                llm_sequence_length=600,
            )

    def fake_tqdm(iterable, *, desc, leave):
        progress_descriptions.append((desc, leave))
        return iterable

    monkeypatch.setattr(cached_extract, "tqdm", fake_tqdm)
    monkeypatch.setattr(cached_extract, "ensure_run_contract", lambda *_a, **_k: None)
    monkeypatch.setattr(cached_extract, "attack_cache_contract_path", lambda **_k: contract_path)
    monkeypatch.setattr(cached_extract, "attack_image_exists", lambda **_k: True)
    monkeypatch.setattr(cached_extract, "load_attack_image", lambda **_k: object())
    monkeypatch.setattr(cached_extract, "cache_path", lambda *_a, **_k: Path("cached.png"))

    engine = FakeEngine()
    run_chunk(
        engine=engine,
        dataset="TextVQA_Open",
        attack="cage",
        cache_attack="cage",
        qa=[{"image_filename": "image-0.png", "question": "Q?"}],
        start=0,
        limit=1,
        cache_root=tmp_path / "cache",
        output_dir=tmp_path,
        overwrite=False,
    )

    assert progress_descriptions == [("cached_chunk", False)]
    assert outputs_complete(
        tmp_path / "cached_chunk.npz",
        tmp_path / "cached_chunk.csv",
        1,
        [0],
        ["image-0.png"],
    )
