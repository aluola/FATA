from __future__ import annotations

import json

import pytest

from fata.runtimes.internvl35.evaluation.datasets import atomic_write_jsonl


def test_internvl_manifest_writer_round_trip(tmp_path):
    destination = tmp_path / "manifest.jsonl"
    atomic_write_jsonl(destination, [{"sample_index": 0}, {"sample_index": 1}])
    assert [json.loads(line) for line in destination.read_text().splitlines()] == [
        {"sample_index": 0},
        {"sample_index": 1},
    ]


def test_internvl_manifest_writer_rejects_final_symlink(tmp_path):
    sentinel = tmp_path / "sentinel.jsonl"
    destination = tmp_path / "manifest.jsonl"
    sentinel.write_text("unchanged", encoding="utf-8")
    destination.symlink_to(sentinel)
    with pytest.raises(ValueError, match="symbolic link"):
        atomic_write_jsonl(destination, [{"sample_index": 0}])
    assert sentinel.read_text(encoding="utf-8") == "unchanged"
