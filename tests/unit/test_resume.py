import json
import pytest

from fata.utils.checkpoint import completed_ids, pending_records


def test_resume_skips_completed_and_duplicate_input(tmp_path):
    checkpoint = tmp_path / "progress.jsonl"
    checkpoint.write_text(json.dumps({"sample_id": "a"}) + "\n", encoding="utf-8")
    done = completed_ids(checkpoint)
    records = [{"sample_id": "a"}, {"sample_id": "b"}, {"sample_id": "b"}]
    assert pending_records(records, done) == [{"sample_id": "b"}]


def test_corrupt_checkpoint_fails(tmp_path):
    checkpoint = tmp_path / "progress.jsonl"
    checkpoint.write_text('{"sample_id":"a"}\n{"sample_id":"a"}\n', encoding="utf-8")
    with pytest.raises(ValueError):
        completed_ids(checkpoint)
