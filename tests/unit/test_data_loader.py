from pathlib import Path

from fata.data.records import iter_records


def test_fixture_loader_reads_open_and_mc():
    fixture = Path(__file__).parents[1] / "fixtures" / "tiny_mapping.jsonl"
    records = list(iter_records(fixture))
    assert [record["type"] for record in records] == ["open", "multiple_choice"]
    assert records[1]["options"] == ["red", "blue"]
