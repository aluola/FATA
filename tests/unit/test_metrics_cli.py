from __future__ import annotations

import csv

import pytest

from fata.cli.metrics import main


def _write_input(path):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "clean_full",
                "attack_full",
                "clean_practical",
                "attack_practical",
            ),
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "clean_full": 1,
                    "attack_full": 1,
                    "clean_practical": 1,
                    "attack_practical": 0,
                },
                {
                    "clean_full": 1,
                    "attack_full": 1,
                    "clean_practical": 1,
                    "attack_practical": 1,
                },
            ]
        )


def test_metrics_cli_writes_json_atomically(tmp_path):
    source = tmp_path / "source.csv"
    output = tmp_path / "metrics.json"
    _write_input(source)
    assert main([str(source), "--output", str(output)]) == 0
    assert '"cbr_denominator": 2' in output.read_text(encoding="utf-8")


def test_metrics_cli_rejects_final_output_symlink(tmp_path):
    source = tmp_path / "source.csv"
    sentinel = tmp_path / "sentinel.json"
    output = tmp_path / "metrics.json"
    _write_input(source)
    sentinel.write_text("unchanged", encoding="utf-8")
    output.symlink_to(sentinel)
    with pytest.raises(ValueError, match="symbolic link"):
        main([str(source), "--output", str(output)])
    assert sentinel.read_text(encoding="utf-8") == "unchanged"
