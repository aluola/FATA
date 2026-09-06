"""Compute auditable paper metrics from a sample-aligned CSV."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path

from fata.evaluation.metrics import compute_paper_metrics
from fata.utils.paths import resolve_owned_output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--clean-full", default="clean_full")
    parser.add_argument("--attack-full", default="attack_full")
    parser.add_argument("--clean-practical", default="clean_practical")
    parser.add_argument("--attack-practical", default="attack_practical")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_path = args.input.expanduser().resolve()
    output_path = None
    if args.output:
        requested_output = args.output.expanduser()
        output_path = resolve_owned_output_path(
            requested_output.parent, requested_output.name
        )
    if output_path == input_path:
        raise ValueError("--output must not overwrite the input CSV")
    names = (args.clean_full, args.attack_full, args.clean_practical, args.attack_practical)
    with input_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = set(names) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"missing metric columns: {sorted(missing)}")
        columns = {name: [] for name in names}
        for row in reader:
            for name in names:
                columns[name].append(float(row[name]))
    result = compute_paper_metrics(*(columns[name] for name in names)).to_dict()
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{output_path.name}.tmp.", dir=output_path.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(rendered)
                handle.flush()
                os.fsync(handle.fileno())
            fd = -1
            output_path = resolve_owned_output_path(
                output_path.parent, output_path.name
            )
            os.replace(temporary, output_path)
        finally:
            if fd >= 0:
                os.close(fd)
            if temporary.exists():
                temporary.unlink()
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
