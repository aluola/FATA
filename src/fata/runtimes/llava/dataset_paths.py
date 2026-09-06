"""Central dataset-root configuration shared by legacy experiment entrypoints."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


DEFAULT_DATASET_ROOT = os.environ.get("FATA_DATA_ROOT")


def add_dataset_root_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help=(
            "Root containing <dataset>/<dataset>_mapping.jsonl. "
            "Defaults to FATA_DATA_ROOT. No server-specific fallback is embedded."
        ),
    )


def resolve_dataset_paths(dataset_root: str | Path, dataset: str) -> tuple[str, str]:
    if dataset_root is None:
        raise ValueError("set FATA_DATA_ROOT or pass --dataset-root")
    root = Path(dataset_root).expanduser().resolve()
    dataset_directory = root / dataset
    mapping = dataset_directory / f"{dataset}_mapping.jsonl"
    return str(dataset_directory), str(mapping)
