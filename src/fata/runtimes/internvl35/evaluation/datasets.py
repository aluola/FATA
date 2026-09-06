"""Dataset loading and deterministic manifest construction.

The mapping order is authoritative.  No manifest builder in this project may
filter on correctness, tile count, or an attack outcome.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

from fata.utils.paths import resolve_dataset_relative_path, resolve_owned_output_path


DATASET_NAMES: tuple[str, ...] = (
    "TextVQA_Open",
    "VQAv2_Open",
    "ScienceQA_MC",
    "VQAv2_MC",
)

PRACTICAL_RATIOS: dict[str, str] = {
    "TextVQA_Open": "1/9",
    "VQAv2_Open": "1/9",
    "ScienceQA_MC": "1/18",
    "VQAv2_MC": "1/18",
}


@dataclass(frozen=True)
class DatasetSample:
    dataset: str
    sample_index: int
    image_filename: str
    image_path: Path
    task_type: str
    question: str
    answers: tuple[str, ...]
    options: tuple[str, ...]
    ground_truth_text: str | None

    @property
    def image_id(self) -> str:
        return Path(self.image_filename).stem

    def as_manifest_record(self) -> dict[str, object]:
        return {
            "dataset": self.dataset,
            "sample_index": self.sample_index,
            "image_id": self.image_id,
            "image_filename": self.image_filename,
            "image_path": str(self.image_path),
            "type": self.task_type,
            "question": self.question,
            "answers": list(self.answers),
            "options": list(self.options),
            "ground_truth_text": self.ground_truth_text,
        }


def mapping_path(dataset_root: str | Path, dataset: str) -> Path:
    if dataset not in DATASET_NAMES:
        raise ValueError(f"unsupported dataset {dataset!r}; expected one of {DATASET_NAMES}")
    root = Path(dataset_root).expanduser().resolve()
    return root / dataset / f"{dataset}_mapping.jsonl"


def iter_mapping(dataset_root: str | Path, dataset: str) -> Iterator[DatasetSample]:
    path = mapping_path(dataset_root, dataset)
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{index + 1}") from exc
            filename = str(raw["image_filename"])
            declared_dataset = str(raw.get("dataset", dataset))
            if declared_dataset != dataset:
                raise ValueError(
                    f"dataset mismatch at {path}:{index + 1}: "
                    f"{declared_dataset!r} != {dataset!r}"
                )
            yield DatasetSample(
                dataset=dataset,
                sample_index=index,
                image_filename=filename,
                image_path=resolve_dataset_relative_path(path.parent, filename),
                task_type=str(raw["type"]),
                question=str(raw["question"]),
                answers=tuple(str(answer) for answer in raw.get("answers", ())),
                options=tuple(str(option) for option in raw.get("options", ())),
                ground_truth_text=(
                    None if raw.get("ground_truth_text") is None else str(raw["ground_truth_text"])
                ),
            )


def load_mapping(
    dataset_root: str | Path,
    dataset: str,
    *,
    expected_count: int | None = 1000,
) -> list[DatasetSample]:
    samples = list(iter_mapping(dataset_root, dataset))
    if expected_count is not None and len(samples) != expected_count:
        raise ValueError(f"{dataset} has {len(samples)} rows; expected {expected_count}")
    indices = [sample.sample_index for sample in samples]
    if indices != list(range(len(samples))):
        raise ValueError(f"{dataset} mapping indices are not contiguous in source order")
    return samples


def select_prefix_manifest(
    dataset_root: str | Path,
    per_dataset: int,
    *,
    datasets: Sequence[str] = DATASET_NAMES,
) -> list[DatasetSample]:
    """Select the literal mapping prefix for each dataset.

    This intentionally performs no validity or correctness filtering. Missing
    or corrupt images remain in the manifest and are surfaced by preflight or
    by the runner's per-stage failure records.
    """

    if per_dataset < 1:
        raise ValueError("per_dataset must be positive")
    selected: list[DatasetSample] = []
    for dataset in datasets:
        selected.extend(load_mapping(dataset_root, dataset)[:per_dataset])
    return selected


def atomic_write_jsonl(path: str | Path, records: Iterable[Mapping[str, object]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination = resolve_owned_output_path(destination.parent, destination.name)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.tmp-", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(dict(record), ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        fd = -1
        destination = resolve_owned_output_path(destination.parent, destination.name)
        os.replace(temporary, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if fd >= 0:
            os.close(fd)
        if temporary.exists():
            temporary.unlink()


def write_prefix_manifest(
    path: str | Path,
    dataset_root: str | Path,
    per_dataset: int,
    *,
    datasets: Sequence[str] = DATASET_NAMES,
) -> list[DatasetSample]:
    samples = select_prefix_manifest(dataset_root, per_dataset, datasets=datasets)
    atomic_write_jsonl(path, (sample.as_manifest_record() for sample in samples))
    return samples
