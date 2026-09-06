"""Read-only JSONL loader for both legacy and unique-image mappings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator


def validate_record(record: dict, *, source: str = "record") -> dict:
    missing = [key for key in ("question", "answers", "image_filename") if key not in record]
    if missing:
        raise ValueError(f"{source}: missing fields {missing}")
    if not isinstance(record["answers"], list) or not record["answers"]:
        raise ValueError(f"{source}: answers must be a non-empty list")
    qtype = record.get("type", "open")
    if qtype not in {"open", "multiple_choice"}:
        raise ValueError(f"{source}: unsupported type {qtype!r}")
    return record


def iter_records(mapping: str | Path, *, limit: int | None = None) -> Iterator[dict]:
    path = Path(mapping)
    if limit is not None and limit < 0:
        raise ValueError("limit cannot be negative")
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle, start=1):
            if limit is not None and index > limit:
                break
            if not line.strip():
                continue
            yield validate_record(json.loads(line), source=f"{path}:{index}")
