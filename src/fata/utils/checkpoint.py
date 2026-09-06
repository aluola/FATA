"""Minimal append-only checkpoint helpers for sample-level resume."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def completed_ids(path: str | Path) -> set[str]:
    result: set[str] = set()
    file_path = Path(path)
    if not file_path.exists():
        return result
    with file_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            sample_id = str(record.get("sample_id", "")).strip()
            if not sample_id:
                raise ValueError(f"{file_path}:{line_number}: missing sample_id")
            if sample_id in result:
                raise ValueError(f"{file_path}:{line_number}: duplicate sample_id {sample_id!r}")
            result.add(sample_id)
    return result


def pending_records(records: Iterable[dict], done: set[str]) -> list[dict]:
    pending, seen = [], set(done)
    for record in records:
        sample_id = str(record.get("sample_id", "")).strip()
        if not sample_id:
            raise ValueError("record missing sample_id")
        if sample_id in seen:
            continue
        pending.append(record)
        seen.add(sample_id)
    return pending
