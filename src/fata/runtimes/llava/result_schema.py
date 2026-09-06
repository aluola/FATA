"""Strict answer-plus-score persistence for resumable LLaVA evaluations.

The historical paper CSVs contain only scalar scores.  They remain useful as
provenance, but a scalar alone cannot be checked against the current mapping
and scorer during resume.  New formal runs therefore persist every decoded
answer next to an immutable description of this v2 result schema.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from fata.evaluation.scoring import score_mc, score_vqa


RESULT_SCHEMA_NAME = "llava_answer_score_v2"
RAW_ANSWER_SUFFIX = "__raw_answer"
SCORE_FORMAT = "canonical_fixed_2dp"


def raw_answer_column(score_column: str) -> str:
    """Return the unambiguous raw-answer column paired with one score."""

    if not isinstance(score_column, str) or not score_column:
        raise ValueError("score columns must be non-empty strings")
    return f"{score_column}{RAW_ANSWER_SUFFIX}"


def _validated_columns(columns: Sequence[str], *, label: str) -> list[str]:
    values = list(columns)
    if not values or any(not isinstance(value, str) or not value for value in values):
        raise ValueError(f"{label} must contain non-empty strings")
    if len(values) != len(set(values)):
        raise ValueError(f"{label} must be unique")
    return values


def result_header(prefix_columns: Sequence[str], score_columns: Sequence[str]) -> list[str]:
    """Build the v2 CSV header while retaining historical score names/order."""

    prefix = _validated_columns(prefix_columns, label="prefix columns")
    scores = _validated_columns(score_columns, label="score columns")
    answers = [raw_answer_column(column) for column in scores]
    header = [*prefix, *scores, *answers]
    if len(header) != len(set(header)):
        raise ValueError("prefix, score, and raw-answer columns must not collide")
    return header


def result_schema(score_columns: Sequence[str]) -> dict[str, Any]:
    """Return the exact portable descriptor stored in each run contract."""

    scores = _validated_columns(score_columns, label="score columns")
    return {
        "name": RESULT_SCHEMA_NAME,
        "version": 2,
        "score_format": SCORE_FORMAT,
        "scorers": {
            "multiple_choice": "fata.evaluation.scoring.score_mc",
            "open": "fata.evaluation.scoring.score_vqa",
        },
        "score_answer_pairs": [
            {
                "score_column": column,
                "raw_answer_column": raw_answer_column(column),
            }
            for column in scores
        ],
    }


def canonical_score_text(raw_answer: str, ground_truth: Mapping[str, Any]) -> str:
    """Recompute one score with the canonical scorer and fixed two decimals.

    Empty decoded answers are valid model outputs and are intentionally scored.
    Mapping/scorer errors propagate instead of being converted to a plausible
    all-zero result.
    """

    if not isinstance(raw_answer, str):
        raise TypeError("raw answer must be a string")
    if not isinstance(ground_truth, Mapping):
        raise TypeError("ground truth must be a mapping")
    answers = ground_truth.get("answers")
    if not isinstance(answers, list) or not answers:
        raise ValueError("ground truth answers must be a non-empty list")
    question_type = ground_truth.get("type", "open")
    if question_type == "multiple_choice":
        value = score_mc(raw_answer, answers[0])
    elif question_type == "open":
        value = score_vqa(raw_answer, answers)
    else:
        raise ValueError(f"unsupported question type: {question_type!r}")
    return f"{value:.2f}"


def answer_score_cells(
    score_columns: Sequence[str],
    raw_answers: Mapping[str, str],
    ground_truth: Mapping[str, Any],
) -> list[str]:
    """Serialize scores followed by their raw answers in v2 header order."""

    scores = _validated_columns(score_columns, label="score columns")
    if not isinstance(raw_answers, Mapping):
        raise TypeError("raw_answers must be a mapping keyed by score column")
    missing = [column for column in scores if column not in raw_answers]
    extra = sorted(set(raw_answers).difference(scores))
    if missing or extra:
        raise ValueError(
            "raw answer keys do not exactly match score columns: "
            f"missing={missing} extra={extra}"
        )
    answers = [raw_answers[column] for column in scores]
    if any(not isinstance(answer, str) for answer in answers):
        raise TypeError("every raw answer must be a string")
    score_text = [canonical_score_text(answer, ground_truth) for answer in answers]
    return [*score_text, *answers]


def _ground_truth_by_id(
    rows: Iterable[Mapping[str, Any]], id_key: str
) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError(f"ground-truth row {index} is not a mapping")
        identifier = row.get(id_key)
        if not isinstance(identifier, str) or not identifier:
            raise ValueError(f"ground-truth row {index} has invalid {id_key!r}")
        if identifier in indexed:
            raise ValueError(f"duplicate ground-truth identifier: {identifier}")
        indexed[identifier] = row
    if not indexed:
        raise ValueError("ground-truth cohort is empty")
    return indexed


def read_completed_result_ids(
    path: str | Path,
    *,
    prefix_columns: Sequence[str],
    score_columns: Sequence[str],
    ground_truth_rows: Iterable[Mapping[str, Any]],
    id_key: str = "image_filename",
    id_column: str = "Image_ID",
    expected_values: Mapping[str, Mapping[str, str]] | None = None,
) -> set[str]:
    """Strictly validate/re-score a v2 CSV and return completed identifiers.

    This deliberately rejects historical score-only CSVs.  Formal resume is
    allowed only when every persisted score can be recomputed from its raw
    answer and the current mapping ground truth.
    """

    source = Path(path)
    if not source.exists():
        return set()
    prefix = _validated_columns(prefix_columns, label="prefix columns")
    scores = _validated_columns(score_columns, label="score columns")
    header = result_header(prefix, scores)
    try:
        identifier_index = header.index(id_column)
    except ValueError as error:
        raise ValueError(f"id column is absent from result header: {id_column}") from error
    truth = _ground_truth_by_id(ground_truth_rows, id_key)
    if expected_values is not None:
        unknown_ids = set(expected_values).difference(truth)
        if unknown_ids:
            raise ValueError(
                f"expected values contain identifiers outside ground truth: {sorted(unknown_ids)}"
            )
        unknown_columns = {
            column
            for values in expected_values.values()
            for column in values
            if column not in header
        }
        if unknown_columns:
            raise ValueError(
                f"expected-value columns are absent from header: {sorted(unknown_columns)}"
            )

    score_indices = {column: header.index(column) for column in scores}
    answer_indices = {
        column: header.index(raw_answer_column(column)) for column in scores
    }
    completed: set[str] = set()
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        stored_header = next(reader, None)
        if stored_header != header:
            raise RuntimeError(
                f"unexpected LLaVA answer+score v2 CSV header in {source}"
            )
        for line_number, row in enumerate(reader, start=2):
            if len(row) != len(header):
                raise RuntimeError(
                    f"incomplete CSV row at {source}:{line_number}: "
                    f"expected {len(header)} columns, got {len(row)}"
                )
            identifier = row[identifier_index]
            if not identifier:
                raise RuntimeError(f"empty result identifier at {source}:{line_number}")
            if identifier in completed:
                raise RuntimeError(f"duplicate result identifier in {source}: {identifier}")
            if identifier not in truth:
                raise RuntimeError(f"unexpected result identifier in {source}: {identifier}")
            if expected_values is not None:
                for column, expected in expected_values.get(identifier, {}).items():
                    if row[header.index(column)] != expected:
                        raise RuntimeError(
                            f"{column} mismatch for {identifier} at {source}:{line_number}"
                        )
            for column in scores:
                raw_answer = row[answer_indices[column]]
                try:
                    recomputed = canonical_score_text(raw_answer, truth[identifier])
                except (TypeError, ValueError) as error:
                    raise RuntimeError(
                        f"cannot canonically rescore {column} for {identifier} "
                        f"at {source}:{line_number}: {error}"
                    ) from error
                stored_score = row[score_indices[column]]
                if stored_score != recomputed:
                    raise RuntimeError(
                        f"score/raw-answer mismatch for {column} and {identifier} "
                        f"at {source}:{line_number}: stored={stored_score!r} "
                        f"recomputed={recomputed!r}"
                    )
            completed.add(identifier)
    return completed
