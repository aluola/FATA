"""Task scoring for normalized VQA answers and A/B/C/D choices."""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata
from typing import Iterable

from .output_parsing import (
    MULTIPLE_CHOICE_TASK,
    OPEN_TASK,
    ParsedOutput,
    TaskType,
    normalize_vqa_answer,
    parse_mc_answer,
    parse_open_answer,
)


VQA_SCORING_RULE = "min(exact_normalized_reference_matches / 3, 1)"
MC_SCORING_RULE = "exact parsed option match in A/B/C/D"

_MC_REFERENCE_RE = re.compile(
    r"^\s*(?:option\s+)?[\(\[]?\s*([A-D])\s*[\)\]]?\s*[\.!]?\s*$",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class AnswerScore:
    """Deterministic flattened fields for one generated answer.

    ``score`` is the task accuracy credit: fractional VQA consensus credit for
    open answers and 0/1 for multiple choice. ``correct`` means non-zero task
    credit (at least one normalized reference match); ``full_credit`` records
    whether the score is exactly one. Keeping all three fields avoids silently
    converting a fractional VQA score into an unspecified Boolean policy.
    """

    task_type: TaskType
    raw_output: str
    parsed_answer: str | None
    parse_status: str
    parse_strategy: str
    reference_answers: tuple[str, ...]
    normalized_reference_answers: tuple[str, ...]
    match_count: int
    score: float
    correct: bool
    full_credit: bool
    scoring_rule: str

    def to_record(self) -> dict[str, object]:
        """Return stable field names and ordering for sample-level results."""

        return {
            "task_type": self.task_type,
            "raw_output": self.raw_output,
            "parsed_answer": self.parsed_answer,
            "parse_status": self.parse_status,
            "parse_strategy": self.parse_strategy,
            "reference_answers": list(self.reference_answers),
            "normalized_reference_answers": list(
                self.normalized_reference_answers
            ),
            "match_count": self.match_count,
            "score": self.score,
            "correct": self.correct,
            "full_credit": self.full_credit,
            "scoring_rule": self.scoring_rule,
        }


def _materialize_references(references: Iterable[object]) -> tuple[str, ...]:
    materialized = tuple("" if value is None else str(value) for value in references)
    if not materialized:
        raise ValueError("at least one reference answer is required")
    return materialized


def normalize_mc_reference(reference: object) -> str:
    """Normalize a ground-truth option and reject anything outside A/B/C/D."""

    text = unicodedata.normalize("NFKC", "" if reference is None else str(reference))
    match = _MC_REFERENCE_RE.fullmatch(text)
    if not match:
        raise ValueError(f"invalid multiple-choice reference {reference!r}; expected A/B/C/D")
    return match.group(1).upper()


def _answer_score(
    parsed: ParsedOutput,
    references: tuple[str, ...],
    normalized_references: tuple[str, ...],
    match_count: int,
    score: float,
    scoring_rule: str,
) -> AnswerScore:
    return AnswerScore(
        task_type=parsed.task_type,
        raw_output=parsed.raw_output,
        parsed_answer=parsed.parsed_answer,
        parse_status=parsed.status,
        parse_strategy=parsed.strategy,
        reference_answers=references,
        normalized_reference_answers=normalized_references,
        match_count=match_count,
        score=float(score),
        correct=score > 0.0,
        full_credit=score >= 1.0,
        scoring_rule=scoring_rule,
    )


def score_vqa_answer(
    raw_output: object,
    reference_answers: Iterable[object],
) -> AnswerScore:
    """Score an open answer with exact normalized VQA consensus matching.

    Duplicate human answers are intentionally retained.  If ``m`` references
    exactly equal the normalized prediction, credit is ``min(m / 3, 1)``.  No
    substring fallback is used, so a verbose phrase does not receive credit for
    merely containing a reference answer.
    """

    references = _materialize_references(reference_answers)
    normalized_references = tuple(normalize_vqa_answer(ref) for ref in references)
    parsed = parse_open_answer(raw_output)
    predicted = parsed.parsed_answer
    match_count = (
        normalized_references.count(predicted) if predicted is not None else 0
    )
    score = min(1.0, match_count / 3.0)
    return _answer_score(
        parsed,
        references,
        normalized_references,
        match_count,
        score,
        VQA_SCORING_RULE,
    )


def score_mc_answer(
    raw_output: object,
    reference_answers: Iterable[object] | object,
) -> AnswerScore:
    """Score a generated option against one or more strict A/B/C/D references."""

    if isinstance(reference_answers, (str, bytes)) or reference_answers is None:
        reference_values: Iterable[object] = (reference_answers,)
    else:
        try:
            iter(reference_answers)  # type: ignore[arg-type]
        except TypeError:
            reference_values = (reference_answers,)
        else:
            reference_values = reference_answers  # type: ignore[assignment]

    references = _materialize_references(reference_values)
    normalized_references = tuple(normalize_mc_reference(ref) for ref in references)
    parsed = parse_mc_answer(raw_output)
    predicted = parsed.parsed_answer
    match_count = (
        normalized_references.count(predicted) if predicted is not None else 0
    )
    score = 1.0 if match_count > 0 else 0.0
    return _answer_score(
        parsed,
        references,
        normalized_references,
        match_count,
        score,
        MC_SCORING_RULE,
    )


def score_answer(
    raw_output: object,
    reference_answers: Iterable[object],
    task_type: TaskType,
) -> AnswerScore:
    """Dispatch scoring using the mapping's explicit task type."""

    if task_type == OPEN_TASK:
        return score_vqa_answer(raw_output, reference_answers)
    if task_type == MULTIPLE_CHOICE_TASK:
        return score_mc_answer(raw_output, reference_answers)
    raise ValueError(
        f"unsupported task_type {task_type!r}; expected 'open' or 'multiple_choice'"
    )


__all__ = [
    "AnswerScore",
    "MC_SCORING_RULE",
    "VQA_SCORING_RULE",
    "normalize_mc_reference",
    "score_answer",
    "score_mc_answer",
    "score_vqa_answer",
]
