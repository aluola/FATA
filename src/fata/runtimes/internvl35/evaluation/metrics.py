"""Accuracy-gap, macro/pooled, CBR, and full diagnostic metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from numbers import Real
from statistics import fmean
from typing import Hashable, Iterable, Mapping, Sequence


ScoreValue = bool | int | float


def _credit(value: ScoreValue, *, field_name: str = "score") -> float:
    if not isinstance(value, Real):
        raise TypeError(f"{field_name} must be a real number or bool, got {value!r}")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{field_name} must be finite and in [0, 1], got {value!r}")
    return result


def _credits(values: Iterable[ScoreValue], *, field_name: str) -> tuple[float, ...]:
    materialized = tuple(_credit(value, field_name=field_name) for value in values)
    if not materialized:
        raise ValueError(f"{field_name} must contain at least one sample")
    return materialized


def _boolean(value: object, *, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, Real) and float(value) in {0.0, 1.0}:
        return bool(int(float(value)))
    raise TypeError(f"{field_name} must contain only bool or numeric 0/1, got {value!r}")


def _booleans(values: Iterable[object], *, field_name: str) -> tuple[bool, ...]:
    return tuple(_boolean(value, field_name=field_name) for value in values)


def _require_same_length(*named_sequences: tuple[str, Sequence[object]]) -> int:
    lengths = {name: len(sequence) for name, sequence in named_sequences}
    if len(set(lengths.values())) > 1:
        details = ", ".join(f"{name}={length}" for name, length in lengths.items())
        raise ValueError(f"paired metric inputs must have equal lengths: {details}")
    count = next(iter(lengths.values()), 0)
    if count == 0:
        raise ValueError("paired metric inputs must contain at least one sample")
    return count


def _percentage(value: Real, *, field_name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 100.0:
        raise ValueError(
            f"{field_name} must be a finite percentage in [0, 100], got {value!r}"
        )
    return result


@dataclass(frozen=True)
class AccuracyResult:
    sample_count: int
    total_credit: float
    accuracy_pct: float

    def to_record(self) -> dict[str, object]:
        return {
            "sample_count": self.sample_count,
            "total_credit": self.total_credit,
            "accuracy_pct": self.accuracy_pct,
        }


def accuracy_result(scores: Iterable[ScoreValue]) -> AccuracyResult:
    """Compute mean task credit as a percentage.

    Boolean exact-match values and fractional VQA credits are both accepted.
    """

    values = _credits(scores, field_name="scores")
    total = math.fsum(values)
    return AccuracyResult(
        sample_count=len(values),
        total_credit=total,
        accuracy_pct=100.0 * total / len(values),
    )


def accuracy_percent(scores: Iterable[ScoreValue]) -> float:
    return accuracy_result(scores).accuracy_pct


def gap_pp(clean_accuracy_pct: Real, fata_accuracy_pct: Real) -> float:
    """Return ``Clean Accuracy - FATA Accuracy`` in percentage points."""

    clean = _percentage(clean_accuracy_pct, field_name="clean_accuracy_pct")
    fata = _percentage(fata_accuracy_pct, field_name="fata_accuracy_pct")
    return clean - fata


@dataclass(frozen=True)
class AccuracyGapResult:
    sample_count: int
    clean_total_credit: float
    fata_total_credit: float
    clean_accuracy_pct: float
    fata_accuracy_pct: float
    gap_pp: float

    def to_record(self) -> dict[str, object]:
        return {
            "sample_count": self.sample_count,
            "clean_total_credit": self.clean_total_credit,
            "fata_total_credit": self.fata_total_credit,
            "clean_accuracy_pct": self.clean_accuracy_pct,
            "fata_accuracy_pct": self.fata_accuracy_pct,
            "gap_pp": self.gap_pp,
        }


def paired_accuracy_gap(
    clean_scores: Iterable[ScoreValue],
    fata_scores: Iterable[ScoreValue],
) -> AccuracyGapResult:
    """Compute paired Clean/FATA accuracies and their signed pp difference."""

    clean = _credits(clean_scores, field_name="clean_scores")
    fata = _credits(fata_scores, field_name="fata_scores")
    count = _require_same_length(("clean_scores", clean), ("fata_scores", fata))
    clean_total = math.fsum(clean)
    fata_total = math.fsum(fata)
    clean_pct = 100.0 * clean_total / count
    fata_pct = 100.0 * fata_total / count
    return AccuracyGapResult(
        sample_count=count,
        clean_total_credit=clean_total,
        fata_total_credit=fata_total,
        clean_accuracy_pct=clean_pct,
        fata_accuracy_pct=fata_pct,
        gap_pp=gap_pp(clean_pct, fata_pct),
    )


@dataclass(frozen=True)
class GroupAccuracyGapResult:
    group: Hashable
    sample_count: int
    clean_total_credit: float
    fata_total_credit: float
    clean_accuracy_pct: float
    fata_accuracy_pct: float
    gap_pp: float

    def to_record(self) -> dict[str, object]:
        return {
            "group": self.group,
            "sample_count": self.sample_count,
            "clean_total_credit": self.clean_total_credit,
            "fata_total_credit": self.fata_total_credit,
            "clean_accuracy_pct": self.clean_accuracy_pct,
            "fata_accuracy_pct": self.fata_accuracy_pct,
            "gap_pp": self.gap_pp,
        }


@dataclass(frozen=True)
class GroupedAccuracyGapSummary:
    groups: tuple[GroupAccuracyGapResult, ...]
    macro_clean_accuracy_pct: float
    macro_fata_accuracy_pct: float
    macro_gap_pp: float
    pooled_sample_count: int
    pooled_clean_accuracy_pct: float
    pooled_fata_accuracy_pct: float
    pooled_gap_pp: float

    def to_record(self) -> dict[str, object]:
        return {
            "groups": [group.to_record() for group in self.groups],
            "macro_clean_accuracy_pct": self.macro_clean_accuracy_pct,
            "macro_fata_accuracy_pct": self.macro_fata_accuracy_pct,
            "macro_gap_pp": self.macro_gap_pp,
            "pooled_sample_count": self.pooled_sample_count,
            "pooled_clean_accuracy_pct": self.pooled_clean_accuracy_pct,
            "pooled_fata_accuracy_pct": self.pooled_fata_accuracy_pct,
            "pooled_gap_pp": self.pooled_gap_pp,
        }


def summarize_grouped_accuracy_gaps(
    grouped_scores: Mapping[
        Hashable,
        Sequence[Iterable[ScoreValue]],
    ],
) -> GroupedAccuracyGapSummary:
    """Compute equal-group macro and all-sample pooled Clean/FATA gaps.

    Each mapping value is ``(clean_scores, fata_scores)`` for the same samples.
    Groups are returned in deterministic ``repr(key)`` order.  Macro statistics
    give every group equal weight; pooled statistics weight every sample equally.
    """

    if not grouped_scores:
        raise ValueError("grouped_scores must contain at least one group")

    rows: list[GroupAccuracyGapResult] = []
    for group in sorted(grouped_scores, key=repr):
        pair = grouped_scores[group]
        try:
            clean_scores, fata_scores = pair
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"group {group!r} must map to (clean_scores, fata_scores)"
            ) from exc
        result = paired_accuracy_gap(clean_scores, fata_scores)
        rows.append(
            GroupAccuracyGapResult(
                group=group,
                sample_count=result.sample_count,
                clean_total_credit=result.clean_total_credit,
                fata_total_credit=result.fata_total_credit,
                clean_accuracy_pct=result.clean_accuracy_pct,
                fata_accuracy_pct=result.fata_accuracy_pct,
                gap_pp=result.gap_pp,
            )
        )

    pooled_count = sum(row.sample_count for row in rows)
    pooled_clean_total = math.fsum(row.clean_total_credit for row in rows)
    pooled_fata_total = math.fsum(row.fata_total_credit for row in rows)
    pooled_clean_pct = 100.0 * pooled_clean_total / pooled_count
    pooled_fata_pct = 100.0 * pooled_fata_total / pooled_count
    macro_clean_pct = fmean(row.clean_accuracy_pct for row in rows)
    macro_fata_pct = fmean(row.fata_accuracy_pct for row in rows)

    return GroupedAccuracyGapSummary(
        groups=tuple(rows),
        macro_clean_accuracy_pct=macro_clean_pct,
        macro_fata_accuracy_pct=macro_fata_pct,
        macro_gap_pp=fmean(row.gap_pp for row in rows),
        pooled_sample_count=pooled_count,
        pooled_clean_accuracy_pct=pooled_clean_pct,
        pooled_fata_accuracy_pct=pooled_fata_pct,
        pooled_gap_pp=gap_pp(pooled_clean_pct, pooled_fata_pct),
    )


def macro_accuracy_percent(
    grouped_scores: Mapping[Hashable, Iterable[ScoreValue]],
) -> float:
    """Average group accuracies with equal weight per non-empty group."""

    if not grouped_scores:
        raise ValueError("grouped_scores must contain at least one group")
    return fmean(
        accuracy_percent(grouped_scores[group])
        for group in sorted(grouped_scores, key=repr)
    )


def pooled_accuracy_percent(
    grouped_scores: Mapping[Hashable, Iterable[ScoreValue]],
) -> float:
    """Compute accuracy after pooling every group sample together."""

    if not grouped_scores:
        raise ValueError("grouped_scores must contain at least one group")
    group_credits: list[float] = []
    total_count = 0
    for group in sorted(grouped_scores, key=repr):
        result = accuracy_result(grouped_scores[group])
        group_credits.append(result.total_credit)
        total_count += result.sample_count
    return 100.0 * math.fsum(group_credits) / total_count


@dataclass(frozen=True)
class CBRResult:
    sample_count: int
    denominator: int
    numerator: int
    cbr: float | None
    cbr_pct: float | None

    @property
    def defined(self) -> bool:
        return self.cbr is not None

    def to_record(self) -> dict[str, object]:
        return {
            "sample_count": self.sample_count,
            "cbr_denominator": self.denominator,
            "cbr_numerator": self.numerator,
            "cbr": self.cbr,
            "cbr_pct": self.cbr_pct,
            "cbr_defined": self.defined,
        }


def compute_cbr(
    clean_full_correct: Iterable[object],
    fata_full_correct: Iterable[object],
    fata_practical_correct: Iterable[object],
) -> CBRResult:
    """Compute the project's paired compression-break rate.

    ``denominator = count(CF and AF)`` and
    ``numerator = count(CF and AF and not AP)``.  Clean-practical correctness is
    intentionally absent.  A zero denominator yields ``None`` rather than a
    fabricated zero or NaN.
    """

    clean_full = _booleans(clean_full_correct, field_name="clean_full_correct")
    fata_full = _booleans(fata_full_correct, field_name="fata_full_correct")
    fata_practical = _booleans(
        fata_practical_correct, field_name="fata_practical_correct"
    )
    count = _require_same_length(
        ("clean_full_correct", clean_full),
        ("fata_full_correct", fata_full),
        ("fata_practical_correct", fata_practical),
    )
    denominator = sum(cf and af for cf, af in zip(clean_full, fata_full))
    numerator = sum(
        cf and af and not ap
        for cf, af, ap in zip(clean_full, fata_full, fata_practical)
    )
    rate = numerator / denominator if denominator else None
    return CBRResult(
        sample_count=count,
        denominator=denominator,
        numerator=numerator,
        cbr=rate,
        cbr_pct=None if rate is None else 100.0 * rate,
    )


@dataclass(frozen=True)
class GroupCBRResult:
    group: Hashable
    sample_count: int
    denominator: int
    numerator: int
    cbr: float | None
    cbr_pct: float | None

    def to_record(self) -> dict[str, object]:
        return {
            "group": self.group,
            "sample_count": self.sample_count,
            "cbr_denominator": self.denominator,
            "cbr_numerator": self.numerator,
            "cbr": self.cbr,
            "cbr_pct": self.cbr_pct,
            "cbr_defined": self.cbr is not None,
        }


@dataclass(frozen=True)
class GroupedCBRSummary:
    groups: tuple[GroupCBRResult, ...]
    group_count: int
    defined_group_count: int
    macro_cbr: float | None
    macro_cbr_pct: float | None
    pooled: CBRResult

    def to_record(self) -> dict[str, object]:
        return {
            "groups": [group.to_record() for group in self.groups],
            "group_count": self.group_count,
            "defined_group_count": self.defined_group_count,
            "macro_cbr": self.macro_cbr,
            "macro_cbr_pct": self.macro_cbr_pct,
            "pooled_cbr": self.pooled.cbr,
            "pooled_cbr_pct": self.pooled.cbr_pct,
            "pooled_cbr_denominator": self.pooled.denominator,
            "pooled_cbr_numerator": self.pooled.numerator,
        }


def summarize_grouped_cbr(
    grouped_correctness: Mapping[
        Hashable,
        Sequence[Iterable[object]],
    ],
) -> GroupedCBRSummary:
    """Compute equal-group macro CBR and count-pooled CBR."""

    if not grouped_correctness:
        raise ValueError("grouped_correctness must contain at least one group")

    rows: list[GroupCBRResult] = []
    pooled_sample_count = 0
    pooled_denominator = 0
    pooled_numerator = 0
    for group in sorted(grouped_correctness, key=repr):
        triple = grouped_correctness[group]
        try:
            clean_full, fata_full, fata_practical = triple
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "each group must map to "
                "(clean_full_correct, fata_full_correct, fata_practical_correct)"
            ) from exc
        result = compute_cbr(clean_full, fata_full, fata_practical)
        rows.append(
            GroupCBRResult(
                group=group,
                sample_count=result.sample_count,
                denominator=result.denominator,
                numerator=result.numerator,
                cbr=result.cbr,
                cbr_pct=result.cbr_pct,
            )
        )
        pooled_sample_count += result.sample_count
        pooled_denominator += result.denominator
        pooled_numerator += result.numerator

    defined_rates = [row.cbr for row in rows if row.cbr is not None]
    macro = fmean(defined_rates) if defined_rates else None
    pooled_rate = (
        pooled_numerator / pooled_denominator if pooled_denominator else None
    )
    pooled = CBRResult(
        sample_count=pooled_sample_count,
        denominator=pooled_denominator,
        numerator=pooled_numerator,
        cbr=pooled_rate,
        cbr_pct=None if pooled_rate is None else 100.0 * pooled_rate,
    )
    return GroupedCBRSummary(
        groups=tuple(rows),
        group_count=len(rows),
        defined_group_count=len(defined_rates),
        macro_cbr=macro,
        macro_cbr_pct=None if macro is None else 100.0 * macro,
        pooled=pooled,
    )


@dataclass(frozen=True)
class FullDiagnostic:
    """Full-token accuracy and paired survival statistics for reporting only."""

    sample_count: int
    clean_full_correct_count: int
    fata_full_correct_count: int
    clean_full_accuracy_pct: float
    fata_full_accuracy_pct: float
    survival_denominator: int
    survival_numerator: int
    sr: float | None
    sr_pct: float | None
    purpose: str = field(default="diagnostic_only", init=False)

    def to_record(self) -> dict[str, object]:
        return {
            "sample_count": self.sample_count,
            "clean_full_correct_count": self.clean_full_correct_count,
            "fata_full_correct_count": self.fata_full_correct_count,
            "clean_full_accuracy_pct": self.clean_full_accuracy_pct,
            "fata_full_accuracy_pct": self.fata_full_accuracy_pct,
            "sr_denominator": self.survival_denominator,
            "sr_numerator": self.survival_numerator,
            "sr": self.sr,
            "sr_pct": self.sr_pct,
            "purpose": self.purpose,
        }


def compute_full_diagnostic(
    clean_full_correct: Iterable[object],
    fata_full_correct: Iterable[object],
) -> FullDiagnostic:
    """Report Full accuracies and ``SR = count(CF & AF) / count(CF)``.

    The result contains measurements only.  It has no threshold, pass/fail
    state, candidate rank, or selection behavior.
    """

    clean = _booleans(clean_full_correct, field_name="clean_full_correct")
    fata = _booleans(fata_full_correct, field_name="fata_full_correct")
    count = _require_same_length(
        ("clean_full_correct", clean), ("fata_full_correct", fata)
    )
    clean_count = sum(clean)
    fata_count = sum(fata)
    survived = sum(cf and af for cf, af in zip(clean, fata))
    rate = survived / clean_count if clean_count else None
    return FullDiagnostic(
        sample_count=count,
        clean_full_correct_count=clean_count,
        fata_full_correct_count=fata_count,
        clean_full_accuracy_pct=100.0 * clean_count / count,
        fata_full_accuracy_pct=100.0 * fata_count / count,
        survival_denominator=clean_count,
        survival_numerator=survived,
        sr=rate,
        sr_pct=None if rate is None else 100.0 * rate,
    )


__all__ = [
    "AccuracyGapResult",
    "AccuracyResult",
    "CBRResult",
    "FullDiagnostic",
    "GroupAccuracyGapResult",
    "GroupCBRResult",
    "GroupedAccuracyGapSummary",
    "GroupedCBRSummary",
    "accuracy_percent",
    "accuracy_result",
    "compute_cbr",
    "compute_full_diagnostic",
    "gap_pp",
    "macro_accuracy_percent",
    "paired_accuracy_gap",
    "pooled_accuracy_percent",
    "summarize_grouped_accuracy_gaps",
    "summarize_grouped_cbr",
]
