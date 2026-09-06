"""Definitions for FATA accuracy, transitions, SR, ASR, CBR, and damage."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np


def _vector(values: Iterable[float], name: str) -> np.ndarray:
    result = np.asarray(list(values), dtype=float)
    if result.ndim != 1 or result.size == 0:
        raise ValueError(f"{name} must be a non-empty 1-D vector")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains NaN or infinity")
    if ((result < 0) | (result > 1)).any():
        raise ValueError(f"{name} must contain scores in [0, 1]")
    return result


@dataclass(frozen=True)
class PaperMetrics:
    sample_count: int
    clean_full_accuracy: float
    attack_full_accuracy: float
    clean_practical_accuracy: float
    attack_practical_accuracy: float
    sr_percent: float
    asr_numerator: int
    asr_denominator: int
    asr_percent: float | None
    cbr_numerator: int
    cbr_denominator: int
    cbr_percent: float | None
    full_damage_pp: float
    practical_damage_pp: float
    amplification_pp: float
    cc: int
    cw: int
    wc: int
    ww: int
    net_harm: int

    def to_dict(self) -> dict[str, int | float | None]:
        return asdict(self)


def compute_paper_metrics(
    clean_full: Iterable[float],
    attack_full: Iterable[float],
    clean_practical: Iterable[float],
    attack_practical: Iterable[float],
) -> PaperMetrics:
    cf = _vector(clean_full, "clean_full")
    af = _vector(attack_full, "attack_full")
    cp = _vector(clean_practical, "clean_practical")
    ap = _vector(attack_practical, "attack_practical")
    if len({len(cf), len(af), len(cp), len(ap)}) != 1:
        raise ValueError("metric vectors must be sample-aligned and equal length")
    cf_correct, af_correct = np.isclose(cf, 1.0), np.isclose(af, 1.0)
    cp_correct, ap_correct = np.isclose(cp, 1.0), np.isclose(ap, 1.0)
    clean_acc, attack_acc = float(cf.mean()), float(af.mean())
    if clean_acc <= 0:
        raise ZeroDivisionError("SR is undefined because Clean Full accuracy is zero")
    asr_eligible = cp_correct
    asr_num, asr_den = int((asr_eligible & ~ap_correct).sum()), int(asr_eligible.sum())
    cbr_eligible = cf_correct & af_correct
    cbr_num, cbr_den = int((cbr_eligible & ~ap_correct).sum()), int(cbr_eligible.sum())
    cc = int((cf_correct & af_correct).sum())
    cw = int((cf_correct & ~af_correct).sum())
    wc = int((~cf_correct & af_correct).sum())
    ww = int((~cf_correct & ~af_correct).sum())
    full_damage = 100.0 * (clean_acc - attack_acc)
    practical_damage = 100.0 * (float(cp.mean()) - float(ap.mean()))
    return PaperMetrics(
        sample_count=len(cf), clean_full_accuracy=clean_acc,
        attack_full_accuracy=attack_acc, clean_practical_accuracy=float(cp.mean()),
        attack_practical_accuracy=float(ap.mean()),
        sr_percent=100.0 * attack_acc / clean_acc,
        asr_numerator=asr_num, asr_denominator=asr_den,
        asr_percent=(100.0 * asr_num / asr_den if asr_den else None),
        cbr_numerator=cbr_num, cbr_denominator=cbr_den,
        cbr_percent=(100.0 * cbr_num / cbr_den if cbr_den else None),
        full_damage_pp=full_damage, practical_damage_pp=practical_damage,
        amplification_pp=practical_damage - full_damage,
        cc=cc, cw=cw, wc=wc, ww=ww, net_harm=cw - wc,
    )


def retained_set_diagnostics(clean_indices: Iterable[int], attack_indices: Iterable[int]) -> dict[str, float | int]:
    clean, attack = set(clean_indices), set(attack_indices)
    union = clean | attack
    intersection = clean & attack
    return {
        "clean_count": len(clean), "attack_count": len(attack),
        "intersection": len(intersection), "union": len(union),
        "jaccard": (len(intersection) / len(union) if union else 1.0),
        "flip_rate": (len(clean ^ attack) / len(union) if union else 0.0),
    }
