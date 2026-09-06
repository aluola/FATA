"""Dependency-light detection metrics and calibration-only FPR threshold."""

from __future__ import annotations

import numpy as np


def _finite_vector(values, name: str) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    if array.ndim != 1 or not array.size or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite, non-empty 1-D vector")
    return array


def fit_fpr_threshold(negative_calibration_scores, max_fpr: float = 0.05) -> float:
    """Highest-sensitivity empirical threshold with `score >= threshold` flagged.

    The threshold is fit exclusively on negative calibration scores. At most
    `floor(max_fpr*n)` calibration negatives may be flagged.
    """
    negatives = np.sort(_finite_vector(negative_calibration_scores, "negative scores"))[::-1]
    if not 0 <= max_fpr < 1:
        raise ValueError("max_fpr must be in [0, 1)")
    allowed = int(np.floor(max_fpr * len(negatives)))
    if allowed == 0:
        return float(np.nextafter(negatives[0], np.inf))
    cutoff = negatives[allowed - 1]
    # Ties at the cutoff would exceed the allowance; advance past them.
    if int((negatives >= cutoff).sum()) > allowed:
        return float(np.nextafter(cutoff, np.inf))
    return float(cutoff)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def detection_metrics(negative_scores, positive_scores, *, threshold: float) -> dict[str, float | int]:
    neg = _finite_vector(negative_scores, "negative scores")
    pos = _finite_vector(positive_scores, "positive scores")
    joined = np.concatenate((neg, pos))
    ranks = _average_ranks(joined)
    rank_sum_pos = float(ranks[len(neg):].sum())
    auroc = (rank_sum_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    labels = np.concatenate((np.zeros(len(neg), dtype=int), np.ones(len(pos), dtype=int)))
    order = np.argsort(-joined, kind="mergesort")
    y = labels[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    precision = tp / (tp + fp)
    # Non-interpolated average precision: precision at each positive rank.
    aupr = float(precision[y == 1].sum() / len(pos))
    fpr = float((neg >= threshold).mean())
    tpr = float((pos >= threshold).mean())
    return {
        "negative_count": len(neg), "positive_count": len(pos),
        "threshold": float(threshold), "auroc": float(auroc), "aupr": aupr,
        "fpr": fpr, "tpr_at_threshold": tpr,
    }
