"""Explicit dynamic-token budget semantics."""

from __future__ import annotations

from fata.constants import MC_DATASETS, OPEN_DATASETS


def actual_keep_count(token_count: int, numerator: int, denominator: int) -> int:
    if token_count <= 0 or numerator <= 0 or denominator <= 0 or numerator > denominator:
        raise ValueError("token count and a fraction in (0, 1] are required")
    # Matches the selected Qwen/InternVL runtimes: Python round, then lower bound 1.
    return max(1, int(round(token_count * numerator / denominator)))


def internvl_budget_weights(dataset: str) -> dict[str, float]:
    if dataset in OPEN_DATASETS:
        return {"1/9": 1.0, "1/3": 0.4, "2/9": 0.25, "1/18": 0.0, "1/36": 0.0}
    if dataset in MC_DATASETS:
        return {"1/18": 1.0, "1/9": 0.4, "1/3": 0.2, "2/9": 0.0, "1/36": 0.0}
    raise ValueError(f"unknown paper dataset: {dataset!r}")
