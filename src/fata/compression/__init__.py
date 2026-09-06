"""Compression protocol utilities; model-specific adapters live in runtimes."""

from .budgets import actual_keep_count, internvl_budget_weights

__all__ = ["actual_keep_count", "internvl_budget_weights"]
