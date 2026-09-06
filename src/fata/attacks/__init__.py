"""Attack invariants shared by runtime-specific implementations."""

from .pgd import linf_project_and_clip, validate_linf

__all__ = ["linf_project_and_clip", "validate_linf"]
