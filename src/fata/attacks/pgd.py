"""Small, independently testable L-infinity PGD invariants."""

from __future__ import annotations

import torch


def linf_project_and_clip(clean: torch.Tensor, candidate: torch.Tensor, epsilon: float) -> torch.Tensor:
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")
    delta = torch.clamp(candidate - clean, min=-epsilon, max=epsilon)
    return torch.clamp(clean + delta, min=0.0, max=1.0)


def validate_linf(clean: torch.Tensor, adversarial: torch.Tensor, epsilon: float, atol: float = 1e-7) -> float:
    if clean.shape != adversarial.shape:
        raise ValueError("clean and adversarial tensors have different shapes")
    if not torch.isfinite(adversarial).all():
        raise ValueError("adversarial tensor contains NaN or infinity")
    actual = float((adversarial - clean).abs().max().item())
    if actual > epsilon + atol:
        raise ValueError(f"Linf violation: {actual:.9g} > {epsilon:.9g} (+{atol:g})")
    if float(adversarial.min()) < -atol or float(adversarial.max()) > 1.0 + atol:
        raise ValueError("adversarial pixels are outside [0, 1]")
    return actual
