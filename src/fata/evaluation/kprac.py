"""Clean-only LLaVA K_prac selection with strict provenance checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable, Mapping

from fata.constants import LLAVA_COMPRESSED_BUDGETS


@dataclass(frozen=True)
class KPracSelection:
    dataset: str
    compressor: str
    threshold: float
    full_k: int
    full_accuracy: float
    selected_k: int
    selected_accuracy: float
    retention: float
    sample_count: int
    trajectory: dict[int, float]

    def to_dict(self) -> dict:
        value = asdict(self)
        value["trajectory"] = {str(k): v for k, v in self.trajectory.items()}
        return value


def select_llava_clean_threshold_kprac(
    *, dataset: str, compressor: str, clean_accuracy_by_k: Mapping[int, float],
    sample_count: int, threshold: float = 0.80,
    candidates: Iterable[int] = LLAVA_COMPRESSED_BUDGETS,
) -> KPracSelection:
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be in (0, 1]")
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    supplied = tuple(int(k) for k in candidates)
    requested = tuple(sorted(set(supplied)))
    if len(requested) != len(supplied):
        raise ValueError("duplicate K in candidate list")
    if not requested or 576 in requested:
        raise ValueError("candidates must be non-empty compressed K values and exclude Full K=576")
    required = (576, *requested)
    missing = [k for k in required if k not in clean_accuracy_by_k]
    if missing:
        raise ValueError(f"missing Clean accuracy for K={missing}")
    trajectory = {k: float(clean_accuracy_by_k[k]) for k in required}
    if any(not math.isfinite(v) for v in trajectory.values()):
        raise ValueError("Clean trajectory contains NaN or infinity")
    if any(v < 0 or v > 1 for v in trajectory.values()):
        raise ValueError("Clean accuracy must lie in [0, 1]")
    full = trajectory[576]
    if full <= 0:
        raise ZeroDivisionError("Clean Full accuracy is zero")
    eligible = [k for k in requested if trajectory[k] / full >= threshold]
    if not eligible:
        raise ValueError("no tested compressed K satisfies the Clean threshold")
    selected = min(eligible)
    return KPracSelection(
        dataset=dataset, compressor=compressor, threshold=threshold,
        full_k=576, full_accuracy=full, selected_k=selected,
        selected_accuracy=trajectory[selected], retention=trajectory[selected] / full,
        sample_count=sample_count, trajectory=dict(sorted(trajectory.items(), reverse=True)),
    )
