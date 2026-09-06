#!/usr/bin/env python
"""
Q-FATA-AR: Multi-Budget Boundary Target Builder.

For each sample, defines core and decoy token targets at two compression budgets:
  - K_medium = round(N_full / 3)
  - K_practical = round(N_full / 9) for Open, round(N_full / 18) for MC

Core tokens: clean ranking [K-window, K]
Decoy tokens: clean ranking [K+1, K+window]

Default window = max(8, round(0.05 * N_full))

Target pairs are FIXED for all 100 steps — no re-selection per step.
Multi-budget targets take union but preserve budget labels.

Ranking margin loss: softplus(margin + score_core - score_decoy)
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class BudgetTarget:
    """Target pair for a single compression budget."""
    K: int                        # Target token count for this budget
    window: int                   # Boundary window size
    core_indices: torch.Tensor    # [window] indices of core tokens
    decoy_indices: torch.Tensor   # [window] indices of decoy tokens
    budget_label: str             # "medium" or "practical"


@dataclass
class BoundaryTargetResult:
    """Complete multi-budget boundary target result."""
    N_full: int
    all_core_indices: torch.Tensor    # Union of all core indices
    all_decoy_indices: torch.Tensor   # Union of all decoy indices
    budgets: List[BudgetTarget]
    target_mask_core: torch.Tensor    # [N_full] binary mask for core tokens
    target_mask_decoy: torch.Tensor   # [N_full] binary mask for decoy tokens
    metadata: Dict


class BoundaryTargetBuilder:
    """
    Build boundary targets for multi-budget compressor-aware attack.

    For each K (medium, practical):
    1. Rank tokens by clean importance (descending)
    2. Core tokens: top K tokens -> keep the last `window` of these
       (indices [K-window, K) in the sorted ranking)
    3. Decoy tokens: next `window` tokens after K
       (indices [K, K+window) in the sorted ranking)

    The idea: compressors keep ~K tokens. We want to make the
    boundary tokens (those just inside K) less important AND
    the near-miss tokens (those just outside K) more important,
    maximizing the chance of compressor failure.

    Args:
        window: boundary window size (default: adaptive based on N_full)
        margin: ranking margin for softplus loss (default 0.05)
    """

    def __init__(
        self,
        window: Optional[int] = None,
        margin: float = 0.05,
        is_mc: bool = False,
    ):
        self.window = window
        self.margin = margin
        self.is_mc = is_mc

    def compute_K_values(self, N_full: int) -> Dict[str, int]:
        """Compute K values for medium and practical budgets."""
        K_medium = max(1, int(round(N_full / 3)))
        if self.is_mc:
            K_practical = max(1, int(round(N_full / 18)))
        else:
            K_practical = max(1, int(round(N_full / 9)))
        return {"medium": K_medium, "practical": K_practical}

    def compute_window(self, N_full: int) -> int:
        """Compute adaptive window size."""
        if self.window is not None:
            return self.window
        return max(8, int(round(0.05 * N_full)))

    def build(
        self,
        clean_importance: torch.Tensor,
        N_full: Optional[int] = None,
        budgets: Optional[List[str]] = None,
    ) -> BoundaryTargetResult:
        """
        Build boundary targets from clean importance scores.

        Args:
            clean_importance: [N_full] importance scores (higher = more important)
            N_full: token count (inferred if None)
            budgets: which budgets to build targets for (default: ["medium", "practical"])

        Returns:
            BoundaryTargetResult with all target info
        """
        if N_full is None:
            N_full = clean_importance.shape[0]

        if budgets is None:
            budgets = ["medium", "practical"]

        window = self.compute_window(N_full)
        K_values = self.compute_K_values(N_full)

        # Sort tokens by importance (descending)
        sorted_indices = clean_importance.argsort(descending=True)

        budget_targets = []
        all_core = []
        all_decoy = []

        for budget_label in budgets:
            K = K_values[budget_label]

            # Ensure window fits within bounds
            effective_window = min(window, K, N_full - K)
            if effective_window == 0:
                continue

            # Core tokens: ranking positions [K-window, K)
            core_start = max(0, K - effective_window)
            core_ranking_positions = sorted_indices[core_start:K]
            core_indices = core_ranking_positions.clone()

            # Decoy tokens: ranking positions [K, K+window)
            decoy_end = min(N_full, K + effective_window)
            decoy_ranking_positions = sorted_indices[K:decoy_end]
            decoy_indices = decoy_ranking_positions.clone()

            budget_targets.append(BudgetTarget(
                K=K,
                window=effective_window,
                core_indices=core_indices,
                decoy_indices=decoy_indices,
                budget_label=budget_label,
            ))

            all_core.append(core_indices)
            all_decoy.append(decoy_indices)

        # Build union masks
        all_core_union = torch.cat(all_core).unique() if all_core else torch.tensor([], dtype=torch.long, device=clean_importance.device)
        all_decoy_union = torch.cat(all_decoy).unique() if all_decoy else torch.tensor([], dtype=torch.long, device=clean_importance.device)

        # Binary masks
        core_mask = torch.zeros(N_full, dtype=torch.bool, device=clean_importance.device)
        decoy_mask = torch.zeros(N_full, dtype=torch.bool, device=clean_importance.device)
        if len(all_core_union) > 0:
            core_mask[all_core_union] = True
        if len(all_decoy_union) > 0:
            decoy_mask[all_decoy_union] = True

        metadata = {
            "N_full": N_full,
            "window": window,
            "K_values": K_values,
            "num_budgets": len(budget_targets),
            "total_core_tokens": len(all_core_union),
            "total_decoy_tokens": len(all_decoy_union),
            "margin": self.margin,
            "is_mc": self.is_mc,
        }

        return BoundaryTargetResult(
            N_full=N_full,
            all_core_indices=all_core_union,
            all_decoy_indices=all_decoy_union,
            budgets=budget_targets,
            target_mask_core=core_mask,
            target_mask_decoy=decoy_mask,
            metadata=metadata,
        )

    def compute_ranking_loss(
        self,
        scores: torch.Tensor,
        target: BoundaryTargetResult,
    ) -> torch.Tensor:
        """
        Compute ranking margin loss: softplus(margin + score_core - score_decoy).

        We want core scores to be LOWER than decoy scores by at least margin.
        Loss is minimized when score_decoy > score_core + margin.

        Args:
            scores: [N_full] current (adversarial) importance scores
            target: BoundaryTargetResult from build()

        Returns:
            scalar loss
        """
        if len(target.all_core_indices) == 0 or len(target.all_decoy_indices) == 0:
            return torch.tensor(0.0, device=scores.device, dtype=torch.float32)

        core_scores = scores[target.all_core_indices]
        decoy_scores = scores[target.all_decoy_indices]

        # Expand for all pairs: [num_core, num_decoy]
        # Average over all core-decoy pairs
        score_core = core_scores.mean()
        score_decoy = decoy_scores.mean()

        # softplus(margin + score_core - score_decoy)
        # When score_decoy > score_core + margin, this is near 0
        diff = self.margin + score_core - score_decoy
        loss = F.softplus(diff)

        return loss

    def compute_per_budget_losses(
        self,
        scores: torch.Tensor,
        target: BoundaryTargetResult,
    ) -> Dict[str, torch.Tensor]:
        """Compute ranking loss separately for each budget."""
        losses = {}
        for bt in target.budgets:
            if len(bt.core_indices) == 0 or len(bt.decoy_indices) == 0:
                losses[bt.budget_label] = torch.tensor(0.0, device=scores.device, dtype=torch.float32)
                continue
            score_core = scores[bt.core_indices].mean()
            score_decoy = scores[bt.decoy_indices].mean()
            diff = self.margin + score_core - score_decoy
            losses[bt.budget_label] = F.softplus(diff)
        return losses
