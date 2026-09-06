#!/usr/bin/env python
"""
VisionZIP Surrogate: Continuous score based on dominant + contextual token selection.

Reuses VisionZIP's score logic:
- dominant_score: importance (for dominant token selection)
- contextual_score: distinctiveness (1 - cosine_similarity to dominant tokens)

Produces continuous score per token without discrete top-k.
"""

import torch
import torch.nn.functional as F


class VisionZIPSurrogate:
    """
    Continuous VisionZIP score surrogate.

    Score = importance_weight * norm_importance + (1-importance_weight) * distinctiveness
    where distinctiveness = 1 - max cosine similarity to dominant-like tokens.
    """

    def __init__(self, temperature: float = 1.0, importance_weight: float = 0.75):
        self.temperature = temperature
        self.importance_weight = importance_weight

    def compute_scores(
        self,
        visual_embeddings: torch.Tensor,
        importance: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute continuous VisionZIP token scores.

        Args:
            visual_embeddings: [N_full, D] merged token features
            importance: [N_full] importance scores (feature norms used if None)

        Returns:
            scores: [N_full] continuous scores (higher = more likely to be kept)
        """
        N_full, D = visual_embeddings.shape
        device = visual_embeddings.device

        if importance is None:
            importance = visual_embeddings.float().norm(dim=-1)

        importance = importance.float()

        # Normalize importance
        imp_score = importance / (importance.sum() + 1e-8)

        # Distinctiveness: 1 - cosine similarity to top tokens
        # Find dominating tokens (top 25% by importance proxy for "dominant")
        dominant_num = max(1, N_full // 4)
        _, dominant_indices = torch.topk(importance, dominant_num, largest=True)

        # Normalize features
        feat_norm = F.normalize(visual_embeddings.float(), dim=-1)

        # Cosine similarity to dominant tokens
        dominant_feats = feat_norm[dominant_indices]  # [dom, D]
        sim_to_dominant = feat_norm @ dominant_feats.T  # [N, dom]

        # For each token: max similarity to any dominant token
        max_sim = sim_to_dominant.max(dim=-1).values  # [N]

        # Distinctiveness = 1 - similarity (tokens unlike dominant ones are more distinctive)
        distinct_score = 1.0 - max_sim

        # Combine
        scores = (self.importance_weight * imp_score +
                  (1 - self.importance_weight) * distinct_score)

        scores = scores / self.temperature

        return scores

    def compute_boundary_loss(
        self,
        scores: torch.Tensor,
        core_indices: torch.Tensor,
        decoy_indices: torch.Tensor,
        margin: float = 0.05,
    ) -> torch.Tensor:
        """
        Compute boundary ranking loss for VisionZIP surrogate.

        We want core scores to decrease and decoy scores to increase.
        """
        if len(core_indices) == 0 or len(decoy_indices) == 0:
            return torch.tensor(0.0, device=scores.device, dtype=torch.float32)

        score_core = scores[core_indices].mean()
        score_decoy = scores[decoy_indices].mean()

        diff = margin + score_core - score_decoy
        return F.softplus(diff)
