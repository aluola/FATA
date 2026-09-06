#!/usr/bin/env python
"""
VisPruner Surrogate: Continuous score based on importance + diversity.

Reuses VisPruner's dual-criterion score:
- importance_score: feature norm based
- diversity_score: 1 - pairwise cosine similarity (iterative pruning proxy)

Continuous approximation: weighted sum of importance and diversity scores.
"""

import torch
import torch.nn.functional as F


class VisPrunerSurrogate:
    """Continuous VisPruner score surrogate (50% importance, 50% diversity)."""

    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature

    def compute_scores(
        self,
        visual_embeddings: torch.Tensor,
        importance: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute continuous VisPruner token scores.

        Args:
            visual_embeddings: [N_full, D]
            importance: [N_full] importance scores

        Returns:
            scores: [N_full] continuous scores
        """
        N_full, D = visual_embeddings.shape
        device = visual_embeddings.device

        if importance is None:
            importance = visual_embeddings.float().norm(dim=-1)

        importance = importance.float()

        # 1. Importance score (normalized)
        imp_score = importance / (importance.sum() + 1e-8)

        # 2. Diversity score: for each token, how "unique" it is
        # Approximate iterative pruning: mean distance to nearest neighbors
        feat_norm = F.normalize(visual_embeddings.float(), dim=-1)

        # Cosine similarity matrix in chunks if needed
        if N_full <= 1024:
            sim_matrix = feat_norm @ feat_norm.T  # [N, N]
        else:
            # Chunked computation
            chunk_size = 256
            sim_matrix = torch.zeros(N_full, N_full, device=device)
            for i in range(0, N_full, chunk_size):
                chunk = feat_norm[i:i+chunk_size]
                sim_matrix[i:i+chunk_size] = chunk @ feat_norm.T

        # For each token: mean similarity to all others (excluding self)
        # Lower mean similarity = more diverse
        sim_sum = sim_matrix.sum(dim=1) - 1.0  # subtract self-similarity
        mean_sim = sim_sum / (N_full - 1)

        # Diversity = 1 - mean similarity (higher = more diverse)
        div_score = 1.0 - mean_sim

        # Normalize diversity
        div_score = div_score / (div_score.sum() + 1e-8)

        # Combine 50/50
        scores = (0.5 * imp_score + 0.5 * div_score) / self.temperature

        return scores

    def compute_boundary_loss(
        self,
        scores: torch.Tensor,
        core_indices: torch.Tensor,
        decoy_indices: torch.Tensor,
        margin: float = 0.05,
    ) -> torch.Tensor:
        if len(core_indices) == 0 or len(decoy_indices) == 0:
            return torch.tensor(0.0, device=scores.device, dtype=torch.float32)
        diff = margin + scores[core_indices].mean() - scores[decoy_indices].mean()
        return F.softplus(diff)
