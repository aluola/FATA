#!/usr/bin/env python
"""
PruMerge Surrogate: Continuous score based on importance + spatial coverage.

Reuses PruMerge's selection criteria:
- top-k by importance (attention-based)
- spatial sampling from remaining
- cluster merging via cosine NN

Continuous approximation: importance_score + spatial_coverage_score
"""

import torch
import torch.nn.functional as F
import math


class PruMergeSurrogate:
    """
    Continuous PruMerge score surrogate.

    Score = importance_weight * norm_importance + spatial_weight * spatial_coverage
    where spatial_coverage favors tokens in under-represented grid regions.
    """

    def __init__(self, temperature: float = 1.0, spatial_weight: float = 0.3):
        self.temperature = temperature
        self.spatial_weight = spatial_weight

    def compute_scores(
        self,
        visual_embeddings: torch.Tensor,
        importance: torch.Tensor = None,
        grid_thw: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute continuous PruMerge scores.

        Args:
            visual_embeddings: [N_full, D]
            importance: [N_full] importance scores
            grid_thw: optional [[T, H, W]] for spatial coverage

        Returns:
            scores: [N_full] continuous scores
        """
        N_full, D = visual_embeddings.shape

        if importance is None:
            importance = visual_embeddings.float().norm(dim=-1)

        importance = importance.float()

        # 1. Importance score
        imp_score = importance / (importance.sum() + 1e-8)

        # 2. Spatial coverage score: penalize clustered tokens, reward uniform coverage
        if grid_thw is not None and N_full > 1:
            thw = grid_thw[0] if grid_thw.dim() == 2 else grid_thw
            H_merged = int(thw[1].item()) // 2
            W_merged = int(thw[2].item()) // 2

            if H_merged * W_merged == N_full:
                # Build spatial coverage: for each token, compute local density
                spatial_score = torch.zeros(N_full, device=visual_embeddings.device)

                # Reshape importance to 2D
                imp_2d = imp_score.reshape(H_merged, W_merged)

                # Local importance density (3x3 kernel)
                imp_padded = F.pad(imp_2d.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='reflect')
                kernel = torch.ones(1, 1, 3, 3, device=visual_embeddings.device)
                kernel[0, 0, 1, 1] = 0  # Exclude self
                local_density = F.conv2d(imp_padded, kernel, padding=0).squeeze()

                # Spatial score: inverse density (reward isolation)
                # Normalize to [0, 1]
                spatial_score = 1.0 / (1.0 + local_density.flatten())

                # Normalize spatial score
                spatial_score = spatial_score / (spatial_score.sum() + 1e-8)
            else:
                spatial_score = torch.ones(N_full, device=visual_embeddings.device) / N_full
        else:
            # Without grid info, use diversity as spatial proxy
            feat_norm = F.normalize(visual_embeddings.float(), dim=-1)
            # Mean distance to all others
            if N_full <= 1024:
                sim = feat_norm @ feat_norm.T
                sim_sum = sim.sum(dim=1) - 1.0
                spatial_score = 1.0 - sim_sum / (N_full - 1)
            else:
                spatial_score = torch.ones(N_full, device=visual_embeddings.device) / N_full
            spatial_score = spatial_score / (spatial_score.sum() + 1e-8)

        # Combine
        scores = ((1 - self.spatial_weight) * imp_score +
                  self.spatial_weight * spatial_score) / self.temperature

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
