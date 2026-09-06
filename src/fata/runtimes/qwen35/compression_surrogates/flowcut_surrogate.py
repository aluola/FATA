#!/usr/bin/env python
"""
FlowCut Surrogate: Continuous score based on (relation + semantic) x L1(value).

Reuses FlowCut's exact score formula:
- relation_score = norm_importance / sum(importance)
- semantic_score = softmax(cosine_sim_to_global_mean) / sum(...)
- value_metric = L1 norm of token features
- final_score = (relation_score + semantic_score) * value_metric
"""

import torch
import torch.nn.functional as F


class FlowCutSurrogate:
    """Continuous FlowCut score surrogate."""

    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature

    def compute_scores(
        self,
        visual_embeddings: torch.Tensor,
        importance: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute continuous FlowCut scores.

        Exact formula from FlowCutDynamic.compress():
        score = (relation_score + semantic_score) * L1_norm(features)

        Args:
            visual_embeddings: [N_full, D]
            importance: [N_full] importance scores

        Returns:
            scores: [N_full] continuous FlowCut scores
        """
        N_full, D = visual_embeddings.shape

        if importance is None:
            importance = visual_embeddings.float().norm(dim=-1)

        importance = importance.float()

        # 1. Relation score: normalized importance
        relation_score = importance / (importance.sum() + 1e-8)

        # 2. Semantic score: cosine similarity to global mean, clamped >= 0, normalized
        global_feat = visual_embeddings.float().mean(dim=0, keepdim=True)  # [1, D]
        semantic_raw = F.cosine_similarity(
            visual_embeddings.float().unsqueeze(0),
            global_feat.unsqueeze(1),
            dim=-1,
        ).squeeze(0)  # [N_full]
        semantic_raw = torch.clamp(semantic_raw, min=0)
        semantic_score = semantic_raw / (semantic_raw.sum() + 1e-8)

        # 3. Value metric: L1 norm
        value_metric = visual_embeddings.float().norm(p=1, dim=-1)  # [N_full]

        # 4. FlowCut final score
        scores = (relation_score + semantic_score) * value_metric

        scores = scores / self.temperature

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
