#!/usr/bin/env python
"""
Dynamic FlowCut: Ported from compression_zoo.py FlowCut.
Key algorithm: (relation_score + semantic_score) × L1(value).
Dynamic K = max(1, round(N_full * retention_ratio)).
"""
import torch
import torch.nn.functional as F
from .base import BaseDynamicCompressor


class FlowCutDynamic(BaseDynamicCompressor):
    """
    FlowCut compression with dynamic token count.

    Original algorithm (from compression_zoo.py patch_flowcut_official):
    - relation_score = cls_attn / sum(cls_attn)
    - semantic_score = softmax(cls_value @ value_states^T) / sum(...)
    - final_score = (relation_score + semantic_score) × L1_norm(value)
    - Select top-K by final_score

    Qwen adaptation:
    - relation_score: feature_norm based importance / sum(importance)
    - semantic_score: spatial-softmax importance / sum(...)
    - L1(value): L1 norm of each token's feature vector
    """

    def __init__(self, retention_ratio: float):
        super().__init__(retention_ratio)

    def compress(self, visual_embeddings, importance=None):
        """
        Args:
            visual_embeddings: [N_full, D] or [tiles, tokens_per_tile, D]
            importance: [N_full] importance scores (feature norms used if None)

        Returns:
            compressed: [K, D], selected_indices: [K], metadata: dict
        """
        # Handle 3D input
        if visual_embeddings.ndim == 3:
            N_tiles, N_pt, D = visual_embeddings.shape
            visual_embeddings = visual_embeddings.reshape(-1, D)
            N_full = N_tiles * N_pt
            if importance is not None and importance.ndim == 3:
                importance = importance.reshape(-1)
        else:
            N_full, D = visual_embeddings.shape
        self.set_token_count(N_full)
        K = self.K

        if K >= N_full:
            return (visual_embeddings.clone(),
                    torch.arange(N_full, device=visual_embeddings.device),
                    {"K": K, "N_full": N_full, "method": "identity"})

        # Importance
        if importance is None:
            importance = visual_embeddings.norm(dim=-1)

        # 1. Relation score: normalized importance (proxy for CLS attention)
        relation_score = importance / (importance.sum() + 1e-8)

        # 2. Semantic score: spatial softmax over importance
        semantic_weight = importance.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        # Simulate cls_value @ value_states: use feature similarity to mean
        global_feat = visual_embeddings.mean(dim=0, keepdim=True)  # [1, D]
        semantic_raw = F.cosine_similarity(
            visual_embeddings.unsqueeze(0), global_feat.unsqueeze(1), dim=-1
        ).squeeze(0)  # [N_full]
        semantic_raw = torch.clamp(semantic_raw, min=0)  # Non-negative
        semantic_score = semantic_raw / (semantic_raw.sum() + 1e-8)

        # 3. L1 norm of value features (original uses value_states L1 norm)
        value_metric = visual_embeddings.norm(p=1, dim=-1)  # [N_full]

        # 4. FlowCut final score
        final_score = (relation_score + semantic_score) * value_metric

        # 5. Select top-K
        _, keep_indices = torch.topk(final_score, K, largest=True)
        keep_indices = keep_indices.sort().values

        compressed = visual_embeddings[keep_indices]

        return compressed, keep_indices, {
            "K": K, "N_full": N_full, "K_actual": K,
            "relation_mean": float(relation_score.mean()),
            "semantic_mean": float(semantic_score.mean()),
            "method": "flowcut",
        }
