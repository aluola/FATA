#!/usr/bin/env python
"""
Dynamic VisPruner: Ported from compression_zoo.py VisPruner.
Key algorithm: ~50% tokens by importance, ~50% by pairwise diversity pruning.
Dynamic K = max(1, round(N_full * retention_ratio)).
"""
import torch
import torch.nn.functional as F
from .base import BaseDynamicCompressor


class VisPrunerDynamic(BaseDynamicCompressor):
    """
    VisPruner compression with dynamic token count.

    Original algorithm (from compression_zoo.py patch_vispruner_official):
    - important_token_num = int(K * 0.5), min 1
    - diverse_token_num = K - important_token_num
    - Important tokens: top-k by CLS attention
    - Diverse tokens: iteratively prune the most similar pair from residual set
      until residual size equals diverse_token_num
    - Combine sorted important + diverse indices
    """

    def __init__(self, retention_ratio: float):
        super().__init__(retention_ratio)

    def compress(self, visual_embeddings, importance=None):
        """
        Args:
            visual_embeddings: [N_full, D] or [tiles, tokens_per_tile, D]
            importance: [N_full] attention-based importance scores (feature norms used if None)

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

        # Importance — use feature norms if none provided
        if importance is None:
            importance = visual_embeddings.norm(dim=-1)

        # VisPruner split: 50% importance, 50% diversity
        important_num = int(K * 0.5)
        if important_num <= 0:
            important_num = 1
        diverse_num = K - important_num

        # Step 1: Top-k important tokens
        token_indices_sorted = importance.argsort(dim=-1, descending=True)
        important_indices = token_indices_sorted[:important_num]
        residual_indices = token_indices_sorted[important_num:]

        # Step 2: Iterative diversity pruning on residual set
        features_normalized = visual_embeddings / visual_embeddings.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        while True:
            residual_tokens = features_normalized[residual_indices]
            n_residual = residual_tokens.shape[0]
            r = min(8, n_residual - diverse_num)
            if r <= 0:
                break

            a = residual_tokens[::2]
            b = residual_tokens[1::2]
            min_len = min(a.shape[0], b.shape[0])
            if min_len == 0:
                break
            a, b = a[:min_len], b[:min_len]

            scores = (a @ b.T).diag()  # pairwise scores for adjacent pairs
            # Keep the r pairs with highest similarity (remove the most similar token from each)
            remove_local = scores.argsort(descending=True)
            keep_local = remove_local[r:]

            kept_a = a[keep_local]
            kept_b = b[keep_local]
            # Rebuild residual: keep both tokens from non-pruned pairs
            new_residual = []
            for j in range(min_len):
                if j in keep_local:
                    new_residual.append(residual_indices[2 * j])
                    new_residual.append(residual_indices[2 * j + 1])
                else:
                    new_residual.append(residual_indices[2 * j])  # Keep one, drop the other
            if len(residual_indices) % 2 == 1:
                new_residual.append(residual_indices[-1])

            residual_indices = torch.tensor(new_residual, device=visual_embeddings.device, dtype=torch.long)

        # If residual still has more than diverse_num, truncate
        if len(residual_indices) > diverse_num:
            residual_indices = residual_indices[:diverse_num]

        # Step 3: Combine
        all_indices = torch.cat([important_indices, residual_indices])
        all_indices = all_indices.sort().values

        compressed = visual_embeddings[all_indices]

        # Ensure exact K
        K_actual = compressed.shape[0]
        if K_actual > K:
            compressed = compressed[:K]
            all_indices = all_indices[:K]
        elif K_actual < K:
            pad = compressed[-1:].repeat(K - K_actual, 1)
            compressed = torch.cat([compressed, pad], dim=0)
            pad_idx = all_indices[-1].repeat(K - K_actual)
            all_indices = torch.cat([all_indices, pad_idx])

        return compressed, all_indices, {
            "K": K, "N_full": N_full, "K_actual": K,
            "important_num": important_num, "diverse_num": diverse_num,
            "method": "vispruner",
        }
