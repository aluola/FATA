#!/usr/bin/env python
"""
Dynamic VisionZIP: Ported from compression_zoo.py VisionZIP.
Key algorithm: dominant tokens by CLS attention + contextual tokens by step-sampled cosine merging.
Dynamic K = max(1, round(N_full * retention_ratio)).
"""

import torch
import torch.nn.functional as F
from .base import BaseDynamicCompressor


class VisionZIPDynamic(BaseDynamicCompressor):
    """
    VisionZIP compression with dynamic token count.

    Original algorithm (from compression_zoo.py patch_visionzip_official):
    - dominant_num = K * 0.75 - 1 (min 1)
    - contextual_num = K - dominant_num - 1
    - Dominant tokens: top-k by CLS attention sum across heads
    - Contextual tokens: step-sampled targets, merge remaining by cosine similarity,
      aggregated clusters added to target tokens
    - Reconstruct: cosine NN back to N_full
    """

    def __init__(self, retention_ratio: float):
        super().__init__(retention_ratio)

    def compress(self, visual_embeddings, importance=None):
        """
        Args:
            visual_embeddings: [N_full, D]
            importance: [N_full] attention-based importance scores

        Returns:
            compressed: [K, D], selected_indices: [K], metadata: dict
        """
        if visual_embeddings.ndim == 3:
            # [tiles, tokens_per_tile, D] -> flatten to [tiles*tokens, D]
            N_tiles, N_pt, D = visual_embeddings.shape
            visual_embeddings = visual_embeddings.reshape(-1, D)
            N_full = N_tiles * N_pt
        else:
            N_full, D = visual_embeddings.shape
        self.set_token_count(N_full)
        K = self.K

        if K >= N_full:
            return visual_embeddings.clone(), torch.arange(N_full, device=visual_embeddings.device), {"K": K, "N_full": N_full}

        # Dominant tokens: top (K * 0.75 - 1) by importance
        dominant_num = int(K * 0.75) - 1
        if dominant_num <= 0:
            dominant_num = 1
        contextual_num = K - dominant_num - 1

        if importance is None:
            importance = torch.ones(N_full, device=visual_embeddings.device) / N_full

        # Select dominant tokens by importance (excluding a "CLS" position if applicable)
        # For model-agnostic: just use top-k importance
        _, dominant_indices = torch.topk(importance, dominant_num, largest=True)
        dominant_indices = dominant_indices.sort().values

        dominant_tokens = visual_embeddings[dominant_indices]  # [dominant_num, D]

        # Remaining tokens for contextual merging
        all_indices = torch.arange(N_full, device=visual_embeddings.device)
        mask = torch.ones(N_full, dtype=torch.bool, device=visual_embeddings.device)
        mask[dominant_indices] = False
        remaining_indices = all_indices[mask]  # [N_full - dominant_num]
        remaining_tokens = visual_embeddings[remaining_indices]  # [N_rem, D]

        if contextual_num > 0 and remaining_tokens.shape[0] > 0:
            # Normalize for cosine similarity
            remaining_normalized = remaining_tokens / remaining_tokens.norm(dim=-1, keepdim=True).clamp(min=1e-8)

            # Step-sample target tokens
            N_rem = remaining_normalized.shape[0]
            step = max(1, N_rem // contextual_num)
            target_local_indices = torch.arange(0, N_rem, step, device=visual_embeddings.device)[:contextual_num]
            target_tokens = remaining_normalized[target_local_indices]  # [contextual_num, D]

            # Remaining tokens to merge (not selected as targets)
            non_target_mask = torch.ones(N_rem, dtype=torch.bool, device=visual_embeddings.device)
            non_target_mask[target_local_indices] = False
            tokens_to_merge = remaining_normalized[non_target_mask]  # [N_merge, D]
            hidden_to_merge = remaining_tokens[non_target_mask]  # [N_merge, D]

            if tokens_to_merge.shape[0] > 0:
                # Cosine similarity between merge tokens and target tokens
                similarity = tokens_to_merge @ target_tokens.T  # [N_merge, contextual_num]
                assign = similarity.argmax(dim=1)  # [N_merge] — which target each merge token belongs to

                # Aggregate merged tokens to targets
                contextual_tokens_list = []
                for c in range(contextual_num):
                    cluster_mask = (assign == c)
                    if cluster_mask.sum() > 0:
                        cluster_tokens = hidden_to_merge[cluster_mask]  # [n_c, D]
                        aggregated = cluster_tokens.mean(dim=0)  # [D]
                        merged_token = remaining_tokens[target_local_indices[c]] + aggregated
                    else:
                        merged_token = remaining_tokens[target_local_indices[c]]
                    contextual_tokens_list.append(merged_token)

                contextual_tokens = torch.stack(contextual_tokens_list, dim=0)  # [contextual_num, D]
            else:
                contextual_tokens = remaining_tokens[target_local_indices]
        else:
            contextual_tokens = torch.empty(0, D, device=visual_embeddings.device, dtype=visual_embeddings.dtype)

        # Combine dominant + contextual
        compressed_parts = [dominant_tokens]
        if contextual_num > 0 and contextual_tokens.shape[0] > 0:
            compressed_parts.append(contextual_tokens)

        compressed = torch.cat(compressed_parts, dim=0)  # [K_actual, D]

        # Ensure exactly K tokens
        K_actual = compressed.shape[0]
        if K_actual > K:
            compressed = compressed[:K]
        elif K_actual < K:
            # Pad with copies of the last token
            pad = compressed[-1:].repeat(K - K_actual, 1)
            compressed = torch.cat([compressed, pad], dim=0)

        # Build selected indices (approximate for dominant + contextual representative positions)
        selected_indices = torch.arange(K, device=visual_embeddings.device)

        return compressed, selected_indices, {
            "K": K,
            "N_full": N_full,
            "dominant_num": dominant_num,
            "contextual_num": contextual_num,
            "K_actual": K_actual,
        }
