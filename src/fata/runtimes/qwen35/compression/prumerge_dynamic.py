#!/usr/bin/env python
"""
Dynamic PruMerge: Ported from compression_zoo.py PruMerge.
Key algorithm: top-K attention (50%) + spatial sampling (50%) + cluster merging.
Dynamic K = max(1, round(N_full * retention_ratio)).
"""
import torch
import torch.nn.functional as F
from .base import BaseDynamicCompressor


def complement_idx(idx, dim):
    """Compute complement of selected indices (from compression_zoo.py)."""
    a = torch.arange(dim, device=idx.device)
    ndim = idx.ndim
    dims = idx.shape
    n_idx = dims[-1]
    dims = dims[:-1] + (-1,)
    for i in range(1, ndim):
        a = a.unsqueeze(0)
    a = a.expand(*dims)
    masked = torch.scatter(a, -1, idx, 0)
    compl, _ = torch.sort(masked, dim=-1, descending=False)
    compl = compl.permute(-1, *tuple(range(ndim - 1)))
    compl = compl[n_idx:].permute(*(tuple(range(1, ndim)) + (0,)))
    return compl


class PruMergeDynamic(BaseDynamicCompressor):
    """
    PruMerge compression with dynamic token count.

    Original algorithm (from compression_zoo.py patch_prumerge_official):
    - target_k = int(K), if target_k >= N: target_k = N // 2
    - top_k_num = target_k // 2  (by CLS attention)
    - spatial_num = target_k - top_k_num  (uniform spatial sampling)
    - For each selected token i:
        - Find 32 nearest neighbors by cosine similarity of key features
        - Merge: center = x_i + weighted_sum(cluster_tokens * attention_weights)
    - Reconstruct: cosine NN back to N_full positions, fill with updated_x_others

    Qwen adaptation:
    - CLS attention proxy: feature_norm based importance
    - Key features: normalized visual embeddings (proxy for actual K projection)
    - Cluster merge uses feature cosine similarity
    """

    def __init__(self, retention_ratio: float):
        super().__init__(retention_ratio)

    def compress(self, visual_embeddings, importance=None):
        """
        Args:
            visual_embeddings: [N_full, D] or [tiles, tokens_per_tile, D]
            importance: [N_full] importance scores

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

        # PruMerge target K
        target_k = K
        if target_k >= N_full:
            target_k = max(1, N_full // 2)

        top_k_num = target_k // 2
        if top_k_num <= 0:
            top_k_num = 1
        spatial_num = target_k - top_k_num

        # Step 1: Top-k by importance (proxy for CLS attention)
        _, topk_idx = torch.topk(importance, top_k_num, largest=True)

        # Step 2: Spatial sampling from remaining tokens
        all_idx = torch.arange(N_full, device=visual_embeddings.device)
        mask = ~torch.isin(all_idx, topk_idx)
        remain_idx = all_idx[mask]

        if len(remain_idx) >= spatial_num:
            sample_pos = torch.linspace(0, len(remain_idx) - 1, steps=spatial_num).long()
            sampled = remain_idx[sample_pos]
        else:
            sampled = remain_idx

        selected_idx = torch.cat([topk_idx, sampled])
        selected_idx = selected_idx.sort().values
        K_selected = len(selected_idx)

        # Step 2.5: Get complement (non-selected) indices
        compl = complement_idx(selected_idx.unsqueeze(0), N_full).squeeze(0)

        # Step 3: Cluster merging for each selected token
        x_selected = visual_embeddings[selected_idx]  # [K_selected, D]
        x_non_selected = visual_embeddings[compl]  # [N_full - K_selected, D]

        # Use features as proxy for keys
        key_features = F.normalize(visual_embeddings, p=2, dim=-1)
        key_selected = key_features[selected_idx]
        key_non_selected = key_features[compl]

        # Attention weights for non-selected tokens
        non_selected_attn = importance[compl]
        non_selected_attn = non_selected_attn / (non_selected_attn.sum() + 1e-8)

        # For each selected token, find nearest 32 neighbors and merge
        # Process in chunks to avoid OOM
        updated_selected = x_selected.clone()
        merge_k = min(32, x_non_selected.shape[0])

        # Compute cosine sim in chunks to save memory
        chunk_size = 64
        for chunk_start in range(0, K_selected, chunk_size):
            chunk_end = min(chunk_start + chunk_size, K_selected)
            key_chunk = key_selected[chunk_start:chunk_end]  # [chunk, D]
            cos_chunk = torch.mm(key_chunk, key_non_selected.T)  # [chunk, N-C]

            for i in range(chunk_end - chunk_start):
                global_i = chunk_start + i
                if merge_k == 0:
                    break
                sim_i = cos_chunk[i]
                _, cluster_indices = torch.topk(sim_i, k=merge_k, largest=True)
                cluster_tokens = x_non_selected[cluster_indices]
                weights = non_selected_attn[cluster_indices].unsqueeze(-1)
                weighted_avg = (cluster_tokens * weights).sum(dim=0)
                updated_selected[global_i] = updated_selected[global_i] + weighted_avg

        # Step 4: Cosine NN reconstruction in chunks to avoid OOM
        K_sel = updated_selected.shape[0]
        nearest_idx = torch.zeros(N_full, dtype=torch.long, device=visual_embeddings.device)
        for chunk_start in range(0, N_full, chunk_size):
            chunk_end = min(chunk_start + chunk_size, N_full)
            sim_chunk = F.cosine_similarity(
                visual_embeddings[chunk_start:chunk_end].unsqueeze(1),
                updated_selected.unsqueeze(0), dim=-1
            )  # [chunk, K_selected]
            nearest_idx[chunk_start:chunk_end] = sim_chunk.argmax(dim=-1)

        # Build compressed by gathering from updated_selected
        compressed = updated_selected[nearest_idx]  # [N_full, D]

        # Now select K tokens from the reconstructed set
        # Use the same selected_idx positions to extract K tokens
        compressed = compressed[selected_idx]  # [K_selected, D]

        # Ensure exact K
        K_actual = compressed.shape[0]
        if K_actual > K:
            compressed = compressed[:K]
            selected_idx = selected_idx[:K]
        elif K_actual < K:
            pad = compressed[-1:].repeat(K - K_actual, 1)
            compressed = torch.cat([compressed, pad], dim=0)
            pad_idx = selected_idx[-1].repeat(K - K_actual)
            selected_idx = torch.cat([selected_idx, pad_idx])

        return compressed, selected_idx, {
            "K": K, "N_full": N_full, "K_actual": K,
            "top_k_num": top_k_num, "spatial_num": spatial_num,
            "merge_k": merge_k, "n_merged": K_selected,
            "method": "prumerge",
        }
