"""Source-faithful PruMerge core with explicit tile boundaries."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from .base import (
    CompressionContext,
    CompressionInputError,
    CompressionResult,
    TileCompressionContext,
    TileMetadata,
)
from .common import compress_tiled
from .reconstruction import reconstruct_from_reference


METHOD_NAME = "PruMerge"
CLUSTER_NEIGHBORS = 32


def complement_idx(idx: torch.Tensor, dim: int) -> torch.Tensor:
    """Literal copy of ``compression_zoo.complement_idx`` (lines 88--100)."""

    a = torch.arange(dim, device=idx.device)
    ndim = idx.ndim
    dims = idx.shape
    n_idx = dims[-1]
    dims = dims[:-1] + (-1,)
    for _ in range(1, ndim):
        a = a.unsqueeze(0)
    a = a.expand(*dims)
    masked = torch.scatter(a, -1, idx, 0)
    compl, _ = torch.sort(masked, dim=-1, descending=False)
    compl = compl.permute(-1, *tuple(range(ndim - 1)))
    compl = compl[n_idx:].permute(*(tuple(range(1, ndim)) + (0,)))
    return compl


def _compress_single_tile(
    features: torch.Tensor,
    k: int,
    context: Optional[TileCompressionContext],
    *,
    reconstruct: bool = True,
) -> CompressionResult:
    if context is None:
        raise CompressionInputError("PruMerge requires query/key context")
    if context.cls_query is None or context.cls_key is None or context.patch_key is None:
        raise CompressionInputError("PruMerge requires CLS query, CLS key, and patch keys")

    batch, tokens, width = features.shape
    cls_query, cls_key, patch_key = (
        context.cls_query,
        context.cls_key,
        context.patch_key,
    )
    if cls_query.shape != (batch, width) or cls_key.shape != (batch, width):
        raise CompressionInputError("PruMerge CLS query/key shape must be [B, C]")
    if patch_key.shape != (batch, tokens, width):
        raise CompressionInputError("PruMerge patch keys must have shape [B, N, C]")
    if tokens - 1 < CLUSTER_NEIGHBORS:
        raise CompressionInputError(
            "PruMerge source requests exactly 32 neighbors and requires N_tile >= 33"
        )

    # Only the CLS row is consumed by source line 131.  Computing that row
    # directly is the tile-safe equivalent of the full q @ k^T allocation.
    all_key = torch.cat([cls_key.unsqueeze(1), patch_key], dim=1)
    attn = (cls_query.unsqueeze(1) @ all_key.transpose(-2, -1)) * width**-0.5
    attn = F.softmax(attn, dim=-1)
    cls_attn = attn[:, 0, 1:]

    target_k = int(k)
    if target_k >= tokens:
        target_k = tokens // 2
    top_k_num = target_k // 2
    spatial_num = target_k - top_k_num
    _, topk_idx = torch.topk(cls_attn, top_k_num, dim=1, largest=True)

    idx_list = []
    spatial_list = []
    for b in range(batch):
        cur_topk = topk_idx[b]
        all_idx = torch.arange(tokens, device=features.device)
        mask = ~torch.isin(all_idx, cur_topk)
        remain_idx = all_idx[mask]
        if len(remain_idx) >= spatial_num:
            sample_pos = torch.linspace(
                0, len(remain_idx) - 1, steps=spatial_num, device=remain_idx.device
            ).long()
            sampled = remain_idx[sample_pos]
        else:
            sampled = remain_idx
        final_idx = torch.cat((cur_topk, sampled), dim=0)
        idx_list.append(final_idx)
        spatial_list.append(sampled)

    idx = torch.stack(idx_list, dim=0)
    index = idx.unsqueeze(-1).expand(-1, -1, width)
    x_others = torch.gather(features, dim=1, index=index)
    x_others_attn = torch.gather(cls_attn, dim=1, index=idx)
    key_others = torch.gather(patch_key, dim=1, index=index)
    compl = complement_idx(idx, tokens)
    compl_index = compl.unsqueeze(-1).expand(-1, -1, width)
    non_topk = torch.gather(features, dim=1, index=compl_index)
    non_topk_key = torch.gather(patch_key, dim=1, index=compl_index)
    non_topk_attn = torch.gather(cls_attn, dim=1, index=compl)

    if not reconstruct:
        # Attack-internal surrogate mode: skip the scalar 32-neighbor merge
        # loop entirely and return the unmerged selected centers.  The merge
        # only rewrites compressed values; the retained token identity (the
        # attack's target) is already fixed by the top-k + spatial selection.
        nearest = torch.arange(x_others.shape[1], device=x_others.device).unsqueeze(0).expand(batch, -1)
        return CompressionResult(
            method=METHOD_NAME,
            nominal_k=k,
            selected_indices=idx,
            compressed_tokens=x_others,
            reconstructed_tokens=x_others,
            reconstruction_indices=nearest,
            scores=cls_attn,
            cluster_assignment=None,
            tile_budgets=(k,),
            diagnostics={
                "attention_topk_indices": topk_idx,
                "spatial_indices": torch.stack(spatial_list, dim=0),
                "complement_indices": compl,
                "unmerged_centers": x_others,
                "surrogate_no_merge": True,
            },
        )

    key_others_norm = F.normalize(key_others, p=2, dim=-1)
    non_topk_key_norm = F.normalize(non_topk_key, p=2, dim=-1)
    _, left_tokens, _ = x_others.size()
    updated_x_others = torch.zeros_like(x_others)
    cluster_rows: list[torch.Tensor] = []

    # Lines 171--191 are kept intentionally scalar over centers.  Vectorizing
    # would change tie/order behavior and no longer be a strict source adapter.
    for b in range(batch):
        batch_clusters: list[torch.Tensor] = []
        for i in range(left_tokens):
            key_others_norm_i = key_others_norm[b, i, :].unsqueeze(0).unsqueeze(0)
            before_i_key = key_others_norm[b, :i, :].unsqueeze(0)
            after_i_key = key_others_norm[b, i + 1 :, :].unsqueeze(0)
            before_i_x = x_others[b, :i, :].unsqueeze(0)
            after_i_x = x_others[b, i + 1 :, :].unsqueeze(0)
            rest_x = torch.cat(
                [before_i_x, after_i_x, non_topk[b, :, :].unsqueeze(0)], dim=1
            )
            before_i_attn = x_others_attn[b, :i].unsqueeze(0)
            after_i_attn = x_others_attn[b, i + 1 :].unsqueeze(0)
            rest_attn = torch.cat(
                [before_i_attn, after_i_attn, non_topk_attn[b, :].unsqueeze(0)],
                dim=1,
            )
            rest_keys = torch.cat(
                [
                    before_i_key,
                    after_i_key,
                    non_topk_key_norm[b, :, :].unsqueeze(0),
                ],
                dim=1,
            )
            cos_sim_matrix = torch.bmm(key_others_norm_i, rest_keys.transpose(1, 2))
            _, cluster_indices = torch.topk(
                cos_sim_matrix, k=int(CLUSTER_NEIGHBORS), dim=2, largest=True
            )
            flat_cluster_indices = cluster_indices.squeeze()
            cluster_tokens = rest_x[:, flat_cluster_indices, :]
            weights = rest_attn[:, flat_cluster_indices].unsqueeze(-1)
            weighted_avg = torch.sum(cluster_tokens * weights, dim=1)
            updated_center = x_others[b, i, :] + weighted_avg
            updated_x_others[b, i, :] = updated_center
            batch_clusters.append(flat_cluster_indices)
        cluster_rows.append(torch.stack(batch_clusters, dim=0))

    cluster_assignment = torch.stack(cluster_rows, dim=0)
    if reconstruct:
        reconstructed, nearest = reconstruct_from_reference(
            features, x_others, updated_x_others, return_indices=True
        )
    else:
        reconstructed = updated_x_others
        nearest = torch.arange(
            updated_x_others.shape[1], device=updated_x_others.device
        ).unsqueeze(0).expand(batch, -1)
    return CompressionResult(
        method=METHOD_NAME,
        nominal_k=k,
        selected_indices=idx,
        compressed_tokens=updated_x_others,
        reconstructed_tokens=reconstructed,
        reconstruction_indices=nearest,
        scores=cls_attn,
        cluster_assignment=cluster_assignment,
        tile_budgets=(k,),
        diagnostics={
            "attention_topk_indices": topk_idx,
            "spatial_indices": torch.stack(spatial_list, dim=0),
            "complement_indices": compl,
            "unmerged_centers": x_others,
            "normalized_center_keys": key_others_norm,
        },
    )


def compress(
    features: torch.Tensor,
    *,
    context: CompressionContext,
    k: Optional[int] = None,
    retention_ratio: Optional[float] = None,
    tile_metadata: Optional[TileMetadata] = None,
    reconstruct: bool = True,
) -> CompressionResult:
    return compress_tiled(
        method=METHOD_NAME,
        features=features,
        single_tile_compressor=_compress_single_tile,
        k=k,
        retention_ratio=retention_ratio,
        context=context,
        tile_metadata=tile_metadata,
        reconstruct=reconstruct,
    )

