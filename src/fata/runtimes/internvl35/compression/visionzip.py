"""Source-faithful VisionZIP core with dynamic, tile-local dispatch."""

from __future__ import annotations

from typing import Optional

import torch

from .base import (
    CompressionContext,
    CompressionInputError,
    CompressionResult,
    TileCompressionContext,
    TileMetadata,
)
from .common import compress_tiled
from .reconstruction import reconstruct_to_original_length


METHOD_NAME = "VisionZIP"


def _compress_single_tile(
    features: torch.Tensor,
    k: int,
    context: Optional[TileCompressionContext],
    *,
    reconstruct: bool = True,
) -> CompressionResult:
    if context is None or context.cls_attention_heads is None:
        raise CompressionInputError("VisionZIP requires per-head CLS attention")
    batch, tokens, width = features.shape
    attention = context.cls_attention_heads
    if attention.shape[0] != batch or attention.shape[2] != tokens:
        raise CompressionInputError("VisionZIP attention is not aligned with features")

    dominant_num = int(k * 0.75) - 1
    if dominant_num <= 0:
        dominant_num = 1
    contextual_num = k - dominant_num - 1
    if dominant_num > tokens:
        raise CompressionInputError("VisionZIP dominant budget exceeds tile length")

    # Source uses a head sum (not a mean) for dominant ranking.
    cls_attention_sum = attention.sum(dim=1)
    topk_indices = cls_attention_sum.topk(dominant_num, dim=1).indices
    dominant_mask = torch.zeros(
        (batch, tokens), dtype=torch.bool, device=features.device
    ).scatter_(1, topk_indices, True)
    # masked_select in the source emits tokens in ascending sequence order.
    dominant_tokens = features.masked_select(dominant_mask.unsqueeze(-1)).view(
        batch, dominant_num, width
    )
    dominant_source_indices = (
        torch.arange(tokens, device=features.device)
        .unsqueeze(0)
        .expand(batch, -1)
        .masked_select(dominant_mask)
        .view(batch, dominant_num)
    )
    residual_indices = (
        torch.arange(tokens, device=features.device)
        .unsqueeze(0)
        .expand(batch, -1)
        .masked_select(~dominant_mask)
        .view(batch, tokens - dominant_num)
    )
    hidden_filtered = features.masked_select((~dominant_mask).unsqueeze(-1)).view(
        batch, tokens - dominant_num, width
    )
    metric_normalized = hidden_filtered / hidden_filtered.norm(dim=-1, keepdim=True)

    assignment = None
    if contextual_num > 0:
        step = max(1, metric_normalized.shape[1] // contextual_num)
        target_indices = torch.arange(
            0, metric_normalized.shape[1], step, device=features.device
        )[:contextual_num]
        target_tokens = metric_normalized[:, target_indices, :]
        residual_positions = torch.arange(metric_normalized.shape[1], device=features.device)
        merge_mask = ~torch.isin(residual_positions, target_indices)
        tokens_to_merge = metric_normalized[:, merge_mask, :]
        similarity = torch.bmm(tokens_to_merge, target_tokens.transpose(1, 2))
        assignment = similarity.argmax(dim=2)
        assign_one_hot = torch.zeros(
            tokens_to_merge.shape[0],
            tokens_to_merge.shape[1],
            contextual_num,
            dtype=hidden_filtered.dtype,
            device=features.device,
        )
        assign_one_hot.scatter_(2, assignment.unsqueeze(-1), 1)
        counts = assign_one_hot.sum(dim=1).clamp(min=1).unsqueeze(-1)
        hidden_to_merge = hidden_filtered[:, merge_mask, :]
        aggregated_hidden = (
            torch.bmm(assign_one_hot.transpose(1, 2), hidden_to_merge) / counts
        )
        target_hidden = hidden_filtered[:, target_indices, :]
        contextual_tokens = target_hidden + aggregated_hidden
        compressed = torch.cat([dominant_tokens, contextual_tokens], dim=1)
        contextual_source_indices = residual_indices[:, target_indices]
        selected = torch.cat(
            [dominant_source_indices, contextual_source_indices], dim=1
        )
    else:
        target_indices = torch.empty(0, dtype=torch.long, device=features.device)
        similarity = None
        compressed = dominant_tokens
        selected = dominant_source_indices

    if reconstruct:
        reconstructed, nearest = reconstruct_to_original_length(
            features, compressed, return_indices=True
        )
    else:
        reconstructed = compressed
        nearest = torch.arange(
            compressed.shape[1], device=compressed.device
        ).unsqueeze(0).expand(batch, -1)
    return CompressionResult(
        method=METHOD_NAME,
        nominal_k=k,
        selected_indices=selected,
        compressed_tokens=compressed,
        reconstructed_tokens=reconstructed,
        reconstruction_indices=nearest,
        scores=cls_attention_sum,
        cluster_assignment=assignment,
        tile_budgets=(k,),
        diagnostics={
            "raw_attention_topk_indices": topk_indices,
            "dominant_indices_in_compressed_order": dominant_source_indices,
            "residual_indices": residual_indices,
            "contextual_target_positions": target_indices,
            "contextual_similarity": similarity,
            "dominant_num": dominant_num,
            "contextual_num": contextual_num,
            "source_nominal_minus_actual": k - compressed.shape[1],
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

