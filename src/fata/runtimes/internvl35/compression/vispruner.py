"""Source-faithful VisPruner core with dynamic, tile-local dispatch."""

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


METHOD_NAME = "VisPruner"


def _compress_single_tile(
    features: torch.Tensor,
    k: int,
    context: Optional[TileCompressionContext],
    *,
    reconstruct: bool = True,
) -> CompressionResult:
    if context is None or context.cls_attention_heads is None:
        raise CompressionInputError("VisPruner requires per-head CLS attention")
    batch, tokens, width = features.shape
    attention = context.cls_attention_heads
    if attention.shape[0] != batch or attention.shape[2] != tokens:
        raise CompressionInputError("VisPruner attention is not aligned with features")

    image_attentions = attention.mean(dim=1)
    features_normalized = features / features.norm(dim=-1, keepdim=True)
    important_token_num = int(k * 0.5)
    if important_token_num <= 0:
        important_token_num = 1
    diverse_token_num = k - important_token_num
    token_indices = image_attentions.argsort(dim=-1, descending=True)
    important_indices = token_indices[:, :important_token_num]
    residual_indices = token_indices[:, important_token_num:]
    batch_index = torch.arange(batch, device=features.device)
    pruning_trace: list[dict[str, torch.Tensor | int]] = []

    while True:
        residual_tokens = features_normalized[batch_index, residual_indices]
        r = min(8, residual_tokens.shape[1] - diverse_token_num)
        if r <= 0:
            break
        a, b = residual_tokens[..., ::2, :], residual_tokens[..., 1::2, :]
        pair_scores = a @ b.transpose(-1, -2)
        scores = pair_scores.max(dim=-1).values
        distinct_indices = scores.argsort(dim=-1, descending=True)[:, r:]
        removed_positions = scores.argsort(dim=-1, descending=True)[:, :r]
        even_indices = residual_indices[..., ::2]
        removed_global = even_indices[batch_index, removed_positions]
        residual_indices = torch.cat(
            [even_indices[batch_index, distinct_indices], residual_indices[..., 1::2]],
            dim=-1,
        )
        pruning_trace.append(
            {
                "r": r,
                "max_pair_scores": scores,
                "removed_even_positions": removed_positions,
                "removed_global_indices": removed_global,
                "remaining_indices": residual_indices,
            }
        )

    selected = torch.cat([important_indices, residual_indices], dim=-1)
    selected = torch.sort(selected).values
    index = selected.unsqueeze(-1).expand(-1, -1, width)
    compressed = torch.gather(features, 1, index)
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
        scores=image_attentions,
        tile_budgets=(k,),
        diagnostics={
            "important_indices": important_indices,
            "final_diverse_indices": residual_indices,
            "pruning_trace": tuple(pruning_trace),
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

