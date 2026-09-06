"""Source-faithful FlowCut core with dynamic, tile-local dispatch."""

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
from .reconstruction import reconstruct_to_original_length


METHOD_NAME = "FlowCut"


def _compress_single_tile(
    features: torch.Tensor,
    k: int,
    context: Optional[TileCompressionContext],
    *,
    reconstruct: bool = True,
) -> CompressionResult:
    if context is None or context.cls_attention_heads is None:
        raise CompressionInputError("FlowCut requires per-head CLS attention")
    if context.cls_value is None or context.patch_value is None:
        raise CompressionInputError("FlowCut requires CLS and patch value states")

    batch, tokens, width = features.shape
    attention = context.cls_attention_heads
    cls_value = context.cls_value
    patch_value = context.patch_value
    if attention.shape[0] != batch or attention.shape[2] != tokens:
        raise CompressionInputError("FlowCut attention is not aligned with features")
    if patch_value.shape[:3] != (batch, attention.shape[1], tokens):
        raise CompressionInputError("FlowCut patch values are not aligned with attention")
    if cls_value.shape[:2] != patch_value.shape[:2] or cls_value.shape[-1] != patch_value.shape[-1]:
        raise CompressionInputError("FlowCut CLS value shape is incompatible with patch values")
    if patch_value.shape[1] * patch_value.shape[-1] != width:
        raise CompressionInputError(
            "FlowCut source assumes feature width == num_heads * value head_dim"
        )

    # compression_zoo.py:44--63.  The semantic softmax includes the CLS value.
    cls_attn = attention.mean(dim=1)
    value_states = torch.cat([cls_value.unsqueeze(2), patch_value], dim=2)
    semantic_weight = torch.matmul(
        cls_value.unsqueeze(2), value_states.transpose(-1, -2)
    )
    semantic_attn = F.softmax(semantic_weight, dim=-1).squeeze(2).mean(dim=1)[:, 1:]
    value_metric = patch_value.mean(dim=1)
    relation_score = cls_attn / (cls_attn.sum(dim=-1, keepdim=True) + 1e-8)
    semantic_score = semantic_attn / (semantic_attn.sum(dim=-1, keepdim=True) + 1e-8)
    final_score = (relation_score + semantic_score) * torch.norm(
        value_metric, p=1, dim=-1
    )

    _, keep_indices = torch.topk(final_score, int(k), dim=1)
    keep_indices = keep_indices.sort(dim=1).values
    index = keep_indices.unsqueeze(-1).expand(-1, -1, width)
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
        selected_indices=keep_indices,
        compressed_tokens=compressed,
        reconstructed_tokens=reconstructed,
        reconstruction_indices=nearest,
        scores=final_score,
        tile_budgets=(k,),
        diagnostics={
            "relation_score": relation_score,
            "semantic_attention": semantic_attn,
            "semantic_score": semantic_score,
            "value_metric": value_metric,
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

