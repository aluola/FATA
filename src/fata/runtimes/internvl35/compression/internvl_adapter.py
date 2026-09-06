"""Bridge raw InternVL vision signals to the dynamic compressor contract.

The injection point is deliberately *after* InternVL pixel shuffle and
*before* ``multi_modal_projector``.  For InternVL3.5-8B this is a
``[tiles, 256, 4096]`` tensor.  Compression is reconstructed to 256 tokens per
tile before projection, so Hugging Face's image-placeholder count remains
unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch

from ..architecture.internvl_pixel_shuffle import (
    INTERNVL35_448_LAYOUT,
    PixelShuffleLayout,
)

from .base import (
    CompressionContext,
    CompressionInputError,
    CompressionResult,
    TileMetadata,
    coerce_tile_metadata,
)


def _normalize_raw_sequence(
    tensor: Optional[torch.Tensor],
    name: str,
    raw_sequence_length: int,
) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    if tensor.ndim == 3:  # [T, S, C]
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 4 or tensor.shape[-2] != raw_sequence_length:
        raise CompressionInputError(
            f"{name} must be [T,S,C] or [B,T,S,C] with S={raw_sequence_length}; "
            f"got {tuple(tensor.shape)}"
        )
    return tensor


def _normalize_cls_attention(
    tensor: Optional[torch.Tensor],
    raw_tokens: int,
) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    # Accept full source attention matrices as a convenience.
    if tensor.ndim in (4, 5) and tensor.shape[-2:] == (
        raw_tokens + 1,
        raw_tokens + 1,
    ):
        tensor = tensor[..., 0, 1:]
    if tensor.ndim == 3:  # [T, H, R]
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 4 or tensor.shape[-1] != raw_tokens:
        raise CompressionInputError(
            "raw CLS attention must be [T,H,R], [B,T,H,R], [T,H,S,S], "
            f"or [B,T,H,S,S]; got {tuple(tensor.shape)}"
        )
    return tensor


def group_raw_patch_tensor(
    raw_patch_tensor: torch.Tensor,
    *,
    layout: PixelShuffleLayout = INTERNVL35_448_LAYOUT,
) -> torch.Tensor:
    """Concatenate each real pixel-shuffle group in TL, TR, BL, BR order.

    Input may be ``[T, R, C]`` (one dynamic-tiled sample) or
    ``[B, T, R, C]``.  Output is always ``[B, T*P, 4*C]`` for the checkpoint's
    factor-two spatial reduction.
    """

    if raw_patch_tensor.ndim == 3:
        raw_patch_tensor = raw_patch_tensor.unsqueeze(0)
    if raw_patch_tensor.ndim != 4 or raw_patch_tensor.shape[-2] != layout.raw_token_count:
        raise CompressionInputError(
            "raw_patch_tensor must be [T,R,C] or [B,T,R,C] with "
            f"R={layout.raw_token_count}; got {tuple(raw_patch_tensor.shape)}"
        )
    batch, tiles, _, width = raw_patch_tensor.shape
    groups = layout.post_to_raw(device=raw_patch_tensor.device)
    grouped = raw_patch_tensor[:, :, groups, :]
    return grouped.reshape(
        batch,
        tiles * layout.post_token_count,
        layout.patches_per_post_token * width,
    )


def _repeat_cls_for_group(
    cls_tensor: torch.Tensor,
    groups_per_token: int,
) -> torch.Tensor:
    batch, tiles, width = cls_tensor.shape
    return (
        cls_tensor.unsqueeze(2)
        .expand(batch, tiles, groups_per_token, width)
        .reshape(batch, tiles, groups_per_token * width)
    )


def build_internvl_compression_context(
    *,
    raw_cls_attention_heads: Optional[torch.Tensor] = None,
    raw_query: Optional[torch.Tensor] = None,
    raw_key: Optional[torch.Tensor] = None,
    raw_value: Optional[torch.Tensor] = None,
    num_heads: Optional[int] = None,
    layout: PixelShuffleLayout = INTERNVL35_448_LAYOUT,
) -> CompressionContext:
    """Align raw ViT attention/QKV with post-pixel-shuffle token groups.

    Attention mass over each real 2x2 group is summed.  Patch K/V vectors are
    concatenated in the model's channel order.  CLS Q/K/V vectors are repeated
    once per group slot so their width matches the 4x-wide pre-projector
    features.  These are the minimum unavoidable InternVL adaptations of the
    formulas in ``compression_zoo.py``.
    """

    raw_tokens = layout.raw_token_count
    sequence = raw_tokens + 1
    attention = _normalize_cls_attention(raw_cls_attention_heads, raw_tokens)
    query = _normalize_raw_sequence(raw_query, "raw_query", sequence)
    key = _normalize_raw_sequence(raw_key, "raw_key", sequence)
    value = _normalize_raw_sequence(raw_value, "raw_value", sequence)
    available = [tensor for tensor in (attention, query, key, value) if tensor is not None]
    if not available:
        return CompressionContext()
    batch_tiles = {(tensor.shape[0], tensor.shape[1]) for tensor in available}
    if len(batch_tiles) != 1:
        raise CompressionInputError(
            f"raw attention/QKV disagree on [B,T]: {sorted(batch_tiles)}"
        )
    batch, tiles = next(iter(batch_tiles))
    groups = layout.post_to_raw(device=available[0].device)
    group_size = layout.patches_per_post_token
    post_tokens = layout.post_token_count

    aligned_attention = None
    if attention is not None:
        attention_groups = groups.to(attention.device)
        # [B,T,H,P,G] -> [B,H,T*P]
        aligned_attention = attention[..., attention_groups].sum(dim=-1)
        aligned_attention = aligned_attention.permute(0, 2, 1, 3).reshape(
            batch, attention.shape[2], tiles * post_tokens
        )

    cls_query = None
    if query is not None:
        cls_query = _repeat_cls_for_group(query[:, :, 0], group_size)

    cls_key = patch_key = None
    if key is not None:
        cls_key = _repeat_cls_for_group(key[:, :, 0], group_size)
        patch_key = group_raw_patch_tensor(key[:, :, 1:], layout=layout)

    cls_value = patch_value = None
    if value is not None:
        if num_heads is None or num_heads < 1:
            raise CompressionInputError("num_heads is required with raw_value")
        width = value.shape[-1]
        if width % num_heads:
            raise CompressionInputError(
                f"raw value width {width} is not divisible by num_heads={num_heads}"
            )
        head_dim = width // num_heads
        states = value.view(batch, tiles, sequence, num_heads, head_dim).permute(
            0, 1, 3, 2, 4
        )
        cls_state = states[:, :, :, 0]  # [B,T,H,D]
        cls_value = (
            cls_state.unsqueeze(-2)
            .expand(batch, tiles, num_heads, group_size, head_dim)
            .reshape(batch, tiles, num_heads, group_size * head_dim)
            .permute(0, 2, 1, 3)
        )
        value_groups = groups.to(value.device)
        patch_state = states[:, :, :, 1:]
        # [B,T,H,P,G,D] -> [B,H,T*P,G*D]
        patch_value = patch_state[:, :, :, value_groups, :].reshape(
            batch, tiles, num_heads, post_tokens, group_size * head_dim
        )
        patch_value = patch_value.permute(0, 2, 1, 3, 4).reshape(
            batch, num_heads, tiles * post_tokens, group_size * head_dim
        )

    return CompressionContext(
        cls_attention_heads=aligned_attention,
        cls_query=cls_query,
        cls_key=cls_key,
        patch_key=patch_key,
        cls_value=cls_value,
        patch_value=patch_value,
    )


def internvl35_tile_metadata(
    total_tile_count: int,
    *,
    thumbnail_appended: bool,
) -> TileMetadata:
    """Create metadata for the processor's row-major tiles and final thumbnail."""

    if total_tile_count < 1:
        raise CompressionInputError("total_tile_count must be positive")
    if thumbnail_appended and total_tile_count < 2:
        raise CompressionInputError("a thumbnail cannot be the only tile")
    thumbnail_ids = (total_tile_count - 1,) if thumbnail_appended else ()
    return TileMetadata.from_tile_shapes(
        [INTERNVL35_448_LAYOUT.post_grid] * total_tile_count,
        thumbnail_tile_indices=thumbnail_ids,
    )


@dataclass(frozen=True)
class InternVLCompressionInputs:
    """Ready-to-run pre-projector features, aligned signals, and coordinates."""

    features: torch.Tensor
    context: CompressionContext
    tile_metadata: TileMetadata

    def as_projector_tiles(self, flat_features: torch.Tensor) -> torch.Tensor:
        """Restore ``[B,N,C]`` to the model's ``[B*T,256,C]`` shape."""

        return reshape_for_internvl_projector(flat_features, self.tile_metadata)


def build_internvl_compression_inputs(
    raw_patch_features: torch.Tensor,
    *,
    raw_cls_attention_heads: Optional[torch.Tensor] = None,
    raw_query: Optional[torch.Tensor] = None,
    raw_key: Optional[torch.Tensor] = None,
    raw_value: Optional[torch.Tensor] = None,
    num_heads: Optional[int] = None,
    thumbnail_appended: bool = False,
    layout: PixelShuffleLayout = INTERNVL35_448_LAYOUT,
) -> InternVLCompressionInputs:
    """Build all pure inputs needed at the InternVL pre-projector hook point."""

    features = group_raw_patch_tensor(raw_patch_features, layout=layout)
    raw = raw_patch_features.unsqueeze(0) if raw_patch_features.ndim == 3 else raw_patch_features
    if raw.ndim != 4:
        raise CompressionInputError("raw_patch_features has an invalid rank")
    tiles = raw.shape[1]
    metadata = TileMetadata.from_tile_shapes(
        [layout.post_grid] * tiles,
        thumbnail_tile_indices=(tiles - 1,) if thumbnail_appended else (),
    )
    context = build_internvl_compression_context(
        raw_cls_attention_heads=raw_cls_attention_heads,
        raw_query=raw_query,
        raw_key=raw_key,
        raw_value=raw_value,
        num_heads=num_heads,
        layout=layout,
    )
    return InternVLCompressionInputs(features, context, metadata)


def flatten_internvl_preprojector_features(
    tile_features: torch.Tensor,
) -> tuple[torch.Tensor, int, int]:
    """Normalize model-native tile tensors to ``[B,N_full,C]``.

    ``[T,P,C]`` denotes one sample.  ``[B,T,P,C]`` is supported when every
    batch item has the same tile layout.
    """

    if tile_features.ndim == 3:
        tile_features = tile_features.unsqueeze(0)
    if tile_features.ndim != 4:
        raise CompressionInputError(
            "InternVL pre-projector features must be [T,P,C] or [B,T,P,C]"
        )
    batch, tiles, per_tile, width = tile_features.shape
    return tile_features.reshape(batch, tiles * per_tile, width), tiles, per_tile


def reshape_for_internvl_projector(
    flat_features: torch.Tensor,
    tile_metadata: Any,
) -> torch.Tensor:
    """Restore flat reconstructed tokens to model-native tile-major shape."""

    if flat_features.ndim != 3:
        raise CompressionInputError("flat projector features must be [B,N_full,C]")
    batch, n_full, width = flat_features.shape
    metadata = coerce_tile_metadata(tile_metadata, n_full)
    tile_parts = [
        torch.index_select(
            flat_features,
            1,
            metadata.global_indices_for_tile(tile_id, flat_features.device),
        )
        for tile_id in metadata.tile_ids
    ]
    lengths = {part.shape[1] for part in tile_parts}
    if len(lengths) != 1:
        raise CompressionInputError(
            "InternVL projector reshape requires equal token count for every tile"
        )
    per_tile = next(iter(lengths))
    return torch.stack(tile_parts, dim=1).reshape(
        batch * metadata.num_tiles, per_tile, width
    )


@dataclass(frozen=True)
class InternVLProjectorCompressionOutput:
    """Reconstructed model-native projector input plus full diagnostics.

    ``projector_input`` is ``None`` in the attack-internal surrogate mode
    (``reconstruct=False``), which skips the cosine-nearest reconstruction.
    """

    projector_input: Optional[torch.Tensor]
    compression_result: CompressionResult


def compress_internvl_preprojector(
    method_name: str,
    tile_features: torch.Tensor,
    *,
    context: Optional[CompressionContext],
    tile_metadata: Any,
    k: Optional[int] = None,
    retention_ratio: Optional[float] = None,
    reconstruct: bool = True,
) -> InternVLProjectorCompressionOutput:
    """Run a compressor at the exact InternVL pixel-shuffle/projector seam."""

    flat_features, tile_count, per_tile = flatten_internvl_preprojector_features(
        tile_features
    )
    metadata = coerce_tile_metadata(tile_metadata, flat_features.shape[1])
    if metadata.num_tiles != tile_count:
        raise CompressionInputError(
            f"features contain {tile_count} tiles but metadata contains {metadata.num_tiles}"
        )
    tile_lengths = {
        int(metadata.global_indices_for_tile(tile_id, flat_features.device).numel())
        for tile_id in metadata.tile_ids
    }
    if tile_lengths != {per_tile}:
        raise CompressionInputError(
            f"features contain {per_tile} tokens/tile but metadata lengths are {tile_lengths}"
        )
    # Lazy import avoids a registry/module cycle at package import time.
    from .registry import compress_visual_tokens

    result = compress_visual_tokens(
        method_name,
        flat_features,
        context=context,
        tile_metadata=metadata,
        k=k,
        retention_ratio=retention_ratio,
        reconstruct=reconstruct,
    )
    if not reconstruct:
        return InternVLProjectorCompressionOutput(None, result)
    projector_input = reshape_for_internvl_projector(
        result.reconstructed_tokens, metadata
    )
    return InternVLProjectorCompressionOutput(projector_input, result)


__all__ = [
    "InternVLCompressionInputs",
    "InternVLProjectorCompressionOutput",
    "build_internvl_compression_context",
    "build_internvl_compression_inputs",
    "compress_internvl_preprojector",
    "flatten_internvl_preprojector_features",
    "group_raw_patch_tensor",
    "internvl35_tile_metadata",
    "reshape_for_internvl_projector",
]
