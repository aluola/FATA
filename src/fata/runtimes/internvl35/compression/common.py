"""Dynamic-budget and tile-safe dispatch shared by compressor adapters."""

from __future__ import annotations

from typing import Callable, Optional

import torch

from .base import (
    CompressionContext,
    CompressionInputError,
    CompressionResult,
    TileCompressionContext,
    TileMetadata,
    coerce_tile_metadata,
    compute_k_actual,
)


SingleTileCompressor = Callable[
    [torch.Tensor, int, Optional[TileCompressionContext]], CompressionResult
]


def validate_features(features: torch.Tensor) -> tuple[int, int, int]:
    if not isinstance(features, torch.Tensor) or features.ndim != 3:
        raise CompressionInputError("features must be a [B, N_full, C] tensor")
    batch, tokens, width = features.shape
    if batch < 1 or tokens < 1 or width < 1:
        raise CompressionInputError(f"invalid feature shape: {tuple(features.shape)}")
    if not features.is_floating_point():
        raise CompressionInputError("features must have a floating dtype")
    return batch, tokens, width


def resolve_k(
    n_full: int,
    *,
    k: Optional[int],
    retention_ratio: Optional[float],
) -> int:
    if (k is None) == (retention_ratio is None):
        raise CompressionInputError("provide exactly one of k or retention_ratio")
    if k is not None:
        if isinstance(k, bool) or int(k) != k or k < 1:
            raise CompressionInputError(f"k must be a positive integer, got {k}")
        return int(k)
    assert retention_ratio is not None
    return compute_k_actual(n_full, retention_ratio)


def allocate_tile_budgets(tile_lengths: tuple[int, ...], total_k: int) -> tuple[int, ...]:
    """Hamilton allocation with one representative per independent tile.

    The experiment's InternVL tiles all contain 256 post-shuffle tokens, so its
    five retention ratios always satisfy ``total_k >= tile_count``.  Rejecting
    smaller budgets is safer than silently reconstructing one tile from a
    different tile, which would violate the metadata contract.
    """

    if not tile_lengths or any(length < 1 for length in tile_lengths):
        raise CompressionInputError("tile lengths must all be positive")
    n_full = sum(tile_lengths)
    if total_k < len(tile_lengths):
        raise CompressionInputError(
            f"budget K={total_k} cannot retain one token for each of "
            f"{len(tile_lengths)} independent tiles"
        )
    if total_k > n_full:
        total_k = n_full

    ideals = [total_k * length / n_full for length in tile_lengths]
    budgets = [max(1, int(value)) for value in ideals]

    while sum(budgets) > total_k:
        candidates = [i for i, value in enumerate(budgets) if value > 1]
        if not candidates:
            raise CompressionInputError("unable to allocate a non-empty budget per tile")
        index = min(candidates, key=lambda i: (ideals[i] - budgets[i], -i))
        budgets[index] -= 1

    while sum(budgets) < total_k:
        candidates = [i for i, value in enumerate(budgets) if value < tile_lengths[i]]
        if not candidates:
            break
        index = max(candidates, key=lambda i: (ideals[i] - budgets[i], -i))
        budgets[index] += 1
    return tuple(budgets)


def identity_result(method: str, features: torch.Tensor, nominal_k: int) -> CompressionResult:
    batch, tokens, _ = features.shape
    indices = torch.arange(tokens, device=features.device).unsqueeze(0).expand(batch, -1)
    return CompressionResult(
        method=method,
        nominal_k=nominal_k,
        selected_indices=indices,
        compressed_tokens=features,
        reconstructed_tokens=features,
        reconstruction_indices=indices,
        tile_budgets=(tokens,),
        diagnostics={"bypassed": True},
    )


def compress_tiled(
    *,
    method: str,
    features: torch.Tensor,
    single_tile_compressor: SingleTileCompressor,
    k: Optional[int] = None,
    retention_ratio: Optional[float] = None,
    context: Optional[CompressionContext] = None,
    tile_metadata: Optional[TileMetadata] = None,
    reconstruct: bool = True,
) -> CompressionResult:
    """Run a source-equivalent core independently within each real tile.

    ``reconstruct=False`` is an attack-internal surrogate mode that skips the
    O(N_full x K) cosine-nearest reconstruction entirely; the formal path
    always keeps the default ``True``.
    """

    batch, n_full, _ = validate_features(features)
    nominal_k = resolve_k(n_full, k=k, retention_ratio=retention_ratio)
    metadata = coerce_tile_metadata(tile_metadata, n_full)
    if nominal_k >= n_full:
        result = identity_result(method, features, nominal_k)
        result.tile_budgets = tuple(
            int(metadata.global_indices_for_tile(tile_id, features.device).numel())
            for tile_id in metadata.tile_ids
        )
        return result

    tile_indices = tuple(
        metadata.global_indices_for_tile(tile_id, features.device)
        for tile_id in metadata.tile_ids
    )
    tile_lengths = tuple(int(indices.numel()) for indices in tile_indices)
    budgets = allocate_tile_budgets(tile_lengths, nominal_k)

    selected_parts: list[torch.Tensor] = []
    compressed_parts: list[torch.Tensor] = []
    if reconstruct:
        reconstructed = torch.empty_like(features)
        reconstruction_indices = torch.empty(
            (batch, n_full), dtype=torch.long, device=features.device
        )
    else:
        reconstructed = None
        reconstruction_indices = None
    score_output: Optional[torch.Tensor] = None
    cluster_parts: list[torch.Tensor] = []
    tile_results: list[CompressionResult] = []
    compressed_offset = 0

    for tile_slot, (indices, tile_k) in enumerate(zip(tile_indices, budgets)):
        tile_features = torch.index_select(features, 1, indices)
        tile_context = None if context is None else context.for_tile(tile_slot, indices)
        tile_result = single_tile_compressor(
            tile_features, tile_k, tile_context, reconstruct=reconstruct
        )
        tile_results.append(tile_result)

        mapped_selected = indices[tile_result.selected_indices]
        selected_parts.append(mapped_selected)
        compressed_parts.append(tile_result.compressed_tokens)
        if reconstruct:
            reconstructed.index_copy_(1, indices, tile_result.reconstructed_tokens)
            mapped_reconstruction = tile_result.reconstruction_indices + compressed_offset
            reconstruction_indices.index_copy_(1, indices, mapped_reconstruction)
        compressed_offset += tile_result.actual_k

        if tile_result.scores is not None:
            if score_output is None:
                score_output = torch.empty(
                    (batch, n_full),
                    dtype=tile_result.scores.dtype,
                    device=tile_result.scores.device,
                )
            score_output.index_copy_(1, indices.to(score_output.device), tile_result.scores)
        if tile_result.cluster_assignment is not None:
            cluster_parts.append(tile_result.cluster_assignment)

    selected = torch.cat(selected_parts, dim=1)
    compressed = torch.cat(compressed_parts, dim=1)
    cluster_assignment = None
    if len(cluster_parts) == len(tile_results) and cluster_parts:
        shapes = {part.shape[2:] for part in cluster_parts}
        if len(shapes) == 1:
            cluster_assignment = torch.cat(cluster_parts, dim=1)

    diagnostics = dict(tile_results[0].diagnostics) if len(tile_results) == 1 else {}
    diagnostics.update(
        {
            "tile_ids": metadata.tile_ids,
            "thumbnail_tile_indices": metadata.thumbnail_tile_indices,
            "tile_results": tuple(tile_results),
            "selected_coordinates": tuple(
                metadata.token_coordinates(selected[b]) for b in range(batch)
            ),
        }
    )
    if not reconstruct:
        # surrogate mode: no reconstruction is computed; keep the compressed
        # tokens in the reconstruction slot so downstream consumers never see
        # a None tensor, and note the mode in diagnostics.
        reconstructed = compressed
        reconstruction_indices = torch.arange(
            compressed.shape[1], device=compressed.device
        ).unsqueeze(0).expand(batch, -1)
        diagnostics.update({"surrogate_no_reconstruction": True})
    return CompressionResult(
        method=method,
        nominal_k=nominal_k,
        selected_indices=selected,
        compressed_tokens=compressed,
        reconstructed_tokens=reconstructed,
        reconstruction_indices=reconstruction_indices,
        scores=score_output,
        cluster_assignment=cluster_assignment,
        tile_budgets=budgets,
        diagnostics=diagnostics,
    )
