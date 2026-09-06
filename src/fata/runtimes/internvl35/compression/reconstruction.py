"""Fixed-length reconstruction used by all five reference compressors."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def reconstruct_to_original_length(
    original_patches: torch.Tensor,
    compressed_patches: torch.Tensor,
    *,
    return_indices: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Copy the cosine-nearest compressed token to every original position.

    Lines 11--15 of ``compression_zoo.py`` are intentionally preserved,
    including ``F.cosine_similarity``'s default epsilon and first-index tie
    behavior in ``argmax``.
    """

    sim = F.cosine_similarity(
        original_patches.unsqueeze(2), compressed_patches.unsqueeze(1), dim=-1
    )
    nearest_idx = torch.argmax(sim, dim=-1)
    expanded_idx = nearest_idx.unsqueeze(-1).expand(
        -1, -1, compressed_patches.shape[-1]
    )
    reconstructed = torch.gather(compressed_patches, 1, expanded_idx)
    if return_indices:
        return reconstructed, nearest_idx
    return reconstructed


def reconstruct_from_reference(
    original_patches: torch.Tensor,
    reference_patches: torch.Tensor,
    value_patches: torch.Tensor,
    *,
    return_indices: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """PruMerge's source-specific split between addressing and assigned value."""

    sim = F.cosine_similarity(
        original_patches.unsqueeze(2), reference_patches.unsqueeze(1), dim=-1
    )
    nearest_idx = torch.argmax(sim, dim=-1)
    expanded_idx = nearest_idx.unsqueeze(-1).expand(-1, -1, value_patches.shape[-1])
    reconstructed = torch.gather(value_patches, 1, expanded_idx)
    if return_indices:
        return reconstructed, nearest_idx
    return reconstructed

