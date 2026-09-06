"""Exact token alignment for InternVL's channel-last pixel shuffle.

InternVL uses a *downsampling* operation named ``pixel_shuffle``.  For the
8B checkpoint a 32 x 32 patch grid is rearranged into a 16 x 16 grid.  The
four source patch vectors are concatenated in row-major order before the MLP
projector sees them::

    top-left, top-right, bottom-left, bottom-right

The helpers in this module deliberately operate on patch indices as well as
features.  That makes the alignment independently testable and gives attack
and compression code one canonical mapping.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Literal

import torch
from torch import Tensor


Reduction = Literal["mean", "sum", "max"]


def _downsample_factor(scale_factor: float) -> int:
    """Return the integer spatial reduction implied by ``scale_factor``."""

    if not 0.0 < scale_factor <= 1.0:
        raise ValueError(f"scale_factor must be in (0, 1], got {scale_factor!r}")
    factor = round(1.0 / scale_factor)
    if factor < 1 or abs(scale_factor * factor - 1.0) > 1e-12:
        raise ValueError(
            "InternVL alignment requires the reciprocal of an integer scale; "
            f"got {scale_factor!r}"
        )
    return factor


@dataclass(frozen=True)
class PixelShuffleLayout:
    """Shape metadata for one InternVL image tile.

    Spatial coordinates and token indices are row-major.  ``raw`` excludes
    the ViT CLS token because the model removes CLS before pixel shuffle.
    """

    raw_height: int
    raw_width: int
    scale_factor: float = 0.5

    def __post_init__(self) -> None:
        if not isinstance(self.raw_height, Integral) or self.raw_height <= 0:
            raise ValueError(f"raw_height must be a positive integer, got {self.raw_height!r}")
        if not isinstance(self.raw_width, Integral) or self.raw_width <= 0:
            raise ValueError(f"raw_width must be a positive integer, got {self.raw_width!r}")
        factor = _downsample_factor(self.scale_factor)
        if self.raw_height % factor or self.raw_width % factor:
            raise ValueError(
                f"raw grid {(self.raw_height, self.raw_width)} is not divisible "
                f"by downsample factor {factor}"
            )

    @property
    def factor(self) -> int:
        return _downsample_factor(self.scale_factor)

    @property
    def post_height(self) -> int:
        return self.raw_height // self.factor

    @property
    def post_width(self) -> int:
        return self.raw_width // self.factor

    @property
    def raw_token_count(self) -> int:
        return self.raw_height * self.raw_width

    @property
    def post_token_count(self) -> int:
        return self.post_height * self.post_width

    @property
    def patches_per_post_token(self) -> int:
        return self.factor * self.factor

    @property
    def post_grid(self) -> tuple[int, int]:
        return self.post_height, self.post_width

    @property
    def raw_grid(self) -> tuple[int, int]:
        return self.raw_height, self.raw_width

    def post_to_raw(self, *, device: torch.device | str | None = None) -> Tensor:
        """Return ``[post_tokens, factor**2]`` source patch indices."""

        return postshuffle_to_raw_indices(
            self.raw_height,
            self.raw_width,
            scale_factor=self.scale_factor,
            device=device,
        )

    def raw_to_post(self, *, device: torch.device | str | None = None) -> Tensor:
        """Return ``[raw_tokens, 2]`` pairs of (post index, channel group)."""

        inverse = torch.empty(
            (self.raw_token_count, 2), dtype=torch.long, device=device
        )
        groups = self.post_to_raw(device=device)
        post_ids = torch.arange(self.post_token_count, device=device)
        post_ids = post_ids[:, None].expand_as(groups)
        slots = torch.arange(self.patches_per_post_token, device=device)
        slots = slots[None, :].expand_as(groups)
        inverse[groups.reshape(-1), 0] = post_ids.reshape(-1)
        inverse[groups.reshape(-1), 1] = slots.reshape(-1)
        return inverse


INTERNVL35_448_LAYOUT = PixelShuffleLayout(32, 32, 0.5)


def internvl_pixel_shuffle(features: Tensor, scale_factor: float = 0.5) -> Tensor:
    """Apply InternVL's exact channel-last spatial downsampling rearrangement.

    Args:
        features: Tensor shaped ``[batch, height, width, channels]``.
        scale_factor: Reciprocal integer spatial scale.  InternVL3.5-8B uses
            ``0.5``.

    Returns:
        Tensor shaped ``[batch, height*scale, width*scale,
        channels/(scale**2)]``.  This is a pure rearrangement.
    """

    if features.ndim != 4:
        raise ValueError(
            "features must have shape [batch, height, width, channels], "
            f"got {tuple(features.shape)}"
        )
    batch_size, height, width, channels = features.shape
    layout = PixelShuffleLayout(height, width, scale_factor)
    factor = layout.factor

    # These four operations intentionally mirror the model source.  The first
    # view packs adjacent columns into channels; after the transpose, the
    # second view packs adjacent rows.  The final transpose restores H/W.
    shuffled = features.view(
        batch_size, height, width // factor, channels * factor
    )
    shuffled = shuffled.permute(0, 2, 1, 3).contiguous()
    shuffled = shuffled.view(
        batch_size,
        width // factor,
        height // factor,
        channels * factor * factor,
    )
    return shuffled.permute(0, 2, 1, 3).contiguous()


def postshuffle_to_raw_indices(
    raw_height: int,
    raw_width: int,
    *,
    scale_factor: float = 0.5,
    device: torch.device | str | None = None,
) -> Tensor:
    """Map every flattened post-shuffle token to its raw patch indices.

    Both token axes are row-major.  The last axis follows the feature channel
    concatenation order used by :func:`internvl_pixel_shuffle`.
    """

    layout = PixelShuffleLayout(raw_height, raw_width, scale_factor)
    rows = torch.arange(raw_height, dtype=torch.long, device=device).reshape(
        layout.post_height, layout.factor
    )
    cols = torch.arange(raw_width, dtype=torch.long, device=device).reshape(
        layout.post_width, layout.factor
    )
    raw_indices = rows[:, None, :, None] * raw_width + cols[None, :, None, :]
    return raw_indices.reshape(
        layout.post_token_count, layout.patches_per_post_token
    )


def raw_to_postshuffle_indices(
    raw_height: int,
    raw_width: int,
    *,
    scale_factor: float = 0.5,
    device: torch.device | str | None = None,
) -> Tensor:
    """Inverse mapping as ``[raw_tokens, (post_index, channel_group)]``."""

    return PixelShuffleLayout(raw_height, raw_width, scale_factor).raw_to_post(
        device=device
    )


def map_raw_patch_scores(
    raw_scores: Tensor,
    layout: PixelShuffleLayout = INTERNVL35_448_LAYOUT,
    *,
    reduction: Reduction = "mean",
) -> Tensor:
    """Aggregate raw-patch scores onto post-pixel-shuffle visual tokens.

    The final dimension of ``raw_scores`` must contain exactly one value per
    raw patch.  Leading dimensions (batch, layer, attention head, ...) are
    preserved.
    """

    if raw_scores.ndim < 1 or raw_scores.shape[-1] != layout.raw_token_count:
        raise ValueError(
            f"raw_scores last dimension must be {layout.raw_token_count}, got "
            f"{tuple(raw_scores.shape)}"
        )
    groups = layout.post_to_raw(device=raw_scores.device)
    grouped = raw_scores[..., groups]
    if reduction == "mean":
        return grouped.mean(dim=-1)
    if reduction == "sum":
        return grouped.sum(dim=-1)
    if reduction == "max":
        return grouped.amax(dim=-1)
    raise ValueError(f"unsupported reduction {reduction!r}; expected mean, sum, or max")


__all__ = [
    "INTERNVL35_448_LAYOUT",
    "PixelShuffleLayout",
    "internvl_pixel_shuffle",
    "map_raw_patch_scores",
    "postshuffle_to_raw_indices",
    "raw_to_postshuffle_indices",
]
