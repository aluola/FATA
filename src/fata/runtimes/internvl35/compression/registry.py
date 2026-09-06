"""Canonical router restricted to the final four paper compressors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch

from . import flowcut, prumerge, visionzip, vispruner
from .base import (
    CompressionContext,
    CompressionInputError,
    CompressionResult,
    TileMetadata,
)


METHOD_NAMES: tuple[str, ...] = (
    "VisionZIP",
    "VisPruner",
    "PruMerge",
    "FlowCut",
)

_COMPRESSORS: dict[str, Callable[..., CompressionResult]] = {
    "visionzip": visionzip.compress,
    "vispruner": vispruner.compress,
    "prumerge": prumerge.compress,
    "flowcut": flowcut.compress,
}


def canonical_method_name(method_name: str) -> str:
    key = method_name.strip().lower()
    if key not in _COMPRESSORS:
        raise CompressionInputError(
            f"unknown compression method {method_name!r}; expected one of {METHOD_NAMES}"
        )
    return METHOD_NAMES[list(_COMPRESSORS).index(key)]


def get_compressor(method_name: str) -> Callable[..., CompressionResult]:
    canonical_method_name(method_name)
    return _COMPRESSORS[method_name.strip().lower()]


def compress_visual_tokens(
    method_name: str,
    features: torch.Tensor,
    *,
    context: Optional[CompressionContext] = None,
    tile_metadata: Optional[TileMetadata] = None,
    k: Optional[int] = None,
    retention_ratio: Optional[float] = None,
    reconstruct: bool = True,
) -> CompressionResult:
    """Compress then reconstruct to ``N_full`` for HF placeholder parity.

    ``features`` are InternVL visual tokens after the model's exact pixel
    shuffle and before ``multi_modal_projector``.  Attention and QKV signals
    in ``context`` must describe the same grouped token order.  Reconstruction
    occurs before the projector, retaining every HF placeholder.  Full-token
    evaluation must bypass this router instead of naming a fake compressor.

    ``reconstruct=False`` is a documented attack-internal surrogate mode: the
    O(N_full x K) cosine-nearest reconstruction is skipped and
    ``reconstructed_tokens`` is set to the compressed tokens.  The formal
    evaluation path always keeps the default ``True``.
    """

    compressor = get_compressor(method_name)
    kwargs = {
        "k": k,
        "retention_ratio": retention_ratio,
        "tile_metadata": tile_metadata,
        "reconstruct": reconstruct,
    }
    if context is None:
        raise CompressionInputError(f"{canonical_method_name(method_name)} requires context")
    kwargs["context"] = context
    return compressor(features, **kwargs)


@dataclass(frozen=True)
class InternVLCompressionAdapter:
    """Configured callable used by InternVL evaluation code."""

    method_name: str
    retention_ratio: float

    def __post_init__(self) -> None:
        canonical_method_name(self.method_name)
        if not 0 < self.retention_ratio < 1:
            raise CompressionInputError(
                "a compressor retention ratio must be in (0, 1); Full bypasses compression"
            )

    def __call__(
        self,
        features: torch.Tensor,
        *,
        context: Optional[CompressionContext] = None,
        tile_metadata: Optional[TileMetadata] = None,
    ) -> CompressionResult:
        return compress_visual_tokens(
            self.method_name,
            features,
            context=context,
            tile_metadata=tile_metadata,
            retention_ratio=self.retention_ratio,
        )


apply_compression = compress_visual_tokens


__all__ = [
    "InternVLCompressionAdapter",
    "METHOD_NAMES",
    "apply_compression",
    "canonical_method_name",
    "compress_visual_tokens",
    "get_compressor",
]
