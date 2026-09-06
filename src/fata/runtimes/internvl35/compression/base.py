"""Types and invariants shared by the source-faithful compressors.

The reference implementation operates on one 24 x 24 patch sequence at a
time.  InternVL supplies a dynamic number of independently encoded tiles.
``TileMetadata`` keeps that distinction explicit: no compressor is allowed to
reshape the concatenated sequence into a synthetic two-dimensional image.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import torch


RETENTION_RATIOS: tuple[float, ...] = (1 / 3, 2 / 9, 1 / 9, 1 / 18, 1 / 36)


class CompressionInputError(ValueError):
    """Raised when inputs cannot represent the source algorithm faithfully."""


def compute_k_actual(n_full: int, retention_ratio: float) -> int:
    """Return the experiment-specified dynamic token budget.

    This intentionally uses Python's ``round`` because the experiment prompt
    defines ``max(1, round(N_full * retention_ratio))`` verbatim.
    """

    if n_full < 1:
        raise CompressionInputError(f"n_full must be positive, got {n_full}")
    if not 0 < retention_ratio <= 1:
        raise CompressionInputError(
            f"retention_ratio must be in (0, 1], got {retention_ratio}"
        )
    return max(1, round(n_full * retention_ratio))


def _as_cpu_long(values: Sequence[int] | torch.Tensor, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.long, device="cpu")
    if tensor.ndim != 1:
        raise CompressionInputError(f"{name} must be one-dimensional")
    return tensor.contiguous()


@dataclass(frozen=True)
class TileMetadata:
    """Coordinate metadata for every token in a concatenated InternVL stream.

    Entries in the four tensors describe tokens; ``global_index`` maps each
    entry to the matching position in the feature sequence.  Consequently the
    entries need not themselves be stored in global order.  Tile order is the
    order of each tile's first global token, matching InternVL concatenation.
    """

    tile_index: torch.Tensor
    local_row: torch.Tensor
    local_col: torch.Tensor
    global_index: torch.Tensor
    thumbnail_tile_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "tile_index", _as_cpu_long(self.tile_index, "tile_index"))
        object.__setattr__(self, "local_row", _as_cpu_long(self.local_row, "local_row"))
        object.__setattr__(self, "local_col", _as_cpu_long(self.local_col, "local_col"))
        object.__setattr__(self, "global_index", _as_cpu_long(self.global_index, "global_index"))
        object.__setattr__(
            self,
            "thumbnail_tile_indices",
            tuple(int(v) for v in self.thumbnail_tile_indices),
        )
        self.validate()

    @property
    def num_tokens(self) -> int:
        return int(self.global_index.numel())

    @property
    def tile_ids(self) -> tuple[int, ...]:
        if self.num_tokens == 0:
            return ()
        order = torch.argsort(self.global_index)
        seen: set[int] = set()
        result: list[int] = []
        for value in self.tile_index[order].tolist():
            tile_id = int(value)
            if tile_id not in seen:
                seen.add(tile_id)
                result.append(tile_id)
        return tuple(result)

    @property
    def num_tiles(self) -> int:
        return len(self.tile_ids)

    def validate(self, expected_tokens: Optional[int] = None) -> None:
        sizes = {
            self.tile_index.numel(),
            self.local_row.numel(),
            self.local_col.numel(),
            self.global_index.numel(),
        }
        if len(sizes) != 1:
            raise CompressionInputError(
                "tile_index, local_row, local_col, and global_index must have equal length"
            )
        n = self.num_tokens
        if n < 1:
            raise CompressionInputError("tile metadata must describe at least one token")
        if expected_tokens is not None and n != expected_tokens:
            raise CompressionInputError(
                f"tile metadata describes {n} tokens, expected {expected_tokens}"
            )
        if torch.any(self.tile_index < 0):
            raise CompressionInputError("tile_index values must be non-negative")
        if torch.any(self.local_row < 0) or torch.any(self.local_col < 0):
            raise CompressionInputError("local coordinates must be non-negative")
        expected = torch.arange(n, dtype=torch.long)
        if not torch.equal(torch.sort(self.global_index).values, expected):
            raise CompressionInputError("global_index must be a permutation of range(N_full)")
        coordinates = set()
        for item in zip(
            self.tile_index.tolist(), self.local_row.tolist(), self.local_col.tolist()
        ):
            if item in coordinates:
                raise CompressionInputError(f"duplicate tile-local coordinate: {item}")
            coordinates.add(item)
        unknown_thumbnails = set(self.thumbnail_tile_indices).difference(self.tile_ids)
        if unknown_thumbnails:
            raise CompressionInputError(
                f"thumbnail tile ids are absent from metadata: {sorted(unknown_thumbnails)}"
            )

    def global_indices_for_tile(self, tile_id: int, device: torch.device) -> torch.Tensor:
        mask = self.tile_index == int(tile_id)
        indices = self.global_index[mask]
        return torch.sort(indices).values.to(device=device)

    def token_coordinates(self, global_indices: torch.Tensor) -> torch.Tensor:
        """Return ``[tile, row, col, global]`` rows for global token indices."""

        inverse = torch.empty(self.num_tokens, dtype=torch.long)
        inverse[self.global_index] = torch.arange(self.num_tokens, dtype=torch.long)
        positions = inverse[global_indices.detach().to(device="cpu", dtype=torch.long)]
        return torch.stack(
            (
                self.tile_index[positions],
                self.local_row[positions],
                self.local_col[positions],
                self.global_index[positions],
            ),
            dim=-1,
        )

    @classmethod
    def single_grid(cls, rows: int, cols: int, tile_id: int = 0) -> "TileMetadata":
        if rows < 1 or cols < 1:
            raise CompressionInputError("grid dimensions must be positive")
        n = rows * cols
        return cls(
            tile_index=torch.full((n,), int(tile_id), dtype=torch.long),
            local_row=torch.arange(rows, dtype=torch.long).repeat_interleave(cols),
            local_col=torch.arange(cols, dtype=torch.long).repeat(rows),
            global_index=torch.arange(n, dtype=torch.long),
        )

    @classmethod
    def from_tile_shapes(
        cls,
        tile_shapes: Sequence[tuple[int, int]],
        *,
        tile_ids: Optional[Sequence[int]] = None,
        thumbnail_tile_indices: Sequence[int] = (),
    ) -> "TileMetadata":
        if not tile_shapes:
            raise CompressionInputError("tile_shapes must not be empty")
        ids = list(range(len(tile_shapes))) if tile_ids is None else list(tile_ids)
        if len(ids) != len(tile_shapes) or len(set(ids)) != len(ids):
            raise CompressionInputError("tile_ids must be unique and match tile_shapes")
        tile_values: list[int] = []
        row_values: list[int] = []
        col_values: list[int] = []
        for tile_id, (rows, cols) in zip(ids, tile_shapes):
            if rows < 1 or cols < 1:
                raise CompressionInputError("all tile grid dimensions must be positive")
            for row in range(rows):
                for col in range(cols):
                    tile_values.append(int(tile_id))
                    row_values.append(row)
                    col_values.append(col)
        n = len(tile_values)
        return cls(
            tile_index=torch.tensor(tile_values),
            local_row=torch.tensor(row_values),
            local_col=torch.tensor(col_values),
            global_index=torch.arange(n),
            thumbnail_tile_indices=tuple(thumbnail_tile_indices),
        )

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "TileMetadata":
        aliases = {
            "tile_index": ("tile_index", "tile_indices", "tile_id"),
            "local_row": ("local_row", "local_rows", "row"),
            "local_col": ("local_col", "local_cols", "col"),
            "global_index": ("global_index", "global_indices", "token_index"),
        }
        resolved: dict[str, Any] = {}
        for canonical, names in aliases.items():
            for name in names:
                if name in values:
                    resolved[canonical] = values[name]
                    break
            else:
                raise CompressionInputError(f"tile metadata is missing {canonical}")
        thumbnails = values.get(
            "thumbnail_tile_indices", values.get("thumbnail_tile_ids", ())
        )
        return cls(**resolved, thumbnail_tile_indices=tuple(thumbnails))


def coerce_tile_metadata(metadata: Any, n_full: int) -> TileMetadata:
    """Accept this package's metadata, mappings, or equivalent dataclasses."""

    if metadata is None:
        side = int(n_full**0.5)
        rows, cols = (side, side) if side * side == n_full else (1, n_full)
        return TileMetadata.single_grid(rows, cols)
    if isinstance(metadata, TileMetadata):
        metadata.validate(expected_tokens=n_full)
        return metadata
    if isinstance(metadata, Mapping):
        result = TileMetadata.from_mapping(metadata)
        result.validate(expected_tokens=n_full)
        return result
    if hasattr(metadata, "token_metadata_mapping"):
        return coerce_tile_metadata(metadata.token_metadata_mapping(), n_full)
    fields = {}
    for name in ("tile_index", "local_row", "local_col", "global_index"):
        if not hasattr(metadata, name):
            raise CompressionInputError(
                "metadata must be TileMetadata, a mapping, or expose tile_index, "
                "local_row, local_col, and global_index"
            )
        fields[name] = getattr(metadata, name)
    fields["thumbnail_tile_indices"] = getattr(
        metadata,
        "thumbnail_tile_indices",
        getattr(metadata, "thumbnail_tile_ids", ()),
    )
    result = TileMetadata(**fields)
    result.validate(expected_tokens=n_full)
    return result


@dataclass(frozen=True)
class TileCompressionContext:
    """Signals aligned to one source-equivalent tile."""

    cls_attention_heads: Optional[torch.Tensor] = None  # [B, H, N_tile]
    cls_query: Optional[torch.Tensor] = None  # [B, C]
    cls_key: Optional[torch.Tensor] = None  # [B, C]
    patch_key: Optional[torch.Tensor] = None  # [B, N_tile, C]
    cls_value: Optional[torch.Tensor] = None  # [B, H, D]
    patch_value: Optional[torch.Tensor] = None  # [B, H, N_tile, D]


@dataclass(frozen=True)
class CompressionContext:
    """Attention/QKV signals aligned with a dynamic multi-tile token stream."""

    cls_attention_heads: Optional[torch.Tensor] = None  # [B, H, N_full]
    cls_query: Optional[torch.Tensor] = None  # [B, T, C]
    cls_key: Optional[torch.Tensor] = None  # [B, T, C]
    patch_key: Optional[torch.Tensor] = None  # [B, N_full, C]
    cls_value: Optional[torch.Tensor] = None  # [B, H, T, D]
    patch_value: Optional[torch.Tensor] = None  # [B, H, N_full, D]

    @classmethod
    def from_source_tensors(
        cls,
        *,
        attentions: Optional[torch.Tensor] = None,
        query: Optional[torch.Tensor] = None,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        num_heads: Optional[int] = None,
    ) -> "CompressionContext":
        """Build a one-tile context from tensors used by ``compression_zoo``."""

        attention_heads = None if attentions is None else attentions[:, :, 0, 1:]
        cls_query = None if query is None else query[:, 0].unsqueeze(1)
        cls_key = None if key is None else key[:, 0].unsqueeze(1)
        patch_key = None if key is None else key[:, 1:]
        cls_value = None
        patch_value = None
        if value is not None:
            if num_heads is None or num_heads < 1:
                raise CompressionInputError("num_heads is required with a value tensor")
            batch, sequence, width = value.shape
            if width % num_heads:
                raise CompressionInputError(
                    f"value width {width} is not divisible by num_heads={num_heads}"
                )
            head_dim = width // num_heads
            states = value.view(batch, sequence, num_heads, head_dim).transpose(1, 2)
            cls_value = states[:, :, 0].unsqueeze(2)
            patch_value = states[:, :, 1:]
        return cls(
            cls_attention_heads=attention_heads,
            cls_query=cls_query,
            cls_key=cls_key,
            patch_key=patch_key,
            cls_value=cls_value,
            patch_value=patch_value,
        )

    def for_tile(
        self,
        tile_slot: int,
        global_indices: torch.Tensor,
    ) -> TileCompressionContext:
        def select_patch(tensor: Optional[torch.Tensor], dim: int) -> Optional[torch.Tensor]:
            if tensor is None:
                return None
            return torch.index_select(tensor, dim, global_indices.to(tensor.device))

        def select_cls(tensor: Optional[torch.Tensor], dim: int) -> Optional[torch.Tensor]:
            if tensor is None:
                return None
            index = torch.tensor([tile_slot], device=tensor.device)
            return torch.index_select(tensor, dim, index).squeeze(dim)

        return TileCompressionContext(
            cls_attention_heads=select_patch(self.cls_attention_heads, 2),
            cls_query=select_cls(self.cls_query, 1),
            cls_key=select_cls(self.cls_key, 1),
            patch_key=select_patch(self.patch_key, 1),
            cls_value=select_cls(self.cls_value, 2),
            patch_value=select_patch(self.patch_value, 2),
        )


@dataclass
class CompressionResult:
    """Observable result and parity diagnostics for a compressor invocation."""

    method: str
    nominal_k: int
    selected_indices: torch.Tensor
    compressed_tokens: torch.Tensor
    reconstructed_tokens: torch.Tensor
    reconstruction_indices: torch.Tensor
    scores: Optional[torch.Tensor] = None
    cluster_assignment: Optional[torch.Tensor] = None
    tile_budgets: tuple[int, ...] = ()
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def actual_k(self) -> int:
        return int(self.compressed_tokens.shape[1])
