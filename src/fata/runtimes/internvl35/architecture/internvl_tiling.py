"""Deterministic metadata for InternVL dynamic image tiling.

The official processor chooses a width-by-height tile canvas, emits its crops
in row-major order, and appends a thumbnail when the canvas has more than one
tile.  These helpers reproduce that metadata without resizing image pixels.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache


@lru_cache(maxsize=16)
def supported_tile_grids(
    min_patches: int = 1, max_patches: int = 12
) -> tuple[tuple[int, int], ...]:
    """Return official ``(num_columns, num_rows)`` candidates in source order."""

    if min_patches < 1 or max_patches < min_patches:
        raise ValueError(
            f"invalid tile bounds min_patches={min_patches}, max_patches={max_patches}"
        )
    candidates = [
        (columns, rows)
        for columns in range(1, max_patches + 1)
        for rows in range(1, max_patches + 1)
        if min_patches <= columns * rows <= max_patches
    ]
    # Python's sort is stable.  This preserves the width-major order used by
    # the processor among candidates with the same tile count.
    return tuple(sorted(candidates, key=lambda grid: grid[0] * grid[1]))


def select_tile_grid(
    image_height: int,
    image_width: int,
    *,
    tile_height: int = 448,
    tile_width: int = 448,
    min_patches: int = 1,
    max_patches: int = 12,
) -> tuple[int, int]:
    """Reproduce ``get_optimal_tiled_canvas`` from Transformers 4.55.0."""

    if min(image_height, image_width, tile_height, tile_width) <= 0:
        raise ValueError("image and tile dimensions must be positive")
    image_aspect_ratio = image_width / image_height
    image_area = image_width * image_height
    best_ratio_difference = float("inf")
    best_grid = (1, 1)
    for grid in supported_tile_grids(min_patches, max_patches):
        ratio_difference = abs(image_aspect_ratio - grid[0] / grid[1])
        if ratio_difference < best_ratio_difference:
            best_ratio_difference = ratio_difference
            best_grid = grid
        elif ratio_difference == best_ratio_difference:
            tiled_area = tile_height * tile_width * grid[0] * grid[1]
            if image_area > 0.5 * tiled_area:
                best_grid = grid
    return best_grid


@dataclass(frozen=True)
class TileLocation:
    """One emitted tile; thumbnail coordinates are ``None``."""

    tile_index: int
    grid_row: int | None
    grid_col: int | None
    is_thumbnail: bool


@dataclass(frozen=True)
class TileTokenLocation:
    """Duck-typed token metadata shared with compression adapters."""

    tile_index: int
    local_row: int
    local_col: int
    global_index: int
    is_thumbnail: bool = False


@dataclass(frozen=True)
class DynamicTileLayout:
    """Dynamic tiling and post-shuffle token counts for one input image."""

    image_height: int
    image_width: int
    num_columns: int
    num_rows: int
    use_thumbnail: bool = True
    tokens_per_tile: int = 256

    @classmethod
    def from_image_size(
        cls,
        image_height: int,
        image_width: int,
        *,
        tile_height: int = 448,
        tile_width: int = 448,
        min_patches: int = 1,
        max_patches: int = 12,
        use_thumbnail: bool = True,
        tokens_per_tile: int = 256,
    ) -> "DynamicTileLayout":
        columns, rows = select_tile_grid(
            image_height,
            image_width,
            tile_height=tile_height,
            tile_width=tile_width,
            min_patches=min_patches,
            max_patches=max_patches,
        )
        return cls(
            image_height=image_height,
            image_width=image_width,
            num_columns=columns,
            num_rows=rows,
            use_thumbnail=use_thumbnail,
            tokens_per_tile=tokens_per_tile,
        )

    @property
    def grid_tile_count(self) -> int:
        return self.num_columns * self.num_rows

    @property
    def has_thumbnail(self) -> bool:
        return self.use_thumbnail and self.grid_tile_count != 1

    @property
    def tile_count(self) -> int:
        return self.grid_tile_count + int(self.has_thumbnail)

    @property
    def thumbnail_tile_index(self) -> int | None:
        return self.tile_count - 1 if self.has_thumbnail else None

    @property
    def thumbnail_tile_indices(self) -> frozenset[int]:
        index = self.thumbnail_tile_index
        return frozenset() if index is None else frozenset((index,))

    @property
    def visual_token_count(self) -> int:
        return self.tile_count * self.tokens_per_tile

    def tiles(self) -> tuple[TileLocation, ...]:
        locations = [
            TileLocation(
                tile_index=index,
                grid_row=index // self.num_columns,
                grid_col=index % self.num_columns,
                is_thumbnail=False,
            )
            for index in range(self.grid_tile_count)
        ]
        if self.has_thumbnail:
            locations.append(
                TileLocation(
                    tile_index=self.grid_tile_count,
                    grid_row=None,
                    grid_col=None,
                    is_thumbnail=True,
                )
            )
        return tuple(locations)

    def token_locations(
        self, *, post_height: int = 16, post_width: int = 16
    ) -> tuple[TileTokenLocation, ...]:
        """Return tile-major, then row-major post-shuffle token metadata."""

        if post_height * post_width != self.tokens_per_tile:
            raise ValueError(
                f"post grid {(post_height, post_width)} contains "
                f"{post_height * post_width} tokens, expected {self.tokens_per_tile}"
            )
        result = []
        for tile in self.tiles():
            for local_row in range(post_height):
                for local_col in range(post_width):
                    result.append(
                        TileTokenLocation(
                            tile_index=tile.tile_index,
                            local_row=local_row,
                            local_col=local_col,
                            global_index=len(result),
                            is_thumbnail=tile.is_thumbnail,
                        )
                    )
        return tuple(result)

    def token_metadata_mapping(
        self, *, post_height: int = 16, post_width: int = 16
    ) -> dict[str, tuple[int, ...]]:
        """Return aggregate fields accepted by compression ``TileMetadata``."""

        locations = self.token_locations(
            post_height=post_height, post_width=post_width
        )
        return {
            "tile_index": tuple(item.tile_index for item in locations),
            "local_row": tuple(item.local_row for item in locations),
            "local_col": tuple(item.local_col for item in locations),
            "global_index": tuple(item.global_index for item in locations),
            "thumbnail_tile_indices": tuple(self.thumbnail_tile_indices),
        }


__all__ = [
    "DynamicTileLayout",
    "TileLocation",
    "TileTokenLocation",
    "select_tile_grid",
    "supported_tile_grids",
]
