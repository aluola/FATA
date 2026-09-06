"""The final four InternVL pre-projector compressor adapters."""

from .flowcut import compress as compress_flowcut
from .prumerge import compress as compress_prumerge
from .visionzip import compress as compress_visionzip
from .vispruner import compress as compress_vispruner

__all__ = ["compress_visionzip", "compress_vispruner", "compress_prumerge", "compress_flowcut"]
