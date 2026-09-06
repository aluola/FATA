"""Base class for dynamic compressors."""

from abc import ABC, abstractmethod
import torch


class BaseDynamicCompressor(ABC):
    """Base class for compression algorithms with dynamic K = round(N_full * ratio)."""

    def __init__(self, retention_ratio: float):
        self.retention_ratio = retention_ratio
        self.K = None
        self.N_full = None

    def set_token_count(self, N_full: int):
        """Compute dynamic K from N_full and retention ratio."""
        self.N_full = N_full
        self.K = max(1, min(N_full, int(round(N_full * self.retention_ratio))))

    @abstractmethod
    def compress(self, visual_embeddings, importance=None):
        """
        Compress visual embeddings from [N_full, D] to [K, D].

        Args:
            visual_embeddings: [N_full, D]
            importance: optional [N_full] importance scores

        Returns:
            compressed: [K, D]
            selected_indices: [K] indices of selected tokens
            metadata: dict with compression info
        """
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(r={self.retention_ratio:.4f}, K={self.K})"
