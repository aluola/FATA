"""Runtime-neutral assertions for compressor adapter outputs."""

from __future__ import annotations


def validate_compressor_output(original_tokens, compressed_tokens, requested_k: int) -> int:
    if getattr(original_tokens, "ndim", None) != 3 or getattr(compressed_tokens, "ndim", None) != 3:
        raise ValueError("token tensors must have shape [batch, tokens, hidden]")
    if original_tokens.shape[0] != compressed_tokens.shape[0] or original_tokens.shape[2] != compressed_tokens.shape[2]:
        raise ValueError("compressor changed batch or hidden dimension")
    actual = int(compressed_tokens.shape[1])
    if actual != int(requested_k):
        raise ValueError(f"compressor returned {actual} tokens; requested {requested_k}")
    if actual > int(original_tokens.shape[1]):
        raise ValueError("compressor increased token count")
    return actual
