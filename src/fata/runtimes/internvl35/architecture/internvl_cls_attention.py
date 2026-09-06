"""Exact, differentiable InternVL vision CLS-attention extraction.

The HF InternVL vision attention implementation can use SDPA, which does not
promise to return attention probabilities.  InternVL nevertheless exposes the
same learned query/key projections to every backend.  Computing the single
CLS row from those projections is exact, substantially cheaper than forming
the full 1025 x 1025 matrix, and remains differentiable with respect to image
pixels.

The softmax is always taken over *CLS plus every patch*.  The CLS column is
removed only afterwards; normalizing over patches alone would not reproduce
the model's eager attention.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class AttentionValidation:
    """Numerical agreement between manual QK attention and eager attention."""

    cosine_min: float
    cosine_mean: float
    topk_overlap_min: float
    topk_overlap_mean: float
    max_abs_error: float
    finite: bool
    cosine_threshold: float
    topk_overlap_threshold: float
    topk: int
    passed: bool

    def to_dict(self) -> dict[str, float | int | bool]:
        return asdict(self)

    def assert_passed(self) -> None:
        if not self.passed:
            raise AssertionError(
                "exact CLS-attention validation failed: "
                f"finite={self.finite}, cosine_min={self.cosine_min:.8f} "
                f"(required {self.cosine_threshold}), "
                f"topk_overlap_min={self.topk_overlap_min:.8f} "
                f"(required {self.topk_overlap_threshold}), "
                f"max_abs_error={self.max_abs_error:.8g}"
            )


def _module_int(module: nn.Module, names: Sequence[str]) -> int | None:
    for name in names:
        value = getattr(module, name, None)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    config = getattr(module, "config", None)
    if config is not None:
        for name in names:
            value = getattr(config, name, None)
            if isinstance(value, int) and not isinstance(value, bool):
                return value
    return None


def _normalization_width(norm: nn.Module | None) -> int | None:
    if norm is None or isinstance(norm, nn.Identity):
        return None
    normalized_shape = getattr(norm, "normalized_shape", None)
    if normalized_shape is not None:
        if isinstance(normalized_shape, int):
            return normalized_shape
        if len(normalized_shape) == 1:
            return int(normalized_shape[0])
    weight = getattr(norm, "weight", None)
    if isinstance(weight, Tensor) and weight.ndim == 1:
        return int(weight.numel())
    return None


def _apply_qk_norm(
    projected: Tensor,
    per_head: Tensor,
    norm: nn.Module | None,
    *,
    projection_name: str,
) -> Tensor:
    """Apply optional Q/K normalization at its declared feature width."""

    if norm is None or isinstance(norm, nn.Identity):
        return per_head
    width = _normalization_width(norm)
    if width == projected.shape[-1]:
        batch, tokens = projected.shape[:2]
        normalized = norm(projected)
        return normalized.reshape(batch, tokens, per_head.shape[1], per_head.shape[-1]).transpose(1, 2)
    if width in (None, per_head.shape[-1]):
        return norm(per_head)
    raise ValueError(
        f"{projection_name} normalization width {width} matches neither "
        f"projection width {projected.shape[-1]} nor head width {per_head.shape[-1]}"
    )


def project_internvl_query_key(
    attention_module: nn.Module, hidden_states: Tensor
) -> tuple[Tensor, Tensor]:
    """Run the real attention Q/K projections and split them into heads.

    Returns tensors shaped ``[batch, heads, tokens, head_dim]``.  Official HF
    InternVL uses separate ``q_proj`` and ``k_proj`` layers; fused ``qkv`` is
    accepted as a compatibility aid for older custom-code checkpoints.
    """

    if hidden_states.ndim != 3:
        raise ValueError(
            "hidden_states must have shape [batch, tokens, hidden], got "
            f"{tuple(hidden_states.shape)}"
        )
    q_proj = getattr(attention_module, "q_proj", None)
    k_proj = getattr(attention_module, "k_proj", None)
    if callable(q_proj) and callable(k_proj):
        query_projected = q_proj(hidden_states)
        key_projected = k_proj(hidden_states)
    else:
        qkv = getattr(attention_module, "qkv", None)
        if not callable(qkv):
            raise TypeError(
                "attention module exposes neither separate q_proj/k_proj nor a fused qkv projection"
            )
        fused = qkv(hidden_states)
        if fused.shape[-1] % 3:
            raise ValueError(
                f"fused qkv width must be divisible by three, got {fused.shape[-1]}"
            )
        query_projected, key_projected, _ = fused.chunk(3, dim=-1)

    num_heads = _module_int(
        attention_module, ("num_heads", "num_attention_heads")
    )
    head_dim = _module_int(attention_module, ("head_dim",))
    if num_heads is None and head_dim is None:
        raise AttributeError("attention module must expose num_heads or head_dim")
    if num_heads is None:
        assert head_dim is not None
        if query_projected.shape[-1] % head_dim:
            raise ValueError("query projection width is not divisible by head_dim")
        num_heads = query_projected.shape[-1] // head_dim
    if head_dim is None:
        if query_projected.shape[-1] % num_heads:
            raise ValueError("query projection width is not divisible by num_heads")
        head_dim = query_projected.shape[-1] // num_heads
    if query_projected.shape[-1] != num_heads * head_dim:
        raise ValueError(
            f"query projection width {query_projected.shape[-1]} != "
            f"num_heads*head_dim ({num_heads}*{head_dim})"
        )
    if key_projected.shape[-1] != num_heads * head_dim:
        raise ValueError(
            "InternVL vision Q and K must have the same number of heads; "
            f"key width is {key_projected.shape[-1]}"
        )

    batch_size, token_count = hidden_states.shape[:2]
    query = query_projected.reshape(
        batch_size, token_count, num_heads, head_dim
    ).transpose(1, 2)
    key = key_projected.reshape(
        batch_size, token_count, num_heads, head_dim
    ).transpose(1, 2)
    query = _apply_qk_norm(
        query_projected,
        query,
        getattr(attention_module, "q_norm", None),
        projection_name="query",
    )
    key = _apply_qk_norm(
        key_projected,
        key,
        getattr(attention_module, "k_norm", None),
        projection_name="key",
    )
    return query, key


def _cls_mask_row(attention_mask: Tensor, cls_index: int, logits: Tensor) -> Tensor:
    """Select/broadcast the CLS query row of a standard attention mask."""

    mask = attention_mask.to(device=logits.device)
    if mask.ndim == 4:
        if mask.shape[-2] not in (1, logits.shape[-1]):
            raise ValueError(f"invalid 4-D attention mask shape {tuple(mask.shape)}")
        mask = mask[..., 0 if mask.shape[-2] == 1 else cls_index, :]
    elif mask.ndim == 3:
        if mask.shape[-2] not in (1, logits.shape[-1]):
            raise ValueError(f"invalid 3-D attention mask shape {tuple(mask.shape)}")
        mask = mask[..., 0 if mask.shape[-2] == 1 else cls_index, :]
    elif mask.ndim == 2:
        mask = mask[:, None, :]
    elif mask.ndim != 1:
        raise ValueError(f"unsupported attention mask shape {tuple(mask.shape)}")
    return mask


def compute_cls_attention_row(
    attention_module: nn.Module,
    hidden_states: Tensor,
    *,
    cls_index: int = 0,
    attention_mask: Tensor | None = None,
) -> Tensor:
    """Compute the exact pre-dropout CLS row, including the CLS key column.

    Output shape is ``[batch, heads, tokens]``.  Softmax stays in the Q dtype,
    matching InternVL 4.55.0 eager attention (which explicitly does not
    upcast its attention weights).
    """

    query, key = project_internvl_query_key(attention_module, hidden_states)
    token_count = query.shape[-2]
    if not -token_count <= cls_index < token_count:
        raise IndexError(f"cls_index {cls_index} is outside {token_count} tokens")
    cls_index %= token_count
    scale = getattr(attention_module, "scaling", None)
    if scale is None:
        scale = getattr(attention_module, "scale", None)
    if scale is None:
        scale = 1.0 / sqrt(query.shape[-1])

    logits = torch.matmul(
        query[:, :, cls_index : cls_index + 1, :], key.transpose(-2, -1)
    ).squeeze(-2)
    logits = logits * float(scale)
    if attention_mask is not None:
        mask = _cls_mask_row(attention_mask, cls_index, logits)
        if mask.dtype == torch.bool:
            logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        else:
            logits = logits + mask.to(dtype=logits.dtype)
    return F.softmax(logits, dim=-1)


def compute_cls_to_patch_attention(
    attention_module: nn.Module,
    hidden_states: Tensor,
    *,
    cls_index: int = 0,
    attention_mask: Tensor | None = None,
) -> Tensor:
    """Compute exact ``CLS -> patch`` probabilities as ``[B, H, N-1]``."""

    full_row = compute_cls_attention_row(
        attention_module,
        hidden_states,
        cls_index=cls_index,
        attention_mask=attention_mask,
    )
    cls_index %= full_row.shape[-1]
    if cls_index == 0:
        return full_row[..., 1:]
    return torch.cat((full_row[..., :cls_index], full_row[..., cls_index + 1 :]), dim=-1)


# A concise name for call sites in attack code.
exact_cls_attention = compute_cls_to_patch_attention


def eager_cls_to_patch_attention(eager_attention: Tensor, *, cls_index: int = 0) -> Tensor:
    """Extract the comparable CLS-to-patch row from eager full attention."""

    if eager_attention.ndim != 4 or eager_attention.shape[-2] != eager_attention.shape[-1]:
        raise ValueError(
            "eager_attention must have shape [batch, heads, tokens, tokens], "
            f"got {tuple(eager_attention.shape)}"
        )
    token_count = eager_attention.shape[-1]
    if not -token_count <= cls_index < token_count:
        raise IndexError(f"cls_index {cls_index} is outside {token_count} tokens")
    cls_index %= token_count
    row = eager_attention[..., cls_index, :]
    if cls_index == 0:
        return row[..., 1:]
    return torch.cat((row[..., :cls_index], row[..., cls_index + 1 :]), dim=-1)


def compare_cls_attention(
    manual: Tensor,
    eager: Tensor,
    *,
    topk: int | None = None,
    cosine_threshold: float = 0.9999,
    topk_overlap_threshold: float = 0.999,
) -> AttentionValidation:
    """Compare like-shaped CLS rows and apply the required validation gates."""

    if manual.shape != eager.shape:
        raise ValueError(
            f"manual/eager shape mismatch: {tuple(manual.shape)} vs {tuple(eager.shape)}"
        )
    if manual.ndim < 1 or manual.shape[-1] < 1:
        raise ValueError("attention rows must have a non-empty key dimension")
    key_count = manual.shape[-1]
    if topk is None:
        topk = min(64, key_count)
    if not 1 <= topk <= key_count:
        raise ValueError(f"topk must be in [1, {key_count}], got {topk}")

    manual_f = manual.detach().to(torch.float32)
    eager_f = eager.detach().to(device=manual.device, dtype=torch.float32)
    finite = bool(torch.isfinite(manual_f).all() and torch.isfinite(eager_f).all())
    if finite:
        cosine = F.cosine_similarity(manual_f, eager_f, dim=-1, eps=1e-12)
        manual_top = manual_f.topk(topk, dim=-1).indices
        eager_top = eager_f.topk(topk, dim=-1).indices
        overlap = (manual_top[..., :, None] == eager_top[..., None, :]).any(dim=-1)
        overlap = overlap.to(torch.float32).mean(dim=-1)
        cosine_min = float(cosine.amin().item())
        cosine_mean = float(cosine.mean().item())
        overlap_min = float(overlap.amin().item())
        overlap_mean = float(overlap.mean().item())
        max_abs_error = float((manual_f - eager_f).abs().amax().item())
    else:
        cosine_min = cosine_mean = float("nan")
        overlap_min = overlap_mean = 0.0
        max_abs_error = float("nan")
    passed = bool(
        finite
        and cosine_min >= cosine_threshold
        and overlap_min >= topk_overlap_threshold
    )
    return AttentionValidation(
        cosine_min=cosine_min,
        cosine_mean=cosine_mean,
        topk_overlap_min=overlap_min,
        topk_overlap_mean=overlap_mean,
        max_abs_error=max_abs_error,
        finite=finite,
        cosine_threshold=cosine_threshold,
        topk_overlap_threshold=topk_overlap_threshold,
        topk=topk,
        passed=passed,
    )


def validate_cls_attention_against_eager(
    attention_module: nn.Module,
    hidden_states: Tensor,
    eager_attention: Tensor,
    *,
    cls_index: int = 0,
    attention_mask: Tensor | None = None,
    topk: int | None = None,
    cosine_threshold: float = 0.9999,
    topk_overlap_threshold: float = 0.999,
) -> AttentionValidation:
    """Compute manual CLS attention, compare with eager, and return metrics."""

    manual = compute_cls_to_patch_attention(
        attention_module,
        hidden_states,
        cls_index=cls_index,
        attention_mask=attention_mask,
    )
    eager = eager_cls_to_patch_attention(eager_attention, cls_index=cls_index)
    return compare_cls_attention(
        manual,
        eager,
        topk=topk,
        cosine_threshold=cosine_threshold,
        topk_overlap_threshold=topk_overlap_threshold,
    )


def _vision_encoder_layers(model: nn.Module) -> Sequence[nn.Module]:
    candidates: list[Any] = [model]
    nested_model = getattr(model, "model", None)
    if nested_model is not None:
        candidates.append(nested_model)
    for root in candidates:
        towers = [getattr(root, "vision_tower", None)]
        if getattr(root, "encoder", None) is not None:
            towers.append(root)  # A bare InternVLVisionModel is also accepted.
        for tower in towers:
            if tower is None:
                continue
            encoder = getattr(tower, "encoder", None)
            if encoder is None:
                continue
            layers = getattr(encoder, "layer", None)
            if layers is None:
                layers = getattr(encoder, "layers", None)
            if layers is not None:
                return layers
    raise AttributeError(
        "could not locate InternVL vision layers at vision_tower.encoder.layer"
    )


def resolve_vision_attention_modules(
    model: nn.Module, layer_indices: Iterable[int] = (-1,)
) -> list[tuple[int, nn.Module]]:
    """Resolve possibly-negative layer indices to InternVL attention modules."""

    layers = _vision_encoder_layers(model)
    resolved: list[tuple[int, nn.Module]] = []
    seen: set[int] = set()
    for requested in layer_indices:
        index = requested + len(layers) if requested < 0 else requested
        if not 0 <= index < len(layers):
            raise IndexError(
                f"vision layer index {requested} resolves outside {len(layers)} layers"
            )
        if index in seen:
            continue
        attention = getattr(layers[index], "attention", None)
        if attention is None:
            raise AttributeError(f"vision layer {index} has no attention module")
        seen.add(index)
        resolved.append((index, attention))
    if not resolved:
        raise ValueError("at least one vision layer index is required")
    return resolved


class ExactCLSAttentionCapture:
    """Context manager that captures exact CLS rows during a vision forward.

    The pre-hooks see precisely the normalized hidden states consumed by each
    selected attention layer.  Captured tensors retain autograd history unless
    ``detach=True`` is requested.
    """

    def __init__(
        self,
        model: nn.Module,
        layer_indices: Iterable[int] = (-1,),
        *,
        cls_index: int = 0,
        detach: bool = False,
    ) -> None:
        self.modules = resolve_vision_attention_modules(model, layer_indices)
        self.cls_index = cls_index
        self.detach = detach
        self.rows: dict[int, Tensor] = {}
        self._handles: list[Any] = []

    def _hook(self, layer_index: int):
        def capture(module: nn.Module, args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> None:
            hidden_states = kwargs.get("hidden_states")
            if hidden_states is None:
                if not args:
                    raise RuntimeError(
                        f"attention layer {layer_index} received no hidden_states"
                    )
                hidden_states = args[0]
            attention_mask = kwargs.get("attention_mask")
            row = compute_cls_to_patch_attention(
                module,
                hidden_states,
                cls_index=self.cls_index,
                attention_mask=attention_mask,
            )
            self.rows[layer_index] = row.detach() if self.detach else row

        return capture

    def __enter__(self) -> "ExactCLSAttentionCapture":
        if self._handles:
            raise RuntimeError("capture context is already active")
        self.rows.clear()
        for layer_index, module in self.modules:
            handle = module.register_forward_pre_hook(
                self._hook(layer_index), with_kwargs=True
            )
            self._handles.append(handle)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    @property
    def layer_indices(self) -> tuple[int, ...]:
        return tuple(index for index, _ in self.modules)

    def stacked(self) -> Tensor:
        """Return rows as ``[layers, batch, heads, raw_patches]``."""

        missing = [index for index in self.layer_indices if index not in self.rows]
        if missing:
            raise RuntimeError(f"selected attention layers were not executed: {missing}")
        return torch.stack([self.rows[index] for index in self.layer_indices], dim=0)


__all__ = [
    "AttentionValidation",
    "ExactCLSAttentionCapture",
    "compare_cls_attention",
    "compute_cls_attention_row",
    "compute_cls_to_patch_attention",
    "eager_cls_to_patch_attention",
    "exact_cls_attention",
    "project_internvl_query_key",
    "resolve_vision_attention_modules",
    "validate_cls_attention_against_eager",
]
