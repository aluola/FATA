"""Lossless 8-bit serialization for pixel-space L-infinity attacks."""

from __future__ import annotations

import math
from typing import Any


IMAGE_SERIALIZATION_CONTRACT = "png_u8_round_project_linf_v1"


def quantize_linf_image(clean_01: Any, adversarial_01: Any, epsilon: float):
    """Return an RGB PIL image with a post-decode discrete L-inf guarantee.

    Attack optimization is continuous, while PNG pixels are 8-bit.  Generic
    ``ToPILImage`` truncation can add a third intensity level to an eps=2/255
    perturbation.  This routine rounds both tensors to their explicit 8-bit
    reference, projects every adversarial channel to the allowed integer box,
    and verifies the decoded representation before returning it.
    """

    import numpy as np
    import torch
    from PIL import Image

    if not math.isfinite(float(epsilon)) or epsilon < 0:
        raise ValueError("epsilon must be finite and non-negative")
    clean = torch.as_tensor(clean_01).detach().float().cpu()
    adversarial = torch.as_tensor(adversarial_01).detach().float().cpu()
    if clean.shape != adversarial.shape:
        raise ValueError("clean and adversarial tensors must have identical shapes")
    if clean.ndim == 4 and clean.shape[0] == 1:
        clean = clean[0]
        adversarial = adversarial[0]
    if clean.ndim != 3 or clean.shape[0] != 3:
        raise ValueError("expected one RGB tensor with shape [1,3,H,W] or [3,H,W]")
    if not torch.isfinite(clean).all() or not torch.isfinite(adversarial).all():
        raise ValueError("image tensor contains NaN or infinity")

    max_levels = int(math.floor(float(epsilon) * 255.0 + 1e-6))
    clean_u8 = torch.round(clean.clamp(0, 1) * 255.0).to(torch.int16)
    adv_u8 = torch.round(adversarial.clamp(0, 1) * 255.0).to(torch.int16)
    lower = (clean_u8 - max_levels).clamp(0, 255)
    upper = (clean_u8 + max_levels).clamp(0, 255)
    adv_u8 = torch.maximum(lower, torch.minimum(upper, adv_u8))
    observed_levels = int((adv_u8 - clean_u8).abs().max().item())
    if observed_levels > max_levels:
        raise RuntimeError(
            f"post-quantization L_inf violation: {observed_levels}/255 > {max_levels}/255"
        )
    array = adv_u8.to(torch.uint8).permute(1, 2, 0).contiguous().numpy()
    image = Image.fromarray(np.asarray(array), mode="RGB")
    decoded = torch.from_numpy(np.asarray(image, dtype=np.uint8).copy()).permute(2, 0, 1)
    decoded_levels = int((decoded.to(torch.int16) - clean_u8).abs().max().item())
    if decoded_levels > max_levels:
        raise RuntimeError("serialized PNG/PIL representation violates the L_inf budget")
    return image
