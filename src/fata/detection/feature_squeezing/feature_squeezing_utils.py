# fata_detection_defense/feature_squeezing/feature_squeezing_utils.py

from __future__ import annotations

import io
from typing import Callable, Dict

import numpy as np
import torch
from PIL import Image, ImageFilter


DETECTION_LABELS = {
    "clean": 0,
    "random": 0,
    "base": 1,
    "fata": 1,
    "cage": 1,
    "caa": 1,
}


def detection_label(attack: str) -> int:
    """Return the fixed binary label used by detector evaluation."""

    try:
        return DETECTION_LABELS[attack]
    except KeyError as error:
        raise ValueError(f"unknown detection attack: {attack}") from error


def ensure_pil_rgb(image) -> Image.Image:
    """
    Convert input image to PIL RGB.
    """
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    if torch.is_tensor(image):
        x = image.detach().cpu()
        if x.ndim == 4:
            x = x[0]
        if x.ndim == 3 and x.shape[0] in (1, 3):
            x = x.permute(1, 2, 0)
        x = x.numpy()

        if x.max() <= 1.0:
            x = x * 255.0
        x = np.clip(x, 0, 255).astype(np.uint8)
        return Image.fromarray(x).convert("RGB")

    if isinstance(image, np.ndarray):
        x = image
        if x.max() <= 1.0:
            x = x * 255.0
        x = np.clip(x, 0, 255).astype(np.uint8)
        return Image.fromarray(x).convert("RGB")

    raise TypeError(f"Unsupported image type: {type(image)}")


def reduce_bit_depth(image: Image.Image, bits: int = 5) -> Image.Image:
    """
    Reduce color bit depth from 8 bits to `bits`.
    """
    assert 1 <= bits <= 8
    image = ensure_pil_rgb(image)
    arr = np.asarray(image).astype(np.uint8)

    if bits == 8:
        return image

    levels = 2 ** bits
    arr = np.floor(arr.astype(np.float32) / 256.0 * levels)
    arr = np.clip(arr, 0, levels - 1)
    arr = np.round(arr * (255.0 / (levels - 1)))
    arr = np.clip(arr, 0, 255).astype(np.uint8)

    return Image.fromarray(arr).convert("RGB")


def median_filter(image: Image.Image, size: int = 3) -> Image.Image:
    """
    Apply median filter.
    """
    image = ensure_pil_rgb(image)
    return image.filter(ImageFilter.MedianFilter(size=size))


def jpeg_compress(image: Image.Image, quality: int = 75) -> Image.Image:
    """
    JPEG compress and reload image.
    """
    image = ensure_pil_rgb(image)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def get_squeezers() -> Dict[str, Callable[[Image.Image], Image.Image]]:
    """
    Squeezing transformations used for detection.
    """
    return {
        "bit5": lambda img: reduce_bit_depth(img, bits=5),
        "bit4": lambda img: reduce_bit_depth(img, bits=4),
        "median3": lambda img: median_filter(img, size=3),
        "jpeg75": lambda img: jpeg_compress(img, quality=75),
        "jpeg50": lambda img: jpeg_compress(img, quality=50),
    }


@torch.no_grad()
def clip_image_feature(
    image: Image.Image,
    clip_model,
    clip_processor,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Return normalized CLIP image feature.
    This assumes HuggingFace CLIPModel-like interface.
    """
    image = ensure_pil_rgb(image)
    inputs = clip_processor(images=image, return_tensors="pt").to(device)

    if hasattr(clip_model, "get_image_features"):
        feat = clip_model.get_image_features(**inputs)
    else:
        outputs = clip_model.vision_model(**inputs)
        feat = outputs.pooler_output

    feat = torch.nn.functional.normalize(feat.float(), dim=-1)
    return feat[0].detach().cpu()


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    1 - cosine similarity.
    """
    a = torch.nn.functional.normalize(a.float(), dim=0)
    b = torch.nn.functional.normalize(b.float(), dim=0)
    return float(1.0 - torch.dot(a, b).item())


def answer_changed(ans1: str, ans2: str) -> int:
    """
    Conservative answer-level instability.
    """
    a = str(ans1).strip().lower()
    b = str(ans2).strip().lower()
    return int(a != b)
