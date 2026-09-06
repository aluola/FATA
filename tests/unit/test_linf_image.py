from __future__ import annotations

import numpy as np
import torch

from fata.attacks.linf_image import quantize_linf_image


def test_quantization_cannot_turn_two_levels_into_three():
    clean = torch.full((1, 3, 2, 2), 6 / 255, dtype=torch.float32)
    adversarial = clean - torch.tensor(2 / 255, dtype=torch.float32)
    image = quantize_linf_image(clean, adversarial, 2 / 255)
    decoded = np.asarray(image, dtype=np.int16)
    assert int(np.abs(decoded - 6).max()) <= 2


def test_quantization_projects_out_of_budget_candidate():
    clean = torch.full((1, 3, 1, 1), 100 / 255)
    adversarial = torch.zeros_like(clean)
    image = quantize_linf_image(clean, adversarial, 2 / 255)
    assert np.asarray(image).tolist() == [[[98, 98, 98]]]


def test_fractional_pixel_budget_uses_floor_of_discrete_levels():
    clean = torch.full((1, 3, 1, 1), 100 / 255)
    adversarial = torch.full_like(clean, 102 / 255)
    half_level = quantize_linf_image(clean, adversarial, 0.5 / 255)
    one_and_half = quantize_linf_image(clean, adversarial, 1.5 / 255)
    assert np.asarray(half_level).tolist() == [[[100, 100, 100]]]
    assert np.asarray(one_and_half).tolist() == [[[101, 101, 101]]]
