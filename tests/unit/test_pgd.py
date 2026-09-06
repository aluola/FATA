import pytest
import torch

from fata.attacks.pgd import linf_project_and_clip, validate_linf


def test_projection_and_pixel_clipping():
    clean = torch.tensor([0.0, .5, 1.0])
    adv = linf_project_and_clip(clean, torch.tensor([-1.0, .9, 2.0]), epsilon=.1)
    assert torch.allclose(adv, torch.tensor([0.0, .6, 1.0]))
    assert validate_linf(clean, adv, .1) == pytest.approx(.1)


def test_linf_violation_fails():
    with pytest.raises(ValueError):
        validate_linf(torch.zeros(2), torch.tensor([0.0, .2]), .1)
