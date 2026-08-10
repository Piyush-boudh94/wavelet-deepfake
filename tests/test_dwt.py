"""DWT tests: shapes, hand-computed values, recursion, LL exclusion."""
import numpy as np
import pytest
import torch

from src.models.dwt import HaarDWT, MultiLevelHaarDWT


def test_shapes():
    x = torch.randn(2, 3, 32, 32)
    bands = HaarDWT()(x)
    assert set(bands) == {"LL", "LH", "HL", "HH"}
    for b in bands.values():
        assert b.shape == (2, 3, 16, 16)


def test_known_values():
    # Single 2x2 block [[a,b],[c,d]] = [[1,2],[3,4]]:
    #   LL = (a+b+c+d)/2 = 5,  LH = (a+b-c-d)/2 = -2,
    #   HL = (a-b+c-d)/2 = -1, HH = (a-b-c+d)/2 = 0
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).view(1, 1, 2, 2)
    b = HaarDWT()(x)
    assert torch.isclose(b["LL"].squeeze(), torch.tensor(5.0))
    assert torch.isclose(b["LH"].squeeze(), torch.tensor(-2.0))
    assert torch.isclose(b["HL"].squeeze(), torch.tensor(-1.0))
    assert torch.isclose(b["HH"].squeeze(), torch.tensor(0.0))


def test_constant_image_has_zero_high_freq():
    x = torch.full((1, 3, 16, 16), 7.0)
    b = HaarDWT()(x)
    for k in ("LH", "HL", "HH"):
        assert torch.allclose(b[k], torch.zeros_like(b[k]), atol=1e-6)
    # LL of a constant image is 2*value (filters sum to 2 with the 1/2 factor)
    assert torch.allclose(b["LL"], torch.full_like(b["LL"], 14.0))


def test_energy_preservation():
    # Haar with 1/2 scaling is orthonormal: sum of squared coefficients
    # across the 4 bands equals the input energy.
    x = torch.randn(1, 1, 64, 64)
    b = HaarDWT()(x)
    energy = sum((b[k] ** 2).sum() for k in b)
    assert torch.isclose(energy, (x ** 2).sum(), rtol=1e-4)


def test_multilevel_shapes_and_ll_exclusion():
    ml = MultiLevelHaarDWT(4)
    out = ml(torch.randn(1, 3, 224, 224))
    assert len(out) == 4
    for lvl, t in enumerate(out, start=1):
        s = 224 // (2 ** lvl)
        assert t.shape == (1, 9, s, s)      # 3 bands x 3 ch -- LL never exposed


def test_odd_size_rejected():
    with pytest.raises(ValueError):
        HaarDWT()(torch.randn(1, 3, 15, 16))


def test_wrong_ndim_rejected():
    with pytest.raises(ValueError):
        HaarDWT()(torch.randn(3, 16, 16))
