"""DCConv tests: activation bounds, Eq. 5/6 geometry, init behavior, gradients."""
import math

import pytest
import torch

from src.models.dcconv import DCConv


def test_forward_shape():
    m = DCConv(9, 16, kernel_length=9, axis="x")
    y = m(torch.randn(2, 9, 28, 28))
    assert y.shape == (2, 16, 28, 28)


def test_offset_and_angle_bounds():
    """Tanh/Sigmoid ranges actually respected, even with adversarial predictor
    weights (Tier 3: angle must be [0, pi/2], not [0, 2pi])."""
    m = DCConv(4, 4, kernel_length=9, axis="x")
    with torch.no_grad():   # blow up the predictors on purpose
        m.offset_conv.weight.fill_(100.0); m.offset_conv.bias.fill_(100.0)
        m.angle_conv.weight.fill_(100.0); m.angle_conv.bias.fill_(100.0)
    x = torch.randn(1, 4, 8, 8)
    deltas = torch.tanh(m.offset_conv(x))
    theta = torch.sigmoid(m.angle_conv(x)) * m.angle_max
    assert deltas.abs().max() <= 1.0 + 1e-6
    assert theta.min() >= 0.0 and theta.max() <= math.pi / 2 + 1e-6

    # worst-case displacement bound: |c|<=4, |cumsum|<=4 -> norm <= sqrt(32)
    pos = m.kernel_positions(x)
    ys, xs = torch.meshgrid(torch.arange(8.), torch.arange(8.), indexing="ij")
    center = torch.stack([xs, ys], -1).view(1, 1, 8, 8, 2)
    disp = (pos - center).norm(dim=-1)
    assert disp.max() <= math.sqrt(32) + 1e-4


def test_zero_init_is_straight_axis_kernel():
    """At init the kernel must lie exactly along its axis (paper: 'initially
    aligned along a predefined coordinate axis'; Tier 3 degenerate-start)."""
    x = torch.randn(1, 3, 8, 8)
    for axis in ("x", "y"):
        m = DCConv(3, 3, kernel_length=9, axis=axis)
        pos = m.kernel_positions(x)                     # (1, 9, 8, 8, 2)
        ys, xs = torch.meshgrid(torch.arange(8.), torch.arange(8.), indexing="ij")
        steps = torch.arange(-4.0, 5.0)
        if axis == "x":
            expect_x = xs.view(1, 1, 8, 8) + steps.view(1, 9, 1, 1)
            expect_y = ys.view(1, 1, 8, 8).expand(1, 9, 8, 8)
        else:
            expect_x = xs.view(1, 1, 8, 8).expand(1, 9, 8, 8)
            expect_y = ys.view(1, 1, 8, 8) + steps.view(1, 9, 1, 1)
        # angle bias -6 -> theta ~ 0.22 deg; tolerance covers that residual
        assert torch.allclose(pos[..., 0], expect_x, atol=0.05), axis
        assert torch.allclose(pos[..., 1], expect_y, atol=0.05), axis


def test_zero_init_matches_plain_1d_conv():
    """With zero offsets/angle, DCConv must equal a straight 1D conv with the
    same weights (border padding vs conv zero-padding differs at edges, so
    compare interior only)."""
    torch.manual_seed(0)
    m = DCConv(2, 3, kernel_length=9, axis="x")
    x = torch.randn(1, 2, 16, 16)
    y = m(x)
    ref = torch.nn.functional.conv2d(
        x, m.weight.unsqueeze(2), bias=m.bias, padding=(0, 4))
    # sigmoid never reaches exactly 0: angle bias -6 leaves theta ~= 0.22 deg,
    # displacing the outermost points by ~0.016 px through bilinear interp.
    # That designed residual bounds the deviation at ~0.03 here.
    assert torch.allclose(y[..., 4:-4], ref[..., 4:-4], atol=0.05)


def test_rotation_geometry():
    """Force theta = pi/2 on an x-axis kernel: it must become a y-axis kernel
    ((c, d) . R(pi/2) = (-d, c) -- the moving coordinate transfers to y)."""
    m = DCConv(1, 1, kernel_length=5, axis="x")
    with torch.no_grad():
        m.angle_conv.weight.zero_(); m.angle_conv.bias.fill_(100.0)  # sigmoid->1
    x = torch.randn(1, 1, 8, 8)
    pos = m.kernel_positions(x)
    ys, xs = torch.meshgrid(torch.arange(8.), torch.arange(8.), indexing="ij")
    steps = torch.arange(-2.0, 3.0)
    assert torch.allclose(pos[..., 0], xs.view(1, 1, 8, 8).expand(1, 5, 8, 8), atol=1e-4)
    assert torch.allclose(pos[..., 1], ys.view(1, 1, 8, 8) + steps.view(1, 5, 1, 1), atol=1e-4)


def test_gradients_flow_to_predictors():
    m = DCConv(3, 4, kernel_length=9, axis="y")
    x = torch.randn(2, 3, 12, 12, requires_grad=True)
    m(x).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert m.angle_conv.bias.grad is not None
    assert m.offset_conv.weight.grad is not None
    assert torch.isfinite(m.weight.grad).all()


def test_invalid_args_rejected():
    with pytest.raises(ValueError):
        DCConv(3, 3, kernel_length=8)       # even
    with pytest.raises(ValueError):
        DCConv(3, 3, kernel_length=1)       # too short
    with pytest.raises(ValueError):
        DCConv(3, 3, axis="z")
