"""Dynamic Contour Convolution (DCConv), per Sec. 3.3 / Eqs. 5-6 of WMamba.

A 1D deformable kernel of odd length k (paper: k=9, Table 10) whose shape adapts
per pixel. For each output pixel i at (x_i, y_i), the layer predicts:

  * offsets  {delta_{i,+-c} : c=1..(k-1)/2}, Tanh-bounded to [-1, 1]; delta_{i,0}=0.
    Offsets are accumulated iteratively outward from the kernel center
    (the sum in Eqs. 5-6), so adjacent kernel points never drift apart by more
    than one pixel -- this is what makes the kernel trace smooth slender contours.
  * a rotation angle theta_i, Sigmoid-then-scaled to [0, pi/2] (Eq. text), which
    rotates the whole kernel about its center via
        R = [[cos t, sin t], [-sin t, cos t]]   (row-vector convention).

x-axis initialization (Eq. 5):  K_{i+-c} = (x_i, y_i) + (+-c, sum_j delta_{i,+-j}) . R
y-axis initialization (Eq. 6):  K_{i+-c} = (x_i, y_i) + (sum_j delta_{i,+-j}, +-c) . R

Feature values at the resulting fractional positions are read with bilinear
interpolation (as in deformable convolution), then combined by a learnable
1D kernel across the k sampled points.

Bounds note (audit Tier 3): with k=9, |c| <= 4 and |cumsum(delta)| <= 4, so the
worst-case displacement is ||(4,4)|| = 5.66 px before rotation (rotation preserves
norm). Sampling uses grid_sample(padding_mode="border"), which clamps
out-of-bounds coordinates to the edge instead of silently returning zeros --
border clamping keeps gradients alive at the image boundary.

Init note (audit Tier 3): offset- and angle-predictor convs are zero-initialized,
and the angle predictor's bias is set so theta ~= 0 at initialization. The layer
therefore starts as an exact straight, axis-aligned 1D convolution -- the paper's
"initially aligned along a predefined coordinate axis" -- and deforms only as it
learns. Large random initial offsets are a documented degenerate-start failure
mode for deformable convs.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

_ANGLE_BIAS_INIT = -6.0  # sigmoid(-6) ~= 0.0025 -> theta_0 ~= 0.2 deg ~ axis-aligned


class DCConv(nn.Module):
    """Deformable 1D contour conv. (B, C_in, H, W) -> (B, C_out, H, W)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_length: int = 9,
        axis: str = "x",
        angle_max: float = math.pi / 2,
    ) -> None:
        super().__init__()
        if kernel_length < 3 or kernel_length % 2 == 0:
            raise ValueError(f"kernel_length must be odd and >= 3, got {kernel_length}")
        if axis not in ("x", "y"):
            raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")
        self.k = kernel_length
        self.half = kernel_length // 2
        self.axis = axis
        self.angle_max = float(angle_max)

        # Predictors (Sec. 3.3: "two standard 2D convolutional layers to
        # independently generate pixel-wise offsets and rotation angles").
        # k-1 offsets per pixel: c = 1..half on each side; delta_{i,0} is fixed 0.
        self.offset_conv = nn.Conv2d(in_channels, self.k - 1, 3, padding=1)
        self.angle_conv = nn.Conv2d(in_channels, 1, 3, padding=1)
        # 1D kernel applied across the k sampled points, fused with the channel
        # mixing: equivalent to Conv2d over the stacked samples.
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, self.k))
        self.bias = nn.Parameter(torch.zeros(out_channels))

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)
        nn.init.zeros_(self.angle_conv.weight)
        nn.init.constant_(self.angle_conv.bias, _ANGLE_BIAS_INIT)

    def kernel_positions(self, x: torch.Tensor) -> torch.Tensor:
        """Absolute fractional sampling positions, (B, k, H, W, 2) as (px, py).

        Exposed separately so tests can assert the Eq. 5/6 geometry and the
        activation bounds directly.
        """
        b, _, h, w = x.shape
        dev, dt = x.device, torch.float32

        # --- learned quantities, bounded exactly as the paper specifies -------
        deltas = torch.tanh(self.offset_conv(x.float()))            # (B, k-1, H, W) in [-1,1]
        theta = torch.sigmoid(self.angle_conv(x.float())) * self.angle_max  # (B,1,H,W) in [0, angle_max]

        # Split per side and accumulate outward from the center (delta_{i,0}=0):
        # position c uses sum_{j<=c} delta_j. cumsum implements the iterative
        # "step-by-step" calculation in the paper.
        neg = torch.cumsum(deltas[:, : self.half], dim=1)           # c = -1..-half
        pos = torch.cumsum(deltas[:, self.half :], dim=1)           # c = +1..+half
        zero = torch.zeros(b, 1, h, w, device=dev, dtype=dt)
        # kernel order: -half..-1, 0, +1..+half  -> index axis size k
        drift = torch.cat([neg.flip(1), zero, pos], dim=1)          # (B, k, H, W)
        steps = torch.arange(-self.half, self.half + 1, device=dev, dtype=dt)
        steps = steps.view(1, self.k, 1, 1).expand(b, self.k, h, w)  # +-c

        # (a, b) . [[cos,sin],[-sin,cos]] = (a cos - b sin, a sin + b cos)
        if self.axis == "x":
            a, off = steps, drift        # Eq. 5: moving coord is x, drift is y
        else:
            a, off = drift, steps        # Eq. 6: moving coord is y, drift is x
        cos_t, sin_t = torch.cos(theta), torch.sin(theta)            # (B,1,H,W)
        dx = a * cos_t - off * sin_t
        dy = a * sin_t + off * cos_t

        # absolute positions
        ys = torch.arange(h, device=dev, dtype=dt).view(1, 1, h, 1)
        xs = torch.arange(w, device=dev, dtype=dt).view(1, 1, 1, w)
        px = xs + dx
        py = ys + dy
        return torch.stack([px, py], dim=-1)                         # (B, k, H, W, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        pos = self.kernel_positions(x)                               # (B, k, H, W, 2)

        # Normalize to [-1, 1] for grid_sample (align_corners=True maps
        # 0 -> -1 and (size-1) -> +1 exactly).
        gx = pos[..., 0] * (2.0 / max(w - 1, 1)) - 1.0
        gy = pos[..., 1] * (2.0 / max(h - 1, 1)) - 1.0
        grid = torch.stack([gx, gy], dim=-1).reshape(b, self.k * h, w, 2)

        sampled = F.grid_sample(
            x.float(), grid, mode="bilinear", padding_mode="border", align_corners=True
        )                                                            # (B, C, k*H, W)
        sampled = sampled.view(b, c, self.k, h, w)

        # y[b,o,h,w] = sum_{c,k} W[o,c,k] * sampled[b,c,k,h,w] + bias
        out = torch.einsum("ock,bckhw->bohw", self.weight.float(), sampled)
        out = out + self.bias.float().view(1, -1, 1, 1)
        return out.to(x.dtype)
