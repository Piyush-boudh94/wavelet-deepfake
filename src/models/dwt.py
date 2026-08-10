"""Haar Discrete Wavelet Transform, exactly per Eq. 1 of WMamba (arXiv:2501.09617).

The four filters are fixed (non-learnable) 2x2 kernels applied with stride 2,
independently per input channel:

    f_LL = 1/2 [[ 1,  1], [ 1,  1]]
    f_LH = 1/2 [[ 1,  1], [-1, -1]]
    f_HL = 1/2 [[ 1, -1], [ 1, -1]]
    f_HH = 1/2 [[ 1, -1], [-1,  1]]

Implemented directly as a grouped conv2d rather than via a wavelet library so the
filter bank matches the paper's Eq. 1 bit-for-bit.

The model consumes only LH/HL/HH: the paper's ablation (Table 3) shows that
including LL *hurts* cross-dataset performance. LL is still computed internally
because the multi-level pyramid recurses on it (Sec. 3.2.1).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

# Eq. 1 filter bank, order: LL, LH, HL, HH.
_HAAR = torch.tensor(
    [
        [[1.0, 1.0], [1.0, 1.0]],
        [[1.0, 1.0], [-1.0, -1.0]],
        [[1.0, -1.0], [1.0, -1.0]],
        [[1.0, -1.0], [-1.0, 1.0]],
    ]
) * 0.5


class HaarDWT(nn.Module):
    """One DWT level. (B, C, H, W) -> dict of 4 sub-bands, each (B, C, H/2, W/2)."""

    def __init__(self) -> None:
        super().__init__()
        # (4, 1, 2, 2); registered as a buffer so .to(device)/.half() follow the
        # module but the filters are never trainable.
        self.register_buffer("filters", _HAAR.unsqueeze(1), persistent=False)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError(f"HaarDWT expects (B,C,H,W), got shape {tuple(x.shape)}")
        b, c, h, w = x.shape
        if h % 2 or w % 2:
            raise ValueError(f"HaarDWT needs even spatial dims, got {h}x{w}")
        # Repeat the 4-filter bank per channel and convolve with groups=C, so each
        # channel is decomposed independently: out (B, C*4, H/2, W/2).
        weight = self.filters.to(x.dtype).repeat(c, 1, 1, 1)  # (C*4, 1, 2, 2)
        y = F.conv2d(x, weight, stride=2, groups=c)
        y = y.view(b, c, 4, h // 2, w // 2)
        return {
            "LL": y[:, :, 0],
            "LH": y[:, :, 1],
            "HL": y[:, :, 2],
            "HH": y[:, :, 3],
        }


class MultiLevelHaarDWT(nn.Module):
    """Recursive DWT pyramid (Sec. 3.2.1): LL of level l feeds level l+1.

    Returns, for each level, the concatenation of the three high-frequency
    sub-bands only -- (B, 3*C, H/2^l, W/2^l), channel order [LH | HL | HH].
    LL is used solely for recursion and never exposed to the model.
    """

    def __init__(self, levels: int) -> None:
        super().__init__()
        if levels < 1:
            raise ValueError(f"levels must be >= 1, got {levels}")
        self.levels = levels
        self.dwt = HaarDWT()

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        out: list[torch.Tensor] = []
        cur = x
        for _ in range(self.levels):
            bands = self.dwt(cur)
            out.append(torch.cat([bands["LH"], bands["HL"], bands["HH"]], dim=1))
            cur = bands["LL"]
        return out
