"""Hierarchical Wavelet Feature Extraction Branch (HWFEB), per Sec. 3.2.1.

Pipeline per scale l (l = 1..4):

    wavelet bands (B, 9, H/2^l, W/2^l)   [LH|HL|HH of the RGB input, 3ch each]
        -> WFEM: three parallel branches
             DCConv (x-axis init) | DCConv (y-axis init) | standard Conv2d 3x3
           -> concat -> projection -> spatial attention map (B, 1, h_l, w_l)
        -> spatial gating into VMamba stage l:
             gated = feat * attn + feat        (dot-product + skip connection)

Resolution note (documented inference, not in the paper's text): DWT level l has
resolution H/2^l (112, 56, 28, 14 for 224 input) while VMamba stage l runs at
H/2^(l+1) (56, 28, 14, 7). Figure 3 nevertheless pairs the four DWT levels with
the four stages, so each WFEM's projection uses stride 2 -- the only mapping that
makes the four pairings dimensionally consistent.

Channel note (documented inference): the paper says the branch outputs are
"concatenated and projected to create the desired spatial attention map" without
giving a channel count. A *spatial* attention map is single-channel; we project
to 1 channel and apply a sigmoid so the gate is a bounded weighting.
"""
from __future__ import annotations

import torch
from torch import nn

from .dcconv import DCConv
from .dwt import MultiLevelHaarDWT


class WFEM(nn.Module):
    """Wavelet Feature Extraction Module for one scale."""

    def __init__(self, in_channels: int, mid_channels: int, kernel_length: int = 9,
                 angle_max: float | None = None) -> None:
        super().__init__()
        kw = {} if angle_max is None else {"angle_max": angle_max}
        self.dc_x = DCConv(in_channels, mid_channels, kernel_length, axis="x", **kw)
        self.dc_y = DCConv(in_channels, mid_channels, kernel_length, axis="y", **kw)
        self.conv = nn.Conv2d(in_channels, mid_channels, 3, padding=1)
        self.norm = nn.BatchNorm2d(3 * mid_channels)
        self.act = nn.GELU()
        # stride-2 projection: DWT-level resolution -> VMamba-stage resolution
        self.proj = nn.Conv2d(3 * mid_channels, 1, 3, stride=2, padding=1)

    def forward(self, bands: torch.Tensor) -> torch.Tensor:
        f = torch.cat([self.dc_x(bands), self.dc_y(bands), self.conv(bands)], dim=1)
        f = self.act(self.norm(f))
        return torch.sigmoid(self.proj(f))          # (B, 1, h/2, w/2), in (0,1)


class SpatialGate(nn.Module):
    """gated = feat * attn + feat  (Sec. 3.2.1: dot-product + skip connection)."""

    def forward(self, feat: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
        if feat.shape[-2:] != attn.shape[-2:]:
            raise ValueError(
                f"spatial gate mismatch: feat {tuple(feat.shape[-2:])} vs "
                f"attn {tuple(attn.shape[-2:])}"
            )
        return feat * attn + feat


class HWFEB(nn.Module):
    """Multi-level DWT -> one WFEM per enabled stage -> attention maps.

    `stages` selects which VMamba stages receive gating (paper Table 8: all four).
    Returns a dict {stage_index: attention map} with stage_index in 1..4.
    """

    def __init__(
        self,
        levels: int = 4,
        in_channels: int = 3,
        wfem_channels: int = 32,
        kernel_length: int = 9,
        stages: tuple[int, ...] = (1, 2, 3, 4),
        angle_max: float | None = None,
    ) -> None:
        super().__init__()
        if not stages or any(s < 1 or s > levels for s in stages):
            raise ValueError(f"stages must be within 1..{levels}, got {stages}")
        self.stages = tuple(sorted(stages))
        self.dwt = MultiLevelHaarDWT(levels)
        # 3 sub-bands x in_channels each
        self.wfems = nn.ModuleDict(
            {
                str(s): WFEM(3 * in_channels, wfem_channels, kernel_length,
                             angle_max=angle_max)
                for s in self.stages
            }
        )

    def forward(self, image: torch.Tensor) -> dict[int, torch.Tensor]:
        pyramid = self.dwt(image)                    # level l at index l-1
        return {s: self.wfems[str(s)](pyramid[s - 1]) for s in self.stages}
