"""Spectral-Wavelet Feature Extraction Branch (SWFEB), per Phase 3 implementation.

Extends the standard HWFEB with a parallel global spectral branch (DCT + FFT)
fused via content-adaptive spatial gating.
"""
from __future__ import annotations

import torch
from torch import nn

from .dcconv import DCConv
from .dwt import MultiLevelHaarDWT
from .spectral import SpectralBranch, SpectralEncoder


class SWFEM(nn.Module):
    """Spectral-Wavelet Feature Extraction Module for one scale."""

    def __init__(self, in_channels: int, mid_channels: int, 
                 in_channels_dct: int = 192, in_channels_fft: int = 3, 
                 kernel_length: int = 9, angle_max: float | None = None) -> None:
        super().__init__()
        kw = {} if angle_max is None else {"angle_max": angle_max}
        # Wavelet Branch
        self.dc_x = DCConv(in_channels, mid_channels, kernel_length, axis="x", **kw)
        self.dc_y = DCConv(in_channels, mid_channels, kernel_length, axis="y", **kw)
        self.conv = nn.Conv2d(in_channels, mid_channels, 3, padding=1)
        self.norm = nn.BatchNorm2d(3 * mid_channels)
        self.act = nn.GELU()
        # stride-2 projection: DWT-level resolution -> VMamba-stage resolution
        self.proj = nn.Conv2d(3 * mid_channels, 1, 3, stride=2, padding=1)

        # Spectral Branch Encoder
        self.spec_enc = SpectralEncoder(in_channels_dct, in_channels_fft, mid_channels)
        
        # Spatial Fusion mechanism
        self.fusion_mlp = nn.Sequential(
            nn.Linear(2, 16),
            nn.GELU(),
            nn.Linear(16, 2)
        )

    def forward(self, bands: torch.Tensor, x_dct: torch.Tensor, x_fft: torch.Tensor) -> torch.Tensor:
        # 1. Wavelet Attention Map
        f_wave = torch.cat([self.dc_x(bands), self.dc_y(bands), self.conv(bands)], dim=1)
        f_wave = self.act(self.norm(f_wave))
        a_wave = torch.sigmoid(self.proj(f_wave))          # (B, 1, h/2, w/2)

        # 2. Spectral Attention Map
        a_spec = self.spec_enc(x_dct, x_fft, a_wave.shape[-2:])  # (B, 1, h/2, w/2)

        # 3. Content-adaptive Fusion
        pool_wave = a_wave.mean(dim=[-2, -1])  # (B, 1)
        pool_spec = a_spec.mean(dim=[-2, -1])  # (B, 1)
        pool = torch.cat([pool_wave, pool_spec], dim=1)  # (B, 2)
        
        beta = torch.softmax(self.fusion_mlp(pool), dim=1)  # (B, 2)
        beta_wave = beta[:, 0:1, None, None]
        beta_spec = beta[:, 1:2, None, None]
        
        a_fused = beta_wave * a_wave + beta_spec * a_spec
        return a_fused


class SWFEB(nn.Module):
    """Multi-level DWT + Global Spectral -> one SWFEM per enabled stage.

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
        self.spectral = SpectralBranch(in_channels)
        
        # 3 sub-bands x in_channels each
        self.swfems = nn.ModuleDict(
            {
                str(s): SWFEM(3 * in_channels, wfem_channels, 
                              in_channels_dct=64*in_channels, in_channels_fft=in_channels,
                              kernel_length=kernel_length, angle_max=angle_max)
                for s in self.stages
            }
        )

    def forward(self, image: torch.Tensor) -> dict[int, torch.Tensor]:
        pyramid = self.dwt(image)                    # level l at index l-1
        x_dct, x_fft = self.spectral(image)
        return {s: self.swfems[str(s)](pyramid[s - 1], x_dct, x_fft) for s in self.stages}
