"""Spectral feature extraction branch for S-WMamba (Candidate A).

Implements Block DCT and FFT extraction, which are then fused with wavelet
features to provide global frequency context.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def _generate_dct_basis_8x8() -> torch.Tensor:
    """Generate 8x8 DCT basis filters.
    
    Returns:
        Tensor of shape (64, 1, 8, 8)
    """
    basis = torch.zeros(64, 1, 8, 8)
    for u in range(8):
        for v in range(8):
            for i in range(8):
                for j in range(8):
                    cu = math.sqrt(1 / 8) if u == 0 else math.sqrt(2 / 8)
                    cv = math.sqrt(1 / 8) if v == 0 else math.sqrt(2 / 8)
                    basis[u * 8 + v, 0, i, j] = (
                        cu * cv
                        * math.cos((2 * i + 1) * u * math.pi / 16)
                        * math.cos((2 * j + 1) * v * math.pi / 16)
                    )
    return basis


class BlockDCT8x8(nn.Module):
    """Computes 8x8 block DCT using a fixed convolutional layer."""
    
    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        self.in_channels = in_channels
        basis = _generate_dct_basis_8x8()
        # Repeat basis for each input channel, keeping groups=in_channels
        basis = basis.repeat(in_channels, 1, 1, 1)
        self.conv = nn.Conv2d(
            in_channels, 
            64 * in_channels, 
            kernel_size=8, 
            stride=8, 
            groups=in_channels, 
            bias=False
        )
        self.conv.weight.data.copy_(basis)
        self.conv.weight.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            (B, 64*C, H/8, W/8)
        """
        return self.conv(x)


class GlobalFFT(nn.Module):
    """Computes the log magnitude of the 2D Real FFT."""
    
    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            mag: (B, C, H, W/2+1)
        """
        fft_c = torch.fft.rfft2(x, norm="ortho")
        mag = torch.log(torch.abs(fft_c) + self.eps)
        return mag


class SpectralBranch(nn.Module):
    """Computes global DCT and FFT once per input."""
    
    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        self.dct = BlockDCT8x8(in_channels)
        self.fft = GlobalFFT()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_dct = self.dct(x)
        x_fft = self.fft(x)
        return x_dct, x_fft


class SpectralEncoder(nn.Module):
    """Encodes resampled spectral features into a spectral attention map for a specific stage."""
    
    def __init__(self, in_channels_dct: int, in_channels_fft: int, mid_channels: int = 32) -> None:
        super().__init__()
        # 1. 2-layer lightweight spectral encoder
        self.s_encoder = nn.Sequential(
            nn.Conv2d(in_channels_dct + in_channels_fft, mid_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.GELU()
        )
        # 2. frequency-selective filters (F3-Net style)
        self.sconv_a = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1)
        self.sconv_b = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1)
        # 3. projection to attention map
        self.project = nn.Conv2d(mid_channels * 2, 1, kernel_size=3, padding=1)

    def forward(self, x_dct: torch.Tensor, x_fft: torch.Tensor, target_shape: tuple[int, int]) -> torch.Tensor:
        """
        Args:
            x_dct: (B, C_dct, H_dct, W_dct)
            x_fft: (B, C_fft, H_fft, W_fft)
            target_shape: (h_l, w_l) of the VMamba stage
        Returns:
            (B, 1, h_l, w_l)
        """
        # Resample to stage-i spatial size
        h, w = target_shape
        x_dct_r = F.interpolate(x_dct, size=(h, w), mode="bilinear", align_corners=False)
        x_fft_r = F.interpolate(x_fft, size=(h, w), mode="bilinear", align_corners=False)
        
        feat = torch.cat([x_dct_r, x_fft_r], dim=1)
        s = self.s_encoder(feat)
        
        g1 = self.sconv_a(s)
        g2 = self.sconv_b(s)
        
        a_spec = self.project(torch.cat([g1, g2], dim=1))
        return torch.sigmoid(a_spec)
