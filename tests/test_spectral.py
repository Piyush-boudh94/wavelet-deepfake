"""Unit tests for Spectral-Wavelet Feature Extraction (Candidate A)."""

import pytest
import torch

from src.models.spectral import BlockDCT8x8, SpectralBranch, SpectralEncoder
from src.models.swfeb import SWFEB, SWFEM


def test_block_dct_8x8():
    batch_size, channels, h, w = 2, 3, 224, 224
    x = torch.randn(batch_size, channels, h, w)
    
    dct = BlockDCT8x8(in_channels=channels)
    out = dct(x)
    
    # 64 basis functions per channel
    assert out.shape == (batch_size, channels * 64, h // 8, w // 8)
    # Ensure it requires no gradients (parameter-free feature extractor)
    assert not dct.conv.weight.requires_grad


def test_spectral_branch():
    batch_size, channels, h, w = 2, 3, 224, 224
    x = torch.randn(batch_size, channels, h, w)
    
    branch = SpectralBranch(in_channels=channels)
    x_dct, x_fft = branch(x)
    
    assert x_dct.shape == (batch_size, channels * 64, h // 8, w // 8)
    assert x_fft.shape == (batch_size, channels, h, w // 2 + 1)


def test_spectral_encoder():
    batch_size = 2
    x_dct = torch.randn(batch_size, 192, 28, 28)
    x_fft = torch.randn(batch_size, 3, 224, 113)
    
    target_shape = (56, 56)  # Stage 1 VMamba spatial size
    
    encoder = SpectralEncoder(in_channels_dct=192, in_channels_fft=3, mid_channels=32)
    a_spec = encoder(x_dct, x_fft, target_shape)
    
    assert a_spec.shape == (batch_size, 1, 56, 56)
    assert torch.all((a_spec >= 0) & (a_spec <= 1))  # Sigmoid applied


def test_swfem():
    batch_size = 2
    bands = torch.randn(batch_size, 9, 112, 112)  # Level 1 DWT bands
    x_dct = torch.randn(batch_size, 192, 28, 28)
    x_fft = torch.randn(batch_size, 3, 224, 113)
    
    swfem = SWFEM(in_channels=9, mid_channels=32, in_channels_dct=192, in_channels_fft=3)
    
    # Should fuse to stage 1 spatial size (112/2 = 56)
    a_fused = swfem(bands, x_dct, x_fft)
    
    assert a_fused.shape == (batch_size, 1, 56, 56)
    assert torch.all((a_fused >= 0) & (a_fused <= 1))


def test_swfeb_all_stages():
    batch_size, channels, h, w = 2, 3, 224, 224
    x = torch.randn(batch_size, channels, h, w)
    
    swfeb = SWFEB(levels=4, in_channels=channels, wfem_channels=32)
    
    attn_maps = swfeb(x)
    
    assert set(attn_maps.keys()) == {1, 2, 3, 4}
    assert attn_maps[1].shape == (batch_size, 1, 56, 56)
    assert attn_maps[2].shape == (batch_size, 1, 28, 28)
    assert attn_maps[3].shape == (batch_size, 1, 14, 14)
    assert attn_maps[4].shape == (batch_size, 1, 7, 7)
