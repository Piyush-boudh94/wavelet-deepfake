"""Unit tests for Temporal WMamba (Candidate B)."""

import pytest
import torch

from src.models.wmamba import WMamba
from src.models.temporal import T_WMamba


def test_twmamba_forward_backward():
    # Keep sizes very small for the CPU test environment to avoid OOM
    batch_size = 1
    frames = 4
    channels, h, w = 3, 224, 224
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        pytest.skip("VMamba backbone requires CUDA for cross_scan_fn")
    
    # 1. Instantiate the spatial backbone (with default settings)
    spatial_model = WMamba(hwfeb_enabled=False).to(device)
    
    # 2. Wrap it with the temporal sequence model
    t_model = T_WMamba(spatial_backbone=spatial_model, d_model=768, max_frames=32).to(device)
    
    # 3. Create dummy video tensor (B, T, C, H, W)
    x = torch.randn(batch_size, frames, channels, h, w, device=device)
    
    # 4. Forward pass
    logits = t_model(x)
    
    assert logits.shape == (batch_size, 2)  # Binary classification by default
    
    # 5. Backward pass
    loss = logits.sum()
    loss.backward()
    
    # Check that gradients flow to the positional encoding and temporal Mamba parameters
    assert t_model.pos_enc.grad is not None
    assert any(p.grad is not None for p in t_model.temporal_mamba.parameters())
    
    # Check that gradients flow to the spatial backbone
    assert any(p.grad is not None for p in spatial_model.parameters())
