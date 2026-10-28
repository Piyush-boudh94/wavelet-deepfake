"""Temporal-WMamba (Candidate B).

Replaces per-frame max-pooling with a learned, bidirectional Mamba sequence 
model across frames for video-level deepfake detection.
"""
from __future__ import annotations

import torch
from torch import nn

from .wmamba import WMamba


class T_WMamba(nn.Module):
    """Temporal sequence modeling over WMamba spatial features."""

    def __init__(
        self, 
        spatial_backbone: WMamba, 
        d_model: int = 768, 
        max_frames: int = 32
    ) -> None:
        super().__init__()
        self.spatial_backbone = spatial_backbone
        
        # Disable the internal classifier head of the backbone to prevent confusion
        # (we use it explicitly at the end)
        self.classifier = self.spatial_backbone.backbone.vssm.classifier.head
        
        try:
            from mamba_ssm import Mamba
        except ImportError:
            raise ImportError(
                "mamba_ssm is required for T_WMamba. Ensure it is installed in the environment."
            )
        
        # Mamba layer expects (B, T, D)
        self.temporal_mamba = Mamba(
            d_model=d_model,
            d_state=16,
            d_conv=4,
            expand=2,
        )
        
        # Learned positional encoding for temporal ordering
        self.pos_enc = nn.Parameter(torch.randn(1, max_frames, d_model) * 0.02)
        
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C, H, W) video tensor
        Returns:
            (B, num_classes) video logits
        """
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)
        
        # 1. Spatial feature extraction (WMamba)
        attn = self.spatial_backbone.hwfeb(x_flat) if self.spatial_backbone.hwfeb is not None else None
        features = self.spatial_backbone.backbone.forward_features(x_flat, attn)  # (B*T, d_model, h, w)
        
        # 2. Global Average Pooling -> (B*T, d_model)
        z = features.flatten(2).mean(2)
        z = z.view(B, T, -1)
        
        # 3. Add Temporal Positional Encoding
        z = z + self.pos_enc[:, :T, :]
        
        # 4. Bidirectional Temporal Mamba Scan
        z_fwd = self.temporal_mamba(z)
        z_bwd = self.temporal_mamba(z.flip(dims=[1])).flip(dims=[1])
        
        z_temporal = self.norm(z_fwd + z_bwd)
        
        # 5. Video-level aggregation
        z_video = z_temporal.mean(dim=1)  # (B, d_model)
        
        return self.classifier(z_video)

    @torch.no_grad()
    def predict_fakeness(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample P(fake), used by video-level evaluation."""
        return torch.softmax(self.forward(x), dim=1)[:, 1]
