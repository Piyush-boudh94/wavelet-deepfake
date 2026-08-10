"""WMamba: HWFEB + VMamba-S, per Sec. 3 of arXiv:2501.09617.

Input  : (B, 3, 224, 224) RGB face crops in [0, 1] (normalized upstream).
Output : (B, 2) logits -- index 0 = real, index 1 = fake.
Loss   : plain cross-entropy (Sec. 3.1 -- deliberately no auxiliary losses).

The HWFEB consumes the *raw image* (multi-level Haar DWT of the input, Fig. 3),
not backbone features. Its four attention maps gate the inputs of the four
VMamba stages (resolutions 56/28/14/7 for 224 input).
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .hwfeb import HWFEB
from .vmamba_backbone import VMambaSBackbone


class WMamba(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        pretrained_path: str | Path | None = None,
        hwfeb_enabled: bool = True,
        dwt_levels: int = 4,
        stages: tuple[int, ...] = (1, 2, 3, 4),
        wfem_channels: int = 32,
        dcconv_kernel_length: int = 9,
        angle_max: float | None = None,
        drop_path_rate: float = 0.3,
    ) -> None:
        super().__init__()
        self.backbone = VMambaSBackbone(
            pretrained_path=pretrained_path,
            num_classes=num_classes,
            drop_path_rate=drop_path_rate,
        )
        self.hwfeb = (
            HWFEB(
                levels=dwt_levels,
                in_channels=3,
                wfem_channels=wfem_channels,
                kernel_length=dcconv_kernel_length,
                stages=stages,
                angle_max=angle_max,
            )
            if hwfeb_enabled
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.hwfeb(x) if self.hwfeb is not None else None
        return self.backbone(x, attn)

    @torch.no_grad()
    def predict_fakeness(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample P(fake), used by video-level evaluation."""
        return torch.softmax(self.forward(x), dim=1)[:, 1]

    def param_groups(self) -> list[dict]:
        """Two groups so training can monitor/tune pretrained-backbone vs
        fresh-HWFEB separately (audit Tier 3: per-group gradient norms)."""
        groups = [{"name": "backbone", "params": list(self.backbone.parameters())}]
        if self.hwfeb is not None:
            groups.append({"name": "hwfeb", "params": list(self.hwfeb.parameters())})
        return groups


def build_wmamba(cfg) -> WMamba:
    """Construct from the OmegaConf config (configs/base.yaml `model` node)."""
    m = cfg.model
    return WMamba(
        num_classes=m.num_classes,
        pretrained_path=m.backbone.pretrained_path if m.backbone.pretrained else None,
        hwfeb_enabled=m.hwfeb.enabled,
        dwt_levels=m.hwfeb.dwt_levels,
        stages=tuple(m.hwfeb.stages),
        wfem_channels=m.hwfeb.wfem_channels,
        dcconv_kernel_length=m.dcconv.kernel_length,
        angle_max=m.dcconv.angle_max,
        drop_path_rate=m.backbone.drop_path_rate,
    )
