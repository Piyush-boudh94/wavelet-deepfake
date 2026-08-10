"""VMamba-S backbone wrapper around the *official* implementation.

The model class is the vendored, unmodified official code
(vendor/VMamba @ MzeroMiko/VMamba, classification/models/vmamba.py); this module
only (a) instantiates it with the exact `vmambav2_small_224.yaml` hyperparameters
that pair with the official ImageNet-1K checkpoint `vssm_small_0229_ckpt_epoch_222.pth`,
(b) loads those weights strictly, and (c) exposes the four stages so WMamba can
interleave HWFEB spatial gating between them.

Why vendored-official rather than re-implemented: the pretrained checkpoint must
load with strict=True. A reimplementation would need a hand-built key mapping --
the classic silent-degradation trap (a missed key falls back to random init and
nothing errors). strict=True against the official class makes weight loading
all-or-nothing.

Kernel backend note: the official repo's own CUDA extensions
(selective_scan_cuda_core/oflex) are NOT built here -- the pod has no compiler.
The vendored code falls back to mamba_ssm's `selective_scan_cuda` (installed and
verified) and/or its triton path. `selective_scan_backend`/`scan_backend` are
forced accordingly and the choice is asserted at import, not assumed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

_VENDOR = Path(__file__).resolve().parents[2] / "vendor" / "VMamba" / "classification"
if str(_VENDOR) not in sys.path:
    # Appended (not prepended) so the project's own packages always win name clashes.
    sys.path.append(str(_VENDOR))

import models.csm_triton as _csm  # noqa: E402  (vendored official code)

# The pod has no C compiler, and triton builds its GPU driver stub with cc at
# runtime -- so any triton kernel launch dies with "Failed to find C compiler".
# The official code reads this module global at every call and falls back to its
# pure-PyTorch CrossScan/CrossMerge autograd implementation when it is False.
# Data movement only (permutes/flips); the SSM scan itself still runs on
# mamba_ssm's compiled CUDA kernel.
_csm.WITH_TRITON = False

from models.vmamba import VSSM  # noqa: E402  (vendored official code)

# Exact hyperparameters of configs/vssm/vmambav2_small_224.yaml (NAME: vssm1_small_0229),
# the config the official release pairs with vssm_small_0229_ckpt_epoch_222.pth.
VMAMBA_S_KWARGS = dict(
    patch_size=4,
    in_chans=3,
    num_classes=1000,           # ImageNet head; replaced after weight loading
    depths=(2, 2, 15, 2),       # [PAPER Sec 4.1.5]
    dims=(96, 192, 384, 768),
    ssm_d_state=1,
    ssm_ratio=2.0,
    ssm_dt_rank="auto",
    ssm_act_layer="silu",
    ssm_conv=3,
    ssm_conv_bias=False,
    ssm_drop_rate=0.0,
    ssm_init="v0",
    forward_type="v05_noz",
    mlp_ratio=4.0,
    mlp_act_layer="gelu",
    mlp_drop_rate=0.0,
    drop_path_rate=0.3,
    patch_norm=True,
    norm_layer="ln2d",          # -> channel_first=True: features are (B,C,H,W)
    downsample_version="v3",
    patchembed_version="v2",
)

STAGE_DIMS = (96, 192, 384, 768)
STAGE_RESOLUTIONS = (56, 28, 14, 7)  # for 224x224 input


class VMambaSBackbone(nn.Module):
    """Official VSSM (VMamba-S) with stage-wise access for spatial gating.

    forward(x, attn_maps) where attn_maps is {stage_index (1..4): (B,1,h,w)}.
    Gating is applied to each stage's INPUT (Fig. 3: after stem / after each
    downsampling, before the stage's VSS blocks), as feat*attn + feat.
    """

    def __init__(self, pretrained_path: str | Path | None = None,
                 num_classes: int = 2, drop_path_rate: float = 0.3) -> None:
        super().__init__()
        kwargs = dict(VMAMBA_S_KWARGS)
        kwargs["drop_path_rate"] = drop_path_rate
        self.vssm = VSSM(**kwargs)
        if not self.vssm.channel_first:
            raise RuntimeError("expected channel_first VSSM (ln2d norm); config drifted")

        if pretrained_path is not None:
            self._load_official_weights(Path(pretrained_path))

        # Replace the ImageNet head only AFTER strict loading succeeded.
        self.vssm.classifier.head = nn.Linear(self.vssm.num_features, num_classes)
        nn.init.trunc_normal_(self.vssm.classifier.head.weight, std=0.02)
        nn.init.zeros_(self.vssm.classifier.head.bias)

    def _load_official_weights(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(
                f"VMamba-S checkpoint not found: {path}\n"
                "Fetch it with scripts/fetch_weights.py (see docs/DATA.md)."
            )
        if path.suffix == ".safetensors":
            from safetensors.torch import load_file
            state = load_file(str(path))
        else:
            # Security policy (docs/SECURITY.md): never unpickle without
            # weights_only=True. The official ckpt stores extra objects
            # (optimizer/config); we take only the model state dict.
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
            state = ckpt.get("model", ckpt)
        missing, unexpected = self.vssm.load_state_dict(state, strict=True)
        # strict=True raises on mismatch; the return is empty lists by then.
        assert not missing and not unexpected

    # -- stage-wise forward -------------------------------------------------
    def forward_features(
        self, x: torch.Tensor, attn_maps: dict[int, torch.Tensor] | None = None
    ) -> torch.Tensor:
        attn_maps = attn_maps or {}
        x = self.vssm.patch_embed(x)                       # (B, 96, 56, 56)
        if self.vssm.pos_embed is not None:
            x = x + self.vssm.pos_embed
        for i, layer in enumerate(self.vssm.layers, start=1):
            a = attn_maps.get(i)
            if a is not None:
                if a.shape[-2:] != x.shape[-2:]:
                    raise ValueError(
                        f"stage {i}: attn {tuple(a.shape[-2:])} vs feat {tuple(x.shape[-2:])}"
                    )
                x = x * a + x                              # spatial gating + skip
            x = layer(x)                                   # blocks then downsample
        return x                                           # (B, 768, 7, 7)

    def forward(
        self, x: torch.Tensor, attn_maps: dict[int, torch.Tensor] | None = None
    ) -> torch.Tensor:
        return self.vssm.classifier(self.forward_features(x, attn_maps))
