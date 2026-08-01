"""Full-model GPU tests. Skipped automatically when CUDA is absent (they run in
the pod via scripts/pod.sh; the head node has no GPU)."""
import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU pod")

CKPT = "checkpoints/pretrained/vssm_small_0229_ckpt_epoch_222.pth"


@cuda
def test_full_forward_backward():
    from src.models.wmamba import WMamba
    m = WMamba(pretrained_path=None).cuda()      # random init: fast, no I/O
    x = torch.rand(2, 3, 224, 224, device="cuda")
    y = m(x)
    assert y.shape == (2, 2)
    loss = torch.nn.functional.cross_entropy(y, torch.tensor([0, 1], device="cuda"))
    loss.backward()
    assert torch.isfinite(loss)
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


@cuda
def test_hwfeb_attention_maps_match_stage_resolutions():
    from src.models.hwfeb import HWFEB
    h = HWFEB(levels=4).cuda()
    maps = h(torch.rand(1, 3, 224, 224, device="cuda"))
    expect = {1: 56, 2: 28, 3: 14, 4: 7}
    assert set(maps) == set(expect)
    for s, r in expect.items():
        assert maps[s].shape == (1, 1, r, r), s
        assert maps[s].min() >= 0.0 and maps[s].max() <= 1.0   # sigmoid-bounded


@cuda
def test_bf16_autocast_finite():
    """Tier 3: bf16 Mamba has documented silent-NaN failure modes; assert the
    full model is finite fwd+bwd under autocast bf16."""
    from src.models.wmamba import WMamba
    m = WMamba(pretrained_path=None).cuda()
    x = torch.rand(2, 3, 224, 224, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y = m(x)
        loss = torch.nn.functional.cross_entropy(
            y, torch.tensor([0, 1], device="cuda"))
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all()
               for p in m.parameters() if p.grad is not None)


@cuda
def test_pretrained_checkpoint_loads_strict():
    from pathlib import Path
    if not Path(CKPT).exists():
        pytest.skip("pretrained ckpt not fetched")
    from src.models.vmamba_backbone import VMambaSBackbone
    m = VMambaSBackbone(pretrained_path=CKPT)   # strict=True inside
    assert m.vssm.classifier.head.out_features == 2


@cuda
def test_gating_changes_output():
    """Gating must actually modulate the backbone (not a dead code path)."""
    from src.models.vmamba_backbone import VMambaSBackbone
    m = VMambaSBackbone(pretrained_path=None).cuda().eval()
    x = torch.rand(1, 3, 224, 224, device="cuda")
    with torch.no_grad():
        base = m(x)
        half = {i: torch.full((1, 1, r, r), 0.5, device="cuda")
                for i, r in zip((1, 2, 3, 4), (56, 28, 14, 7))}
        gated = m(x, half)
    assert not torch.allclose(base, gated)


@cuda
def test_wrong_attn_resolution_rejected():
    from src.models.vmamba_backbone import VMambaSBackbone
    m = VMambaSBackbone(pretrained_path=None).cuda()
    x = torch.rand(1, 3, 224, 224, device="cuda")
    with pytest.raises(ValueError, match="stage 1"):
        m(x, {1: torch.ones(1, 1, 28, 28, device="cuda")})
