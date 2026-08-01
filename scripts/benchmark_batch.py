#!/usr/bin/env python3
"""Measure the real per-batch VRAM footprint of WMamba on this GPU (Stage 3).

Runs COMPLETE training steps -- forward, CE loss, backward, AdamW step -- so
optimizer moments (2x fp32 params) and activation memory are both in the
measurement, with bf16 autocast matching configs/base.yaml. Batch is stepped
up until OOM; each size reports torch.cuda.max_memory_allocated.

Run inside the pod:
    scripts/pod.sh exec '.venv/bin/python scripts/benchmark_batch.py'
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.wmamba import WMamba  # noqa: E402

CKPT = "checkpoints/pretrained/vssm_small_0229_ckpt_epoch_222.pth"
STEPS_PER_SIZE = 3          # step 1 allocates AdamW moments; 3 covers fragmentation
CANDIDATES = [2, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64]


def try_batch(model, optimizer, bs: int) -> float:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    loss_fn = torch.nn.CrossEntropyLoss()
    for _ in range(STEPS_PER_SIZE):
        x = torch.rand(bs, 3, 224, 224, device="cuda")
        y = torch.randint(0, 2, (bs,), device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = loss_fn(model(x), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1024**3


def main() -> None:
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"device: {torch.cuda.get_device_name(0)}  total {total:.1f} GiB")

    ckpt = CKPT if Path(CKPT).exists() else None
    model = WMamba(pretrained_path=ckpt).cuda()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.05)

    results: dict[int, float] = {}
    for bs in CANDIDATES:
        try:
            peak = try_batch(model, optimizer, bs)
            results[bs] = peak
            print(f"batch {bs:3d}: peak {peak:5.2f} GiB "
                  f"({peak/total*100:4.1f}% of card)", flush=True)
        except torch.cuda.OutOfMemoryError:
            print(f"batch {bs:3d}: OOM", flush=True)
            break

    if not results:
        print("nothing fit -- even batch 2 OOMs")
        return
    # largest batch whose peak leaves >= ~12.5% headroom (cap at 87.5% usage)
    budget = 0.875 * total
    fitting = [b for b, p in results.items() if p <= budget]
    pick = max(fitting) if fitting else min(results)
    print(f"\nbudget {budget:.1f} GiB (87.5% of {total:.1f})")
    print(f"RECOMMENDED micro_batch_size: {pick}  "
          f"(peak {results[pick]:.2f} GiB, headroom "
          f"{total - results[pick]:.2f} GiB)")
    for accum in range(1, 65):
        if pick * accum == 64:
            print(f"grad_accum_steps: {accum}  -> effective batch 64 exactly")
            break
    else:
        print(f"WARNING: 64 not divisible by {pick}; nearest effective batches: "
              f"{pick * (64 // pick)} or {pick * (64 // pick + 1)}")


if __name__ == "__main__":
    main()
