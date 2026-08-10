"""Measure TRAINING THROUGHPUT (steps/s), not just peak memory.

benchmark_batch.py answered "does batch 64 fit?" (it does, 13.08 GiB of 16.0).
This answers the different question the wall-clock estimate depends on: how
many full train steps per second does the MIG slice actually sustain, and is
the limit compute or VRAM?

Synthetic tensors on purpose -- this isolates GPU compute from disk/dataloader
so the two can be attributed separately.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.wmamba import WMamba  # noqa: E402


def bench(model, batch: int, steps: int, device: str = "cuda") -> dict:
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.05)
    x = torch.randn(batch, 3, 224, 224, device=device)
    y = torch.randint(0, 2, (batch,), device=device)
    lossf = torch.nn.CrossEntropyLoss()

    for _ in range(3):                      # warmup: cuDNN autotune + alloc
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = lossf(model(x), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    for _ in range(steps):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = lossf(model(x), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    return {
        "batch": batch,
        "steps_per_s": steps / dt,
        "images_per_s": steps * batch / dt,
        "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, nargs="*", default=[64])
    ap.add_argument("--steps", type=int, default=20)
    args = ap.parse_args()

    torch.backends.cudnn.benchmark = True
    # random init: throughput is independent of weight values, and this keeps
    # the benchmark runnable without the pretrained checkpoint
    model = WMamba().cuda().train()
    p = torch.cuda.get_device_properties(0)
    print(f"{p.name}  {p.total_memory / 2**30:.2f} GiB  {p.multi_processor_count} SMs\n")
    print(f"{'batch':>6} {'steps/s':>9} {'images/s':>10} {'peak GiB':>9}  {'% of 16':>8}")

    rows = []
    for b in args.batches:
        try:
            r = bench(model, b, args.steps)
        except torch.cuda.OutOfMemoryError:
            print(f"{b:>6}  OOM")
            torch.cuda.empty_cache()
            continue
        rows.append(r)
        print(f"{r['batch']:>6} {r['steps_per_s']:>9.2f} {r['images_per_s']:>10.1f} "
              f"{r['peak_gib']:>9.2f} {100 * r['peak_gib'] / 16.0:>7.1f}%")
        torch.cuda.empty_cache()

    if rows:
        best = max(rows, key=lambda r: r["images_per_s"])
        ips = best["images_per_s"]
        print(f"\nbest: {ips:.1f} images/s at batch {best['batch']}")
        for name, n_videos, frames in (("LAV-DF subsample", 20000, 8),
                                       ("LAV-DF full", 78703, 8),
                                       ("DFDC current", 1120, 8)):
            imgs = n_videos * frames
            ep = imgs / ips
            print(f"  {name:18s} {imgs:>8,} img/epoch  {ep / 60:6.1f} min/epoch  "
                  f"{200 * ep / 3600:7.1f} h for 200 epochs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
