"""Benchmark inference latency (Gate 11: Cost, Latency & Scalability).

Measures p50/p95/p99 inference latency on pre-cropped face images
to establish an SLA baseline.

READ-ONLY: does not modify any model weights.
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training.train import load_config
from src.models.wmamba import build_wmamba
from src.data.datasets import normalize_tensor, IMAGE_SIZE


def main():
    cfg = load_config("configs/self_created.yaml")
    device = "cuda"

    model = build_wmamba(cfg).to(device)
    model.load_state_dict(load_file("checkpoints/self_created/best/model.safetensors"), strict=True)
    model.eval()

    # Generate synthetic face crops for benchmarking
    dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)

    # Warmup
    print("Warming up GPU...")
    for _ in range(20):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            _ = model(dummy)
    torch.cuda.synchronize()

    batch_sizes = [1, 4, 8, 16, 32, 64]
    n_iterations = 100

    print("=" * 70)
    print("INFERENCE LATENCY BENCHMARK")
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"Model: WMamba (self_created/best)")
    print(f"Iterations per batch size: {n_iterations}")
    print("=" * 70)

    for bs in batch_sizes:
        batch = torch.randn(bs, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
        latencies = []

        for _ in range(n_iterations):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                _ = model(batch)
            torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000)

        latencies = np.array(latencies)
        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)
        throughput = bs / (p50 / 1000)

        print(f"\nBatch {bs:>3}:")
        print(f"  p50: {p50:>8.2f} ms  |  p95: {p95:>8.2f} ms  |  p99: {p99:>8.2f} ms")
        print(f"  Throughput (at p50): {throughput:>8.1f} images/sec")

    # Single-video end-to-end estimate
    print(f"\n{'=' * 70}")
    print("ESTIMATED END-TO-END VIDEO INFERENCE")
    print("(32 frames × 1 face per frame = 32 crops, batch size 32)")
    batch = torch.randn(32, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
    latencies = []
    for _ in range(n_iterations):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            _ = model(batch)
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000)

    latencies = np.array(latencies)
    print(f"  Model inference only: p50={np.percentile(latencies, 50):.1f} ms, "
          f"p95={np.percentile(latencies, 95):.1f} ms")
    print(f"  + RetinaFace + video decode: estimated ~2-5x overhead")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
