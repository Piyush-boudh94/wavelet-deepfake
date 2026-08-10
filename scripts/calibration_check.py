"""Probability calibration check: reliability diagram + ECE.

Gate 13 (Advanced) requires probability calibration verification.
This script checks whether the model's predicted probabilities correspond
to actual observed frequencies. If not, it recommends Platt scaling.

READ-ONLY: does not modify any model weights or training code.
"""
import sys
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training.train import load_config
from src.models.wmamba import build_wmamba
from src.eval.protocols import _score_dataset
from src.data.datasets import VideoEvalDataset
from src.eval.metrics import frame_scores_to_video_scores


def expected_calibration_error(y_true, y_prob, n_bins=10):
    """Compute Expected Calibration Error (ECE)."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bins_data = []
    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if i == n_bins - 1:  # include right edge for last bin
            mask = mask | (y_prob == bin_edges[i + 1])
        if mask.sum() == 0:
            bins_data.append({
                "bin": f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}",
                "count": 0,
                "avg_confidence": 0,
                "avg_accuracy": 0,
                "gap": 0
            })
            continue
        avg_conf = y_prob[mask].mean()
        avg_acc = y_true[mask].mean()
        gap = abs(avg_acc - avg_conf)
        ece += mask.sum() * gap
        bins_data.append({
            "bin": f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}",
            "count": int(mask.sum()),
            "avg_confidence": round(float(avg_conf), 4),
            "avg_accuracy": round(float(avg_acc), 4),
            "gap": round(float(gap), 4)
        })
    ece /= len(y_true)
    return float(ece), bins_data


def main():
    cfg = load_config("configs/self_created.yaml")
    device = "cuda"

    model = build_wmamba(cfg).to(device)
    model.load_state_dict(load_file("checkpoints/self_created/best/model.safetensors"), strict=True)
    model.eval()

    ds = VideoEvalDataset(
        processed_root=Path(cfg.paths.processed) / "self_created",
        dataset_name="self_created",
        split="test",
        frames_per_video=32
    )

    scores, labels, video_ids = _score_dataset(model, ds, batch_size=64, device=device, num_workers=8)

    label_of = {}
    for l, v in zip(labels, video_ids):
        label_of[v] = int(l)

    vids, vscores = frame_scores_to_video_scores(scores, video_ids)
    vlabels = np.array([label_of[v] for v in vids])

    # Clip scores to [0, 1] for calibration analysis
    vscores_clipped = np.clip(vscores, 0, 1)

    ece, bins = expected_calibration_error(vlabels, vscores_clipped, n_bins=10)

    print("=" * 60)
    print("PROBABILITY CALIBRATION REPORT")
    print("=" * 60)
    print(f"\nExpected Calibration Error (ECE): {ece:.4f}")
    print(f"  (< 0.05 = well calibrated, 0.05-0.15 = moderate, > 0.15 = poor)")
    print()

    print(f"{'Bin':>12} {'Count':>6} {'Avg Conf':>10} {'Avg Acc':>10} {'Gap':>8}")
    print("-" * 50)
    for b in bins:
        if b["count"] > 0:
            print(f"{b['bin']:>12} {b['count']:>6} {b['avg_confidence']:>10.4f} "
                  f"{b['avg_accuracy']:>10.4f} {b['gap']:>8.4f}")

    # Score distribution
    print(f"\n{'=' * 60}")
    print("SCORE DISTRIBUTION")
    print(f"{'=' * 60}")
    real_scores = vscores_clipped[vlabels == 0]
    fake_scores = vscores_clipped[vlabels == 1]
    print(f"Real videos: mean={real_scores.mean():.4f}, std={real_scores.std():.4f}, "
          f"median={np.median(real_scores):.4f}")
    print(f"Fake videos: mean={fake_scores.mean():.4f}, std={fake_scores.std():.4f}, "
          f"median={np.median(fake_scores):.4f}")
    print(f"Separation: {fake_scores.mean() - real_scores.mean():.4f}")

    # Recommendation
    print(f"\n{'=' * 60}")
    if ece < 0.05:
        verdict = "WELL CALIBRATED — no post-hoc correction needed"
    elif ece < 0.15:
        verdict = "MODERATELY CALIBRATED — Platt scaling recommended for threshold-based decisions"
    else:
        verdict = "POORLY CALIBRATED — Platt scaling or isotonic regression strongly recommended"
    print(f"VERDICT: {verdict}")
    print(f"{'=' * 60}")

    results = {
        "ece": round(ece, 5),
        "bins": bins,
        "real_score_mean": round(float(real_scores.mean()), 5),
        "real_score_std": round(float(real_scores.std()), 5),
        "fake_score_mean": round(float(fake_scores.mean()), 5),
        "fake_score_std": round(float(fake_scores.std()), 5),
        "verdict": verdict
    }
    with open("calibration_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nWrote results to calibration_results.json")


if __name__ == "__main__":
    main()
