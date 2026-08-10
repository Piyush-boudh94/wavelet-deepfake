"""Slice-level evaluation: break down aggregate AUC by video source.

Gate 3 requires quality checked on slices, not just aggregate. This script
identifies which sub-populations of the self_created dataset the model
performs well or poorly on.

READ-ONLY: does not modify any model weights or training code.
"""
import sys
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training.train import load_config
from src.models.wmamba import build_wmamba
from src.eval.protocols import _score_dataset
from src.data.datasets import VideoEvalDataset
from src.eval.metrics import frame_scores_to_video_scores, video_auc


def classify_source(video_id: str) -> str:
    """Infer the original dataset source from the video_id naming convention."""
    vid = video_id.lower()
    if "dfdc" in vid:
        return "dfdc"
    elif "ffpp" in vid or "ff++" in vid:
        return "ffpp"
    elif "lavdf" in vid or "lav" in vid:
        return "lavdf"
    else:
        return "other"


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

    # Build per-video scores and labels
    label_of = {}
    for l, v in zip(labels, video_ids):
        label_of[v] = int(l)

    vids, vscores = frame_scores_to_video_scores(scores, video_ids)
    vlabels = np.array([label_of[v] for v in vids])

    # Group by source
    source_groups: dict[str, dict] = defaultdict(lambda: {"scores": [], "labels": [], "vids": []})
    for vid, vs, vl in zip(vids, vscores, vlabels):
        src = classify_source(vid)
        source_groups[src]["scores"].append(vs)
        source_groups[src]["labels"].append(vl)
        source_groups[src]["vids"].append(vid)

    # Compute per-slice AUC
    print("=" * 60)
    print("SLICE-LEVEL EVALUATION RESULTS")
    print("=" * 60)

    results = {}
    for src in sorted(source_groups.keys()):
        g = source_groups[src]
        s = np.array(g["scores"])
        l = np.array(g["labels"])
        n_real = int((l == 0).sum())
        n_fake = int((l == 1).sum())
        total = len(l)

        if len(np.unique(l)) < 2:
            auc_val = "N/A (single class)"
            print(f"\n[{src.upper()}] {total} videos (real={n_real}, fake={n_fake}) -- AUC: {auc_val}")
        else:
            from sklearn.metrics import roc_auc_score
            auc_val = float(roc_auc_score(l, s))
            print(f"\n[{src.upper()}] {total} videos (real={n_real}, fake={n_fake}) -- AUC: {auc_val:.5f}")

            # Per-class accuracy at threshold 0.5
            preds = (s >= 0.5).astype(int)
            tp = int(((preds == 1) & (l == 1)).sum())
            tn = int(((preds == 0) & (l == 0)).sum())
            fp = int(((preds == 1) & (l == 0)).sum())
            fn = int(((preds == 0) & (l == 1)).sum())
            acc = (tp + tn) / total * 100
            print(f"  Accuracy@0.5: {acc:.1f}%  TP={tp} TN={tn} FP={fp} FN={fn}")

        results[src] = {
            "total": total,
            "real": n_real,
            "fake": n_fake,
            "auc": auc_val if isinstance(auc_val, str) else round(auc_val, 5)
        }

    # Overall
    from sklearn.metrics import roc_auc_score
    overall_auc = float(roc_auc_score(vlabels, vscores))
    print(f"\n{'=' * 60}")
    print(f"OVERALL: {len(vids)} videos -- AUC: {overall_auc:.5f}")
    print(f"{'=' * 60}")

    results["overall"] = {"total": len(vids), "auc": round(overall_auc, 5)}

    with open("slice_eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nWrote detailed results to slice_eval_results.json")


if __name__ == "__main__":
    main()
