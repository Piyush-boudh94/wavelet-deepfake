import sys
import json
import torch
import numpy as np
from pathlib import Path
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training.train import load_config
from src.models.wmamba import build_wmamba
from src.eval.protocols import _score_dataset
from src.data.datasets import VideoEvalDataset
from src.eval.metrics import frame_scores_to_video_scores

def main():
    cfg = load_config("configs/self_created.yaml")
    device = "cuda"
    
    model = build_wmamba(cfg).to(device)
    ckpt_path = "checkpoints/self_created/best/model.safetensors"
    model.load_state_dict(load_file(ckpt_path), strict=True)
    model.eval()

    ds = VideoEvalDataset(
        processed_root=Path(cfg.paths.processed) / "self_created",
        dataset_name="self_created",
        split="test",
        frames_per_video=32
    )
    
    s, l, v = _score_dataset(model, ds, batch_size=64, device=device, num_workers=8)
    
    label_of = {}
    for label, vid in zip(l, v):
        label_of[vid] = int(label)
        
    vids, vscores = frame_scores_to_video_scores(s, v)
    
    results = []
    for vid, score in zip(vids, vscores):
        results.append({
            "video_id": vid,
            "true_label": label_of[vid],
            "pred_score": float(score)
        })
        
    # Sort by how "wrong" they are
    # False Positives: true_label == 0, pred_score is high
    fps = sorted([r for r in results if r["true_label"] == 0], key=lambda x: x["pred_score"], reverse=True)
    
    # False Negatives: true_label == 1, pred_score is low
    fns = sorted([r for r in results if r["true_label"] == 1], key=lambda x: x["pred_score"])
    
    out = {
        "top_false_positives": fps[:20],
        "top_false_negatives": fns[:20]
    }
    
    with open("error_analysis.json", "w") as f:
        json.dump(out, f, indent=2)
        
    print("Wrote top errors to error_analysis.json")

if __name__ == "__main__":
    main()
