"""Evaluation protocols: cross-dataset (Table 1) and cross-manipulation (Table 2).

Each protocol enumerates (name, processed_root) pairs and runs the same
score-everything -> video_auc pipeline. Roots are DISTINCT per dataset and
asserted distinct (Tier-5 audit: a copy-paste path bug otherwise produces two
identical result sets with no error).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.datasets import EVAL_FRAMES_PER_VIDEO, VideoEvalDataset
from .metrics import video_auc

CROSS_DATASET = ("cdf", "dfdc", "dfdcp", "ffiw")
CROSS_MANIPULATION = ("DF", "F2F", "FS", "NT")


def _score_dataset(model, ds: VideoEvalDataset, batch_size: int,
                   device: str = "cuda", num_workers: int = 8):
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    was_training = model.training
    model.eval()
    scores, labels, vids = [], [], []
    with torch.no_grad():
        for imgs, lbls, video_ids in loader:
            p = model.predict_fakeness(imgs.to(device, non_blocking=True))
            scores.append(p.float().cpu().numpy())
            labels.append(lbls.numpy())
            vids.extend(video_ids)
    if was_training:
        model.train()   # Tier-4 audit: re-assert mode after nested eval
    return np.concatenate(scores), np.concatenate(labels), vids


def evaluate_within_dataset(model, processed_root: str | Path, dataset_name: str,
                            batch_size: int, device: str = "cuda",
                            num_workers: int = 8,
                            splits: tuple[str, ...] = ("val", "test"),
                            frames_per_video: int = EVAL_FRAMES_PER_VIDEO,
                            ) -> dict[str, float]:
    """Video-level AUC on a dataset's OWN held-out splits.

    Replaces the cross-dataset protocol for the DFDC and LAV-DF runs: those are
    trained and evaluated inside one dataset, so there is no second dataset to
    generalise to. Results are NOT comparable to the paper's Table 1.
    """
    root = Path(processed_root)
    results: dict[str, float] = {}
    for split in splits:
        ds = VideoEvalDataset(root, dataset_name=dataset_name, split=split,
                              frames_per_video=frames_per_video)
        s, l, v = _score_dataset(model, ds, batch_size, device, num_workers)
        results[split] = video_auc(s, l, v)
    return results


def evaluate_cross_dataset(model, processed_base: str | Path, batch_size: int,
                           device: str = "cuda", num_workers: int = 8,
                           datasets: tuple[str, ...] = CROSS_DATASET) -> dict[str, float]:
    base = Path(processed_base)
    roots = {name: base / name for name in datasets}
    if len(set(roots.values())) != len(roots):
        raise AssertionError(f"duplicate eval roots: {roots}")
    results: dict[str, float] = {}
    for name, root in roots.items():
        ds = VideoEvalDataset(root, dataset_name=name, split="test")
        s, l, v = _score_dataset(model, ds, batch_size, device, num_workers)
        results[name] = video_auc(s, l, v)
    return results


def evaluate_cross_manipulation(model, ffpp_processed: str | Path, batch_size: int,
                                device: str = "cuda", num_workers: int = 8,
                                manipulations: tuple[str, ...] = CROSS_MANIPULATION
                                ) -> dict[str, float]:
    """FF++ test split: real videos vs one manipulation type at a time.

    Preprocessing writes each manipulation's frames under its own manifest
    (dataset name 'ffpp_DF', 'ffpp_F2F', ...) that already includes the real
    test videos, so each manifest is a complete binary eval set.
    """
    base = Path(ffpp_processed)
    results: dict[str, float] = {}
    seen_roots: set[Path] = set()
    for m in manipulations:
        root = base / f"ffpp_{m}"
        if root in seen_roots:
            raise AssertionError(f"duplicate manipulation root: {root}")
        seen_roots.add(root)
        ds = VideoEvalDataset(root, dataset_name=f"ffpp_{m}", split="test")
        s, l, v = _score_dataset(model, ds, batch_size, device, num_workers)
        results[m] = video_auc(s, l, v)
    return results
