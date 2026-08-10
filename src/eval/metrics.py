"""Evaluation metrics, per WMamba Sec. 4.1 / Appendix B.4.

Video-level AUC protocol (the paper's number, and the only one we report):
  1. score every sampled frame with P(fake);
  2. average frame scores PER VIDEO;
  3. compute ROC-AUC over the per-video scores.

Never AUC over raw frames, and never per-batch AUC (a single-class batch makes
AUC undefined and sklearn would either crash or silently emit garbage --
Tier-5 audit). Aggregation happens over the entire split, then one AUC call.

Multi-face frames: preprocessing stores one crop per detected face and scoring
takes the MAX fakeness among faces of a frame (Appendix B.4) before the
per-video average; `video_auc` receives per-crop scores plus (video_id,
frame_id) and applies max-then-mean here so the rule lives in exactly one place.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score


def frame_scores_to_video_scores(
    scores: np.ndarray,
    video_ids: list[str],
    frame_ids: list[str] | None = None,
) -> tuple[list[str], np.ndarray]:
    """Aggregate per-crop scores -> per-video scores.

    If frame_ids is given, crops of the same (video, frame) are first reduced
    by MAX (multi-face rule), then frames are averaged per video.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or len(scores) != len(video_ids):
        raise ValueError(
            f"scores ({scores.shape}) and video_ids ({len(video_ids)}) disagree"
        )
    if np.isnan(scores).any():
        raise ValueError("NaN in scores -- refusing to aggregate")

    if frame_ids is not None:
        if len(frame_ids) != len(scores):
            raise ValueError("frame_ids length mismatch")
        per_frame: dict[tuple[str, str], float] = {}
        for s, v, f in zip(scores, video_ids, frame_ids):
            key = (v, f)
            per_frame[key] = max(per_frame.get(key, -np.inf), float(s))
        agg: dict[str, list[float]] = defaultdict(list)
        for (v, _), s in per_frame.items():
            agg[v].append(s)
    else:
        agg = defaultdict(list)
        for s, v in zip(scores, video_ids):
            agg[v].append(float(s))

    vids = sorted(agg)
    return vids, np.array([np.mean(agg[v]) for v in vids])


def video_auc(
    scores: np.ndarray,
    labels: np.ndarray,
    video_ids: list[str],
    frame_ids: list[str] | None = None,
) -> float:
    """The paper's metric. labels are per-crop but must be constant per video."""
    labels = np.asarray(labels)
    if len(labels) != len(scores):
        raise ValueError("labels/scores length mismatch")

    label_of: dict[str, int] = {}
    for l, v in zip(labels, video_ids):
        prev = label_of.setdefault(v, int(l))
        if prev != int(l):
            raise ValueError(f"video {v!r} has inconsistent labels {prev} vs {int(l)}")

    vids, vscores = frame_scores_to_video_scores(scores, video_ids, frame_ids)
    vlabels = np.array([label_of[v] for v in vids])
    if len(np.unique(vlabels)) < 2:
        raise ValueError(
            "video-level AUC undefined: split contains a single class "
            f"({len(vids)} videos, labels all {vlabels[0]})"
        )
    return float(roc_auc_score(vlabels, vscores))
