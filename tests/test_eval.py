"""Video-level AUC tests (Tier 5): per-video averaging, multi-face max rule,
single-class refusal, label consistency."""
import numpy as np
import pytest

from src.eval.metrics import frame_scores_to_video_scores, video_auc


def test_video_averaging():
    scores = np.array([0.2, 0.4, 0.9, 0.7])
    vids = ["a", "a", "b", "b"]
    v, s = frame_scores_to_video_scores(scores, vids)
    assert v == ["a", "b"]
    assert np.allclose(s, [0.3, 0.8])


def test_video_auc_matches_manual():
    # video a: real, mean 0.3 ; video b: fake, mean 0.8 ; video c: real, 0.1
    scores = np.array([0.2, 0.4, 0.9, 0.7, 0.1])
    labels = np.array([0, 0, 1, 1, 0])
    vids = ["a", "a", "b", "b", "c"]
    # perfect separation -> AUC 1.0
    assert video_auc(scores, labels, vids) == 1.0


def test_video_level_not_frame_level():
    """Construct a case where frame-AUC and video-AUC disagree; verify we get
    the video-level answer."""
    from sklearn.metrics import roc_auc_score
    # fake video: frames [0.9, 0.1]; real videos: [0.4] and [0.45]
    scores = np.array([0.9, 0.1, 0.4, 0.45])
    labels = np.array([1, 1, 0, 0])
    vids = ["f", "f", "r1", "r2"]
    frame_auc = roc_auc_score(labels, scores)
    v_auc = video_auc(scores, labels, vids)
    assert v_auc == 1.0                # 0.5 > 0.45 and 0.4 -> perfect
    assert frame_auc != v_auc          # frame-level would be 0.5


def test_multiface_max_rule():
    # one frame with two faces: 0.1 and 0.95 -> frame score must be 0.95
    scores = np.array([0.1, 0.95, 0.2])
    vids = ["v", "v", "v"]
    frames = ["f1", "f1", "f2"]
    _, s = frame_scores_to_video_scores(scores, vids, frames)
    assert np.allclose(s, [(0.95 + 0.2) / 2])


def test_single_class_refused():
    with pytest.raises(ValueError, match="single class"):
        video_auc(np.array([0.5, 0.6]), np.array([1, 1]), ["a", "b"])


def test_inconsistent_video_labels_refused():
    with pytest.raises(ValueError, match="inconsistent"):
        video_auc(np.array([0.5, 0.6]), np.array([0, 1]), ["a", "a"])


def test_nan_scores_refused():
    with pytest.raises(ValueError, match="NaN"):
        frame_scores_to_video_scores(np.array([0.5, np.nan]), ["a", "b"])
