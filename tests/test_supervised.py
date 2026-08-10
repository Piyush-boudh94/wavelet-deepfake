"""Supervised (non-SBI) loader + margin re-cropping.

Covers the path DFDC and LAV-DF training uses: real labels off disk, and the
stored-wide-crop -> arbitrary-margin re-crop that keeps [PAPER B.2]'s random
4-20% train margin reachable from a single stored image.
"""
import json

import cv2
import numpy as np
import pytest

from src.data.datasets import (
    IMAGE_SIZE,
    SupervisedTrainDataset,
    VideoEvalDataset,
    recrop_to_margin,
)

STORE = 256
# a face box occupying the middle of a crop stored at 20% margin:
# 0.20 / (1 + 2*0.20) == 0.1428...
BOX = [0.1428, 0.1428, 0.8572, 0.8572]


def _make_dataset(root, dataset="dfdc", n_per_class=3, split="train", boxes=True):
    entries = []
    for label in (0, 1):
        for i in range(n_per_class):
            vid = f"{split}_{label}_{i}"
            d = root / "frames" / vid
            d.mkdir(parents=True)
            frames = []
            for f in range(4):
                img = np.full((STORE, STORE, 3), (label + 1) * 40 + f, np.uint8)
                cv2.imwrite(str(d / f"{f:04d}.png"), img)
                frames.append(f"frames/{vid}/{f:04d}.png")
            e = {"video_id": vid, "label": label, "split": split, "frames": frames}
            if boxes:
                e["boxes"] = [BOX] * 4
            entries.append(e)
    (root / "manifest.json").write_text(json.dumps(
        {"dataset": dataset, "store_margin": 0.20, "store_size": STORE,
         "entries": entries}))
    return root


def test_recrop_tighter_margin_keeps_face_and_resizes():
    img = np.zeros((STORE, STORE, 3), np.uint8)
    img[36:220, 36:220] = 255                      # the "face" region
    out = recrop_to_margin(img, BOX, 0.04, IMAGE_SIZE)
    assert out.shape == (IMAGE_SIZE, IMAGE_SIZE, 3)
    # a 4% margin crop is nearly all face -> overwhelmingly white
    assert out.mean() > 200


def test_recrop_wider_margin_includes_more_background():
    img = np.zeros((STORE, STORE, 3), np.uint8)
    img[36:220, 36:220] = 255
    tight = recrop_to_margin(img, BOX, 0.04, IMAGE_SIZE).mean()
    wide = recrop_to_margin(img, BOX, 0.20, IMAGE_SIZE).mean()
    assert wide < tight, "wider margin must pull in more (black) background"


def test_recrop_rejects_degenerate_box():
    img = np.zeros((STORE, STORE, 3), np.uint8)
    with pytest.raises(ValueError):
        recrop_to_margin(img, [0.5, 0.5, 0.5, 0.5], 0.1)


def test_supervised_train_yields_both_labels(tmp_path):
    ds = SupervisedTrainDataset(_make_dataset(tmp_path), dataset_name="dfdc", seed=0)
    assert len(ds) == 6
    assert ds.class_counts() == {0: 3, 1: 3}
    img, label = ds[0]
    assert img.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert int(label) in (0, 1)
    assert {int(ds[i][1]) for i in range(len(ds))} == {0, 1}


def test_supervised_train_rejects_single_class(tmp_path):
    root = tmp_path / "one"
    root.mkdir()
    _make_dataset(root, n_per_class=2)
    doc = json.loads((root / "manifest.json").read_text())
    doc["entries"] = [e for e in doc["entries"] if e["label"] == 0]
    (root / "manifest.json").write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="both classes"):
        SupervisedTrainDataset(root, dataset_name="dfdc")


def test_supervised_resamples_frames_across_epochs(tmp_path):
    """Re-seeding must change which frame is drawn, else the stored frames
    collapse into one fixed image per video."""
    ds = SupervisedTrainDataset(_make_dataset(tmp_path), dataset_name="dfdc", seed=1)
    first = [ds[0][0].mean().item() for _ in range(20)]
    assert len(set(round(v, 4) for v in first)) > 1


def test_eval_dataset_applies_fixed_margin(tmp_path):
    root = _make_dataset(tmp_path, split="test")
    ds = VideoEvalDataset(root, dataset_name="dfdc", split="test", frames_per_video=4)
    assert len(ds) == 6 * 4
    img, label, vid = ds[0]
    assert img.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert vid.startswith("test_")


def test_eval_dataset_is_deterministic(tmp_path):
    """No augmentation at eval time -- two reads must be bit-identical."""
    root = _make_dataset(tmp_path, split="test")
    ds = VideoEvalDataset(root, dataset_name="dfdc", split="test", frames_per_video=4)
    assert np.array_equal(ds[3][0].numpy(), ds[3][0].numpy())


def test_manifest_without_boxes_still_loads(tmp_path):
    """Backward compatibility with manifests predating stored boxes."""
    ds = SupervisedTrainDataset(_make_dataset(tmp_path, boxes=False),
                                dataset_name="dfdc", seed=0)
    img, _ = ds[0]
    assert img.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
