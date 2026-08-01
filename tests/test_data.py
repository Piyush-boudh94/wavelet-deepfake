"""Data pipeline tests (Tier 2): manifest validation fails loudly, leakage
guard, SBI degeneracy guard, augmentation-set disjointness, crop margins."""
import json

import numpy as np
import pytest

from src.data.augment import assert_disjoint_policy, build_real_augment, build_stg_augment
from src.data.datasets import assert_no_video_overlap, load_manifest
from src.data.faces import EVAL_MARGIN, expand_box, sample_train_margin
from src.data.sbi import SelfBlender, deform_mask, landmarks_to_hull_mask


# ---------------------------------------------------------------- manifests
def _write_manifest(root, doc):
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps(doc))


def _valid_doc(root):
    (root / "frames/v0").mkdir(parents=True, exist_ok=True)
    import cv2
    cv2.imwrite(str(root / "frames/v0/0.png"), np.zeros((8, 8, 3), np.uint8))
    return {
        "dataset": "cdf",
        "entries": [{"video_id": "v0", "label": 1, "split": "test",
                     "frames": ["frames/v0/0.png"]}],
    }


def test_valid_manifest_loads(tmp_path):
    _write_manifest(tmp_path, _valid_doc(tmp_path))
    entries = load_manifest(tmp_path, expected_dataset="cdf")
    assert entries[0]["video_id"] == "v0"


def test_missing_manifest(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_manifest(tmp_path)


def test_invalid_json(tmp_path):
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "manifest.json").write_text("{not json")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_manifest(tmp_path)


def test_dataset_name_mismatch(tmp_path):
    _write_manifest(tmp_path, _valid_doc(tmp_path))
    with pytest.raises(ValueError, match="mismatch"):
        load_manifest(tmp_path, expected_dataset="dfdc")


def test_zero_frames_rejected(tmp_path):
    doc = _valid_doc(tmp_path)
    doc["entries"][0]["frames"] = []
    _write_manifest(tmp_path, doc)
    with pytest.raises(ValueError, match="zero frames"):
        load_manifest(tmp_path)


def test_bad_label_rejected(tmp_path):
    doc = _valid_doc(tmp_path)
    doc["entries"][0]["label"] = 2
    _write_manifest(tmp_path, doc)
    with pytest.raises(ValueError, match="label"):
        load_manifest(tmp_path)


def test_missing_frame_file_rejected(tmp_path):
    doc = _valid_doc(tmp_path)
    doc["entries"][0]["frames"] = ["frames/v0/missing.png"]
    _write_manifest(tmp_path, doc)
    with pytest.raises(FileNotFoundError):
        load_manifest(tmp_path)


def test_duplicate_video_id_rejected(tmp_path):
    doc = _valid_doc(tmp_path)
    doc["entries"].append(dict(doc["entries"][0]))
    _write_manifest(tmp_path, doc)
    with pytest.raises(ValueError, match="duplicate"):
        load_manifest(tmp_path)


# ---------------------------------------------------------------- leakage guard
def test_video_overlap_detected():
    a = [{"video_id": "x"}, {"video_id": "y"}]
    b = [{"video_id": "z"}, {"video_id": "x"}]
    with pytest.raises(AssertionError, match="leakage"):
        assert_no_video_overlap(a, b)
    assert_no_video_overlap(a, [{"video_id": "z"}])   # disjoint -> fine


# ---------------------------------------------------------------- augmentation
def test_augmentation_sets_disjoint_policy():
    assert_disjoint_policy()


def test_stg_and_real_pipelines_build_and_run():
    img = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
    assert build_stg_augment()(image=img)["image"].shape == img.shape
    assert build_real_augment()(image=img)["image"].shape == img.shape


# ---------------------------------------------------------------- crop margins
def test_expand_box_margins():
    box = (10.0, 10.0, 30.0, 30.0)
    x1, y1, x2, y2 = expand_box(box, 0.125, 100, 100)
    # 12.5% of 20 = 2.5 per side; Python banker's rounding: 7.5->8, 32.5->32
    assert (x1, y1, x2, y2) == (8, 8, 32, 32)
    assert EVAL_MARGIN == 0.125


def test_expand_box_clips_to_image():
    x1, y1, x2, y2 = expand_box((0.0, 0.0, 50.0, 50.0), 0.2, 60, 60)
    assert x1 == 0 and y1 == 0 and x2 == 60 and y2 == 60


def test_train_margin_range():
    rng = np.random.default_rng(0)
    ms = [sample_train_margin(rng) for _ in range(500)]
    assert min(ms) >= 0.04 and max(ms) <= 0.20


def test_invalid_box_rejected():
    with pytest.raises(ValueError):
        expand_box((30.0, 10.0, 10.0, 30.0), 0.1, 100, 100)


# ---------------------------------------------------------------- SBI
def _face_landmarks():
    # 81 points roughly on an ellipse (a plausible face hull)
    t = np.linspace(0, 2 * np.pi, 81)
    return np.stack([64 + 30 * np.cos(t), 64 + 40 * np.sin(t)], 1).astype(np.float32)


def test_hull_mask():
    m = landmarks_to_hull_mask(_face_landmarks(), 128, 128)
    assert m.shape == (128, 128) and 0.0 <= m.min() and m.max() <= 1.0
    assert m.sum() > 1000                     # ellipse interior filled


def test_deform_mask_stays_bounded():
    rng = np.random.default_rng(0)
    m = deform_mask(landmarks_to_hull_mask(_face_landmarks(), 128, 128), rng)
    assert m.shape == (128, 128) and m.min() >= 0.0 and m.max() <= 1.0


def test_sbi_generates_distinct_fake():
    rng = np.random.default_rng(0)
    img = (np.random.rand(128, 128, 3) * 255).astype(np.uint8)
    res = SelfBlender().generate(img, _face_landmarks(), rng)
    assert res.fake.shape == img.shape and res.fake.dtype == np.uint8
    assert res.difference >= 0.5              # degeneracy guard held
    assert not np.array_equal(res.fake, img)  # actually different from real


def test_sbi_rejects_degenerate_landmarks():
    rng = np.random.default_rng(0)
    img = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
    collapsed = np.full((81, 2), 32.0, dtype=np.float32)   # all points identical
    with pytest.raises(ValueError, match="degenerate"):
        SelfBlender().generate(img, collapsed, rng)


def test_sbi_rejects_bad_image():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        SelfBlender().generate(np.zeros((64, 64), np.uint8), _face_landmarks(), rng)
