"""Datasets and loaders.

All datasets are driven by a *manifest* (manifest.json) that the preprocessing
step writes next to the extracted frames:

    data/processed/<dataset>/manifest.json
    {
      "dataset": "ffpp",
      "entries": [
        {"video_id": "000", "label": 0, "split": "train",
         "frames": ["frames/000/0001.png", ...],
         "landmarks": ["landmarks/000/0001.npy", ...]},   # train/real only
        ...
      ]
    }

Design rules enforced here (audit Tier 2):
  * Splits operate at VIDEO level. A frame can never cross splits because the
    unit of membership is the manifest entry (one entry = one video).
  * Malformed manifests, missing files, unreadable images, empty videos all
    raise immediately with the offending path -- nothing is skipped silently.
  * Normalization constants are defined ONCE here and imported everywhere.
  * Train-time SBI runs on the fly: each real video yields a (real, fake) pair
    per draw, exactly the SBI training scheme the paper adopts.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .augment import build_real_augment, build_robust_augment
from .sbi import SelfBlender

# Single source of truth for normalization (ImageNet, matching the pretrained
# VMamba backbone). Tier-2 audit: never redefine these elsewhere.
NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD = (0.229, 0.224, 0.225)
IMAGE_SIZE = 224           # [PAPER B.4]
TRAIN_FRAMES_PER_VIDEO = 8   # [PAPER B.4]
EVAL_FRAMES_PER_VIDEO = 32   # [PAPER B.4]


def normalize_tensor(img_uint8_rgb: np.ndarray) -> torch.Tensor:
    """uint8 HWC RGB -> float32 CHW normalized."""
    t = torch.from_numpy(np.ascontiguousarray(img_uint8_rgb)).float().div_(255.0)
    t = t.permute(2, 0, 1)
    mean = torch.tensor(NORM_MEAN).view(3, 1, 1)
    std = torch.tensor(NORM_STD).view(3, 1, 1)
    return (t - mean) / std


def _read_image_strict(path: Path) -> np.ndarray:
    """Read RGB image or raise with the path (no silent None from cv2)."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"unreadable/corrupt image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def recrop_to_margin(img: np.ndarray, box_norm: list[float], margin: float,
                     out_size: int = IMAGE_SIZE) -> np.ndarray:
    """Re-crop a stored face crop to an arbitrary margin, then resize.

    Preprocessing saves each face at the WIDEST margin the loader may ask for
    (STORE_MARGIN) and records the face box in saved-crop coordinates. That is
    what makes [PAPER B.2]'s random 4-20% train margin and fixed 12.5% eval
    margin reachable from one stored image -- baking a single margin in at
    preprocessing time would have destroyed the augmentation.

    box_norm is (x1, y1, x2, y2) normalised to the saved crop.
    """
    h, w = img.shape[:2]
    x1, y1, x2, y2 = (box_norm[0] * w, box_norm[1] * h, box_norm[2] * w, box_norm[3] * h)
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        raise ValueError(f"degenerate stored box {box_norm}")
    cx1 = max(int(round(x1 - margin * bw)), 0)
    cy1 = max(int(round(y1 - margin * bh)), 0)
    cx2 = min(int(round(x2 + margin * bw)), w)
    cy2 = min(int(round(y2 + margin * bh)), h)
    if cx2 <= cx1 or cy2 <= cy1:
        raise ValueError(f"re-crop collapsed: box={box_norm} margin={margin}")
    crop = img[cy1:cy2, cx1:cx2]
    return cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_LINEAR)


def load_manifest(root: str | Path, expected_dataset: str | None = None) -> list[dict]:
    """Load and validate a manifest. Fails loudly on every malformation."""
    root = Path(root)
    mf = root / "manifest.json"
    if not mf.exists():
        raise FileNotFoundError(f"manifest missing: {mf}")
    try:
        doc = json.loads(mf.read_text())
    except json.JSONDecodeError as e:
        raise ValueError(f"manifest is not valid JSON: {mf}: {e}") from e
    if not isinstance(doc, dict) or "entries" not in doc:
        raise ValueError(f"manifest lacks 'entries': {mf}")
    if expected_dataset and doc.get("dataset") != expected_dataset:
        raise ValueError(
            f"manifest dataset mismatch: expected {expected_dataset!r}, "
            f"got {doc.get('dataset')!r} in {mf}"
        )
    entries = doc["entries"]
    if not entries:
        raise ValueError(f"manifest has zero entries: {mf}")
    seen_ids: set[str] = set()
    for i, e in enumerate(entries):
        for key in ("video_id", "label", "split", "frames"):
            if key not in e:
                raise ValueError(f"manifest entry {i} missing {key!r}: {mf}")
        if e["label"] not in (0, 1):
            raise ValueError(f"entry {e['video_id']}: label must be 0/1, got {e['label']}")
        if not e["frames"]:
            raise ValueError(f"entry {e['video_id']}: zero frames (truncated video?)")
        if e["video_id"] in seen_ids:
            raise ValueError(f"duplicate video_id {e['video_id']} in {mf}")
        seen_ids.add(e["video_id"])
        first = root / e["frames"][0]
        if not first.exists():
            raise FileNotFoundError(f"entry {e['video_id']}: frame missing: {first}")
    return entries


def assert_no_video_overlap(*entry_lists: list[dict]) -> None:
    """Tier-2 leakage guard: no video_id may appear in more than one split/set."""
    seen: dict[str, int] = {}
    for li, entries in enumerate(entry_lists):
        for e in entries:
            vid = e["video_id"]
            if vid in seen and seen[vid] != li:
                raise AssertionError(
                    f"video-level leakage: {vid!r} appears in sets {seen[vid]} and {li}"
                )
            seen[vid] = li


class SBITrainDataset(Dataset):
    """FF++ REAL videos only; SBI generates the fake of each pair on the fly.

    __getitem__(i) -> (images (2,3,H,W), labels (2,)) : [real, fake].
    The trainer flattens pairs into the batch. Every epoch re-samples frames
    and re-rolls SBI, so effective fakes never repeat.
    """

    def __init__(self, processed_root: str | Path, landmark_model_path: str | Path,
                 image_size: int = IMAGE_SIZE, seed: int = 0,
                 augmentation_profile: str = "default") -> None:
        self.root = Path(processed_root)
        entries = load_manifest(self.root, expected_dataset="ffpp")
        self.entries = [e for e in entries if e["split"] == "train" and e["label"] == 0]
        if not self.entries:
            raise ValueError(f"no real train videos in {self.root} manifest")
        for e in self.entries:
            if not e.get("landmarks"):
                raise ValueError(
                    f"train entry {e['video_id']} lacks landmarks -- "
                    "SBI cannot build masks; re-run preprocessing"
                )
        self.image_size = image_size
        self.seed = seed
        self.blender = SelfBlender()
        self.real_aug = build_robust_augment() if augmentation_profile == "robust" else build_real_augment()
        # per-worker rng; re-seeded by worker_init_fn with worker id mixed in
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.entries)

    def reseed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        self.real_aug.set_random_seed(seed)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        e = self.entries[idx]
        fi = int(self.rng.integers(0, len(e["frames"])))
        img = _read_image_strict(self.root / e["frames"][fi])
        lms = np.load(self.root / e["landmarks"][fi])
        if lms.shape != (81, 2):
            raise ValueError(
                f"{e['video_id']} frame {fi}: landmarks shape {lms.shape}, want (81,2)"
            )

        fake = self.blender.generate(img, lms, self.rng).fake
        # whole-sample augmentation (REAL set) on each, independently
        self.real_aug.set_random_seed(int(self.rng.integers(0, 2**32 - 1)))
        real_a = self.real_aug(image=img)["image"]
        self.real_aug.set_random_seed(int(self.rng.integers(0, 2**32 - 1)))
        fake_a = self.real_aug(image=fake)["image"]

        s = self.image_size
        real_a = cv2.resize(real_a, (s, s), interpolation=cv2.INTER_LINEAR)
        fake_a = cv2.resize(fake_a, (s, s), interpolation=cv2.INTER_LINEAR)
        images = torch.stack([normalize_tensor(real_a), normalize_tensor(fake_a)])
        labels = torch.tensor([0, 1], dtype=torch.long)
        return images, labels


class SupervisedTrainDataset(Dataset):
    """Real/fake training on a dataset's OWN labels -- no SBI.

    Used for DFDC and LAV-DF, which ship genuine labelled fakes. SBI's
    real-only scheme would discard all of them (see configs/*.yaml headers).

    __getitem__(i) -> (image (3,H,W), label ()) : one video, one random frame,
    a fresh random [PAPER B.2] 4-20% crop margin, and the REAL augmentation set.
    Re-sampling every epoch is what keeps 8-16 stored frames from collapsing
    into 8-16 fixed training images.

    VIDEO-only: frames are decoded from the video stream and nothing in this
    pipeline ever opens an audio stream.
    """

    def __init__(self, processed_root: str | Path, dataset_name: str,
                 split: str = "train", image_size: int = IMAGE_SIZE,
                 margin_range: tuple[float, float] = (0.04, 0.20),
                 seed: int = 0, augmentation_profile: str = "default",
                 num_frames: int = 1) -> None:
        self.root = Path(processed_root)
        entries = load_manifest(self.root, expected_dataset=dataset_name)
        self.entries = [e for e in entries if e["split"] == split]
        if not self.entries:
            raise ValueError(f"no {split!r} entries in {self.root} manifest")
        labels = {e["label"] for e in self.entries}
        if labels != {0, 1}:
            raise ValueError(
                f"{split!r} split has labels {sorted(labels)}; supervised "
                "training needs both classes present"
            )
        self.image_size = image_size
        self.margin_range = margin_range
        self.real_aug = build_robust_augment() if augmentation_profile == "robust" else build_real_augment()
        self.rng = np.random.default_rng(seed)
        self.num_frames = num_frames

    def __len__(self) -> int:
        return len(self.entries)

    def reseed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        self.real_aug.set_random_seed(seed)

    def class_counts(self) -> dict[int, int]:
        out = {0: 0, 1: 0}
        for e in self.entries:
            out[e["label"]] += 1
        return out

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        e = self.entries[idx]
        total_frames = len(e["frames"])
        
        # Select frames (random contiguous or uniform subset for sequence)
        if self.num_frames > 1:
            k = min(self.num_frames, total_frames)
            picks = np.linspace(0, total_frames - 1, num=k).round().astype(int)
            # Pad if we don't have enough frames (rare, but safe)
            if len(picks) < self.num_frames:
                picks = np.pad(picks, (0, self.num_frames - len(picks)), mode='edge')
        else:
            picks = [int(self.rng.integers(0, total_frames))]
            
        margin = float(self.rng.uniform(*self.margin_range))
        boxes = e.get("boxes")
        aug_seed = int(self.rng.integers(0, 2**32 - 1))
        
        tensors = []
        for fi in picks:
            img = _read_image_strict(self.root / e["frames"][fi])

            if boxes:
                img = recrop_to_margin(img, boxes[fi], margin, self.image_size)
            else:
                img = cv2.resize(img, (self.image_size, self.image_size),
                                 interpolation=cv2.INTER_LINEAR)

            self.real_aug.set_random_seed(aug_seed) # same seed per frame in sequence
            img = self.real_aug(image=img)["image"]
            if img.shape[:2] != (self.image_size, self.image_size):
                img = cv2.resize(img, (self.image_size, self.image_size),
                                 interpolation=cv2.INTER_LINEAR)
            tensors.append(normalize_tensor(img))
            
        if self.num_frames == 1:
            return tensors[0], torch.tensor(e["label"], dtype=torch.long)
        else:
            return torch.stack(tensors), torch.tensor(e["label"], dtype=torch.long)


class VideoEvalDataset(Dataset):
    """Frame-level eval samples carrying video ids for video-level AUC.

    NO augmentation -- resize + normalize only (Tier-2/5 audit).
    """

    def __init__(self, processed_root: str | Path, dataset_name: str,
                 split: str = "test", image_size: int = IMAGE_SIZE,
                 frames_per_video: int = EVAL_FRAMES_PER_VIDEO,
                 eval_margin: float = 0.125,
                 return_sequences: bool = False) -> None:   # [PAPER B.2]
        self.root = Path(processed_root)
        entries = load_manifest(self.root, expected_dataset=dataset_name)
        entries = [e for e in entries if e["split"] == split]
        if not entries:
            raise ValueError(f"no {split!r} entries in {self.root} manifest")
        self.image_size = image_size
        self.eval_margin = eval_margin
        self.return_sequences = return_sequences
        # if return_sequences: (picks, boxes, label, video_id)
        # else: (frame_path, box_or_None, label, video_id)
        self.index = []
        self.video_labels: dict[str, int] = {}
        for e in entries:
            frames = e["frames"]
            boxes = e.get("boxes")
            # deterministic, evenly spaced selection of <= frames_per_video
            k = min(frames_per_video, len(frames))
            picks = np.linspace(0, len(frames) - 1, num=k).round().astype(int)
            picks = sorted(set(picks.tolist()))
            self.video_labels[e["video_id"]] = int(e["label"])
            
            if self.return_sequences:
                seq_frames = [frames[p] for p in picks]
                seq_boxes = [boxes[p] if boxes else None for p in picks]
                self.index.append((seq_frames, seq_boxes, int(e["label"]), e["video_id"]))
            else:
                for p in picks:
                    self.index.append((frames[p], boxes[p] if boxes else None,
                                       int(e["label"]), e["video_id"]))

    def __len__(self) -> int:
        return len(self.index)

    def _load_frame(self, rel: str, box: list[float] | None) -> torch.Tensor:
        img = _read_image_strict(self.root / rel)
        if box is not None:
            img = recrop_to_margin(img, box, self.eval_margin, self.image_size)
        else:
            img = cv2.resize(img, (self.image_size, self.image_size),
                             interpolation=cv2.INTER_LINEAR)
        return normalize_tensor(img)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int, str]:
        if self.return_sequences:
            frames, boxes, label, vid = self.index[i]
            tensors = [self._load_frame(f, b) for f, b in zip(frames, boxes)]
            return torch.stack(tensors), label, vid
        else:
            rel, box, label, vid = self.index[i]
            return self._load_frame(rel, box), label, vid


class SelfCreatedDataset(VideoEvalDataset):
    """Phase-2 loader for data/self_created. Schema-flexible on purpose: it
    accepts any manifest satisfying load_manifest()'s contract, and nothing
    more specific until the actual data format is known. It inherits the
    strict validation, so malformed input still fails loudly."""

    def __init__(self, processed_root: str | Path, **kw) -> None:
        super().__init__(processed_root, dataset_name="self_created", **kw)
