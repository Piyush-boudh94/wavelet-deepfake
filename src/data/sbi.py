"""Self-Blended Images (SBI) pseudo-fake generation, per WMamba Appendix B.1.

A real face image is passed through:
  * STG (Source-Target Generator): appearance transforms (augment.py STG set)
    plus slight resize/translate, producing a pseudo *source*; the original
    acts as the *target* (randomly swapped).
  * MG (Mask Generator): a blending mask from the dlib 81-point landmarks'
    convex hull, deformed and blurred for diversity.
Blending source into target under the mask yields the pseudo-fake.
The method follows the official SBI reference (Shiohara & Yamasaki, CVPR 2022,
repo mapooon/SelfBlendedImages, MIT license), which the paper adopts wholesale.

Degeneracy guard (audit Tier 2): if the sampled STG parameters produce a
source ~identical to the target AND the elastic-deformed mask is ~empty, the
"fake" would be indistinguishable from real -- label noise. `generate()`
enforces a minimum blended-difference and re-rolls up to `max_retries`,
then fails loudly rather than emitting a mislabeled sample.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .augment import build_stg_augment

_RNG_MAX = 2**32 - 1


def landmarks_to_hull_mask(landmarks: np.ndarray, h: int, w: int) -> np.ndarray:
    """Convex-hull mask (float32 in [0,1], (H,W)) from 81-point landmarks."""
    if landmarks.ndim != 2 or landmarks.shape[1] != 2:
        raise ValueError(f"landmarks must be (N,2), got {landmarks.shape}")
    mask = np.zeros((h, w), dtype=np.uint8)
    hull = cv2.convexHull(landmarks.astype(np.int32))
    cv2.fillConvexPoly(mask, hull, 255)
    return mask.astype(np.float32) / 255.0


def deform_mask(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Mask augmentation per SBI: random piecewise-affine-like jitter via
    elastic displacement, then Gaussian blur with two randomly sized kernels."""
    h, w = mask.shape
    # elastic offset field on a coarse grid
    grid = 8
    dx = rng.uniform(-1, 1, (grid, grid)).astype(np.float32)
    dy = rng.uniform(-1, 1, (grid, grid)).astype(np.float32)
    amp = rng.uniform(0.0, 0.12) * min(h, w)
    dx = cv2.resize(dx, (w, h), interpolation=cv2.INTER_CUBIC) * amp
    dy = cv2.resize(dy, (w, h), interpolation=cv2.INTER_CUBIC) * amp
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    warped = cv2.remap(mask, xs + dx, ys + dy, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    # two-stage blur (SBI): hard blur to erode, then soft blur on the boundary
    k1 = int(rng.integers(2, 8)) * 2 + 1
    k2 = int(rng.integers(2, 8)) * 2 + 1
    blurred = cv2.GaussianBlur(warped, (k1, k1), 0)
    blurred = np.where(blurred > 0.5, blurred, 0.0).astype(np.float32)
    blurred = cv2.GaussianBlur(blurred, (k2, k2), 0)
    return np.clip(blurred, 0.0, 1.0)


def _stg_geometry(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Slight resize + translate of the source (SBI's geometric component) --
    this is what creates the blending-boundary misalignment artifact."""
    h, w = img.shape[:2]
    fx = rng.uniform(0.95, 1.05)
    fy = rng.uniform(0.95, 1.05)
    tx = rng.uniform(-0.03, 0.03) * w
    ty = rng.uniform(-0.015, 0.015) * h
    m = np.float32([[fx, 0, tx + (1 - fx) * w / 2], [0, fy, ty + (1 - fy) * h / 2]])
    return cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REPLICATE)


@dataclass
class SBIResult:
    fake: np.ndarray          # uint8 HWC RGB pseudo-fake
    mask: np.ndarray          # float32 HW blending mask actually used
    difference: float         # mean abs difference in the masked region


class SelfBlender:
    """Callable SBI generator. One instance per DataLoader worker (holds its
    own albumentations pipeline; seed it via worker_init_fn)."""

    def __init__(self, min_difference: float = 0.5, max_retries: int = 5) -> None:
        self.stg = build_stg_augment()
        self.min_difference = float(min_difference)
        self.max_retries = int(max_retries)

    def generate(self, image: np.ndarray, landmarks: np.ndarray,
                 rng: np.random.Generator) -> SBIResult:
        """image: uint8 HWC RGB. landmarks: (81, 2) pixel coords."""
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"expected uint8 HWC RGB image, got {image.dtype} {image.shape}")
        h, w = image.shape[:2]
        base_mask = landmarks_to_hull_mask(landmarks, h, w)
        if base_mask.sum() < 16:
            raise ValueError("degenerate landmark hull: mask nearly empty")

        last_diff = 0.0
        for _ in range(self.max_retries):
            # re-seed albumentations from our rng so workers do not collapse
            self.stg.set_random_seed(int(rng.integers(0, _RNG_MAX)))

            source = image.copy()
            target = image.copy()
            # appearance transforms hit either source or target (SBI coin flip)
            if rng.random() < 0.5:
                source = self.stg(image=source)["image"]
            else:
                target = self.stg(image=target)["image"]
            source = _stg_geometry(source, rng)

            mask = deform_mask(base_mask, rng)
            m3 = mask[..., None]
            blended = (source.astype(np.float32) * m3
                       + target.astype(np.float32) * (1.0 - m3))

            region = mask > 0.01
            if region.any():
                last_diff = float(
                    np.abs(blended[region] - target.astype(np.float32)[region]).mean()
                )
                if last_diff >= self.min_difference:
                    return SBIResult(
                        fake=np.clip(blended, 0, 255).astype(np.uint8),
                        mask=mask,
                        difference=last_diff,
                    )

        raise RuntimeError(
            f"SBI degenerate blend: difference {last_diff:.4f} < "
            f"{self.min_difference} after {self.max_retries} retries -- refusing "
            "to emit a 'fake' indistinguishable from real (label noise)."
        )
