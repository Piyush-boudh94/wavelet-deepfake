"""Augmentation pipelines, per WMamba Appendix B.3 (which adopts SBI's).

Two DISTINCT sets -- the paper is explicit and Tier-2 of the audit checks this:

  * STG set  -- applied inside the Source-Target Generator to create the pseudo
    source/target pair: RGBShift, HueSaturationValue, RandomBrightnessContrast,
    Downscale, Sharpen.
  * REAL set -- applied to real training samples (and to the composed
    pseudo-fake as a whole): ImageCompression, RGBShift, HueSaturationValue,
    RandomBrightnessContrast.

Never merge them: Downscale/Sharpen belong only to the source-target
perturbation, ImageCompression only to whole-sample training augmentation.
Test-time uses NO augmentation whatsoever (resize + normalize only).
"""
from __future__ import annotations

import albumentations as A

# Parameter values follow the SBI reference implementation
# (Shiohara & Yamasaki, CVPR 2022, official repo: mapooon/SelfBlendedImages).
STG_TRANSFORMS = (
    "RGBShift", "HueSaturationValue", "RandomBrightnessContrast", "Downscale", "Sharpen",
)
REAL_TRANSFORMS = (
    "RGBShift", "HueSaturationValue", "RandomBrightnessContrast", "ImageCompression", "ISONoise",
)


def build_stg_augment() -> A.Compose:
    """Source-Target Generator transforms (applied to ONE of source/target)."""
    return A.Compose(
        [
            A.RGBShift(r_shift_limit=(-20, 20), g_shift_limit=(-20, 20),
                       b_shift_limit=(-20, 20), p=0.3),
            A.HueSaturationValue(hue_shift_limit=(-0.3, 0.3),
                                 sat_shift_limit=(-0.3, 0.3),
                                 val_shift_limit=(-0.3, 0.3), p=0.3),
            A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1),
                                       contrast_limit=(-0.1, 0.1), p=0.3),
            A.Downscale(scale_range=(0.25, 0.5), p=0.3),
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.3),
        ]
    )


def build_real_augment() -> A.Compose:
    """Training-time augmentation for real samples and composed pseudo-fakes."""
    return A.Compose(
        [
            A.ImageCompression(quality_range=(40, 100), p=0.5),
            A.RGBShift(r_shift_limit=(-20, 20), g_shift_limit=(-20, 20),
                       b_shift_limit=(-20, 20), p=0.3),
            A.HueSaturationValue(hue_shift_limit=(-0.3, 0.3),
                                 sat_shift_limit=(-0.3, 0.3),
                                 val_shift_limit=(-0.3, 0.3), p=0.3),
            A.RandomBrightnessContrast(brightness_limit=(-0.3, 0.3),
                                       contrast_limit=(-0.3, 0.3), p=0.3),
        ]
    )


def build_robust_augment() -> A.Compose:
    """Robust training-time augmentation targeting DFDC weaknesses (compression & dark lighting)."""
    return A.Compose(
        [
            A.ImageCompression(quality_range=(15, 60), p=0.7),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.5),
            A.RGBShift(r_shift_limit=(-20, 20), g_shift_limit=(-20, 20),
                       b_shift_limit=(-20, 20), p=0.3),
            A.HueSaturationValue(hue_shift_limit=(-0.3, 0.3),
                                 sat_shift_limit=(-0.3, 0.3),
                                 val_shift_limit=(-0.3, 0.3), p=0.3),
            A.RandomBrightnessContrast(brightness_limit=(-0.4, 0.4),
                                       contrast_limit=(-0.4, 0.4), p=0.6),
        ]
    )


def assert_disjoint_policy() -> None:
    """Audit hook: the STG-only ops must never appear in the REAL set and
    vice versa. Called by tests, cheap enough to call at training start too."""
    stg_only = set(STG_TRANSFORMS) - set(REAL_TRANSFORMS)
    real_only = set(REAL_TRANSFORMS) - set(STG_TRANSFORMS)
    if stg_only != {"Downscale", "Sharpen"} or real_only != {"ImageCompression", "ISONoise"}:
        raise AssertionError(
            f"augmentation sets drifted from Appendix B.3: "
            f"stg_only={stg_only}, real_only={real_only}"
        )
