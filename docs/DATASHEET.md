# Datasheet: WMamba Training Data

*Following [Gebru et al., 2021](https://arxiv.org/abs/1803.09010) — "Datasheets for Datasets"*

## Motivation

### Purpose
This dataset was assembled to train and evaluate a face-forgery (deepfake) detection model based on the WMamba architecture. It combines videos from three public research datasets into a single curated collection for supervised binary classification (real vs. fake).

### Creators
Assembled by the project owner from publicly available research datasets. The original datasets were created by their respective research teams (see Sources below).

### Funding
Academic research project; no commercial funding.

## Composition

### Overview

| Split | Real Videos | Fake Videos | Total | Fake % |
|---|---|---|---|---|
| Train | 1,150 | 1,150 | 2,300 | 50.0% |
| Val | ~247 | ~247 | ~494 | ~50.0% |
| Test | ~247 | ~247 | ~494 | ~50.0% |
| **Total** | **~1,644** | **~1,644** | **~3,288** | **~50.0%** |

### Sources

| Source | Description | Original Size | Used |
|---|---|---|---|
| **DFDC** (Deepfake Detection Challenge) | Facebook AI's large-scale deepfake dataset; face-swapped videos using multiple generation methods | 1,600 videos (public subset) | Subset |
| **FF++** (FaceForensics++) | Academic benchmark with 4 manipulation methods (DeepFakes, Face2Face, FaceSwap, NeuralTextures) | 1,000 original + 4,000 manipulated | Subset |
| **LAV-DF** (Large-scale Audio-Visual Deepfake) | Videos with independently forged audio and video tracks | 136,304 videos | Subset |

### Instance Format
Each instance is a video (`.mp4`). During preprocessing, videos are converted to:
- **Face crops:** 256×256 PNG images extracted via RetinaFace detection
- **Metadata:** Bounding box coordinates stored in `manifest.json` for re-cropping at different margins
- **Frames:** 16 per video (training), 32 per video (evaluation), evenly spaced

### Labels
Binary: `0` = Real, `1` = Fake.
- **DFDC:** Labels from official `metadata.json`
- **FF++:** Labels inferred from directory structure (original vs. manipulated)
- **LAV-DF:** `modify_video` flag (visual forgery only; audio-only fakes deliberately excluded)

### Data Quality
- **Integrity audit** (`scripts/audit_datasets.py`): Zero corrupt, empty, orphan, or missing files across all sources
- **Duplicate check:** 2 exact-duplicate pairs found in LAV-DF upstream (same split, same label — harmless)
- **Known issue (DFDC):** Some DFDC "fake" videos may contain audio-only manipulation with untouched visual frames. These are mislabeled for a visual-only detector but unresolvable without an audio flag in the metadata.

## Collection Process

### DFDC
Downloaded via Kaggle API from the DFDC public dataset. Rate-limited to 1 worker with 2-second intervals to avoid HTTP 429 errors. 3,608 videos downloaded successfully; 35 failed (recorded in `download_failures.json`).

### FF++
Sourced from the project owner's prior data collection.

### LAV-DF
Extracted from a 23 GB archive. Verified against official metadata: train 78,703 / dev 31,501 / test 26,100. Subsampled to 30,000 videos for Phase 1 due to compute constraints (~59 hours to preprocess full set on MIG slice).

### self_created Assembly
The project owner assembled the `self_created` dataset from clips across the above sources. The exact selection criteria are defined by `dataset_manifest.csv` files within the `data/self_created/` directory tree.

## Preprocessing

1. **Face Detection:** RetinaFace (PyTorch implementation) detects faces in each frame
2. **Crop & Store:** Faces cropped at 20% margin (widest needed), saved as 256×256 PNG
3. **Box Recording:** Normalized face bounding box stored per frame in manifest for re-cropping
4. **Train Augmentation:** Random 4–20% crop margin (per paper §B.2), plus albumentations (ImageCompression, RGBShift, HueSaturationValue, RandomBrightnessContrast)
5. **Eval:** Fixed 12.5% crop margin, resize to 224×224, normalize only (no augmentation)

## Splits

- **DFDC:** Custom identity-grouped split (`scripts/build_dfdc_splits.py`). Fakes and their source reals are placed in the same split to prevent identity leakage.
- **LAV-DF:** Official train/dev/test splits used as-is.
- **self_created:** Split defined by directory structure within the dataset.

## Uses

### Intended Use
Training and evaluating face-forgery detection models for academic research.

### Non-Recommended Uses
- Training models for creating deepfakes
- Any use that violates the original dataset licenses (DFDC Terms of Use, FF++ license)

## Distribution & Licensing

| Dataset | License |
|---|---|
| DFDC | Facebook Deepfake Detection Challenge Terms of Use |
| FF++ | Academic research use |
| LAV-DF | Academic research use |

The assembled `self_created` dataset inherits the most restrictive license terms from its component sources.

## Maintenance

This dataset is a static snapshot assembled for a specific research project. It is not actively maintained or updated. Future deepfake generation methods will not be represented.
