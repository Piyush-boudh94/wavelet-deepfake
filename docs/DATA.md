# Datasets

**Nothing here is auto-downloaded.** FF++, CDF, DFDC, DFDCP and FFIW-10K are gated
academic datasets requiring signed usage agreements with their original authors.
You obtain them; the pipeline starts once they land in `data/raw/`.

## What you actually need (and what you don't)

This matters, because the naive reading is "download five deepfake datasets" — roughly
1.5 TB — and there is only **1.3 TiB free on a shared filesystem**.

The paper trains on **FF++ real videos only** (SBI synthesises the fakes on the fly) and
uses the other four datasets **exclusively for testing**. So:

| Dataset | What's needed | Approx. size | Why |
|---|---|---|---|
| **FF++** | `original_sequences/youtube/c23` — the 720 **real** training videos | ~10 GB | The only training data |
| FF++ | `manipulated_sequences/{DF,F2F,FS,NT}/c23`, **test split only** | ~15 GB | Cross-manipulation eval (Table 2) |
| **CDF-v2** | test split (518 videos) | ~3 GB | Cross-dataset eval |
| **DFDC** | **test set only** (5,000 videos) — *not* the 470 GB train set | ~5 GB | Cross-dataset eval |
| **DFDCP** | test split | ~10 GB | Cross-dataset eval |
| **FFIW-10K** | test split | ~60 GB | Cross-dataset eval |

Total: roughly **100–150 GB**, not 1.5 TB. Do not pull the DFDC training set — it is
never used and it alone would consume a third of the remaining free space.

## Layout expected

```
data/raw/
├── ffpp/
│   ├── original_sequences/youtube/c23/videos/*.mp4     # REAL - training source
│   └── manipulated_sequences/{Deepfakes,Face2Face,FaceSwap,NeuralTextures}/c23/videos/*.mp4
├── cdf/    {Celeb-real,Celeb-synthesis,YouTube-real}/*.mp4 + List_of_testing_videos.txt
├── dfdc/   test videos + labels.csv
├── dfdcp/  videos + dataset.json
└── ffiw/   target/{source,forgery}/*.mp4 + split lists
```

Splits follow the official FF++ train/val/test json lists — these must be the standard
ones, or cross-dataset numbers are not comparable to the paper.

## Transferring onto the cluster

The head node has internet and 40 cores. `~` is NFS and visible from both the head node
and the pod, so stage directly into `data/raw/`:

```bash
rsync -avP --info=progress2 /local/path/ffpp/ \
      dgx-s-bmu-cse-240577@<headnode>:~/research/data/raw/ffpp/
```

Check free space **before** starting a multi-hundred-GB transfer — the filesystem is
shared with other users and was at 77% when this project was set up:

```bash
df -h ~
```

## Preprocessing

Video decode + RetinaFace detection + landmarking is CPU-bound, and the head node has
40 cores versus the pod's 16. Frame extraction is therefore a **head-node** job; only
training touches the GPU. Extracted frames go to `data/processed/` and are what the
loaders read.

## Model weights (not datasets, but also fetched)

| Weight | Source | Notes |
|---|---|---|
| VMamba-S ImageNet-1K | official VMamba release (MzeroMiko/VMamba) | required; paper does not train the backbone from scratch |
| dlib 81-point landmarks | must come from the **original** publisher | commonly mirrored on random sites — do not use those; see `docs/SECURITY.md` |
| RetinaFace weights | `retinaface-pytorch` package release | pinned + checksummed |

## Phase 2 — `data/self_created/`

Empty until you add data after Phase 1. The loader for it is deliberately
schema-flexible but **fails loudly** on malformed input rather than silently skipping
samples. Describe the actual format when you add the data and the loader gets
specialised then — not before.
