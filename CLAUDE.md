# WMamba project — session context (read this first, don't re-derive)

Reproduction of WMamba (Peng et al., ACM MM 2025, arXiv:2501.09617) — wavelet+Mamba
face-forgery detector. Paper PDF: `~/papers/WMamba.Wavelet-based Mamba for Face Forgery Detection.pdf`
(text sections already extracted once; Eq.1 = Haar filters, Eqs.5-6 = DCConv, §4.1.5 = hyperparams,
Appendix B = SBI/preprocessing/augs).

## Current status (updated 2026-08-05)

- **ML PRODUCTION GATES FRAMEWORK SAVED**: The full 12-category rubric for production readiness has been saved to `docs/ML_PRODUCTION_GATES.md` (2026-08-05).
- **PRE-FLIGHT AUDIT PASSED**: Pass 3 of the audit is complete (2026-08-05). All 56/56 tests passed live in the pod. The `SupervisedTrainDataset` loader was implemented and verified.
- **Stages 0–5 of the master prompt are COMPLETE.** Model, data pipeline, eval, trainer,
  k8s manifests, 47/47 tests passing, batch benchmark done, AUDIT.md pass 1 written.
- **🔀 SCOPE CHANGED 2026-08-02 (user decision, do not relitigate).** The paper's protocol
  is abandoned. FF++, Celeb-DF-v2, DFDC-test, DFDCP and FFIW were never transferred and
  will not be. **Train on DFDC and LAV-DF ONLY, as two separate runs.**
  Consequence: the paper's Table-1 cross-dataset AUC cannot be reproduced. Within-dataset
  video AUC on each dataset's own test split replaces it.
- **Datasets are migrated, organized and audited** (see "Datasets" below). `data/raw/`
  holds 137,904 videos; integrity audit clean.
- **⛔ TRAINING IS STILL GATED** on the user explicitly saying **"start Phase 1 training"**.
  The dataset-migration gate is now satisfied; the explicit-consent gate is NOT.
  Re-run the AUDIT.md checklist before any launch (append, never overwrite).
- **Pipeline is now complete end-to-end (2026-08-03), 56/56 tests pass:**
  `scripts/preprocess.py` (videos -> face crops + manifest, resumable),
  `SupervisedTrainDataset` + `recrop_to_margin` (`src/data/datasets.py`),
  `evaluate_within_dataset` (`src/eval/protocols.py`).
  Crops are stored at the WIDEST margin (0.20) at 256px with the face box recorded
  per frame, so [PAPER B.2]'s random 4-20% train margin and fixed 12.5% eval margin
  are both reachable from one stored image. Do not "simplify" this to fixed crops.
- **COMPLETE: LAV-DF preprocessing**, subsampled to
  `train=20000,val=4000,test=6000`, run as **6 parallel shards**.
  All 6 shards successfully processed 38,893 videos and were merged into `manifest.json`.
  The preprocessing pipeline is fully finished.
  **Video selection is frozen in `data/processed/lavdf/selection.json`.** It used to
  be redrawn each run from `hash(split+cls)`, but Python randomises str hashing per
  process — every restart drew a different subset (which is why train/real ended up
  at 11,171 instead of 10,000). Never go back to `hash()`; use `zlib.crc32`.
  **Full LAV-DF (136,304) was rejected: ~59 h to preprocess and ~9 days to train
  200 epochs on the MIG slice. Subsampling is a capacity necessity, not a preference.**
- **COMPLETE: DFDC download** to the balanced ceiling.
  3,608 videos successfully downloaded (35 failed due to Kaggle API limits, as expected).
  Data gathering for DFDC is fully finished.
  **⚠ Kaggle rate-limits PER ACCOUNT. 6 parallel workers earned a blanket HTTP 429
  and 2,592 consecutive failures. Never raise `--workers` above 2.** The script now
  has a circuit breaker (15 consecutive 429s -> exit 75) and the wrapper backs off
  15m/30m/1h/2h. Proven-safe setting: 1 worker, `MIN_INTERVAL_OVERRIDE=2.0`, ~10 files/min.
- **ARMED: `scripts/train_when_ready.sh`** — waits for LAV-DF preprocessing, validates
  the manifest, runs a 2-step smoke test (`--smoke-steps`), then launches training.
  Kill it with `pkill -f train_when_ready` to cancel before it starts.
- **RUNNING: `scripts/supervisor.sh`** — the answer to R1 (external namespace deletion).
  Every 60 s it reconciles: pod missing -> `kubectl apply`; preprocessing/training/download
  not running but not finished -> relaunch (all three resume). Stop with
  `touch STOP_SUPERVISOR`. **VERIFIED 2026-08-03 by deleting the pod deliberately:
  detected in 24 s, pod Running again at 16 s, preprocessing relaunched and logged
  `resuming: 15300 videos already processed` — zero work lost.** Check it is alive
  before trusting any long run; `scripts/status.sh` shows its state first.
- Jobs are `nohup`'d with `parent=1, tty=?` — they survive SSH disconnect, VS Code
  closing and the user's laptop shutting down. They do NOT survive a head-node reboot.
- **Timing, measured 2026-08-03** (`scripts/benchmark_throughput.py`): 44.3 img/s at
  batch 64. Preprocessing 1.20 s/video at 16 frames, 1.56 s at 32.
  LAV-DF training ≈ 7.5 min/epoch (12.9 min including validation), ~6.5 h for 30 epochs (Supervised loader yields 1 random frame/video, not 8).
  Per-epoch val uses `eval.val_frames_per_video: 8` (NOT the paper's 32) purely for the
  early-stopping probe — at 32 it would add ~8 h across the run. The REPORTED test
  metric still uses 32 (`test_frames_per_video`). Do not conflate these two.
- Phase 2 (fine-tune on `data/self_created/`) comes only after user reviews Phase-1 results.

## Environment facts (measured — do NOT re-probe unless something breaks)

- Head node `bmu-headnode`: 40 CPU, 125 GiB RAM, **no GPU**, internet OK, Python 3.12.
  `~` is local XFS on the head node, NFS-exported to pods (persistent, survives pod death).
- **Slurm is dead** (no config source). **Kubernetes is the scheduler.**
  Namespace `dgx-s-bmu-cse-240577-restricted`, quota = 16 CPU / 32 Gi / **1× mig-1g.18gb** (whole namespace).
  Full-GPU and bigger-MIG quotas are hard 0.
- GPU = H200 MIG slice: **16.0 GiB usable, 16 SMs**. `torch.cuda.get_device_name` → "NVIDIA H200 MIG 1g.18gb".
- Pod image `pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime`: **NO gcc/nvcc/git/curl — prebuilt wheels only, no source builds, no triton kernels** (triton needs cc at runtime).
- venv at `~/research/.venv` (created IN the pod, py3.11, `--system-site-packages`):
  torch **2.10.0+cu128** (shadows image's 2.4.0 — always use `.venv/bin/python`),
  mamba_ssm 2.3.2.post1 + causal_conv1d 1.6.2.post1 (checksum-verified wheels in `vendor/wheels/`,
  CUDA kernels verified fwd+bwd), dlib-bin, opencv-python-headless (GUI build breaks: no libxcb).
- **Namespace gets wiped externally** (~2×/24h, API-level deletes incl. Deployment objects; actor
  unknown, admin should be asked). Recovery: `kubectl apply -f scripts/k8s/dev-pod.yaml`. NFS always survives.
- Pod names change on reschedule — use `scripts/pod.sh {name|exec|py|pip|shell}`, never hardcode.
- uid 1109 has no /etc/passwd entry in the image → manifests set USER/LOGNAME env (torch 2.10 crashes without).

## Key decisions already made (don't relitigate)

- bf16 autocast → **paper's full batch 64 fits**: measured 13.08 GiB peak of 16.0 (benchmark:
  `scripts/benchmark_batch.py`). `micro_batch_size 64, grad_accum_steps 1`. fp32+accum rejected
  (would corrupt HWFEB BatchNorm stats at micro-batch 16).
- Official VMamba vendored at `vendor/VMamba` (gitignored except PROVENANCE.md), triton disabled
  via `csm_triton.WITH_TRITON = False` in `src/models/vmamba_backbone.py`; strict=True +
  weights_only=True checkpoint load. VMamba-S ckpt: `checkpoints/pretrained/vssm_small_0229_ckpt_epoch_222.pth`
  (sha256 c540366e…bd30cb); dlib 81-pt: `shape_predictor_81_face_landmarks.dat` (8cae4375…ff869).
- 2 torch advisories stay open (PYSEC-2026-139 no fix; PYSEC-2025-194 needs torch 2.13, no mamba
  wheels) — documented docs/SECURITY.md, do NOT trade the CUDA kernels to close them.
- Paper deviations (6, each justified): see AUDIT.md "Deviations" section.
- **Two separate runs, never a merged one** (user instruction 2026-08-02):
  `configs/dfdc.yaml` and `configs/lavdf.yaml`. Each `extends: base.yaml` (one-level merge
  in `train.load_config`), so shared hyperparameters cannot drift. `run_name` scopes
  `checkpoints/<run_name>` — previously hardcoded `phase1`, which would have made the two
  runs overwrite each other and auto-resume from the wrong checkpoint.
- **AUDIO IS ENTIRELY OUT OF SCOPE (user, restated 2026-08-03: "image only, not at all
  audio").** Nothing in the pipeline opens an audio stream: `cv2.VideoCapture.read()`
  decodes video frames only, the model consumes 224x224 RGB crops, and the DFDC
  audio-only-fake check found 0/285. Never add an audio branch or an audio-derived label.
- **LAV-DF label = `modify_video`, NOT `n_fakes`.** Audio forgery is explicitly out of
  scope (user). 33,170 LAV-DF videos have forged audio but untouched frames; labelling
  them fake would poison ~24% of the set for a visual detector. Under the visual rule the
  splits land at 48.7 / 49.3 / 49.1% fake — no reweighting needed.
- **SBI is off for both runs.** Both datasets ship genuine labelled fakes (67,503 of them);
  SBI's `real_only` mode would discard every one. This is a deliberate deviation from the
  paper's method, driven by the dataset change above.
- DFDC had **no official split** (all 1,600 entries say `split: train`). Built one with
  `scripts/build_dfdc_splits.py`: identity-grouped (fake → its `original`, real → itself),
  groups assigned atomically, so a fake and its source real never straddle a split.

## Datasets (organized + audited 2026-08-02)

| | videos | REAL | FAKE | fake% |
|---|---|---|---|---|
| `data/raw/dfdc/` (7.5 G) | 1,600 | 800 | 800 | 50.0 |
| `data/raw/lavdf/` (25 G) | 136,304 | 69,601 | 66,703 | 48.9 |
| **combined** | **137,904** | **70,401** | **67,503** | **48.9** |

- DFDC: `videos/` + `metadata.json` (labels 1,600/1,600) + `splits.json` (560/560, 120/120,
  120/120 — exactly 50% fake per split) + `build_scripts/`.
- LAV-DF: official `train/` 78,703, `dev/` 31,501 (used as val), `test/` 26,100 + metadata.
  Extracted and verified against metadata (`scripts/extract_lavdf.py`, resumable).
- `data/self_created/` (12 G) — the earlier "satyanetra" project, **kept, out of scope**.
- Audit (`scripts/audit_datasets.py`): no corrupt/empty/orphan/missing files. Only finding
  is 2 exact-duplicate pairs inside LAV-DF upstream (both genuine reals, same label, same
  split — harmless).
- **DFDC is 1.2% of the combined data** — irrelevant while the runs are separate, but it
  would be swamped in any merged run.
- Caveat: DFDC's public release contains some audio-only fakes and this metadata has no
  audio flag, so a few of the 800 DFDC fakes may be audio-only. Only 285/800 fakes resolve
  to an original that is present, so at most ~36% is checkable. Unresolved.

## Where things live

| What | Where |
|---|---|
| Full audit trail (append-only) | `AUDIT.md` |
| Environment measurements | `docs/ENVIRONMENT.md` |
| Dataset plan + rsync instructions | `docs/DATA.md` |
| Security decisions + CVE record | `docs/SECURITY.md` |
| Shared hyperparams ([PAPER]/[CLUSTER] tagged) | `configs/base.yaml` (parent, not run directly) |
| Per-run configs | `configs/{dfdc,lavdf}.yaml` — `extends: base.yaml` |
| DFDC split builder / dataset audit / LAV-DF extractor | `scripts/{build_dfdc_splits,audit_datasets,extract_lavdf}.py` |
| Dev pod / training Job manifests | `scripts/k8s/{dev-pod,train-job}.yaml` |
| Models (DWT/DCConv/HWFEB/VMamba/WMamba) | `src/models/` |
| SBI + loaders + augs (two disjoint sets!) | `src/data/` |
| Video-level AUC + protocols | `src/eval/` |
| Trainer (auto-resume, canary, heartbeat) | `src/training/train.py` |
| Tests (47, GPU ones auto-skip off-pod) | `tests/` — run: `./scripts/pod.sh exec '.venv/bin/python -m pytest tests/ -q'` |

## Standing commands

```bash
./scripts/pod.sh name|shell|exec …        # reach the dev pod
kubectl apply -f scripts/k8s/dev-pod.yaml # recreate pod after namespace wipe
./scripts/discover_env.sh                 # re-run env discovery if needed
# training (ONLY after both gates open):
kubectl delete -f scripts/k8s/dev-pod.yaml && kubectl apply -f scripts/k8s/train-job.yaml
```

## User's working rules (from the master prompt — binding)

- Hard STOP gates; never proceed on inferred consent. Show diffs for fixes; present
  tradeoffs and wait on judgment calls.
- Append to AUDIT.md with timestamps, never overwrite; re-audit before every launch.
- Eviction root cause belongs to the cluster admin — build resilience, don't guess causes.
