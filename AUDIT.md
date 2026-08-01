# WMamba failure-mode audit log

Append-only. Each pass is re-run before every training launch (Phase 1 and 2).

---

## Audit pass 1 — 2026-08-01 (pre-Phase-1, Stages 1–4 of the master prompt)

Scope: everything built in this pass (models, data pipeline, eval, trainer,
k8s manifests). Method: code-path tracing plus live verification runs in the
pod; nothing asserted from memory. Verdicts: **PASS** / **FIXED** / **RISK**.

### Stage 1 — eviction resilience (context for Tier 6)

- Diagnostic: the namespace was found EMPTY again at Stage-1 start — pod,
  ReplicaSet and Deployment all gone (second occurrence in ~24 h; the first
  took the original `research-gpu` deployment). Event log already expired
  (TTL); `kubectl top` unavailable. Only API-level deletes can remove
  controller objects, so this was not a node eviction — but the actor is not
  identifiable from inside the namespace. **Flagged for the cluster admin;
  no root cause is claimed.** NFS state (venv, wheels, code) survived intact
  both times.
- Defense in place: training as a Job (`restartPolicy: OnFailure`,
  `backoffLimit: 8`); all state on NFS; unconditional auto-resume from the
  newest *verified* checkpoint; `ckpt_every_epochs: 2` early (bounding lost
  work to <2 epochs); heartbeat log line every 300 s; recovery from even a
  deleted Job object is one `kubectl apply`.

### Tier 1 — environment & dependencies

| Item | Verdict | Evidence |
|---|---|---|
| Exact pins in lockfile | **PASS** | `requirements.lock.txt`: 126 packages, every line `==`, zero loose specs |
| mamba kernels match CUDA | **PASS** | wheels `cu12torch2.10cxx11abiTRUE-cp311` vs live `torch 2.10.0+cu128`, python 3.11; `causal_conv1d 1.6.2.post1`, `mamba_ssm 2.3.2.post1`; kernels exercised fwd+bwd in tests, finite |
| pip+conda double-installs | **RISK (mitigated, accepted)** | the image's conda base carries torch 2.4.0; the venv (`--system-site-packages`) shadows it with 2.10.0. Verified `importlib.util.find_spec("torch")` resolves into `.venv`. Residual risk: running bare `/opt/conda/bin/python` gets the old torch — mitigated by every entry point (Job manifest, pod.sh, tests) using `.venv/bin/python` explicitly. Cannot uninstall from the image (no root). |
| CUDA available + device count | **PASS** | `device_count: 1`, matches the single mig-1g.18gb allocation |
| Seeds: python/numpy/torch/cuda | **PASS** | `train.py::set_all_seeds` sets all four; traced |
| cudnn deterministic/benchmark intentional | **PASS** | set from `cfg.deterministic` with inline justification (fixed 224×224 shapes → benchmark=True; determinism not claimed — SBI is stochastic by design, mamba scan kernels nondeterministic) |

### Tier 2 — data pipeline

| Item | Verdict | Evidence |
|---|---|---|
| Video-level splits, no frame leakage | **PASS (code) / PENDING (data)** | split membership is per manifest *entry* = one video, so frame-level leakage is structurally impossible; `assert_no_video_overlap` guard + test. Cross-dataset identity overlap vs FF++ can only be checked when real data arrives — preprocessing must call the guard across all manifests. |
| SBI degenerate blends | **PASS** | `SelfBlender.generate` enforces min masked-difference 0.5/255-scale, 5 retries, then raises (label noise refused); tests cover the guard and the collapsed-landmarks case |
| Truncated/corrupt videos fail loudly | **PASS** | `_read_image_strict` raises on unreadable frames (cv2's silent `None` intercepted); manifest validation rejects zero-frame entries, missing files, bad labels, duplicate ids — all tested |
| Class balance logged, not assumed | **PASS** | balance is structural: every draw yields exactly one real + one SBI fake (labels `[0,1]`), so every batch is 50/50 by construction; trainer logs the pair composition at start |
| No train augs in test path | **PASS** | `VideoEvalDataset.__getitem__` is resize+normalize only; `grep` proves `src/eval/` imports no augmentation module; STG/REAL sets policed by `assert_disjoint_policy()` (called at training start + tested) |
| Normalization defined once | **PASS** | `NORM_MEAN/STD` exist exactly once (`src/data/datasets.py`); grep over `src/` confirms no duplicate literals |

### Tier 3 — model & numerical stability

| Item | Verdict | Evidence |
|---|---|---|
| DCConv OOB sampling | **PASS** | worst-case displacement is bounded: ‖(4, 4)‖ = 5.66 px (rotation preserves norm); `grid_sample(padding_mode="border")` clamps rather than zero-fills, keeping edge gradients alive; adversarial-weights test asserts the bound |
| Rotation angle ∈ [0, π/2] | **PASS** | `sigmoid × π/2`; test drives predictor weights to ±100 and asserts range holds |
| bf16 + Mamba fp32-critical params | **PASS** | all 474 model params are fp32 (autocast never casts parameters); the 42/42 SSM-critical tensors (`A_log`, `Ds`) verified fp32 in the instantiated model; full fwd+bwd under bf16 autocast asserts finite loss and finite grads (test_bf16_autocast_finite) |
| NaN/Inf on loss AND grads | **PASS** | loss checked every step; total grad norm checked at **every** optimizer step (free — `clip_grad_norm_` returns it); either aborts the run loudly |
| DCConv offset-predictor init | **PASS** | offset/angle convs zero-init, angle bias −6 (θ₀ ≈ 0.22°): test proves the layer starts as an exact straight axis-aligned kernel and matches a plain 1D conv within the designed 0.03 residual |
| Per-group grad norms (HWFEB vs backbone) | **PASS** | `model.param_groups()` two groups; norms logged every `log_every_steps`; live smoke run showed backbone 1.26 vs hwfeb 0.053 |

### Tier 4 — training-loop mechanics

| Item | Verdict | Evidence |
|---|---|---|
| train()/eval() re-asserted | **PASS** | canary block: eval → no_grad forward → `model.train()` immediately after; `_score_dataset` records prior mode and restores it |
| Distinct DataLoader worker seeds | **PASS** | `worker_init_fn` derives per-worker seeds from `torch.initial_seed()+worker_id` and reseeds python/numpy AND the dataset's own rng + albumentations (`dataset.reseed`) |
| Grad-accum math | **PASS** | `(loss / accum).backward()` — divide before backward; `step`+`zero_grad` only on boundary; config validated `micro × accum == effective` at startup. (Post-benchmark accum = 1, so the path is dormant but correct.) |
| Checkpoint restores full state | **PASS** | optimizer, scheduler, epoch, global_step AND all four RNG streams (torch/cuda/numpy/python) saved and restored; auto-resume is unconditional |
| autocast vs custom ops | **PASS** | DCConv deliberately pins its geometry+sampling to fp32 (`.float()` internally, cast back on return) — precision over speed for a tiny branch; DWT is a plain conv2d, natively autocast-aware; both verified finite under bf16 |
| Sharded optimizer / DeepSpeed clipping | **N/A** | single GPU, single process; quota forbids a second device. If this ever changes, this item reopens. |

### Tier 5 — evaluation & metrics

| Item | Verdict | Evidence |
|---|---|---|
| Video-level AUC per paper | **PASS** | frame scores → per-video mean → one AUC over videos; test constructs a case where frame-AUC ≠ video-AUC and asserts we produce the video answer |
| Multi-face max rule | **PASS** | implemented once (`frame_scores_to_video_scores`, max per (video, frame) then mean per video); tested |
| AUC over whole split, never per batch | **PASS** | aggregation collects the entire loader before one `roc_auc_score` call; single-class splits raise instead of emitting garbage; inconsistent per-video labels raise |
| Frame sampling 8 train / 32 eval | **PASS (code) / PENDING (data)** | eval subsampling to ≤32 frames evenly spaced is implemented+deterministic; the 8-frame training extraction is a preprocessing-time contract (script lands with the data; constants already defined in one place) |
| No threshold tuned on test | **PASS** | AUC is threshold-free; no threshold metric exists anywhere; flag stands if one is ever added |
| Distinct eval data dirs | **PASS** | duplicate-root assertions in both protocols + per-manifest `dataset` name validation (a copied dir with the wrong manifest name is rejected) |

### Tier 6 — hardware / infrastructure

| Item | Verdict | Evidence |
|---|---|---|
| Canary batch | **PASS** | fixed batch captured at step 1, scored every 500 steps in eval+no_grad, loss logged for drift correlation |
| Checkpoint integrity | **PASS** | atomic tmp-dir → `os.replace` publish; every save immediately reloaded and 8 tensors spot-checksummed BEFORE publish; corrupt candidate is deleted and the save fails loudly; `find_latest_checkpoint` skips unreadable dirs |
| Multi-GPU seed discipline | **N/A** | single device by quota |
| Cross-hardware reproduction | **NOTE** | recorded: non-reproduction across GPU generations ≠ bug (kernel differences); this run is on H200 MIG 1g.18gb |

### Tier 7 — reproducibility

| Item | Verdict | Evidence |
|---|---|---|
| Checkpoint ↔ config/commit/lockfile | **PASS** | `meta.json` per checkpoint: full resolved config, git commit hash, lockfile SHA-256, spot-checksums, timestamp |
| Deviations from paper recorded | **PASS** | full list below |
| FF++ aging flag | **FLAGGED (not built)** | FF++-trained detectors degrade against newer forgery methods; monitoring requirement if this ever deploys. Deliberately not built now. |

### Deviations from the paper (each with reason)

1. **bf16 autocast** — paper implies fp32 (~30 GB at batch 64). fp32 batch 64
   cannot fit in 16 GiB; bf16 measured at 13.08 GiB and makes the *paper's own
   batch size* feasible. Finiteness verified fwd+bwd. The alternative
   (fp32 + accum) was rejected: it changes BatchNorm statistics in HWFEB
   (micro-batch 16 vs 64), which is a *worse* fidelity loss than bf16.
2. **WFEM internals** (32 channels, 1-channel sigmoid attention, stride-2
   projection) — the paper specifies the branch structure but not channel
   counts; stride-2 is the only mapping making DWT levels 1–4 (112/56/28/14)
   meet stage resolutions (56/28/14/7). Param count reconciles (~50M backbone).
3. **Angle-predictor bias −6** — paper is silent on predictor init; this makes
   "initialized along the x/y axis" literally true at step 0 (θ₀ ≈ 0.22°).
4. **`ckpt_every_epochs: 2`** — this cluster kills namespaces; bounding lost
   work beats the paper's (unspecified) cadence. Widen once stable.
5. **Triton cross-scan disabled** — no compiler in the pod; official
   pure-PyTorch fallback used for data movement; the SSM scan itself still
   runs mamba_ssm's compiled CUDA kernel. Speed cost only, no numerics change.
6. **drop_path 0.3** — from the official VMamba-S recipe; paper is silent.

### Stage-3 benchmark (recorded)

Full train steps (fwd+bwd+AdamW+clip), bf16, pretrained weights, on the
mig-1g.18gb slice (16.0 GiB): linear scaling from batch 2 (1.15 GiB) to
**batch 64 → 13.08 GiB peak (81.8%, headroom 2.92 GiB ≥ the 12.5% floor)**.
Chosen: `micro_batch_size 64, grad_accum_steps 1, effective 64` — the paper's
exact optimizer semantics with zero accumulation compromise.

### Open RISK items (require user awareness, not code changes)

- **R1 — namespace deletion actor unknown.** Two API-level teardowns in 24 h.
  A multi-day training Job WILL likely be deleted mid-run; auto-resume bounds
  the damage to ≤2 epochs, but only the cluster admin can stop the deletions.
- **R2 — conda/venv torch shadowing.** Safe under our entry points; running
  anything with `/opt/conda/bin/python` silently gets torch 2.4.
- **R3 — dataset-dependent checks pending.** Identity-overlap guard across
  datasets, the 8-frame extraction, and RetinaFace/dlib on real faces can only
  be exercised after migration. Preprocessing lands with the data.
- **R4 — 2 torch advisories remain open** (PYSEC-2026-139 no fix exists;
  PYSEC-2025-194 fixed only in 2.13, no mamba wheels) — standing decision in
  docs/SECURITY.md, re-affirmed this pass.
