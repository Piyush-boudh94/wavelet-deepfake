# WMamba project — session context (read this first, don't re-derive)

Reproduction of WMamba (Peng et al., ACM MM 2025, arXiv:2501.09617) — wavelet+Mamba
face-forgery detector. Paper PDF: `~/papers/WMamba.Wavelet-based Mamba for Face Forgery Detection.pdf`
(text sections already extracted once; Eq.1 = Haar filters, Eqs.5-6 = DCConv, §4.1.5 = hyperparams,
Appendix B = SBI/preprocessing/augs).

## Current status (updated 2026-08-01)

- **Stages 0–5 of the master prompt are COMPLETE.** Model, data pipeline, eval, trainer,
  k8s manifests, 47/47 tests passing, batch benchmark done, AUDIT.md pass 1 written.
- **⛔ TRAINING IS GATED.** Two conditions, BOTH required, NEITHER met yet:
  1. user confirms datasets migrated + verified in `data/raw/` (currently **empty — 0 files**)
  2. user explicitly says **"start Phase 1 training"**
  Never submit `scripts/k8s/train-job.yaml` before both. Re-run the AUDIT.md checklist
  before any launch (append, never overwrite).
- Waiting on: user's dataset rsync. When they send local file counts/bytes, verify the
  DGX side (`find -type f | wc -l`, `du -sb`, spot `sha256sum`) before declaring migration done.
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
- Datasets: only ~100–150 GB needed (test splits only for CDF/DFDC/DFDCP/FFIW; **never** the
  470 GB DFDC train set). FF++ real c23 is the only training data (SBI makes fakes on the fly).

## Where things live

| What | Where |
|---|---|
| Full audit trail (append-only) | `AUDIT.md` |
| Environment measurements | `docs/ENVIRONMENT.md` |
| Dataset plan + rsync instructions | `docs/DATA.md` |
| Security decisions + CVE record | `docs/SECURITY.md` |
| All hyperparams ([PAPER]/[CLUSTER] tagged) | `configs/base.yaml` |
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
