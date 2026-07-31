# WMamba on the BMU cluster

Reproduction of **WMamba: Wavelet-based Mamba for Face Forgery Detection**
(Peng et al., ACM MM 2025, arXiv:2501.09617), adapted to this DGX/Kubernetes cluster.

Paper PDF: `~/papers/WMamba.Wavelet-based Mamba for Face Forgery Detection.pdf`

**Status: environment ready, model not yet implemented.** Step 0 (discovery), the
workspace, and the full dependency stack are done and verified. Model code, data
pipeline, and tests are next. No training job has been submitted.

---

## The three things that actually shape this project

1. **The GPU is a 16 GiB MIG slice with 16 SMs — about 12% of one H200.**
   The paper trains at batch 64 in ~30 GB. That does not fit. We hold the paper's
   *effective* batch of 64 via gradient accumulation, so optimizer semantics are
   unchanged; only the micro-batch shrinks.
2. **Slurm is installed but dead** (`Could not establish a configuration source`).
   **Kubernetes is authoritative.** There are no sbatch scripts here.
3. **There is no compiler in the pod** — no `gcc`, `make`, or `nvcc`, and no root to
   install them. Every dependency must be a prebuilt wheel. This drove the whole
   dependency strategy, including which torch version we are allowed to run.

Full measurements: [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md).

## Layout

```
research/
├── configs/base.yaml     # every hyperparameter; [PAPER] vs [CLUSTER] tagged
├── src/
│   ├── data/             # SBI synthesis, RetinaFace crop, landmarks, loaders
│   ├── models/           # dwt.py, dcconv.py, hwfeb.py, vmamba_backbone.py, wmamba.py
│   ├── training/
│   └── eval/             # video-level AUC, cross-dataset + cross-manipulation
├── scripts/
│   ├── pod.sh            # resolve + exec into the dev pod (never hardcode pod names)
│   ├── discover_env.sh   # re-run Step 0
│   ├── fetch_wheels.py   # download + SHA-256 verify CUDA wheels
│   └── k8s/dev-pod.yaml  # workspace Deployment
├── vendor/wheels/        # verified CUDA wheels (gitignored)
├── data/                 # you populate this; see docs/DATA.md
└── docs/                 # ENVIRONMENT.md, DATA.md, SECURITY.md
```

## Working here

The head node and the pod share `~` over NFS, so edit on the head node and run in the pod.

```bash
cd ~/research

./scripts/discover_env.sh            # re-check the machine
./scripts/pod.sh name                # current pod
./scripts/pod.sh shell               # interactive shell in the pod
./scripts/pod.sh py -c "import torch; print(torch.cuda.get_device_name(0))"
./scripts/pod.sh pip list
```

Do **not** run Python on the head node for anything GPU-related — the head node has no
GPU, and its Python is 3.12 while the pod's is 3.11.

### Pod lifecycle

The dev pod claims the **entire** namespace quota (16 CPU / 32 Gi / the single MIG
slice), so a training Job cannot start while it is up:

```bash
kubectl apply  -f scripts/k8s/dev-pod.yaml    # start / update
kubectl delete -f scripts/k8s/dev-pod.yaml    # free the quota before training
```

The venv lives on NFS at `.venv/`, so it survives pod restarts. Pod names change on
every reschedule — always resolve via `scripts/pod.sh`.

## Environment summary

| | |
|---|---|
| GPU | NVIDIA H200, `MIG 1g.18gb` — **16.0 GiB, 16 SMs** |
| Quota | 16 CPU / 32 Gi / 1× `mig-1g.18gb`; full-GPU quota is **0** |
| Scheduler | Kubernetes (`dgx-s-bmu-cse-240577-restricted`) |
| Image | `pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime`, runs as uid 1109 |
| torch | **2.10.0+cu128** in `.venv` (shadows the image's 2.4.0) |
| Kernels | `mamba-ssm 2.3.2.post1`, `causal-conv1d 1.6.2.post1` — real CUDA kernels, fwd+bwd verified |
| Head node | 40 cores, 125 GiB RAM, **no GPU** — use for data prep and `kubectl` |

## Security

28 dependency advisories found on first audit, **26 fixed**, 2 documented as
structurally unfixable. See [`docs/SECURITY.md`](docs/SECURITY.md) for the full
record and the one real tradeoff (torch version is pinned by the availability of
precompiled mamba kernels, because we cannot build from source).

```bash
./scripts/pod.sh exec '.venv/bin/pip-audit'
```

## Next steps

1. Implement `src/models/` — DWT (Eq. 1), DCConv (Eqs. 5–6), WFEM/HWFEB, VMamba-S wrapper.
2. Fetch the VMamba-S ImageNet-1K checkpoint and the dlib 81-point predictor (checksummed).
3. Implement the SBI data pipeline and the eval protocols.
4. `pytest` suite, then benchmark the real micro-batch size on the MIG slice.
5. **Stop** and report before submitting any training job.
