# Environment discovery (Step 0)

Measured on the BMU cluster, 2026-07-31. **Every number here was read off the machine, not assumed.**
Re-run `scripts/discover_env.sh` to refresh.

## Head node (`bmu-headnode`)

| Item | Value |
|---|---|
| CPU | 2× Intel Xeon Silver 4210R, 40 threads total |
| RAM | 125 GiB (115 GiB available) |
| GPU | **none** — `nvidia-smi` not present |
| CUDA toolkit | 12.9 (`nvcc` present, but no GPU to use it on) |
| Python | 3.12.12 (`/cm/local/apps/python312`) |
| Storage | `/` 5.3 T, **1.3 T free** (77% used, shared) |
| Internet | yes (pypi, github, arxiv reachable) |

**Consequence:** the head node is for code, data staging, and `kubectl` only. No training,
no benchmarking, no CUDA extension compilation targeting the real GPU.

## Scheduler

| Item | Value |
|---|---|
| Slurm | binaries present (`/cm/local/apps/slurm/25.05`) but **non-functional** — `sinfo`/`squeue`/`sbatch` all fail with `resolve_ctls_from_dns_srv: Unknown host` / `Could not establish a configuration source` |
| Kubernetes | **functional and authoritative** |
| Context | `dgx-s-bmu-cse-240577@BMU-Cluster` |
| Namespace | `dgx-s-bmu-cse-240577-restricted` |

**Consequence:** job submission is `kubectl apply`. There are no sbatch scripts in this repo.
The original plan's "sbatch or kubectl, whichever step 0 determines" resolves to **kubectl**.

## Namespace quota (`resourcequota/user-quota`)

| Resource | Used | Hard |
|---|---|---|
| `requests.cpu` / `limits.cpu` | 16 | **16** |
| `requests.memory` / `limits.memory` | 32 Gi | **32 Gi** |
| `requests.nvidia.com/gpu` (full GPU) | 0 | **0** |
| `requests.nvidia.com/mig-1g.18gb` | 1 | **1** |
| `requests.nvidia.com/mig-2g.35gb` | 0 | **0** |
| `requests.nvidia.com/mig-3g.71gb` | 0 | **0** |

No `LimitRange`. No `PersistentVolumeClaim`s, no `StorageClass` — storage is NFS-backed home only.

**Two hard consequences:**

1. **No full-GPU access.** Only the `mig-1g.18gb` profile has any quota, and it is `1/1`.
   The larger MIG profiles (`2g.35gb`, `3g.71gb`) are quota'd to zero — asking for one is rejected
   by the API server, not merely left pending.
2. **The entire namespace budget is currently consumed** by the running `research-gpu` deployment
   (16 CPU / 32 Gi / 1 MIG). A second GPU pod cannot be scheduled. Training must either reuse
   that pod or replace it.

## The GPU actually available

Read from inside the running pod:

| Item | Value |
|---|---|
| Physical GPU | NVIDIA **H200**, driver 580.105.08, CUDA 13.0, MIG enabled |
| Slice | `MIG 1g.18gb` |
| **Usable VRAM** | **16.0 GiB** (not 18 — the profile name is nominal) |
| **SMs** | **16** (an H200 has 132 → this slice is ~12% of the card) |

## The running pod

| Item | Value |
|---|---|
| Deployment | `research-gpu` (1/1, running 10 h) |
| Image | `pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime` |
| Resources | 16 CPU / 32 Gi / 1× `mig-1g.18gb` |
| Node | `bmu-worker` |
| Runs as | **root** (uid 0) |
| Python / torch | 3.11.9 / torch 2.4.0+cu124, `torch.cuda.is_available() == True` |
| **`nvcc`** | **absent** — this is a `-runtime` image, not `-devel` |
| Volume | NFS `10.141.255.254:/home/dgx-s-bmu-cse-240577` → `/home/dgx-s-bmu-cse-240577` |
| Internet | yes (pypi, github, huggingface all reachable) |

**Consequence:** no CUDA toolchain in the pod, so `mamba-ssm`, `causal-conv1d`, and VMamba's
`selective_scan` kernels **cannot be built from source** in the current image. Either prebuilt
wheels matching (torch 2.4 / cu12.4 / cp311) work, or the image must change to `-devel`.

## Filesystem notes

- `~` and `~/research` are the **same NFS export** mounted in both the head node and the pod,
  so code written on the head node is immediately visible to the pod. This is the intended workflow.
- `~/research` was created root-owned by an earlier setup and was **not writable from the head node**.
  Fixed by `chown -R 1109:1109` executed as root from inside the pod. Verified writable from the head node.
- Anything the pod writes lands as **uid 0** unless the container sets `runAsUser: 1109`. The
  training manifests in `scripts/k8s/` set `runAsUser`/`runAsGroup`/`fsGroup` to 1109 so
  checkpoints and logs stay owned by you.
- Python differs between head node (3.12) and pod (3.11). The project venv **must** be created
  inside the pod so extension modules match the runtime. Do not reuse a head-node venv.

## Storage budget for the datasets

1.3 TiB free, shared with other users on `/`. The paper only ever *tests* on CDF/DFDC/DFDCP/FFIW —
it trains on FF++ real videos alone — so only FF++ needs a train split, and only test splits are
needed for the rest. That is the difference between ~1.5 TB (all of DFDC) and roughly 150–200 GB.
See `docs/DATA.md`.
