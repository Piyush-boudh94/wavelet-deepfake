# Security posture

Requirements from the project brief, and what was actually done about each.
Findings are recorded with their resolution, not just logged.

## 1. No hardcoded credentials

No secret is required to train this model. `.env.example` is the committed template;
`.env` is gitignored. Config values resolve through `${oc.env:...}` so nothing sensitive
is ever baked into a Python file or a YAML default.

Run before any push:

```bash
git grep -nEI '(api[_-]?key|secret|token|password|BEGIN [A-Z ]*PRIVATE KEY)' -- \
    ':!docs/SECURITY.md' ':!.env.example'
```

## 2. Kubernetes credentials stay out of the repo

`kubectl` reads `~/.kube/config`, which lives outside the project tree. `.gitignore`
additionally blocks `kubeconfig`, `.kube/`, `**/*service-account*.json`, `*.pem`, `*.key`.
No manifest in `scripts/k8s/` contains a token — the pod uses its own projected
ServiceAccount token, mounted by the API server at runtime and never written to disk here.

Job manifests must not `echo`/log env vars. Never `kubectl config view --raw` into a file
under the repo.

## 3. Dependency CVEs (`pip-audit`)

First audit of the resolved environment returned **28 advisories**; a second pass after
the torch upgrade surfaced a further 34 inherited from the base conda image (the venv
uses `--system-site-packages`, so image-level packages are in scope until shadowed).

**Final state: 2 remaining, both on `torch`, both structurally unfixable.**

| Package | Was | Now | Advisories closed |
|---|---|---|---|
| `torch` | 2.4.0 | **2.10.0** | 20 of 22 (CVE-2025-2148/2149/2998/2999/3001, PYSEC-2026-1970/2286, …) |
| `urllib3` | 2.2.2 | 2.7.0 | 6 |
| `pillow` | 10.4.0 | 12.3.0 | 15 |
| `cryptography` | 42.0.5 | 50.0.0 | 5 |
| `jinja2` | 3.1.4 | 3.1.6 | 3 |
| `filelock` | 3.13.1 | ≥3.20.3 | 2 |
| `requests` | 2.32.3 | 2.34.2 | 2 |
| `soupsieve` | 2.5 | ≥2.8.4 | 2 |
| `idna` | 3.7 | ≥3.15 | 1 |
| `pygments` | 2.15.1 | ≥2.20.0 | 1 |
| `brotli` | 1.0.9 | ≥1.2.0 | 1 |

Each upgrade was followed by a re-verification that `torch.cuda`, the mamba CUDA
kernels, OpenCV, dlib and albumentations all still import and run — no fix was applied
blind.

The torch upgrade was not free, and is the one place where a security fix collided with a
functionality requirement:

> **The tradeoff.** `mamba-ssm` and `causal-conv1d` ship *precompiled* CUDA kernels bound
> to an exact `(torch minor, python, CUDA, C++ ABI)` tuple. The pod has **no compiler and
> no `nvcc`**, so a source build is impossible — the pinned wheel dictates the torch
> version, not the other way round. Upgrading torch therefore required finding upstream
> wheels built against the *same* newer torch.
>
> Resolution: upgraded torch 2.4.0 → **2.10.0**, and moved to
> `mamba-ssm 2.3.2.post1` + `causal-conv1d 1.6.2.post1`, whose `cu12torch2.10cxx11abiTRUE
> cp311` wheels match exactly. Nothing was downgraded and no kernel was replaced by a
> slow Python fallback.
>
> **Residual, accepted (2 advisories):** `PYSEC-2026-139` currently has **no fixed
> version published upstream** — there is nothing to upgrade to.
> `PYSEC-2025-194` is only marked fixed in torch 2.13.0. No
> `mamba-ssm` wheel exists for torch 2.13 (upstream tops out at 2.10), so closing it would
> mean giving up the CUDA kernels entirely and running the SSM scan in pure PyTorch —
> a large slowdown on a GPU slice that is already only ~12% of an H200. Given this is a
> single-tenant training pod that loads no untrusted model files and exposes no network
> service, the exposure is not reachable. Revisit when upstream ships torch 2.13 wheels.

Re-audit at any time:

```bash
scripts/pod.sh pip-audit
```

## 4. Third-party model weights

`dlib`'s 81-point landmark predictor and RetinaFace weights are widely re-hosted on
unofficial mirrors. Policy: **fetch only from the original publisher, and verify a
checksum where one is published.**

Wheels in `vendor/wheels/` were downloaded from the upstream GitHub release APIs
(`state-spaces/mamba`, `Dao-AILab/causal-conv1d`) and verified against the SHA-256
digests GitHub publishes for each asset. This was not ceremonial — the first
`causal-conv1d` download was silently **truncated** (73 MB of an expected 151 MB) by a
timeout and failed verification. An unverified install would have produced a corrupt
CUDA extension. `scripts/fetch_weights.py` applies the same download-then-verify
discipline to model weights and refuses to install on mismatch.

## 5. No pickle on untrusted data

- `torch.load(..., weights_only=True)` everywhere; never bare `torch.load`.
- Our own checkpoints are written as `safetensors` where the format allows it.
- Upstream `.pth` backbone weights (VMamba, dlib, RetinaFace) are `weights_only=True`
  loaded and checksum-verified before first use, then re-serialised to `safetensors`.
- No `pickle.load()` anywhere in `src/`.

## 6. Container posture

The previous `research-gpu` pod ran as **uid 0**, which is why `~/research` ended up
root-owned and unwritable from the head node. `scripts/k8s/dev-pod.yaml` sets
`runAsUser/runAsGroup/fsGroup: 1109`, so the container runs unprivileged and everything
written to NFS is owned by you. Verified: `id` inside the pod reports uid 1109, and CUDA
still initialises correctly without root.
