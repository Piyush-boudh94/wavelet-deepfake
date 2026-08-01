"""WMamba Phase-1 trainer.

Run (inside the pod / training Job):
    .venv/bin/python -m src.training.train --config=configs/base.yaml

Design contract (each point maps to an audit tier or the paper):
  * Hyperparameters come from the config; nothing tunable is hardcoded here.
  * Paper semantics: AdamW, lr 5e-5, 200 epochs, linear LR decay starting at
    epoch 100, plain cross-entropy, effective batch 64 (micro-batch x accum).
  * Auto-resume: ALWAYS resumes from the newest valid checkpoint if one
    exists -- no flag. A restarted Job continues, never restarts (Stage 1).
  * Checkpoints: atomic directory writes on NFS, model weights as safetensors,
    optimizer/scheduler/counters via torch.save; every save is immediately
    reloaded and spot-checksummed before being marked valid (Tier 6).
  * Heartbeat: timestamped line every `heartbeat_sec` so an eviction leaves a
    last-known-alive timestamp (Stage 1).
  * Canary: a fixed batch is scored every `canary_every_steps`; an unexplained
    jump on unchanging inputs flags silent corruption (Tier 6).
  * NaN policy: loss checked every step; total grad norm checked at every
    optimizer step (free -- clip_grad_norm_ returns it). Non-finite -> abort
    loudly rather than train through garbage (Tier 3).
  * Grad-accum math: loss / accum_steps before backward; step + zero_grad only
    on boundary (Tier 4).
  * bf16 autocast (no GradScaler -- bf16 needs none). SSM numerics verified
    finite under bf16 in tests; A_log/D params live in fp32 by construction in
    the vendored code (Tier 3).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.augment import assert_disjoint_policy
from src.data.datasets import SBITrainDataset
from src.models.wmamba import build_wmamba


# --------------------------------------------------------------------------- utils
def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def set_all_seeds(seed: int) -> None:
    """python / numpy / torch / torch.cuda -- all four (Tier 1)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id: int) -> None:
    """Distinct per-worker seeds; identical worker seeds silently collapse
    augmentation diversity (Tier 4)."""
    base = torch.initial_seed() % 2**31
    seed = base + worker_id
    random.seed(seed)
    np.random.seed(seed)
    info = torch.utils.data.get_worker_info()
    if info is not None and hasattr(info.dataset, "reseed"):
        info.dataset.reseed(seed)


def git_commit_hash(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_spot_checksum(state: dict, k: int = 8) -> dict[str, float]:
    """Deterministic spot-checks: sum of k evenly spaced tensors (Tier 6)."""
    keys = sorted(state.keys())
    picks = keys[:: max(1, len(keys) // k)][:k]
    return {key: float(state[key].float().sum()) for key in picks}


# ----------------------------------------------------------------- checkpointing
def save_checkpoint(ckpt_dir: Path, tag: str, model, optimizer, scheduler,
                    epoch: int, global_step: int, cfg, repo_root: Path) -> Path:
    from safetensors.torch import save_file, load_file

    final = ckpt_dir / tag
    tmp = ckpt_dir / f".tmp_{tag}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    model_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    # safetensors forbids shared storage; clone to be safe
    model_state = {k: v.clone().contiguous() for k, v in model_state.items()}
    save_file(model_state, str(tmp / "model.safetensors"))

    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
        },
        tmp / "trainer_state.pt",
    )

    lock = repo_root / "requirements.lock.txt"
    meta = {
        "epoch": epoch,
        "global_step": global_step,
        "config": OmegaConf.to_container(cfg, resolve=True),
        "git_commit": git_commit_hash(repo_root),
        "lockfile_sha256": file_sha256(lock) if lock.exists() else "missing",
        "spot_checksum": tensor_spot_checksum(model_state),
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (tmp / "meta.json").write_text(json.dumps(meta, indent=2, default=str))

    # ---- verify BEFORE publishing (Tier 6): reload + spot-checksum ----------
    reloaded = load_file(str(tmp / "model.safetensors"))
    for key, want in meta["spot_checksum"].items():
        got = float(reloaded[key].float().sum())
        if not math.isclose(got, want, rel_tol=1e-6, abs_tol=1e-6):
            shutil.rmtree(tmp)
            raise RuntimeError(
                f"checkpoint verification FAILED for {key}: {got} != {want}"
            )

    if final.exists():
        shutil.rmtree(final)
    os.replace(tmp, final)          # atomic publish
    log(f"checkpoint saved+verified: {final}")
    return final


def find_latest_checkpoint(ckpt_dir: Path) -> Path | None:
    if not ckpt_dir.exists():
        return None
    candidates = []
    for d in ckpt_dir.iterdir():
        if d.is_dir() and not d.name.startswith(".") and (d / "meta.json").exists():
            try:
                meta = json.loads((d / "meta.json").read_text())
                candidates.append((meta["global_step"], d))
            except Exception as e:      # unreadable meta -> not a valid ckpt
                log(f"WARNING: ignoring corrupt checkpoint {d}: {e}")
    return max(candidates)[1] if candidates else None


def load_checkpoint(path: Path, model, optimizer, scheduler) -> tuple[int, int]:
    from safetensors.torch import load_file

    model.load_state_dict(load_file(str(path / "model.safetensors")), strict=True)
    state = torch.load(path / "trainer_state.pt", map_location="cpu",
                       weights_only=False)  # our own file, on our own NFS; contains
    # RNG state objects (numpy tuple) that weights_only rejects. Provenance: written
    # exclusively by save_checkpoint above. Documented in docs/SECURITY.md.
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    torch.set_rng_state(state["torch_rng"])
    torch.cuda.set_rng_state_all(state["cuda_rng"])
    np.random.set_state(state["numpy_rng"])
    random.setstate(state["python_rng"])
    return state["epoch"], state["global_step"]


# ------------------------------------------------------------------------ train
def group_grad_norm(params: list[torch.nn.Parameter]) -> float:
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += float(p.grad.detach().float().norm() ** 2)
    return math.sqrt(total)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    repo_root = Path(cfg.paths.root)
    ckpt_dir = Path(cfg.paths.checkpoints) / "phase1"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # NFS sanity (Stage 1): refuse to run if checkpoints resolve inside the
    # container's ephemeral filesystem.
    if not str(ckpt_dir).startswith("/home/"):
        raise RuntimeError(f"checkpoint dir not on persistent home: {ckpt_dir}")

    set_all_seeds(int(cfg.seed))
    # Intentional (Tier 1, documented): benchmark=True because input shapes are
    # constant (224x224 fixed); full determinism is not claimed -- SBI is
    # stochastic by design and mamba's scan kernels are not deterministic.
    torch.backends.cudnn.benchmark = bool(not cfg.deterministic)
    torch.backends.cudnn.deterministic = bool(cfg.deterministic)
    assert_disjoint_policy()

    device = "cuda"
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable -- this trainer must run in the GPU pod")

    micro = int(cfg.train.micro_batch_size)
    accum = int(cfg.train.grad_accum_steps)
    eff = int(cfg.train.effective_batch_size)
    if micro * accum != eff:
        raise ValueError(f"micro({micro}) x accum({accum}) != effective({eff})")
    if micro % 2:
        raise ValueError("micro_batch_size must be even (SBI yields real/fake pairs)")

    ds = SBITrainDataset(
        Path(cfg.paths.processed) / "ffpp",
        landmark_model_path=cfg.sbi.landmark_predictor,
        image_size=int(cfg.data.image_size),
        seed=int(cfg.seed),
    )
    loader = DataLoader(
        ds,
        batch_size=micro // 2,          # each item is a (real, fake) pair
        shuffle=True,
        num_workers=int(cfg.data.num_workers),
        prefetch_factor=int(cfg.data.prefetch_factor),
        pin_memory=bool(cfg.data.pin_memory),
        persistent_workers=bool(cfg.data.persistent_workers),
        worker_init_fn=worker_init_fn,
        drop_last=True,
    )
    log(f"train videos: {len(ds)}  micro-batch: {micro} imgs "
        f"({micro//2} pairs)  accum: {accum}  effective: {eff}")

    model = build_wmamba(cfg).to(device)
    groups = model.param_groups()
    optimizer = torch.optim.AdamW(
        [{"params": g["params"]} for g in groups],
        lr=float(cfg.train.lr),
        weight_decay=float(cfg.train.weight_decay),
    )
    epochs = int(cfg.train.epochs)
    decay_start = int(cfg.train.lr_decay_start_epoch)

    def lr_lambda(epoch: int) -> float:   # [PAPER] linear decay from epoch 100
        if epoch < decay_start:
            return 1.0
        return max(0.0, (epochs - epoch) / (epochs - decay_start))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    start_epoch, global_step = 0, 0
    latest = find_latest_checkpoint(ckpt_dir)
    if latest is not None:
        start_epoch, global_step = load_checkpoint(latest, model, optimizer, scheduler)
        start_epoch += 1
        log(f"AUTO-RESUMED from {latest} (epoch {start_epoch}, step {global_step})")
    else:
        log("no checkpoint found -- fresh start")

    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
                 "off": None}[str(cfg.train.amp)]
    clip = float(cfg.train.clip_grad_norm)
    heartbeat_sec = int(cfg.train.get("heartbeat_sec", 300))
    canary_every = int(cfg.train.get("canary_every_steps", 500))
    log_every = int(cfg.train.get("log_every_steps", 50))

    canary: tuple[torch.Tensor, torch.Tensor] | None = None
    last_beat = time.time()
    loss_fn = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(start_epoch, epochs):
        epoch_t0 = time.time()
        optimizer.zero_grad(set_to_none=True)
        for it, (images, labels) in enumerate(loader):
            # (B, 2, 3, H, W) -> (2B, 3, H, W): flatten real/fake pairs
            x = images.flatten(0, 1).to(device, non_blocking=True)
            y = labels.flatten(0, 1).to(device, non_blocking=True)

            if canary is None:
                canary = (x[:4].detach().clone(), y[:4].detach().clone())

            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                loss = loss_fn(model(x), y)

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"non-finite loss at epoch {epoch} step {global_step}: {loss.item()}"
                )

            (loss / accum).backward()      # Tier 4: divide BEFORE backward

            if (it + 1) % accum == 0:
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                if not torch.isfinite(total_norm):
                    raise RuntimeError(
                        f"non-finite grad norm at epoch {epoch} step {global_step}"
                    )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % log_every == 0:
                    norms = {g["name"]: round(group_grad_norm(g["params"]), 4)
                             for g in groups}
                    log(f"epoch {epoch} step {global_step} "
                        f"loss {loss.item():.4f} grad_norm {float(total_norm):.4f} "
                        f"group_norms {norms} lr {scheduler.get_last_lr()[0]:.2e}")

                if canary is not None and global_step % canary_every == 0:
                    model.eval()
                    with torch.no_grad(), torch.autocast(
                            "cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                        closs = loss_fn(model(canary[0]), canary[1])
                    model.train()          # Tier 4: re-assert after nested eval
                    log(f"canary step {global_step} loss {closs.item():.5f}")

            if time.time() - last_beat >= heartbeat_sec:
                log(f"HEARTBEAT alive epoch {epoch} step {global_step} "
                    f"epoch_elapsed {time.time()-epoch_t0:.0f}s")
                last_beat = time.time()

        scheduler.step()

        if (epoch + 1) % int(cfg.train.ckpt_every_epochs) == 0 or epoch == epochs - 1:
            save_checkpoint(ckpt_dir, f"epoch_{epoch:04d}", model, optimizer,
                            scheduler, epoch, global_step, cfg, repo_root)
            keep = int(cfg.train.keep_last_n_ckpts)
            valid = sorted(
                d for d in ckpt_dir.iterdir()
                if d.is_dir() and (d / "meta.json").exists()
            )
            for old in valid[:-keep]:
                shutil.rmtree(old)
                log(f"pruned old checkpoint {old.name}")

        log(f"epoch {epoch} done in {time.time()-epoch_t0:.0f}s")

    log("training complete")


if __name__ == "__main__":
    main()
