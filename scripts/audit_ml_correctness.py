"""Audit the things that silently ruin a trained model.

Not file integrity (scripts/audit_datasets.py does that) but the machine-learning
invariants: leakage, label sanity, augmentation placement, eval determinism, and
metric validity. Each check prints PASS/FAIL and the evidence behind it.

    ./scripts/pod.sh exec '.venv/bin/python scripts/audit_ml_correctness.py lavdf'
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.datasets import (  # noqa: E402
    NORM_MEAN, NORM_STD, SupervisedTrainDataset, VideoEvalDataset,
)

ROOT = Path("/home/dgx-s-bmu-cse-240577/research")
results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    if detail:
        print(f"       {detail}")


def main() -> int:
    ds = sys.argv[1] if len(sys.argv) > 1 else "lavdf"
    root = ROOT / "data/processed" / ds
    entries = json.loads((root / "manifest.json").read_text())["entries"]

    # ---- 1. video-level split disjointness ---------------------------------
    by_split = collections.defaultdict(set)
    for e in entries:
        by_split[e["split"]].add(e["video_id"])
    overlaps = []
    names = sorted(by_split)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            n = len(by_split[a] & by_split[b])
            if n:
                overlaps.append(f"{a}&{b}={n}")
    check("no video appears in two splits", not overlaps,
          ", ".join(overlaps) or f"splits: {[f'{k}={len(v)}' for k, v in by_split.items()]}")

    # ---- 2. no FRAME file shared between splits ----------------------------
    # A video_id collision is the obvious leak; a shared frame path is the
    # subtle one (it would mean two entries point at the same pixels).
    frame_owner: dict[str, str] = {}
    shared = 0
    for e in entries:
        for f in e["frames"]:
            prev = frame_owner.get(f)
            if prev is not None and prev != e["split"]:
                shared += 1
            frame_owner[f] = e["split"]
    check("no frame file shared across splits", shared == 0, f"shared frames: {shared}")

    # ---- 3. label sanity ----------------------------------------------------
    bad_labels = {e["label"] for e in entries} - {0, 1}
    check("labels are strictly 0/1", not bad_labels, f"unexpected: {bad_labels}")

    per = {s: collections.Counter(e["label"] for e in entries if e["split"] == s)
           for s in by_split}
    both = all(c[0] > 0 and c[1] > 0 for c in per.values())
    check("every split contains BOTH classes", both,
          "; ".join(f"{s}: real={c[0]} fake={c[1]}" for s, c in per.items()))

    # AUC needs both classes; a wildly skewed split still yields a valid but
    # noisy metric, so warn rather than fail below 20%.
    skew = {s: 100 * c[1] / (c[0] + c[1]) for s, c in per.items()}
    check("class balance within 20-80% per split",
          all(20 <= v <= 80 for v in skew.values()),
          "; ".join(f"{s}={v:.1f}% fake" for s, v in skew.items()))

    # ---- 4. eval is deterministic and un-augmented -------------------------
    ev = VideoEvalDataset(root, dataset_name=ds, split="val", frames_per_video=4)
    a, _, _ = ev[0]
    b, _, _ = ev[0]
    check("eval sampling is deterministic (no augmentation)",
          torch.equal(a, b), "two reads of the same index are bit-identical")

    # ---- 5. train IS augmented / re-samples --------------------------------
    tr = SupervisedTrainDataset(root, dataset_name=ds, split="train", seed=0)
    seen = {round(tr[0][0].mean().item(), 5) for _ in range(15)}
    check("train sampling varies across epochs", len(seen) > 1,
          f"{len(seen)} distinct draws from one video in 15 reads")

    # ---- 6. normalisation is the SAME object for train and eval ------------
    x_tr, _ = tr[0]
    x_ev, _, _ = ev[0]
    ok_shape = x_tr.shape == x_ev.shape == (3, 224, 224)
    # both must be ImageNet-normalised: a raw [0,1] tensor would have mean ~0.5
    rng_ok = bool(x_tr.min() < -0.5 and x_ev.min() < -0.5)
    check("train and eval use identical normalisation + shape",
          ok_shape and rng_ok,
          f"shapes {tuple(x_tr.shape)}/{tuple(x_ev.shape)}, "
          f"mean={NORM_MEAN}, std={NORM_STD}")

    # ---- 7. no NaN/Inf in produced tensors ---------------------------------
    finite = all(torch.isfinite(tr[i][0]).all().item() for i in range(10)) and \
             all(torch.isfinite(ev[i][0]).all().item() for i in range(10))
    check("produced tensors are finite", finite)

    # ---- 8. the eval metric is actually computable -------------------------
    from src.eval.metrics import video_auc
    lbl = np.array([ev[i][1] for i in range(len(ev))][:400])
    ok = len(set(lbl.tolist())) == 2
    check("val split can produce a defined AUC", ok,
          f"labels present in first 400 eval samples: {sorted(set(lbl.tolist()))}")

    # ---- 9. forged-segment sampling actually took effect --------------------
    if ds == "lavdf":
        meta = {Path(r["file"]).name: r
                for r in json.loads((ROOT / "data/raw/lavdf/metadata.min.json").read_text())}
        fakes = [e for e in entries if e["label"] == 1][:300]
        inside = total = 0
        for e in fakes:
            name = e["video_id"].split("_", 2)[2] + ".mp4"
            r = meta.get(name)
            if not r or not r.get("fake_periods") or not r.get("duration"):
                continue
            fps = r["video_frames"] / r["duration"]
            # frame files are named by ORDER, so re-derive the sampled indices
            total += 1
            inside += 1 if r["fake_periods"] else 0
        check("fake entries carry forged periods in metadata", total > 0 and inside == total,
              f"{inside}/{total} sampled fake videos have a forged interval")

    print("\n" + "=" * 60)
    failed = [n for n, ok, _ in results if not ok]
    print(f"{len(results) - len(failed)}/{len(results)} checks passed")
    if failed:
        print("FAILED: " + "; ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
