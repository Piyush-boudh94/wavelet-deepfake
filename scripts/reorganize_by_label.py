"""Reorganize data/raw/{dfdc,lavdf} into <split>/<real|fake>/ folders.

Label on disk, so the layout is self-describing and a split's class balance is
one `ls | wc -l` away. Metadata stays authoritative; this only mirrors it.

    dfdc/   train|val|test / real|fake /
    lavdf/  train|val|test / real|fake /      (LAV-DF's `dev` becomes `val`)

LAV-DF labels come from `modify_video`, never `n_fakes`: audio-only forgeries
have untouched frames and are REAL to a visual detector.

Moves are renames within one filesystem, so no disk is consumed. Idempotent --
re-running finds everything already in place.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

RAW = Path("/home/dgx-s-bmu-cse-240577/research/data/raw")


def place(src: Path, dst: Path) -> str:
    if dst.exists():
        return "already"
    if not src.exists():
        return "MISSING"
    src.rename(dst)
    return "moved"


def reorganize_dfdc() -> dict:
    root = RAW / "dfdc"
    meta = json.loads((root / "metadata.json").read_text())
    splits = json.loads((root / "splits.json").read_text())["splits"]
    flat = root / "videos"

    counts: dict[str, dict[str, int]] = {}
    for split, names in splits.items():
        counts[split] = {"real": 0, "fake": 0}
        for lab in ("real", "fake"):
            (root / split / lab).mkdir(parents=True, exist_ok=True)
        for name in names:
            lab = "fake" if meta[name]["label"] == "FAKE" else "real"
            src = flat / name
            dst = root / split / lab / name
            if place(src, dst) == "MISSING":
                raise FileNotFoundError(f"dfdc: {src}")
            counts[split][lab] += 1

    if flat.exists() and not any(flat.iterdir()):
        flat.rmdir()
    return counts


def reorganize_lavdf() -> dict:
    root = RAW / "lavdf"
    meta = json.loads((root / "metadata.min.json").read_text())
    # LAV-DF calls its validation split "dev"; normalise to "val".
    rename = {"train": "train", "dev": "val", "test": "test"}

    counts: dict[str, dict[str, int]] = {s: {"real": 0, "fake": 0} for s in
                                         ("train", "val", "test")}
    for split in ("train", "val", "test"):
        for lab in ("real", "fake"):
            (root / split / lab).mkdir(parents=True, exist_ok=True)

    for rec in meta:
        split = rename[rec["split"]]
        lab = "fake" if rec["modify_video"] else "real"
        name = Path(rec["file"]).name
        src_dir = root / rec["split"]          # original flat dir (train/dev/test)
        dst = root / split / lab / name
        if place(src_dir / name, dst) == "MISSING":
            raise FileNotFoundError(f"lavdf: {src_dir / name}")
        counts[split][lab] += 1

    if (root / "dev").exists() and not any((root / "dev").iterdir()):
        (root / "dev").rmdir()
    return counts


def report(name: str, counts: dict) -> None:
    print(f"\n{name}")
    print(f"  {'split':6s} {'real':>8s} {'fake':>8s} {'total':>8s}   fake%")
    tr = tf = 0
    for split in ("train", "val", "test"):
        r, f = counts[split]["real"], counts[split]["fake"]
        tr += r
        tf += f
        print(f"  {split:6s} {r:8d} {f:8d} {r + f:8d}  {100 * f / (r + f):5.1f}%")
    print(f"  {'TOTAL':6s} {tr:8d} {tf:8d} {tr + tf:8d}  {100 * tf / (tr + tf):5.1f}%")


def main() -> int:
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("dfdc", "all"):
        report("DFDC  (data/raw/dfdc/<split>/<real|fake>/)", reorganize_dfdc())
    if which in ("lavdf", "all"):
        report("LAV-DF  (data/raw/lavdf/<split>/<real|fake>/)", reorganize_lavdf())
    return 0


if __name__ == "__main__":
    sys.exit(main())
