"""Full integrity + balance audit of data/raw/{dfdc,lavdf} before any deletion.

Checks, per dataset:
  1. every file on disk is a readable MP4 (ftyp box present, non-trivial size)
  2. metadata <-> disk agreement in BOTH directions (no orphans, no missing)
  3. exact-duplicate detection (size + SHA-256 of first 1 MiB, then full compare)
  4. real/fake balance, overall and per split

Read-only. Deletes nothing.
"""

from __future__ import annotations

import collections
import hashlib
import json
import sys
from pathlib import Path

RAW = Path("/home/dgx-s-bmu-cse-240577/research/data/raw")
MIN_BYTES = 10_000


def mp4_header_ok(p: Path) -> bool:
    """MP4/ISO-BMFF files carry a 'ftyp' box within the first few bytes."""
    try:
        with open(p, "rb") as fh:
            head = fh.read(32)
    except OSError:
        return False
    return b"ftyp" in head


def scan(files: list[Path], label: str) -> dict:
    bad_header, tiny, empty = [], [], []
    by_size: dict[int, list[Path]] = collections.defaultdict(list)

    for n, p in enumerate(files, 1):
        try:
            sz = p.stat().st_size
        except OSError:
            bad_header.append(p)
            continue
        if sz == 0:
            empty.append(p)
            continue
        if sz < MIN_BYTES:
            tiny.append(p)
        if not mp4_header_ok(p):
            bad_header.append(p)
        by_size[sz].append(p)
        if n % 20000 == 0:
            print(f"  [{label}] {n}/{len(files)} scanned", flush=True)

    # duplicate candidates: identical size -> hash first 1 MiB -> full hash
    dups: list[tuple[str, list[Path]]] = []
    for sz, group in by_size.items():
        if len(group) < 2:
            continue
        buckets: dict[str, list[Path]] = collections.defaultdict(list)
        for p in group:
            with open(p, "rb") as fh:
                buckets[hashlib.sha256(fh.read(1 << 20)).hexdigest()].append(p)
        for h, cand in buckets.items():
            if len(cand) < 2:
                continue
            full: dict[str, list[Path]] = collections.defaultdict(list)
            for p in cand:
                full[hashlib.sha256(p.read_bytes()).hexdigest()].append(p)
            dups.extend((h2, ps) for h2, ps in full.items() if len(ps) > 1)

    return {"empty": empty, "tiny": tiny, "bad_header": bad_header, "dups": dups}


def report(name: str, res: dict) -> bool:
    clean = True
    for key, tag in (("empty", "ZERO-BYTE"), ("bad_header", "NOT A VALID MP4"),
                     ("tiny", f"under {MIN_BYTES} bytes")):
        items = res[key]
        if items:
            clean = False
            print(f"  {tag}: {len(items)}  e.g. {[p.name for p in items[:3]]}")
    if res["dups"]:
        clean = False
        n = sum(len(ps) - 1 for _, ps in res["dups"])
        print(f"  EXACT DUPLICATES: {len(res['dups'])} groups, {n} redundant files")
        for _, ps in res["dups"][:3]:
            print(f"    {[p.name for p in ps]}")
    if clean:
        print("  no corrupt, empty, or duplicate files")
    return clean


def main() -> int:
    ok = True

    # ---------------------------------------------------------------- DFDC
    # Layout is <split>/<real|fake>/ since the 2026-08-04 expansion; the old
    # flat videos/ directory is gone.
    print("=" * 60)
    print("DFDC")
    meta = json.loads((RAW / "dfdc/metadata.json").read_text())
    splits = json.loads((RAW / "dfdc/splits.json").read_text())["splits"]
    vids = sorted((RAW / "dfdc").glob("*/*/*.mp4"))
    on_disk = {p.name for p in vids}
    print(f"  videos on disk: {len(vids)}   metadata entries: {len(meta)}")
    orphan = on_disk - set(meta)
    missing = set(meta) - on_disk
    print(f"  on disk without label : {len(orphan)}")
    print(f"  labelled but missing  : {len(missing)}")
    ok &= not orphan and not missing
    ok &= report("dfdc", scan(vids, "dfdc"))
    for s in ("train", "val", "test"):
        c = collections.Counter(meta[k]["label"] for k in splits[s])
        n = sum(c.values())
        print(f"  {s:5s} REAL {c['REAL']:5d} FAKE {c['FAKE']:5d}  "
              f"fake={100 * c['FAKE'] / n:.1f}%")

    # --------------------------------------------------------------- LAV-DF
    print("=" * 60)
    print("LAV-DF  (visual label = modify_video; audio-only forgery -> REAL)")
    lm = json.loads((RAW / "lavdf/metadata.min.json").read_text())
    by_split = collections.defaultdict(dict)
    for r in lm:
        by_split[r["split"]][Path(r["file"]).name] = bool(r["modify_video"])

    # Layout is <split>/<real|fake>/ and LAV-DF's `dev` was renamed `val`.
    for split, meta_split in (("train", "train"), ("val", "dev"), ("test", "test")):
        d = RAW / "lavdf" / split
        vids = sorted(d.glob("*/*.mp4"))
        on_disk = {p.name for p in vids}
        exp = by_split[meta_split]
        orphan, missing = on_disk - set(exp), set(exp) - on_disk
        # `missing` is expected here: the run is deliberately subsampled, so
        # most of LAV-DF's videos are on purpose absent from data/raw.
        print(f"  [{split}] on disk {len(vids)}  in metadata {len(exp)}  "
              f"orphan {len(orphan)}  (subsampled, not all metadata used)")
        ok &= not orphan
        ok &= report(split, scan(vids, split))
        r = sum(1 for p in vids if p.parent.name == "real")
        f = sum(1 for p in vids if p.parent.name == "fake")
        if r + f:
            print(f"  [{split}] BALANCE  REAL {r}  FAKE {f}  "
                  f"fake={100 * f / (r + f):.1f}%")

    print("=" * 60)
    print("AUDIT CLEAN" if ok else "AUDIT FOUND ISSUES (see above)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
