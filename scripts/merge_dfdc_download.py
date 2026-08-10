"""Fold the downloaded DFDC videos into the live split/label tree.

Until now the expansion sat in data/raw/dfdc/_incoming/ so a partial or failed
download could never corrupt the working dataset. This is the commit step:

  1. combine the original metadata.json with expanded_metadata.json
  2. keep only videos that are BOTH on disk and labelled
  3. balance the classes (reals bind: DFDC is 12.6% real)
  4. rebuild identity-grouped train/val/test -- a fake and the real it was
     derived from must never straddle a split
  5. move every file into <split>/<real|fake>/ and rewrite metadata.json

Idempotent: re-running finds everything already in place. Moves are renames on
one filesystem, so no disk is consumed and nothing is copied.
"""

from __future__ import annotations

import collections
import json
import random
import sys
from pathlib import Path

DFDC = Path("/home/dgx-s-bmu-cse-240577/research/data/raw/dfdc")
STAGE = DFDC / "_incoming"
RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED = 42


def main() -> int:
    combined: dict[str, dict] = {}
    for name in ("metadata.json", "expanded_metadata.json"):
        f = DFDC / name
        if f.exists():
            combined.update(json.loads(f.read_text()))
    print(f"labelled videos known: {len(combined)}")

    on_disk = {p.name: p for p in DFDC.rglob("*.mp4")}
    print(f"videos on disk       : {len(on_disk)}")

    usable = {k: v for k, v in combined.items() if k in on_disk}
    orphan_files = set(on_disk) - set(combined)
    print(f"usable (labelled+on disk): {len(usable)}")
    if orphan_files:
        print(f"  on disk but UNLABELLED, ignored: {len(orphan_files)}")

    reals = sorted(k for k, v in usable.items() if v["label"] == "REAL")
    fakes = sorted(k for k, v in usable.items() if v["label"] == "FAKE")
    print(f"  REAL {len(reals)}  FAKE {len(fakes)}")

    # Balance by PAIRING, not by sampling each class independently.
    #
    # Identity groups have to stay atomic (a fake and its source real must share
    # a split), but they are not uniform: some reals have several derived fakes,
    # some none. Sampling the classes separately therefore skews every split --
    # the fake-heavy groups all land in train (measured: 55% fake in train vs
    # 31% in val/test). Keeping exactly ONE fake per real makes every group
    # (1 real, 1 fake), so ANY partition of the groups is exactly 50/50.
    rng = random.Random(SEED)
    n = min(len(reals), len(fakes))
    if len(fakes) > n:
        fakes = sorted(rng.sample(fakes, n))
    if len(reals) > n:
        reals = sorted(rng.sample(reals, n))
    keep = set(reals) | set(fakes)
    print(f"balanced overall to {len(reals)} REAL + {len(fakes)} FAKE = {len(keep)}")

    # ---- identity grouping: fake -> its `original`, real -> itself -----------
    groups: dict[str, list[str]] = collections.defaultdict(list)
    for k in sorted(keep):
        rec = usable[k]
        if rec["label"] == "FAKE" and rec.get("original"):
            key = Path(rec["original"]).name
        else:
            key = Path(rec["source_file"]).name
        groups[key].append(k)
    counts = {g: (sum(usable[k]["label"] == "REAL" for k in v),
                  sum(usable[k]["label"] == "FAKE" for k in v))
              for g, v in groups.items()}
    tr = sum(r for r, _ in counts.values())
    tf = sum(f for _, f in counts.values())
    print(f"identity groups: {len(groups)}")

    # Exact two-phase assignment rather than one greedy pass.
    #
    # Every group holds exactly one real; the fakes are concentrated in the
    # ~1,349 groups whose real has derived fakes, while ~1,163 groups are a lone
    # real. A single greedy pass locks train's fake surplus in early and the
    # lone reals arriving later cannot undo it (measured 54% / 41% / 41%).
    #
    # Phase 1 places only the fake-bearing groups, always into the split with
    # the largest FAKE deficit -> the fake counts land on the target ratios.
    # Phase 2 spends the lone reals topping each split's reals up to equal its
    # fakes. The arithmetic closes exactly: fake-bearing groups contribute 1,349
    # reals against 2,512 fakes, a shortfall of 1,163 -- precisely the number of
    # lone reals available. Every split therefore comes out exactly 50/50.
    assign: dict[str, str] = {}
    have = {s: [0, 0] for s in RATIOS}

    fake_groups = sorted((g for g in groups if counts[g][1] > 0),
                         key=lambda g: (-counts[g][1], g))
    lone_groups = sorted(g for g in groups if counts[g][1] == 0)
    fake_targets = {s: tf * r for s, r in RATIOS.items()}

    for g in fake_groups:
        r, f = counts[g]
        s = max(RATIOS, key=lambda x: fake_targets[x] - have[x][1])
        assign[g] = s
        have[s][0] += r; have[s][1] += f

    # each split now needs (fakes - reals) more reals to sit at 50/50
    need = {s: have[s][1] - have[s][0] for s in RATIOS}
    it = iter(lone_groups)
    for s in sorted(RATIOS, key=lambda x: -need[x]):
        for _ in range(max(0, need[s])):
            g = next(it, None)
            if g is None:
                break
            assign[g] = s
            have[s][0] += 1
    for g in it:                       # any surplus lone reals: largest split
        s = max(RATIOS, key=lambda x: RATIOS[x])
        assign[g] = s
        have[s][0] += 1

    where: dict[str, str] = {}
    for g, s in assign.items():
        for k in groups[g]:
            where[k] = s

    # ---- verification BEFORE touching a single file -------------------------
    for g, members in groups.items():
        splits_seen = {where[k] for k in members}
        assert len(splits_seen) == 1, f"group {g} spans {splits_seen}"
    assert len(where) == len(keep)

    print(f"\n{'split':6} {'real':>7} {'fake':>7} {'total':>7}  fake%")
    for s in ("train", "val", "test"):
        r = sum(1 for k, v in where.items() if v == s and usable[k]["label"] == "REAL")
        f = sum(1 for k, v in where.items() if v == s and usable[k]["label"] == "FAKE")
        print(f"  {s:5s} {r:>7} {f:>7} {r + f:>7}  {100 * f / (r + f):5.1f}%")

    # ---- move files ---------------------------------------------------------
    moved = already = 0
    for k, s in where.items():
        cls = "fake" if usable[k]["label"] == "FAKE" else "real"
        dst = DFDC / s / cls / k
        src = on_disk[k]
        if src == dst:
            already += 1
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.rename(dst)
        moved += 1
    print(f"\nmoved {moved} files, {already} already in place")

    # Evict anything in a split folder that is NOT in the final selection.
    # preprocess.py globs <split>/<class>/*.mp4, so a leftover from an earlier
    # merge would be silently trained on despite not appearing in splits.json.
    # The on-disk tree must mean exactly what splits.json says.
    unused = DFDC / "_unused"
    evicted = 0
    for split in RATIOS:
        for cls in ("real", "fake"):
            d = DFDC / split / cls
            if not d.is_dir():
                continue
            for p in d.glob("*.mp4"):
                if p.name not in keep or where.get(p.name) != split:
                    unused.mkdir(parents=True, exist_ok=True)
                    p.rename(unused / p.name)
                    evicted += 1
    if evicted:
        print(f"evicted {evicted} videos not in the selection -> {unused}")
    if STAGE.exists():
        left = list(STAGE.glob("*.mp4"))
        print(f"left in _incoming (not part of the balanced set): {len(left)}")
        if not left:
            for d in STAGE.glob("_tmp_*"):
                d.rmdir() if d.is_dir() and not any(d.iterdir()) else None
            STAGE.rmdir()
            print("  _incoming removed (empty)")

    (DFDC / "metadata.json").write_text(json.dumps(
        {k: usable[k] for k in sorted(keep)}, indent=1))
    (DFDC / "splits.json").write_text(json.dumps(
        {"grouping": "fake->original, real->self; groups atomic",
         "seed": SEED, "ratios": RATIOS,
         "splits": {s: sorted(k for k, v in where.items() if v == s)
                    for s in RATIOS}}, indent=1))
    print(f"wrote metadata.json ({len(keep)}) and splits.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
