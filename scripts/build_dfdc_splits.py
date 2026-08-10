"""Build leakage-free train/val/test splits for DFDC.

The transferred DFDC subset carries no official split -- every entry in
metadata.json is marked "train". Splitting it at random would put a fake and the
real video it was derived from on opposite sides, so the model would meet the
same face in training and testing.

Grouping rule (identity proxy):
    fake -> the `original` it was generated from
    real -> its own source filename
A real and every fake derived from it therefore share one group, and groups are
assigned to splits atomically.

Writes data/raw/dfdc/splits.json. Read-only w.r.t. the videos.
"""

from __future__ import annotations

import collections
import json
import random
from pathlib import Path

DFDC = Path("/home/dgx-s-bmu-cse-240577/research/data/raw/dfdc")
RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED = 42


def main() -> int:
    meta = json.loads((DFDC / "metadata.json").read_text())

    groups: dict[str, list[str]] = collections.defaultdict(list)
    for name, rec in meta.items():
        if rec["label"] == "FAKE" and rec.get("original"):
            key = Path(rec["original"]).name
        else:
            key = Path(rec["source_file"]).name
        groups[key].append(name)

    counts = {
        k: (sum(meta[n]["label"] == "REAL" for n in v),
            sum(meta[n]["label"] == "FAKE" for n in v))
        for k, v in groups.items()
    }
    total_real = sum(r for r, _ in counts.values())
    total_fake = sum(f for _, f in counts.values())
    print(f"videos {len(meta)}  identity groups {len(groups)}  "
          f"REAL {total_real}  FAKE {total_fake}")
    print(f"largest group: {max(len(v) for v in groups.values())} videos")

    # Greedy over groups, largest first. Real and fake counts are balanced as
    # two independent dimensions: a group goes to whichever split minimises the
    # total squared deviation from every split's real AND fake target. Balancing
    # only the totals skews the class ratio badly, because the groups are very
    # unlike each other -- 585 are a lone real, 386 are fakes whose source video
    # was never transferred, and 215 are a real plus the fakes made from it.
    targets = {s: (total_real * r, total_fake * r) for s, r in RATIOS.items()}
    have = {s: [0, 0] for s in RATIOS}
    assign: dict[str, str] = {}

    def cost() -> float:
        return sum((have[s][0] - targets[s][0]) ** 2 + (have[s][1] - targets[s][1]) ** 2
                   for s in RATIOS)

    rng = random.Random(SEED)
    order = sorted(groups, key=lambda k: (-sum(counts[k]), k))
    rng.shuffle(order)
    order.sort(key=lambda k: -sum(counts[k]))

    for key in order:
        r, f = counts[key]
        best, best_cost = None, float("inf")
        for s in RATIOS:
            have[s][0] += r
            have[s][1] += f
            c = cost()
            have[s][0] -= r
            have[s][1] -= f
            if c < best_cost:
                best, best_cost = s, c
        assign[key] = best
        have[best][0] += r
        have[best][1] += f

    splits: dict[str, list[str]] = {s: [] for s in RATIOS}
    for key, s in assign.items():
        splits[s].extend(sorted(groups[key]))
    for s in splits:
        splits[s].sort()

    # ---- verification -------------------------------------------------------
    seen: dict[str, str] = {}
    for s, names in splits.items():
        for n in names:
            assert n not in seen, f"{n} in two splits"
            seen[n] = s
    assert len(seen) == len(meta), f"{len(seen)} assigned vs {len(meta)} videos"

    for key, members in groups.items():
        where = {seen[n] for n in members}
        assert len(where) == 1, f"group {key} split across {where}"

    print("\nsplit          REAL   FAKE  total   fake%")
    for s in ("train", "val", "test"):
        r = sum(meta[n]["label"] == "REAL" for n in splits[s])
        f = len(splits[s]) - r
        print(f"  {s:5s} {r:8d} {f:6d} {r + f:6d}  {100 * f / (r + f):5.1f}%")

    out = DFDC / "splits.json"
    out.write_text(json.dumps(
        {"grouping": "fake->original, real->self; groups are atomic",
         "seed": SEED, "ratios": RATIOS, "splits": splits}, indent=2))
    print(f"\nno group spans two splits -- wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
