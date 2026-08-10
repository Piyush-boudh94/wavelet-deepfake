"""Expand data/raw/dfdc to the largest balanced set the mirror allows.

Measured from the 10 part-metadata files: 19,909 videos, 2,512 REAL (12.6%),
17,397 FAKE. Real videos are the binding constraint, so the balanced ceiling is
2,512 + 2,512 = 5,024 -- 3.1x the 1,600 currently held.

Fake selection prefers fakes whose `original` is a real we are also keeping, so
identity grouping in build_dfdc_splits.py has real pairs to work with rather
than orphan fakes.

Downloads only what is missing, one file at a time, and is safe to re-run after
a pod eviction or a killed connection. Videos land in a staging dir; nothing
touches the live split/label tree until merge_dfdc_download.py runs.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path("/home/dgx-s-bmu-cse-240577/research")
DFDC = ROOT / "data/raw/dfdc"
META_DIR = DFDC / "part_metadata"
STAGE = DFDC / "_incoming"
SLUG = "pranay22077/dfdc-10"
KAGGLE = Path.home() / ".local/bin/kaggle"
PARTS = range(10)
SEED = 42


def remote_path(part: int, fname: str) -> str:
    return f"dfdc_train_part_{part:02d}/dfdc_train_part_{part}/{fname}"


def local_name(part: int, fname: str) -> str:
    return f"p{part:02d}_{fname}"


def build_selection() -> tuple[list[tuple[int, str, str]], dict]:
    """-> [(part, remote_fname, label)], merged metadata for the selection."""
    reals: list[tuple[int, str]] = []
    fakes: list[tuple[int, str, str | None]] = []
    meta_by_part: dict[int, dict] = {}

    for p in PARTS:
        f = META_DIR / f"part_{p:02d}.json"
        if not f.exists():
            raise FileNotFoundError(f"missing part metadata: {f}")
        m = json.loads(f.read_text())
        meta_by_part[p] = m
        for name, rec in m.items():
            if rec["label"] == "REAL":
                reals.append((p, name))
            else:
                fakes.append((p, name, rec.get("original")))

    rng = random.Random(SEED)
    rng.shuffle(reals)
    real_keys = {name for _, name in reals}

    # prefer fakes whose source real is in the kept set -> usable identity groups
    paired = [f for f in fakes if f[2] in real_keys]
    orphan = [f for f in fakes if f[2] not in real_keys]
    rng.shuffle(paired)
    rng.shuffle(orphan)
    chosen_fakes = (paired + orphan)[:len(reals)]

    selection: list[tuple[int, str, str]] = []
    merged: dict = {}
    for p, name in reals:
        selection.append((p, name, "REAL"))
        merged[local_name(p, name)] = {
            "label": "REAL", "split": "train", "source_part": p,
            "source_file": remote_path(p, name),
        }
    for p, name, orig in chosen_fakes:
        selection.append((p, name, "FAKE"))
        merged[local_name(p, name)] = {
            "label": "FAKE", "split": "train", "source_part": p,
            "source_file": remote_path(p, name), "original": orig,
        }
    print(f"selection: {len(reals)} REAL + {len(chosen_fakes)} FAKE = {len(selection)}")
    print(f"  fakes with their source real also kept: "
          f"{sum(1 for f in chosen_fakes if f[2] in real_keys)}")
    return selection, merged


def already_have() -> set[str]:
    """Videos present in the live tree or already staged."""
    have = {p.name for p in DFDC.rglob("*.mp4")}
    return have


_throttle_lock = threading.Lock()
_last_call = [0.0]
# seconds between API calls, ACROSS all threads (env-overridable so the
# rate-limit recovery wrapper can dial it up without editing this file)
MIN_INTERVAL = float(os.environ.get("MIN_INTERVAL_OVERRIDE", "1.0"))


def _throttle() -> None:
    """Global minimum spacing between Kaggle API calls.

    Kaggle rate-limits per account, not per connection, so having each thread
    sleep independently does nothing. 6 unthrottled workers earned a blanket
    HTTP 429 (2,592 consecutive failures); this bounds the aggregate call rate.
    """
    with _throttle_lock:
        wait = MIN_INTERVAL - (time.time() - _last_call[0])
        if wait > 0:
            time.sleep(wait)
        _last_call[0] = time.time()


def download_one(part: int, fname: str, dest: Path,
                 retries: int = 2) -> tuple[bool, str]:
    """Fetch one video. Returns (ok, error). Retries 429 with exponential backoff.

    Each call gets its OWN temp directory: `--force` clears the target dir, so a
    shared per-part dir would let one thread wipe another's in-flight file.
    """
    out = dest / local_name(part, fname)
    if out.exists() and out.stat().st_size > 100_000:
        return True, ""
    tmp = dest / f"_tmp_{local_name(part, fname)[:-4]}"

    err = "unknown"
    for attempt in range(retries):
        tmp.mkdir(parents=True, exist_ok=True)
        _throttle()
        try:
            r = subprocess.run(
                [str(KAGGLE), "datasets", "download", SLUG, "-f",
                 remote_path(part, fname), "-p", str(tmp), "--force", "-q"],
                capture_output=True, text=True, timeout=600,
            )
            got = tmp / fname
            if got.exists() and got.stat().st_size > 0:
                got.rename(out)          # atomic within one filesystem
                shutil.rmtree(tmp, ignore_errors=True)
                return True, ""
            err = ((r.stderr or "") + (r.stdout or "")).strip()[:200] or "no file produced"
        except subprocess.TimeoutExpired:
            err = "timeout"
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

        if "429" in err or "Too Many Requests" in err:
            # back off hard and globally -- the limit is per account
            sleep = min(60, 4 * (2 ** attempt))
            with _throttle_lock:
                _last_call[0] = time.time() + sleep
            time.sleep(sleep)
        else:
            return False, err
    return False, err


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="debug: cap downloads")
    ap.add_argument("--workers", type=int, default=3,
                    help="parallel downloads; keep modest to avoid Kaggle throttling")
    args = ap.parse_args()

    STAGE.mkdir(parents=True, exist_ok=True)
    selection, merged = build_selection()
    (DFDC / "expanded_metadata.json").write_text(json.dumps(merged, indent=1))

    have = already_have()
    todo = [(p, n, l) for p, n, l in selection if local_name(p, n) not in have]
    print(f"already on disk: {len(selection) - len(todo)}   to download: {len(todo)}")
    if args.limit:
        todo = todo[:args.limit]

    # Most of the serial cost was `kaggle` CLI startup (~4 s/file), not
    # bandwidth, so threads help far more than the file sizes suggest.
    ok = fail = done = 0
    t0 = time.time()
    failures: list[str] = []
    lock = threading.Lock()

    err_kinds: collections.Counter = collections.Counter()
    # Circuit breaker. A banned account fails EVERY file, and grinding 3,643 of
    # them through retries is what turns a short rate-limit into a long one.
    # Trip early, let the wrapper wait, come back later.
    BREAKER_AFTER = 15
    consecutive_429 = 0
    tripped = threading.Event()

    def work(task):
        nonlocal ok, fail, done, consecutive_429
        if tripped.is_set():
            return
        p, n, _ = task
        got, err = download_one(p, n, STAGE)
        with lock:
            done += 1
            if got:
                ok += 1
            else:
                fail += 1
                failures.append(remote_path(p, n))
                kind = "429" if "429" in err else (err.split(":")[0][:40] or "unknown")
                err_kinds[kind] += 1
                # surface the FIRST failure of each kind immediately -- the
                # previous version swallowed stderr and ran 2,592 silent 429s
                if err_kinds[kind] == 1:
                    print(f"  !! first {kind!r} failure: {err[:160]}", flush=True)
            if got:
                consecutive_429 = 0
            elif "429" in err:
                consecutive_429 += 1
                if consecutive_429 >= BREAKER_AFTER and not tripped.is_set():
                    tripped.set()
                    print(f"  ** CIRCUIT BREAKER: {consecutive_429} consecutive 429s "
                          f"and {ok} successes -- aborting so the ban can expire",
                          flush=True)
            if done % 25 == 0:
                el = time.time() - t0
                rate = done / el * 60
                left = (len(todo) - done) / rate if rate else 0
                gb = sum(f.stat().st_size for f in STAGE.glob("*.mp4")) / 2**30
                print(f"  {done}/{len(todo)}  ok={ok} fail={fail}  {gb:.1f} GiB  "
                      f"{rate:.0f} files/min  ETA {left:.0f} min", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, todo))

    print(f"\nDONE ok={ok} fail={fail} in {(time.time() - t0) / 60:.1f} min")
    if tripped.is_set():
        print("  exiting 75 (rate-limited) -- wrapper will retry later")
    if err_kinds:
        print("  failure breakdown:", dict(err_kinds))
    if failures:
        (DFDC / "download_failures.json").write_text(json.dumps(failures, indent=1))
        print(f"  failures -> {DFDC / 'download_failures.json'}")
    for t in STAGE.glob("_tmp_*"):
        if t.is_dir():
            shutil.rmtree(t, ignore_errors=True)
    return 75 if tripped.is_set() else 0


if __name__ == "__main__":
    sys.exit(main())
