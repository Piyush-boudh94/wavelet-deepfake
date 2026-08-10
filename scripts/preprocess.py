"""Videos -> face crops + manifest.json, for any <split>/<real|fake>/ dataset.

    data/raw/<ds>/<split>/<real|fake>/*.mp4
        -> data/processed/<ds>/frames/<video_id>/<nnnn>.png
        -> data/processed/<ds>/manifest.json

Margin handling (the reason crops are not stored at their final size):
[PAPER B.2] uses a RANDOM 4-20% crop margin at train time and a fixed 12.5% at
inference. Storing a crop at one margin would freeze that augmentation, so each
frame is saved at the WIDEST margin (20%) and the face box is recorded in
saved-crop coordinates. The loader then re-crops to whatever margin it wants.
Saving at 256px leaves headroom so the tightest re-crop still exceeds the 224
the model consumes.

Resumable: a video whose frames and manifest entry already exist is skipped, so
an evicted pod costs only the video it was mid-way through.
"""

from __future__ import annotations

import argparse
import json
import zlib
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.faces import FaceDetector, LandmarkPredictor, expand_box  # noqa: E402

ROOT = Path("/home/dgx-s-bmu-cse-240577/research")
STORE_MARGIN = 0.20      # widest margin the loader may ask for
STORE_SIZE = 256         # > 224 so a tighter re-crop still has pixels to spare
CLASS_LABEL = {"real": 0, "fake": 1}


def sample_frame_indices(total: int, n: int,
                         periods: list[tuple[float, float]] | None = None,
                         fps: float | None = None) -> list[int]:
    """Frame indices to keep. Evenly spaced, unless forged periods are given.

    LAV-DF is a temporal-LOCALISATION dataset: a video labelled fake has only a
    short manipulated segment, median 7.4% of its duration. Sampling uniformly
    therefore hands the model ~92% genuine frames labelled "fake" -- measured,
    and it capped the first training run at 0.763 val AUC with a flat curve.

    When `periods` (seconds) and `fps` are supplied, frames are drawn only from
    inside the forged intervals, so a "fake" sample actually contains forgery.
    Real videos pass periods=None and keep uniform sampling, which is correct
    for them: every frame is genuine.
    """
    if total <= 0:
        return []
    if periods and fps and fps > 0:
        cand: list[int] = []
        for a, b in periods:
            lo = max(0, int(np.floor(a * fps)))
            hi = min(total - 1, int(np.ceil(b * fps)))
            if hi >= lo:
                cand.extend(range(lo, hi + 1))
        cand = sorted(set(cand))
        if cand:
            k = min(n, len(cand))
            pick = np.linspace(0, len(cand) - 1, num=k).round().astype(int)
            return sorted({cand[i] for i in pick})
        # fall through to uniform if the periods degenerate
    k = min(n, total)
    return sorted({int(round(i)) for i in np.linspace(0, total - 1, num=k)})


def process_video(path: Path, out_dir: Path, detector: FaceDetector,
                  landmarker: LandmarkPredictor | None, n_frames: int,
                  periods: list | None = None) -> dict | None:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"error": "open_failed"}
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        idxs = sample_frame_indices(total, n_frames, periods, fps)
        if not idxs:
            return {"error": "no_frames"}

        frames, boxes, lms_paths = [], [], []
        out_dir.mkdir(parents=True, exist_ok=True)
        for n, i in enumerate(idxs):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, bgr = cap.read()
            if not ok:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            dets = detector.detect(rgb)
            if not dets:
                continue
            # [PAPER B.4] multi-face: highest-scoring detection wins
            box, _ = dets[0]
            h, w = rgb.shape[:2]
            x1, y1, x2, y2 = expand_box(box, STORE_MARGIN, h, w)   # (img_h, img_w)
            crop = rgb[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            ch, cw = crop.shape[:2]
            resized = cv2.resize(crop, (STORE_SIZE, STORE_SIZE),
                                 interpolation=cv2.INTER_AREA)

            rel = f"{n:04d}.png"
            cv2.imwrite(str(out_dir / rel), cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))
            frames.append(rel)
            # face box in SAVED-crop coordinates, normalised 0-1
            bx1, by1, bx2, by2 = box
            boxes.append([
                float((bx1 - x1) / cw), float((by1 - y1) / ch),
                float((bx2 - x1) / cw), float((by2 - y1) / ch),
            ])

            if landmarker is not None:
                try:
                    lm = landmarker.predict(resized)
                except Exception:
                    lm = None
                if lm is not None:
                    lp = f"{n:04d}.npy"
                    np.save(out_dir / lp, lm.astype(np.float32))
                    lms_paths.append(lp)

        if not frames:
            return {"error": "no_face_detected"}
        return {"frames": frames, "boxes": boxes, "landmarks": lms_paths}
    finally:
        cap.release()


def load_fake_periods(dataset: str) -> dict[str, list]:
    """basename -> forged time ranges, for datasets that localise their fakes.

    Only LAV-DF carries this. DFDC fakes are forged throughout, so uniform
    sampling is already correct there and this returns empty.
    """
    if dataset != "lavdf":
        return {}
    mf = ROOT / "data/raw/lavdf/metadata.min.json"
    out: dict[str, list] = {}
    for r in json.loads(mf.read_text()):
        if r.get("modify_video") and r.get("fake_periods"):
            out[Path(r["file"]).name] = r["fake_periods"]
    print(f"loaded forged periods for {len(out)} visual-fake videos")
    return out


def load_all_done(out: Path) -> dict[str, dict]:
    """Every video already processed, across the main manifest AND all shards.

    Shards must see each other's work (and any pre-shard single-process run) or
    they would redo videos other shards already finished.
    """
    done: dict[str, dict] = {}
    for f in [out / "manifest.json", *sorted(out.glob("manifest.shard*.json"))]:
        if not f.exists():
            continue
        try:
            for e in json.loads(f.read_text())["entries"]:
                done[e["video_id"]] = e
        except (json.JSONDecodeError, KeyError):
            # a shard caught mid-write: ignore it this pass rather than crash
            print(f"  (skipping unreadable {f.name}, will re-read next run)")
    return done


def resolve_selection(out: Path, split: str, cls: str, pool: list[Path],
                      per_class: int, done: dict[str, dict],
                      seed: int) -> list[Path]:
    """Which videos of this class belong to the run. Decided ONCE, then frozen.

    Originally the subsample was drawn fresh each run from
    `hash(split + cls)` -- but Python randomises str hashing per process, so
    every restart drew a DIFFERENT subset, and parallel shards would each draw
    their own. The budget would be blown and work duplicated.

    So the choice is persisted in selection.json on first use. Videos already
    processed are adopted into the selection rather than discarded, which keeps
    every completed video useful while still landing exactly on `per_class`.
    """
    sel_path = out / "selection.json"
    sel = {}
    if sel_path.exists():
        try:
            sel = json.loads(sel_path.read_text())
        except json.JSONDecodeError:
            sel = {}

    key = f"{split}/{cls}"
    if key in sel:
        want = set(sel[key])
        return [p for p in pool if p.name in want]

    if per_class >= len(pool):
        chosen = [p.name for p in pool]
    else:
        already = [p.name for p in pool
                   if f"{split}_{cls}_{p.stem}" in done]
        rest = [p.name for p in pool if p.name not in set(already)]
        need = max(0, per_class - len(already))
        salt = zlib.crc32(key.encode()) % 10_000
        rng = np.random.default_rng(seed + salt)
        pick = rng.choice(len(rest), size=min(need, len(rest)), replace=False)
        chosen = sorted(already + [rest[i] for i in pick])
        print(f"  selection[{key}]: {len(already)} already done adopted "
              f"+ {len(pick)} new = {len(chosen)}")

    sel[key] = chosen
    sel_path.write_text(json.dumps(sel, indent=1))
    want = set(chosen)
    return [p for p in pool if p.name in want]


def verify_complete(out: Path, dataset: str) -> int:
    """Exit 0 only when every video in selection.json has a manifest entry.

    Needed because a merge happens whenever the shards exit -- including when
    they crash. Treating "merged" as "finished" started a training run on 1,666
    of 17,387 fake videos. Completion is a property of the DATA, not of a log
    line, so it is measured here against the frozen selection.
    """
    sel_path = out / "selection.json"
    if not sel_path.exists():
        print("VERIFY FAIL: selection.json missing")
        return 1
    sel = json.loads(sel_path.read_text())
    done = load_all_done(out)

    # video_id is "<split>_<class>_<stem>"
    ok = True
    total_missing = 0
    print(f"{'split/class':16} {'chosen':>8} {'done':>8} {'missing':>8}")
    for key, names in sorted(sel.items()):
        split, cls = key.split("/")
        want = {f"{split}_{cls}_{Path(n).stem}" for n in names}
        missing = want - set(done)
        total_missing += len(missing)
        flag = "" if not missing else "  <-- INCOMPLETE"
        print(f"{key:16} {len(want):>8} {len(want) - len(missing):>8} "
              f"{len(missing):>8}{flag}")
        ok &= not missing

    if ok:
        print(f"PREPROCESSING COMPLETE: all {len(done)} selected videos processed")
        return 0
    print(f"PREPROCESSING INCOMPLETE: {total_missing} videos still to do")
    return 1


def merge_shards(out: Path, dataset: str) -> int:
    entries = list(load_all_done(out).values())
    if not entries:
        print("nothing to merge")
        return 1
    (out / "manifest.json").write_text(json.dumps(
        {"dataset": dataset, "store_margin": STORE_MARGIN,
         "store_size": STORE_SIZE, "entries": entries}, indent=1))
    import collections as _c
    by = _c.Counter((e["split"], e["label"]) for e in entries)
    print(f"merged {len(entries)} videos into {out / 'manifest.json'}")
    for s in ("train", "val", "test"):
        print(f"  {s:5s} real={by[(s, 0)]:6d} fake={by[(s, 1)]:6d}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="dfdc | lavdf | self_created")
    ap.add_argument("--splits", nargs="*", default=["train", "val", "test"])
    ap.add_argument("--train-frames", type=int, default=16,
                    help="frames kept per TRAIN video (loader samples 8 of these)")
    ap.add_argument("--eval-frames", type=int, default=32, help="[PAPER B.4]")
    ap.add_argument("--landmarks", action="store_true",
                    help="also store dlib 81-pt landmarks (only needed for SBI)")
    ap.add_argument("--limit", type=int, default=0, help="debug: cap videos per class")
    ap.add_argument("--subsample", type=str, default="",
                    help="per-split video budget, e.g. 'train=20000,val=4000,test=6000'. "
                         "Split evenly across real/fake, chosen by seeded shuffle so the "
                         "selection is reproducible and recorded in the manifest.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shard", type=str, default="",
                    help="'i/N' -- process only every Nth video, offset i. Lets N "
                         "processes share the job. One RetinaFace instance uses "
                         "~0.7 GiB of 16, so a single process leaves the GPU 95%% "
                         "idle; shards fill it. Each writes its own manifest "
                         "shard, merged by --merge.")
    ap.add_argument("--merge", action="store_true",
                    help="combine manifest.shard*.json into manifest.json and exit")
    ap.add_argument("--verify", action="store_true",
                    help="check the manifest against selection.json and exit 0 only "
                         "if every chosen video was processed. 'merged' is NOT proof "
                         "of completion -- shards merge whenever they exit, crash "
                         "included.")
    args = ap.parse_args()

    out_root = ROOT / "data/processed" / args.dataset
    if args.merge:
        return merge_shards(out_root, args.dataset)
    if args.verify:
        return verify_complete(out_root, args.dataset)

    shard_i, shard_n = 0, 1
    if args.shard:
        shard_i, shard_n = (int(x) for x in args.shard.split("/"))
        if not 0 <= shard_i < shard_n:
            raise ValueError(f"bad --shard {args.shard}")

    budget: dict[str, int] = {}
    for part in filter(None, args.subsample.split(",")):
        k, _, v = part.partition("=")
        budget[k.strip()] = int(v)

    raw = ROOT / "data/raw" / args.dataset
    out = ROOT / "data/processed" / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    # Each shard owns ONE manifest file; it reads everyone's to know what is
    # already done, but only ever writes its own. Two processes writing one JSON
    # file would interleave and corrupt it.
    manifest_path = (out / f"manifest.shard{shard_i}.json" if shard_n > 1
                     else out / "manifest.json")

    done = load_all_done(out)
    if done:
        print(f"resuming: {len(done)} videos already processed (all shards)")

    detector = FaceDetector()
    landmarker = LandmarkPredictor() if args.landmarks else None
    fake_periods = load_fake_periods(args.dataset)

    # only this shard's own prior entries -- other shards keep theirs
    mine: dict[str, dict] = {}
    if manifest_path.exists():
        try:
            mine = {e["video_id"]: e
                    for e in json.loads(manifest_path.read_text())["entries"]}
        except (json.JSONDecodeError, KeyError):
            mine = {}
    entries = list(mine.values())
    failures: list[tuple[str, str]] = []
    for split in args.splits:
        n_frames = args.train_frames if split == "train" else args.eval_frames
        for cls, label in CLASS_LABEL.items():
            if args.dataset == "self_created":
                manifest_csv = ROOT / "data" / args.dataset / "splits" / f"{split}_manifest.csv"
                vids = []
                import csv
                with open(manifest_csv) as f:
                    for row in csv.DictReader(f):
                        if (cls == "fake" and int(row["label"]) == 1) or (cls == "real" and int(row["label"]) == 0):
                            vids.append(ROOT / "data" / args.dataset / row["video_path"])
                vids = sorted(vids)
            else:
                d = raw / split / cls
                if not d.is_dir():
                    raise FileNotFoundError(f"missing {d}")
                vids = sorted(d.glob("*.mp4"))
            if split in budget:
                vids = resolve_selection(out, split, cls, vids,
                                         budget[split] // 2, done, args.seed)
            if shard_n > 1:
                vids = vids[shard_i::shard_n]
            if args.limit:
                vids = vids[:args.limit]
            tag = f"shard{shard_i}" if shard_n > 1 else "single"
            print(f"[{tag}] [{split}/{cls}] {len(vids)} videos, "
                  f"{n_frames} frames each", flush=True)
            for n, v in enumerate(vids, 1):
                vid = f"{split}_{cls}_{v.stem}"
                if vid in done:
                    continue
                # forged-segment sampling for fakes; uniform for reals
                per = fake_periods.get(v.name) if cls == "fake" else None
                res = process_video(v, out / "frames" / vid, detector,
                                    landmarker, n_frames, per)
                if res is None or "error" in res:
                    failures.append((vid, (res or {}).get("error", "unknown")))
                    continue
                entries.append({
                    "video_id": vid,
                    "label": label,
                    "split": split,
                    "source": str(v.relative_to(ROOT)),
                    "frames": [f"frames/{vid}/{f}" for f in res["frames"]],
                    "boxes": res["boxes"],
                    "landmarks": [f"frames/{vid}/{f}" for f in res["landmarks"]],
                })
                if n % 100 == 0:
                    print(f"  {n}/{len(vids)}  ok={len(entries)} fail={len(failures)}",
                          flush=True)
                    manifest_path.write_text(json.dumps(
                        {"dataset": args.dataset, "store_margin": STORE_MARGIN,
                         "store_size": STORE_SIZE, "entries": entries}, indent=1))

    manifest_path.write_text(json.dumps(
        {"dataset": args.dataset, "store_margin": STORE_MARGIN,
         "store_size": STORE_SIZE, "subsample": args.subsample or None,
         "seed": args.seed, "entries": entries}, indent=1))

    print(f"\nwrote {manifest_path}: {len(entries)} videos")
    if failures:
        print(f"FAILED {len(failures)} videos (no usable face):")
        for vid, why in failures[:10]:
            print(f"  {why:20s} {vid}")
        (out / "failures.json").write_text(json.dumps(failures, indent=1))
        print(f"  full list -> {out / 'failures.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
