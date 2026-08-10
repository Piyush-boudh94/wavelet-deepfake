"""Find DFDC fakes that are audio-only swaps (video track untouched).

The public DFDC release contains fakes where only the audio was replaced. Our
metadata has no audio flag, so those sit in the set labelled FAKE while their
frames are pixel-identical to the source real. For a visual detector that is
label noise -- the same class of error we excluded from LAV-DF by labelling on
`modify_video`.

Test: decode evenly-spaced frames from the fake and from its `original` and
compare them pixel-for-pixel. Identical frames => the video was never touched.

Only fakes whose `original` is present on disk can be checked (the rest
reference DFDC parts that were never transferred). Writes
data/raw/dfdc/audio_only.json. Read-only w.r.t. the videos.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import cv2

DFDC = Path("/home/dgx-s-bmu-cse-240577/research/data/raw/dfdc")
VIDEOS = DFDC / "videos"
N_FRAMES = 10


def frame_digests(path: Path, n: int) -> tuple[list[str], int, tuple[int, int]]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return [], 0, (0, 0)
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        wh = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
              int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if total <= 0:
            return [], 0, wh
        idxs = [int(i * (total - 1) / max(n - 1, 1)) for i in range(min(n, total))]
        out = []
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, frame = cap.read()
            if not ok:
                return out, total, wh
            out.append(hashlib.md5(frame.tobytes()).hexdigest())
        return out, total, wh
    finally:
        cap.release()


def main() -> int:
    meta = json.loads((DFDC / "metadata.json").read_text())
    on_disk = {p.name for p in VIDEOS.glob("*.mp4")}

    # map bare DFDC name (e.g. 'gtgnhgzrye.mp4') -> our prefixed name
    bare = {Path(rec["source_file"]).name: name for name, rec in meta.items()}

    pairs = []
    unresolved = 0
    for name, rec in meta.items():
        if rec["label"] != "FAKE":
            continue
        orig = rec.get("original")
        src = bare.get(Path(orig).name) if orig else None
        if src and src in on_disk:
            pairs.append((name, src))
        else:
            unresolved += 1

    print(f"FAKE videos: {sum(v['label'] == 'FAKE' for v in meta.values())}")
    print(f"  checkable (original present): {len(pairs)}")
    print(f"  unresolvable (source never transferred): {unresolved}")
    print(f"comparing {N_FRAMES} evenly-spaced frames per pair...\n", flush=True)

    audio_only, video_modified, unreadable = [], [], []
    for n, (fake, real) in enumerate(pairs, 1):
        fd, ftot, fwh = frame_digests(VIDEOS / fake, N_FRAMES)
        rd, rtot, rwh = frame_digests(VIDEOS / real, N_FRAMES)
        if not fd or not rd:
            unreadable.append((fake, real))
        elif fwh != rwh or ftot != rtot:
            video_modified.append((fake, real))   # geometry differs -> re-encoded
        elif fd == rd:
            audio_only.append((fake, real))
        else:
            video_modified.append((fake, real))
        if n % 50 == 0:
            print(f"  {n}/{len(pairs)}  audio_only={len(audio_only)}", flush=True)

    print(f"\nRESULT over {len(pairs)} checkable fakes:")
    print(f"  audio-only (frames identical to source) : {len(audio_only)}")
    print(f"  genuinely video-modified                : {len(video_modified)}")
    print(f"  unreadable                              : {len(unreadable)}")
    if audio_only:
        print(f"  examples: {[f for f, _ in audio_only[:5]]}")

    out = DFDC / "audio_only.json"
    out.write_text(json.dumps({
        "method": f"pixel comparison of {N_FRAMES} evenly-spaced frames vs `original`",
        "checkable": len(pairs),
        "unresolvable": unresolved,
        "audio_only": [f for f, _ in audio_only],
        "unreadable": [f for f, _ in unreadable],
    }, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
