"""Drop every processed LAV-DF FAKE video so it is re-cut from forged segments.

The first pass sampled frames uniformly across each video. LAV-DF localises its
forgeries -- median 7.4% of a fake video's duration is actually manipulated --
so ~92% of those crops were genuine footage labelled "fake". Training plateaued
at 0.763 val AUC on a flat curve, which is the ceiling that noise allows.

Real videos are untouched: uniform sampling is correct for them, every frame is
genuine. Only fakes are purged, so the re-run costs ~half of a full pass.

Destructive but safe: frames are derived data and the raw videos are intact.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

OUT = Path("/home/dgx-s-bmu-cse-240577/research/data/processed/lavdf")


def main() -> int:
    manifests = [OUT / "manifest.json", *sorted(OUT.glob("manifest.shard*.json"))]
    fake_ids: set[str] = set()

    for mf in manifests:
        if not mf.exists():
            continue
        doc = json.loads(mf.read_text())
        keep = [e for e in doc["entries"] if e["label"] == 0]
        dropped = [e for e in doc["entries"] if e["label"] == 1]
        fake_ids.update(e["video_id"] for e in dropped)
        doc["entries"] = keep
        mf.write_text(json.dumps(doc, indent=1))
        print(f"{mf.name:26} kept {len(keep):6d} real, dropped {len(dropped):6d} fake")

    print(f"\nfake videos to re-process: {len(fake_ids)}")
    removed = 0
    for vid in fake_ids:
        d = OUT / "frames" / vid
        if d.is_dir():
            shutil.rmtree(d)
            removed += 1
    print(f"deleted {removed} frame directories")

    # selection.json stays: the WHICH-videos choice is still valid and frozen,
    # only the WHICH-frames decision changed.
    print("selection.json left intact (video choice unchanged, frame choice fixed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
