#!/usr/bin/env python3
"""Fetch the mamba-ssm / causal-conv1d CUDA wheels and verify them.

Run on the HEAD NODE (it has internet and is not competing for the GPU slice).

Why this exists rather than a plain `pip install mamba-ssm`:

  * These packages ship precompiled CUDA kernels keyed to an exact
    (CUDA major, torch minor, CPython, C++ ABI) tuple. PyPI serves an sdist,
    which triggers a source build. The pod has no gcc and no nvcc, so that
    build cannot succeed.
  * The correct wheels live only on the upstream GitHub releases.

Verification is not ceremonial. On first run here, the causal-conv1d download
was silently truncated by a timeout -- 73 MB of an expected 151 MB -- and only
the SHA-256 check caught it. Installing that would have produced a corrupt
CUDA extension failing in some opaque way at training time.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

# Pinned to the environment verified in docs/ENVIRONMENT.md:
# torch 2.10 / CUDA 12 / CPython 3.11 / cxx11abi TRUE.
WHEELS = {
    "state-spaces/mamba":
        "mamba_ssm-2.3.2.post1+cu12torch2.10cxx11abiTRUE-cp311-cp311-linux_x86_64.whl",
    "Dao-AILab/causal-conv1d":
        "causal_conv1d-1.6.2.post1+cu12torch2.10cxx11abiTRUE-cp311-cp311-linux_x86_64.whl",
}
DEST = Path(__file__).resolve().parent.parent / "vendor" / "wheels"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def find_asset(repo: str, name: str) -> dict:
    url = f"https://api.github.com/repos/{repo}/releases?per_page=30"
    with urllib.request.urlopen(url, timeout=60) as r:
        releases = json.load(r)
    for rel in releases:
        for asset in rel.get("assets", []):
            if asset["name"] == name:
                return asset
    raise SystemExit(f"asset not found in {repo}: {name}")


def fetch(repo: str, name: str, force: bool) -> bool:
    DEST.mkdir(parents=True, exist_ok=True)
    out = DEST / name
    asset = find_asset(repo, name)
    expected_digest, expected_size = asset.get("digest"), asset["size"]

    if out.exists() and not force:
        if out.stat().st_size == expected_size and sha256(out) == expected_digest:
            print(f"  [cached, verified] {name}")
            return True
        print(f"  [stale/corrupt, refetching] {name}")

    print(f"  downloading {name}  ({expected_size / 1e6:.0f} MB) from {repo}")
    urllib.request.urlretrieve(asset["browser_download_url"], out)

    size, digest = out.stat().st_size, sha256(out)
    if size != expected_size:
        print(f"  SIZE MISMATCH: got {size}, expected {expected_size} -- truncated download")
        out.unlink()
        return False
    if expected_digest and digest != expected_digest:
        print(f"  SHA-256 MISMATCH\n    published {expected_digest}\n    local     {digest}")
        out.unlink()   # never leave an unverified wheel where pip could pick it up
        return False
    print(f"  VERIFIED {digest}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="re-download even if cached and valid")
    args = ap.parse_args()

    ok = True
    for repo, name in WHEELS.items():
        print(f"{repo}:")
        ok &= fetch(repo, name, args.force)

    if not ok:
        print("\nFAILED -- one or more wheels could not be verified. Not installing.")
        return 1

    print("\nAll wheels verified. Install them with:")
    for name in WHEELS.values():
        print(f"  scripts/pod.sh pip install --no-deps vendor/wheels/{name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
