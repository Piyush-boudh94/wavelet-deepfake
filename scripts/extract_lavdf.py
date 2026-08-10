"""Extract LAV-DF.zip into data/raw/lavdf/, resuming a partial extraction.

The `dev/` split was extracted once before and interrupted (28,137 of 31,501
files). Rather than redo ~5 GB, every entry is compared against the archive's
recorded size: files that already match are skipped, missing or truncated ones
are (re-)written. Safe to re-run.
"""

import sys
import time
import zipfile
from pathlib import Path

ZIP = Path("/home/dgx-s-bmu-cse-240577/research/data/raw/lavdf/LAV-DF.zip")
DEST = Path("/home/dgx-s-bmu-cse-240577/research/data/raw/lavdf")
PREFIX = "LAV-DF/"


def main() -> int:
    written = skipped = repaired = 0
    t0 = time.time()

    with zipfile.ZipFile(ZIP) as z:
        infos = [i for i in z.infolist() if not i.is_dir()]
        total = len(infos)
        print(f"{total} entries in archive", flush=True)

        for n, info in enumerate(infos, 1):
            name = info.filename
            if not name.startswith(PREFIX):
                print(f"unexpected entry outside {PREFIX}: {name}", file=sys.stderr)
                return 1

            target = DEST / name[len(PREFIX):]
            if target.exists():
                if target.stat().st_size == info.file_size:
                    skipped += 1
                    continue
                repaired += 1  # truncated by the interrupted run

            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, open(target, "wb") as dst:
                while chunk := src.read(1 << 20):
                    dst.write(chunk)
            written += 1

            if n % 5000 == 0:
                print(
                    f"{n}/{total}  written={written} repaired={repaired} "
                    f"skipped={skipped}  {time.time() - t0:.0f}s",
                    flush=True,
                )

    print(
        f"DONE  written={written} repaired={repaired} skipped={skipped} "
        f"in {time.time() - t0:.0f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
