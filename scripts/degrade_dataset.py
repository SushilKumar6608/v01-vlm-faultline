"""Make the degraded copies of all dataset clips with the frozen recipe (degradation_v1).

  python scripts/degrade_dataset.py [--workers 16] [--dry-run] [--verify-only] [--decode 20] [--limit N]

Resumable: clips that already have a degraded copy are skipped. Writes data/degraded/<split>/<clip_id>.mp4,
data/degraded/recipe.json and data/degraded_manifest.csv (sha256 per file). Exit code 1 if verification fails.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

from faultline import paths
from faultline.data import degrade, manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--limit", type=int, default=None, help="make at most this many clips (testing)")
    ap.add_argument("--decode", type=int, default=20, help="decode this many random degraded clips when verifying")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify-only", action="store_true")
    a = ap.parse_args()

    m = manifest.read_manifest()
    root = paths.degraded_dir()
    todo = sum(1 for r in m.itertuples() if not degrade.degraded_path(r.split, r.clip_id, root).exists())
    print(f"{len(m)} clips in the manifest; {todo} still need a degraded copy")
    if a.dry_run:
        return
    if not a.verify_only:
        t0 = time.time()
        res = degrade.run(m, workers=a.workers, limit=a.limit)
        print(res, f"in {time.time() - t0:.0f}s")
        if res["remaining"] and a.limit is None:
            sys.exit("some clips are still missing; run again")
    errs = degrade.verify(m, decode_n=a.decode)
    if errs:
        print("\n".join(errs[:20]))
        sys.exit(1)
    out = degrade.write_degraded_manifest(m)
    print(f"verified; wrote {out}")


if __name__ == "__main__":
    main()
