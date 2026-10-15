"""Build dataset v1: spec -> validate -> render clips (resumable) -> manifest.

Examples:
  python scripts/generate_dataset.py --dry-run            # spec + validation only, writes configs/specs_v1.csv
  python scripts/generate_dataset.py --pilot 20           # 20 clips spread over all classes, with timing
  python scripts/generate_dataset.py --workers 6          # full run; safe to stop and restart
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import pandas as pd

from faultline.data import generate, manifest
from faultline.data.spec import build_specs, effective_specs, load_config, spec_path_committed, validate_specs, write_specs


def pick_pilot(specs: pd.DataFrame, n: int) -> pd.DataFrame:
    """n clips spread round-robin over the classes (first rows of each class in spec order)."""
    groups = [g for _, g in specs.groupby("cls", sort=False)]
    picked, i = [], 0
    while len(picked) < n and any(i < len(g) for g in groups):
        for g in groups:
            if i < len(g) and len(picked) < n:
                picked.append(g.iloc[i])
        i += 1
    return pd.DataFrame(picked)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--pilot", type=int, default=0, help="render only N clips spread over the classes, then time the manifest step")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    cfg = load_config(a.config)
    base = build_specs(cfg)
    errs = validate_specs(base, cfg)
    specs = effective_specs(cfg)
    errs += validate_specs(specs, cfg)
    if errs:
        print("SPEC INVALID:\n  " + "\n  ".join(errs))
        sys.exit(1)
    p = write_specs(base, spec_path_committed())
    print(f"spec ok: {len(specs)} clips; wrote {p}")
    if a.dry_run:
        return
    if a.pilot:
        specs = pick_pilot(specs, a.pilot)
    print(f"rendering {len(specs)} clips with {a.workers} worker(s)")
    t0 = time.time()
    res = generate.run(specs, workers=a.workers)
    dt = time.time() - t0
    print(f"{res}  in {dt:.0f}s" + (f"  ({dt / max(res['rendered'], 1):.1f} s/clip wall)" if res["rendered"] else ""))
    t1 = time.time()
    m = manifest.read_sidecars(specs)
    path = manifest.write_manifest(m)
    print(f"manifest: {len(m)} rows -> {path}  (sidecars + write took {time.time() - t1:.1f}s)")
    if res["remaining"]:
        print(f"{res['remaining']} clips still to render: run again to continue")


if __name__ == "__main__":
    main()
