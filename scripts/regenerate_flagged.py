"""Replace clips that fail the visibility rule: same split/class/scene/severity, new seed from the reserve block.

Originals go to data/quarantine/. Repeats until nothing is flagged (max 5 rounds).
  python scripts/regenerate_flagged.py [--dry-run] [--workers N]
"""
from __future__ import annotations

import argparse
import os

from faultline.data import checks, generate, manifest
from faultline.data.regen import regenerate
from faultline.data.spec import effective_specs, load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    a = ap.parse_args()
    cfg = load_config()
    for rnd in range(1, 6):
        specs = effective_specs(cfg)
        m = manifest.read_sidecars(specs)
        flags = checks.visibility_flags(m, cfg)
        print(f"round {rnd}: {len(flags)} flagged")
        if flags.empty:
            break
        changed = regenerate(flags, cfg, dry_run=a.dry_run)
        print(changed[["clip_id_old", "clip_id_new", "reason"]].to_string(index=False))
        if a.dry_run:
            return
        generate.run(effective_specs(cfg), workers=a.workers)
        manifest.write_manifest(manifest.read_sidecars(effective_specs(cfg)))
    else:
        print("still flagged after 5 rounds: investigate")


if __name__ == "__main__":
    main()
