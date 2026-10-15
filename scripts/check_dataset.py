"""Dataset gate: spec validation, integrity (files, sha256, decode), count tables, visibility flags.

  python scripts/check_dataset.py [--partial] [--decode 20] [--write-flags]
Exit code 1 on integrity problems. Visibility flags are reported (and written with --write-flags) but do not fail the run.
"""
from __future__ import annotations

import argparse
import sys

from faultline import paths
from faultline.data import checks, manifest
from faultline.data.spec import effective_specs, load_config, validate_specs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--partial", action="store_true", help="allow a manifest that covers only part of the spec (pilot)")
    ap.add_argument("--decode", type=int, default=20, help="decode this many random clips and check frame count and size")
    ap.add_argument("--write-flags", action="store_true")
    a = ap.parse_args()

    cfg = load_config()
    specs = effective_specs(cfg)
    errs = validate_specs(specs, cfg)
    m = manifest.read_manifest()
    errs += checks.integrity(specs, m, partial=a.partial, decode_n=a.decode)
    print(f"manifest rows: {len(m)} / spec rows: {len(specs)}")
    for name, t in checks.count_tables(m).items():
        print(f"\n{name}\n{t.to_string()}")
    flags = checks.visibility_flags(m, cfg)
    print(f"\nvisibility flags: {len(flags)}")
    if len(flags):
        print(flags.to_string(index=False))
        if a.write_flags:
            out = paths.data_root() / "flagged.csv"
            flags.to_csv(out, index=False)
            print("wrote", out)
    for w in checks.warnings_list(m, cfg):
        print("WARNING:", w)
    if errs:
        print("\nINTEGRITY PROBLEMS:\n  " + "\n  ".join(errs[:30]))
        sys.exit(1)
    print("\nintegrity OK")


if __name__ == "__main__":
    main()
