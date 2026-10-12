"""Phase 0 environment check. Run from the activated env: python scripts\\check_env.py"""

from __future__ import annotations

import importlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from faultline import paths

PACKAGES = ["numpy", "pandas", "scipy", "sklearn", "matplotlib", "PIL", "yaml", "tqdm", "cv2", "pytest"]


def check_packages() -> bool:
    ok = True
    for name in PACKAGES:
        try:
            mod = importlib.import_module(name)
            print(f"  OK    {name:<11} {getattr(mod, '__version__', '')}")
        except Exception as exc:  # noqa: BLE001
            ok = False
            print(f"  FAIL  {name:<11} {exc}")
    return ok


def check_ffmpeg() -> bool:
    exe = shutil.which("ffmpeg")
    if not exe:
        print("  FAIL  ffmpeg not found on PATH (is the conda env activated?)")
        return False
    out = subprocess.run([exe, "-version"], capture_output=True, text=True).stdout.splitlines()[0]
    print(f"  OK    {out}")
    h264 = subprocess.run([exe, "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    has_x264 = "libx264" in h264
    print(f"  {'OK  ' if has_x264 else 'FAIL'}  libx264 encoder {'available' if has_x264 else 'MISSING'}")
    return has_x264


def check_disk_and_write() -> bool:
    root = paths.repo_root()
    total, used, free = shutil.disk_usage(root)
    print(f"  repo root : {root}")
    print(f"  data root : {paths.data_root()}")
    print(f"  free space on this drive: {free / 1e9:.1f} GB of {total / 1e9:.1f} GB")
    try:
        paths.ensure_data_dirs()
        with tempfile.NamedTemporaryFile(dir=paths.data_root(), delete=True) as f:
            f.write(b"x" * 1024)
        print("  OK    data folder is writable")
        return free > 20e9
    except Exception as exc:  # noqa: BLE001
        print(f"  FAIL  cannot write to data folder: {exc}")
        return False


def check_determinism() -> bool:
    import numpy as np

    a = np.random.default_rng(123).random(5)
    b = np.random.default_rng(123).random(5)
    same = bool((a == b).all())
    print(f"  {'OK  ' if same else 'FAIL'}  numpy seeded generator is reproducible")
    return same


def main() -> int:
    print(f"Python {sys.version.split()[0]}")
    results = {}
    for title, fn in [
        ("Packages", check_packages),
        ("ffmpeg", check_ffmpeg),
        ("Disk and data folder", check_disk_and_write),
        ("Determinism", check_determinism),
    ]:
        print(f"\n[{title}]")
        results[title] = fn()
    print()
    failed = [k for k, v in results.items() if not v]
    if failed:
        print("RESULT: FAIL ->", ", ".join(failed))
        return 1
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
