"""Versions recorded next to every clip, so a manifest says which build produced it."""

from __future__ import annotations

import platform
import subprocess
from functools import lru_cache

import cv2
import numpy as np

from ..media import ffmpeg_exe


@lru_cache(maxsize=1)
def ffmpeg_version() -> str:
    out = subprocess.run([ffmpeg_exe(), "-version"], capture_output=True, text=True, check=True).stdout
    return out.splitlines()[0].split("Copyright")[0].strip()


@lru_cache(maxsize=1)
def git_commit() -> str:
    from .. import paths

    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=paths.repo_root(), timeout=10)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def build_info() -> dict:
    return dict(
        ffmpeg_version=ffmpeg_version(),
        opencv_version=cv2.__version__,
        numpy_version=np.__version__,
        python_version=platform.python_version(),
        git_commit=git_commit(),
    )
