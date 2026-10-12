"""ffmpeg helpers: write and read clips. ffmpeg must be on PATH (the conda env provides it)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np

from .sim import config as C

# Same flags for every encode: one thread and bit-exact muxing, so a file depends only on its input.
_REPRO = ["-threads", "1", "-map_metadata", "-1", "-fflags", "+bitexact", "-flags:v", "+bitexact"]


def ffmpeg_exe() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError("ffmpeg not found on PATH; activate the conda environment")
    return exe


def write_mp4(frames: np.ndarray, path: str | Path, fps: int = C.FPS, crf: int = 18, preset: str = "medium") -> None:
    n, h, w, _ = frames.shape
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
           "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p", *_REPRO, str(path)]
    subprocess.run(cmd, input=np.ascontiguousarray(frames).tobytes(), check=True)


def read_frames(path: str | Path, width: int = C.W, height: int = C.H) -> np.ndarray:
    """Decode a clip to (N, height, width, 3) uint8 RGB, rescaling with bicubic if the size differs."""
    cmd = [ffmpeg_exe(), "-loglevel", "error", "-i", str(path), "-vf", f"scale={width}:{height}:flags=bicubic",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out, np.uint8).reshape(-1, height, width, 3)


REPRO_FLAGS = _REPRO
