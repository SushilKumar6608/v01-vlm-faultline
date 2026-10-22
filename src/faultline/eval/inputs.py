"""Model inputs: a clip becomes either a contact sheet or the single-frame control, clean or degraded.

Same pipeline as the Phase 3 blind audit: decode, rescale to 640x360 if needed, sample 8 frames, build tiles,
burn the timestamp in. Sheet: 360x1280x3. Single frame (last tile, t = 5.25 s): 180x320x3. uint8 RGB.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..data.degrade import degraded_path
from ..data.generate import clip_paths
from ..media import read_frames
from ..sample.sheet import build_sheet, single_frame_tile

KINDS = ("single", "sheet")
CONDITIONS = ("clean", "degraded")


def clip_file(split: str, clip_id: str, condition: str, clips_root: Path | None = None, degraded_root: Path | None = None) -> Path:
    if condition == "clean":
        return clip_paths(split, clip_id, clips_root)[0]
    if condition == "degraded":
        return degraded_path(split, clip_id, degraded_root)
    raise ValueError(f"condition must be one of {CONDITIONS}, got {condition!r}")


def make_input(frames: np.ndarray, kind: str) -> np.ndarray:
    if kind == "sheet":
        return build_sheet(frames)
    if kind == "single":
        return single_frame_tile(frames)
    raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")


def load_input(split: str, clip_id: str, condition: str, kind: str, clips_root: Path | None = None,
               degraded_root: Path | None = None) -> np.ndarray:
    return make_input(read_frames(clip_file(split, clip_id, condition, clips_root, degraded_root)), kind)


__all__ = ["KINDS", "CONDITIONS", "clip_file", "make_input", "load_input"]
