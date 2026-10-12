"""Frame sampling and contact sheets.

Sheet layout (fixed, stated to the models): 2 rows x 4 columns, time runs left to right,
top row first. Each tile is 320x180 with its timestamp burned in at build time, after any
degradation. The single-frame control is the last tile (t = 5.25 s).
"""

from __future__ import annotations

import cv2
import numpy as np

from ..sim import config as C


def _tile(frame: np.ndarray, t: float) -> np.ndarray:
    tile = cv2.resize(frame, (C.TILE_W, C.TILE_H), interpolation=cv2.INTER_AREA)
    text = f"t={t:.2f}s"
    org = (5, 15)
    cv2.putText(tile, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(tile, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(tile, (0, 0), (C.TILE_W - 1, C.TILE_H - 1), (30, 30, 30), 1)
    return tile


def sampled_frames(frames: np.ndarray) -> np.ndarray:
    """The 8 sampled frames of a full clip (or the frames unchanged if 8 were passed)."""
    if len(frames) == C.SAMPLE_COUNT:
        return frames
    return frames[C.sample_frame_indices()]


def tiles(frames: np.ndarray) -> list[np.ndarray]:
    eight = sampled_frames(frames)
    return [_tile(f, t) for f, t in zip(eight, C.sample_times())]


def build_sheet(frames: np.ndarray) -> np.ndarray:
    ts = tiles(frames)
    rows = [np.concatenate(ts[r * C.SHEET_COLS : (r + 1) * C.SHEET_COLS], axis=1) for r in range(C.SHEET_ROWS)]
    return np.concatenate(rows, axis=0)


def single_frame_tile(frames: np.ndarray) -> np.ndarray:
    """Single-frame control: the last sampled frame, same resolution and timestamp as in the sheet."""
    return tiles(frames)[-1]
