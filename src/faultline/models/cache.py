"""Tile cache for the CNN baseline: decode each clip once, keep its 8 tiles at half resolution.

One array per (condition, split): uint8, shape (N, 8, 90, 160, 3), rows in manifest order. Tile k is the k-th sampled frame
(t = 0, 0.75, ... 5.25 s), made by the same code as the sheets and the single-frame control (sample.sheet.tiles) and then
halved with area interpolation. The single-frame input is tile 7 (the last), the sheet input is all 8 in order.
Files are written to a temporary name and renamed, so a file that exists is complete.
"""

from __future__ import annotations

import json
import os
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from ..eval.inputs import clip_file
from ..media import read_frames
from ..sample.sheet import tiles

N_TILES = 8
TILE_H, TILE_W = 90, 160  # half of the 180x320 tiles in the sheets


def cache_file(cache_root: Path, condition: str, split: str) -> Path:
    return Path(cache_root) / f"{condition}_{split}.npy"


def tiles_for_frames(frames: np.ndarray) -> np.ndarray:
    return np.stack([cv2.resize(t, (TILE_W, TILE_H), interpolation=cv2.INTER_AREA) for t in tiles(frames)])


def _load(args: tuple) -> np.ndarray:
    split, clip_id, condition, clips_root, degraded_root = args
    p = clip_file(split, clip_id, condition, Path(clips_root) if clips_root else None, Path(degraded_root) if degraded_root else None)
    return tiles_for_frames(read_frames(p))


def build(manifest: pd.DataFrame, condition: str, split: str, cache_root: Path, workers: int = 1,
          clips_root=None, degraded_root=None, progress: bool = False) -> Path:
    """Build the cache file for one (condition, split) unless it exists. Returns its path."""
    out = cache_file(cache_root, condition, split)
    ids = manifest.loc[manifest["split"] == split, "clip_id"].tolist()
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".tmp.npy")
    arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(len(ids), N_TILES, TILE_H, TILE_W, 3))
    jobs = [(split, c, condition, str(clips_root) if clips_root else None, str(degraded_root) if degraded_root else None) for c in ids]
    it = map(_load, jobs) if workers <= 1 else Pool(workers).imap(_load, jobs, chunksize=8)
    if progress:
        from tqdm import tqdm

        it = tqdm(it, total=len(jobs), desc=f"cache {condition}/{split}", unit="clip")
    for i, t in enumerate(it):
        arr[i] = t
    arr.flush()
    del arr
    out.with_suffix(".ids.json").write_text(json.dumps(ids), encoding="utf-8")
    os.replace(tmp, out)
    return out


def load(manifest: pd.DataFrame, condition: str, split: str, cache_root: Path, mmap: bool = False) -> np.ndarray:
    """Load a cache file and check its rows are the manifest's clips of that split, in order."""
    p = cache_file(cache_root, condition, split)
    ids = json.loads(p.with_suffix(".ids.json").read_text(encoding="utf-8"))
    want = manifest.loc[manifest["split"] == split, "clip_id"].tolist()
    if ids != want:
        raise RuntimeError(f"{p} does not match the manifest (different clips or order); delete the cache folder and rebuild")
    return np.load(p, mmap_mode="r" if mmap else None)


def labels(manifest: pd.DataFrame, split: str, classes: list[str]) -> np.ndarray:
    idx = {c: i for i, c in enumerate(classes)}
    return np.array([idx[c] for c in manifest.loc[manifest["split"] == split, "cls"]], dtype=np.int64)
