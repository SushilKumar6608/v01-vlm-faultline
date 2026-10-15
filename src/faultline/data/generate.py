"""Render clips from the spec. Resumable, parallel, atomic.

For each clip: <clip_id>.tmp.mp4 is written and renamed to <clip_id>.mp4, then the sidecar JSON
(ground truth, sha256, build versions) is written to a temp file and renamed. The sidecar is the
completion marker: a clip counts as done only if the sidecar exists, the mp4 exists and its sha256
matches the sidecar. Anything else is re-rendered.
"""

from __future__ import annotations

import hashlib
import json
import os
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

from .. import paths
from ..media import write_mp4
from ..sim.clip import generate_clip
from .buildinfo import build_info


def clip_paths(split: str, clip_id: str, root: Path | None = None) -> tuple[Path, Path]:
    d = (root or paths.clips_dir()) / split
    return d / f"{clip_id}.mp4", d / f"{clip_id}.json"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_done(split: str, clip_id: str, root: Path | None = None) -> bool:
    mp4, js = clip_paths(split, clip_id, root)
    if not (mp4.exists() and js.exists()):
        return False
    try:
        side = json.loads(js.read_text(encoding="utf-8"))
        return side.get("sha256") == sha256_file(mp4)
    except Exception:
        return False


def render_one(args: tuple) -> dict:
    """Worker: render one spec row. args = (row dict, root str)."""
    import cv2

    cv2.setNumThreads(1)
    row, root = args
    root_p = Path(root)
    mp4, js = clip_paths(row["split"], row["clip_id"], root_p)
    mp4.parent.mkdir(parents=True, exist_ok=True)
    sev = None if row["severity"] is None or pd.isna(row["severity"]) else float(row["severity"])
    clip = generate_clip(int(row["seed"]), row["cls"], row["scene_config_id"], sev)
    tmp = mp4.with_name(f"{row['clip_id']}.tmp.mp4")
    write_mp4(clip.frames, tmp)
    os.replace(tmp, mp4)
    side = dict(clip.meta)
    side.update(split=row["split"], sha256=sha256_file(mp4), bytes=mp4.stat().st_size, **build_info())
    tmpj = js.with_name(f"{row['clip_id']}.tmp.json")
    tmpj.write_text(json.dumps(side, sort_keys=True), encoding="utf-8")
    os.replace(tmpj, js)
    return dict(clip_id=row["clip_id"], ok=True)


def run(specs: pd.DataFrame, root: Path | None = None, workers: int = 1, limit: int | None = None, progress: bool = True) -> dict:
    """Render all not-yet-done clips. Returns counts. Stale .tmp files from an interrupted run are removed."""
    root = root or paths.clips_dir()
    for tmp in Path(root).glob("*/*.tmp.*"):
        tmp.unlink(missing_ok=True)
    rows = specs.to_dict("records")
    todo = [r for r in rows if not is_done(r["split"], r["clip_id"], root)]
    skipped = len(rows) - len(todo)
    if limit is not None:
        todo = todo[:limit]
    jobs = [(r, str(root)) for r in todo]
    done = 0
    it = None
    if workers > 1 and len(jobs) > 1:
        pool = Pool(workers)
        it = pool.imap_unordered(render_one, jobs, chunksize=1)
    else:
        pool = None
        it = map(render_one, jobs)
    try:
        if progress:
            from tqdm import tqdm

            it = tqdm(it, total=len(jobs), unit="clip")
        for _ in it:
            done += 1
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    return dict(rendered=done, skipped=skipped, remaining=len(rows) - skipped - done)
