"""Degraded copies of the clean clips, made with the frozen recipe (configs/degradation_v1.yaml).

Layout: <degraded_dir>/<split>/<clip_id>.mp4, same names and splits as the clean clips. Each file is written to
<clip_id>.tmp.mp4 and renamed, so a file that exists is complete. <degraded_dir>/recipe.json records the recipe hash; a run
refuses to add files made with a different recipe to the same folder.
"""

from __future__ import annotations

import json
import os
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .. import paths
from ..degrade.provisional import degrade_mp4
from ..degrade.recipe_v1 import RECIPE_V1, recipe_hash
from .buildinfo import build_info
from .generate import clip_paths, sha256_file


def degraded_path(split: str, clip_id: str, root: Path | None = None) -> Path:
    return (root or paths.degraded_dir()) / split / f"{clip_id}.mp4"


def check_recipe_marker(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    marker = root / "recipe.json"
    want = dict(name=RECIPE_V1["name"], hash=recipe_hash(RECIPE_V1))
    if marker.exists():
        have = json.loads(marker.read_text(encoding="utf-8"))
        if have.get("hash") != want["hash"]:
            raise RuntimeError(f"{root} holds clips made with another recipe ({have}); use an empty folder for {want}")
    else:
        marker.write_text(json.dumps(want, sort_keys=True), encoding="utf-8")


def degrade_one(args: tuple) -> str:
    split, clip_id, seed, src_root, dst_root = args
    src = clip_paths(split, clip_id, Path(src_root) if src_root else None)[0]
    dst = degraded_path(split, clip_id, Path(dst_root) if dst_root else None)
    tmp = dst.with_name(f"{clip_id}.tmp.mp4")
    dst.parent.mkdir(parents=True, exist_ok=True)
    degrade_mp4(src, tmp, int(seed), RECIPE_V1)
    os.replace(tmp, dst)
    return clip_id


def run(manifest: pd.DataFrame, src_root: Path | None = None, dst_root: Path | None = None, workers: int = 1,
        limit: int | None = None, progress: bool = True) -> dict:
    """Make every missing degraded clip. Resumable: existing files are kept, stale .tmp files are removed."""
    root = dst_root or paths.degraded_dir()
    check_recipe_marker(root)
    for tmp in root.glob("*/*.tmp.*"):
        tmp.unlink(missing_ok=True)
    rows = manifest[["split", "clip_id", "seed"]].to_dict("records")
    todo = [r for r in rows if not degraded_path(r["split"], r["clip_id"], root).exists()]
    skipped = len(rows) - len(todo)
    if limit is not None:
        todo = todo[:limit]
    jobs = [(r["split"], r["clip_id"], r["seed"], str(src_root) if src_root else None, str(root)) for r in todo]
    pool = Pool(workers) if workers > 1 and len(jobs) > 1 else None
    it = pool.imap_unordered(degrade_one, jobs, chunksize=1) if pool else map(degrade_one, jobs)
    try:
        if progress:
            from tqdm import tqdm

            it = tqdm(it, total=len(jobs), unit="clip")
        made = sum(1 for _ in it)
    finally:
        if pool:
            pool.close()
            pool.join()
    remaining = sum(1 for r in rows if not degraded_path(r["split"], r["clip_id"], root).exists())
    return dict(made=made, skipped=skipped, remaining=remaining)


def write_degraded_manifest(manifest: pd.DataFrame, dst_root: Path | None = None, out: Path | None = None) -> Path:
    root = dst_root or paths.degraded_dir()
    info = build_info()
    rows = []
    for r in manifest.itertuples():
        p = degraded_path(r.split, r.clip_id, root)
        rows.append(dict(clip_id=r.clip_id, split=r.split, path=f"degraded/{r.split}/{r.clip_id}.mp4", bytes=p.stat().st_size,
                         sha256=sha256_file(p), recipe_name=RECIPE_V1["name"], recipe_hash=recipe_hash(RECIPE_V1),
                         ffmpeg_version=info.get("ffmpeg_version")))
    dest = out or (root.parent / "degraded_manifest.csv")
    pd.DataFrame(rows).to_csv(dest, index=False, lineterminator="\n")
    return dest


def verify(manifest: pd.DataFrame, dst_root: Path | None = None, decode_n: int = 20, seed: int = 0) -> list[str]:
    """Every degraded clip exists; a random sample decodes with the expected frame count and size."""
    root = dst_root or paths.degraded_dir()
    errs = []
    for r in manifest.itertuples():
        if not degraded_path(r.split, r.clip_id, root).exists():
            errs.append(f"missing: {r.split}/{r.clip_id}")
    if errs:
        return errs
    w, h = RECIPE_V1["scale"]
    idx = np.random.default_rng(seed).choice(len(manifest), size=min(decode_n, len(manifest)), replace=False)
    for i in idx:
        r = manifest.iloc[int(i)]
        cap = cv2.VideoCapture(str(degraded_path(r["split"], r["clip_id"], root)))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        got = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        cap.release()
        if n != int(r["n_frames"]) or got != (w, h):
            errs.append(f"{r['clip_id']}: {n} frames at {got}, expected {int(r['n_frames'])} at {(w, h)}")
    return errs
