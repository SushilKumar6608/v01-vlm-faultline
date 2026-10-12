"""Phase 1 visibility gate: render demo clips, degrade them, save sheets and mask overlays.

Usage: python scripts/preview_sim.py [--out DIR]
Output goes to <data root>/preview by default (see faultline.paths).
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import cv2
import numpy as np

from faultline.degrade.provisional import degrade_mp4
from faultline.media import read_frames, write_mp4
from faultline.sample.sheet import build_sheet, single_frame_tile
from faultline.sim import config as C
from faultline.sim.clip import generate_clip

DEMOS = [
    ("normal", "A1", 0.5, 1), ("normal", "D1", 0.5, 2),
    ("stuck_item", "A1", 0.9, 11), ("stuck_item", "A2", 0.2, 12),
    ("gate_open", "A1", 0.9, 21), ("gate_open", "A3", 0.2, 22),
    ("spray", "A1", 0.9, 31), ("spray", "A2", 0.2, 32), ("spray", "D1", 0.5, 33),
    ("blocked_lens", "A1", 0.9, 41), ("blocked_lens", "A3", 0.2, 42),
]


def label(img: np.ndarray, text: str) -> np.ndarray:
    bar = np.full((24, img.shape[1], 3), 20, np.uint8)
    cv2.putText(bar, text, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return np.concatenate([bar, img], axis=0)


def mask_overlay(clip) -> np.ndarray:
    idx = C.sample_frame_indices()
    fr = clip.frames[idx].copy()
    for i in range(len(idx)):
        m = clip.sample_masks[i]
        fr[i][m] = (0.5 * fr[i][m] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
    return build_sheet(fr)


def save(path: Path, rgb: np.ndarray) -> None:
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.out:
        out = Path(a.out)
    else:
        from faultline import paths
        out = Path(paths.data_root()) / "preview" if hasattr(paths, "data_root") else Path("preview")
    out.mkdir(parents=True, exist_ok=True)
    clean_rows, deg_rows = [], []
    print(f"{'clip':28s} {'sev':>5s} {'onset':>5s} {'region_px':>9s} {'vis_tiles':>9s}  visible_px_samples")
    with tempfile.TemporaryDirectory() as td:
        for cls, scene, sev, seed in DEMOS:
            c = generate_clip(seed, cls, scene, severity=None if cls == "normal" else sev)
            m = c.meta
            print(f"{m['clip_id']:28s} {m['severity'] if m['severity'] is not None else '-':>5} "
                  f"{m['onset_s'] if m['onset_s'] is not None else '-':>5} {m['fault_region_px']:9.1f} "
                  f"{m['expected_visible_tiles']:9d}  {m['visible_px_samples']}")
            clean_p, deg_p = Path(td) / "c.mp4", Path(td) / "d.mp4"
            write_mp4(c.frames, clean_p)
            degrade_mp4(clean_p, deg_p, seed)
            deg = read_frames(deg_p)
            name = m["clip_id"] + f"_s{sev}"
            sc, sd = build_sheet(c.frames), build_sheet(deg)
            save(out / f"{name}_clean.png", sc)
            save(out / f"{name}_degraded.png", sd)
            save(out / f"{name}_mask.png", mask_overlay(c))
            save(out / f"{name}_single.png", single_frame_tile(deg))
            clean_rows.append(label(sc, name + " clean"))
            deg_rows.append(label(sd, name + " degraded"))
    save(out / "overview_clean.png", np.concatenate(clean_rows, axis=0))
    save(out / "overview_degraded.png", np.concatenate(deg_rows, axis=0))
    print("wrote", out)


if __name__ == "__main__":
    main()
