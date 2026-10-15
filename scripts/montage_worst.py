"""Worst-case montages: for each fault class, the lowest-visibility clips as contact sheets (from the decoded mp4).

  python scripts/montage_worst.py [--per-class 6] [--splits test val]
Writes data/review/worst_<class>.png. Labels are shown on purpose: this is a label-aware check.
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np

from faultline import paths
from faultline.data import manifest
from faultline.data.generate import clip_paths
from faultline.media import read_frames
from faultline.sample.sheet import build_sheet
from faultline.sim import config as C


def bar(w: int, text: str) -> np.ndarray:
    b = np.full((24, w, 3), 20, np.uint8)
    cv2.putText(b, text, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return b


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=6)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test", "heldout"])
    a = ap.parse_args()
    m = manifest.read_manifest()
    m = m[m.split.isin(a.splits)]
    out = paths.data_root() / "review"
    out.mkdir(parents=True, exist_ok=True)
    for cls in C.FAULT_CLASSES:
        d = m[m.cls == cls].copy()
        d["score"] = d.halted_tiles if cls == "stuck_item" else d.expected_visible_tiles
        d = d.sort_values(["score", "fault_region_px"]).head(a.per_class)
        rows = []
        for r in d.itertuples():
            mp4, _ = clip_paths(r.split, r.clip_id, None)
            sheet = build_sheet(read_frames(mp4))
            txt = f"{r.clip_id} sev={r.severity:.2f} onset={r.onset_s:.2f} visible_tiles={r.expected_visible_tiles}"
            if cls == "stuck_item":
                txt += f" halted={r.halted_tiles} frozen_pairs={r.frozen_tiles} gap_at_onset={r.neighbor_gap_at_onset_px}px"
            rows += [bar(sheet.shape[1], txt), sheet]
        img = np.concatenate(rows, axis=0)
        p = out / f"worst_{cls}.png"
        cv2.imwrite(str(p), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print("wrote", p)


if __name__ == "__main__":
    main()
