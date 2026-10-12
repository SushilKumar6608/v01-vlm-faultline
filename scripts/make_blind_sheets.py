"""Blind human-baseline kit: N random unlabelled clips -> shuffled sheets + a separate answer key.

Usage:  python scripts/make_blind_sheets.py [--n 30] [--degraded] [--seed 777] [--out DIR]
Then:   label sheets in <out>/blind/ by filling <out>/answers_template.csv (column 'label'), and run
        python scripts/score_blind.py <out>
Seeds used here are 900000+ so they never collide with dataset seeds.
Reported as 'author, single rater'. Do not open key.csv before labelling.
"""
from __future__ import annotations

import argparse
import csv
import tempfile
from pathlib import Path

import cv2
import numpy as np

from faultline.degrade.provisional import degrade_mp4
from faultline.media import read_frames, write_mp4
from faultline.sample.sheet import build_sheet
from faultline.sim import config as C
from faultline.sim.clip import generate_clip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--degraded", action="store_true")
    ap.add_argument("--out", default="data/blind")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    out = Path(a.out)
    (out / "blind").mkdir(parents=True, exist_ok=True)
    classes = list(C.CLASSES)
    rows = []
    with tempfile.TemporaryDirectory() as td:
        for i in range(a.n):
            cls = classes[i % len(classes)]  # balanced classes, shuffled below
            scene = str(rng.choice(["A1", "A2", "A3"]))
            seed = 900000 + int(rng.integers(0, 90000))
            c = generate_clip(seed, cls, scene)
            frames = c.frames
            if a.degraded:
                p1, p2 = Path(td) / "c.mp4", Path(td) / "d.mp4"
                write_mp4(frames, p1)
                degrade_mp4(p1, p2, seed)
                frames = read_frames(p2)
            rows.append((c.meta, build_sheet(frames)))
    order = rng.permutation(len(rows))
    key, tmpl = [], []
    for n, j in enumerate(order, 1):
        m, sheet = rows[j]
        sid = f"sheet_{n:03d}"
        cv2.imwrite(str(out / "blind" / f"{sid}.png"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
        key.append(dict(sheet=sid, cls=m["cls"], severity=m["severity"], scene=m["scene_config_id"], clip_id=m["clip_id"]))
        tmpl.append(dict(sheet=sid, label=""))
    for name, data in (("key.csv", key), ("answers_template.csv", tmpl)):
        with open(out / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    print(f"wrote {len(rows)} blind sheets to {out / 'blind'}; classes: {', '.join(classes)}")
    print("Label them in answers_template.csv (save as answers.csv). Do not open key.csv yet.")


if __name__ == "__main__":
    main()
