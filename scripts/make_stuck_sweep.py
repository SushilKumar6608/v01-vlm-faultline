"""Targeted blind pass for stuck_item: 24 stuck sheets (severity 0.15-0.45, early and late onset) + 24 normals, shuffled.

  python scripts/make_stuck_sweep.py [--out data/stuck_sweep] [--seed 4242]
Label each sheet in answers_template.csv as stuck_item or normal (save as answers.csv), then
  python scripts/score_stuck_sweep.py data/stuck_sweep
Do not open key.csv before labelling. Uses the provisional degradation, like the first blind pass.
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
from faultline.sim.clip import generate_clip

SUBRANGES = ((0.15, 0.25), (0.25, 0.35), (0.35, 0.45))
EARLY = (1.0, 2.5)
LATE = (2.75, 3.5)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/stuck_sweep")
    ap.add_argument("--seed", type=int, default=4242)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    out = Path(a.out)
    (out / "blind").mkdir(parents=True, exist_ok=True)
    jobs = []
    seed = 910000
    for lo, hi in SUBRANGES:
        for k in range(8):
            seed += 1
            on = EARLY if k < 4 else LATE
            jobs.append(("stuck_item", seed, float(rng.uniform(lo, hi)), float(rng.uniform(*on))))
    for _ in range(24):
        seed += 1
        jobs.append(("normal", seed, None, None))
    results = []
    with tempfile.TemporaryDirectory() as td:
        for cls, sd, sev, on in jobs:
            scene = str(rng.choice(["A1", "A2", "A3"]))
            c = generate_clip(sd, cls, scene, sev, on)
            p1, p2 = Path(td) / "c.mp4", Path(td) / "d.mp4"
            write_mp4(c.frames, p1)
            degrade_mp4(p1, p2, sd)
            results.append((c.meta, build_sheet(read_frames(p2))))
    order = rng.permutation(len(results))
    key, tmpl = [], []
    for n, j in enumerate(order, 1):
        m, sheet = results[int(j)]
        sid = f"sheet_{n:03d}"
        cv2.imwrite(str(out / "blind" / f"{sid}.png"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
        key.append(dict(sheet=sid, cls=m["cls"], severity=m["severity"], onset_s=m["onset_s"], stop_s=m["severity_physical"],
                        halted_tiles=m["halted_tiles"], frozen_tiles=m["frozen_tiles"], scene=m["scene_config_id"], clip_id=m["clip_id"]))
        tmpl.append(dict(sheet=sid, label=""))
    for name, data in (("key.csv", key), ("answers_template.csv", tmpl)):
        with open(out / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    print(f"wrote {len(results)} sheets to {out / 'blind'}. Label each as stuck_item or normal; save as {out / 'answers.csv'}. Do not open key.csv.")


if __name__ == "__main__":
    main()
