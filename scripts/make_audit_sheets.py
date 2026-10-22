"""Phase 3 blind audit kit: 100 unlabelled degraded sheets (20 per class), shuffled, with a separate key.

Usage:  python scripts/make_audit_sheets.py [--n 100] [--workers 8] [--out data/audit]
Then:   fill the 'label' column of <out>/answers.csv (one of: normal, stuck_item, gate_open, spray, blocked_lens),
        look only at <out>/blind/sheet_XXX.png, then run  python scripts/score_audit.py <out>
Clips are fresh (seed block 950000+), never part of train/val/test/held-out. Fault severities are stratified per class.
Rater: author, single rater. Do not open key.csv before labelling.
"""
from __future__ import annotations

import argparse
import csv
import os
import tempfile
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

from faultline.data.spec import _stratified_severities
from faultline.degrade.provisional import degrade_mp4
from faultline.media import read_frames, write_mp4
from faultline.sample.sheet import build_sheet
from faultline.sim import config as C
from faultline.sim.clip import generate_clip

SEED0 = 950000


def render(job: dict) -> dict:
    cv2.setNumThreads(1)
    with tempfile.TemporaryDirectory() as td:
        c = generate_clip(job["seed"], job["cls"], job["scene"], job["severity"])
        p1, p2 = Path(td) / "clean.mp4", Path(td) / "deg.mp4"
        write_mp4(c.frames, p1)
        degrade_mp4(p1, p2, job["seed"])
        sheet = build_sheet(read_frames(p2))
    return dict(job=job, sheet=sheet, meta=c.meta)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default="data/audit")
    a = ap.parse_args()
    k = len(C.CLASSES)
    if a.n % k:
        raise SystemExit(f"--n must be a multiple of {k}")
    per = a.n // k
    rng = np.random.default_rng(a.seed)
    out = Path(a.out)
    (out / "blind").mkdir(parents=True, exist_ok=True)

    jobs, i = [], 0
    for cls in C.CLASSES:
        sev = _stratified_severities(rng, per) if cls != "normal" else [None] * per
        scenes = rng.permutation([["A1", "A2", "A3"][j % 3] for j in range(per)])
        for j in range(per):
            jobs.append(dict(seed=SEED0 + i, cls=cls, scene=str(scenes[j]), severity=None if sev[j] is None else float(sev[j])))
            i += 1
    print(f"rendering {len(jobs)} audit clips with {a.workers} worker(s) ...")
    if a.workers > 1:
        with Pool(a.workers) as p:
            res = list(p.imap_unordered(render, jobs, chunksize=1))
    else:
        res = [render(j) for j in jobs]

    order = rng.permutation(len(res))
    key, tmpl = [], []
    for n, idx in enumerate(order, 1):
        r = res[idx]
        sid = f"sheet_{n:03d}"
        cv2.imwrite(str(out / "blind" / f"{sid}.png"), cv2.cvtColor(r["sheet"], cv2.COLOR_RGB2BGR))
        m = r["meta"]
        key.append(dict(sheet=sid, cls=m["cls"], severity=m["severity"], severity_bin=C.severity_bin(m["severity"]),
                        scene=m["scene_config_id"], clip_id=m["clip_id"], seed=r["job"]["seed"]))
        tmpl.append(dict(sheet=sid, label=""))
    for name, data in (("key.csv", key), ("answers.csv", tmpl)):
        with open(out / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    print(f"wrote {len(res)} sheets to {out / 'blind'}")
    print(f"Fill the 'label' column in {out / 'answers.csv'} (normal, stuck_item, gate_open, spray, blocked_lens). Do not open key.csv yet.")


if __name__ == "__main__":
    main()
