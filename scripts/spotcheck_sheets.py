"""Label-aware spot check of the test split: a random 10% sample (stratified, seeded) of contact sheets.

  python scripts/spotcheck_sheets.py            # writes data/spotcheck/sheet_*.png and spotcheck.csv
  python scripts/spotcheck_sheets.py --score    # reads spotcheck.csv, reports failures and the 95% bound
Fill 'labelable' with yes or no (no = the labelled fault is not identifiable on the sheet, or the label is wrong) and a note.
Stopping rule: 3 or more 'no' out of 50 means review the whole test split.
"""
from __future__ import annotations

import argparse
import csv

import cv2
import numpy as np

from faultline import paths
from faultline.data import manifest
from faultline.data.generate import clip_paths
from faultline.eval.stats import upper_bound_zero_failures, wilson
from faultline.media import read_frames
from faultline.sample.sheet import build_sheet
from faultline.sim import config as C

ESCALATE_AT = 3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fraction", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--score", action="store_true")
    a = ap.parse_args()
    out = paths.data_root() / "spotcheck"
    csv_path = out / "spotcheck.csv"
    if a.score:
        rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
        done = [r for r in rows if r["labelable"].strip().lower() in ("yes", "no")]
        bad = [r for r in done if r["labelable"].strip().lower() == "no"]
        n = len(done)
        print(f"checked {n} of {len(rows)}; not labelable: {len(bad)}")
        for r in bad:
            print("  ", r["sheet"], r["clip_id"], r["note"])
        if n and not bad:
            print(f"0 of {n} failed; upper 95% bound on the failure rate about {100 * upper_bound_zero_failures(n):.1f}%")
        elif n:
            lo, hi = wilson(len(bad), n)
            print(f"{len(bad)} of {n} failed; Wilson 95% interval {100 * lo:.1f}% to {100 * hi:.1f}%")
        if len(bad) >= ESCALATE_AT:
            print(f"STOPPING RULE: {len(bad)} >= {ESCALATE_AT} failures: review the whole test split")
        return
    m = manifest.read_manifest()
    t = m[m.split == "test"]
    rng = np.random.default_rng(a.seed)
    picks = []
    for cls in C.CLASSES:
        d = t[t.cls == cls]
        k = max(1, int(round(a.fraction * len(d))))
        picks += [d.iloc[int(i)] for i in rng.choice(len(d), size=k, replace=False)]
    order = rng.permutation(len(picks))
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for n, j in enumerate(order, 1):
        r = picks[int(j)]
        mp4, _ = clip_paths(r.split, r.clip_id, None)
        sheet = build_sheet(read_frames(mp4))
        b = np.full((24, sheet.shape[1], 3), 20, np.uint8)
        sev = "-" if r.severity != r.severity else f"{r.severity:.2f}"
        cv2.putText(b, f"{r.cls}  sev={sev}  onset={r.onset_s}  scene={r.scene_config_id}", (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        sid = f"sheet_{n:03d}"
        cv2.imwrite(str(out / f"{sid}.png"), cv2.cvtColor(np.concatenate([b, sheet]), cv2.COLOR_RGB2BGR))
        rows.append(dict(sheet=sid, clip_id=r.clip_id, cls=r.cls, severity=sev, labelable="", note=""))
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} labelled sheets to {out}; fill labelable in {csv_path}")


if __name__ == "__main__":
    main()
