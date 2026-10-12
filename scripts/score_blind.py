"""Score blind labels: per-class precision/recall with n and Wilson 95% intervals, plus confusion matrix.

Usage: python scripts/score_blind.py <dir with key.csv and answers.csv>
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

from faultline.sim import config as C


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main() -> None:
    d = Path(sys.argv[1])
    key = {r["sheet"]: r for r in csv.DictReader(open(d / "key.csv"))}
    ans = {r["sheet"]: r["label"].strip() for r in csv.DictReader(open(d / "answers.csv"))}
    classes = list(C.CLASSES)
    misses = []
    conf = {t: {p: 0 for p in classes} for t in classes}
    for s, r in key.items():
        p = ans.get(s, "")
        if p not in classes:
            print(f"skipping {s}: label {p!r} not in {classes}")
            continue
        conf[r["cls"]][p] += 1
        if p != r["cls"]:
            misses.append((s, r["cls"], p, r["severity"], r["scene"]))
    print("rows = true, columns = your label")
    print(" " * 14 + "".join(f"{c[:11]:>12s}" for c in classes))
    for t in classes:
        print(f"{t:14s}" + "".join(f"{conf[t][p]:12d}" for p in classes))
    print(f"\n{'class':14s}{'n_true':>7s}{'recall':>8s}  95% CI          {'precision':>9s}  95% CI")
    for c in classes:
        n_true = sum(conf[c].values()); tp = conf[c][c]; n_pred = sum(conf[t][c] for t in classes)
        rl = wilson(tp, n_true); pl = wilson(tp, n_pred)
        rec = tp / n_true if n_true else float("nan"); prec = tp / n_pred if n_pred else float("nan")
        print(f"{c:14s}{n_true:7d}{rec:8.2f}  [{rl[0]:.2f},{rl[1]:.2f}]    {prec:9.2f}  [{pl[0]:.2f},{pl[1]:.2f}]")
    if misses:
        print("\nmisses (sheet, true, your label, severity, scene):")
        for m in misses:
            print("  ", *m)
    tot = sum(sum(v.values()) for v in conf.values()); acc = sum(conf[c][c] for c in classes)
    print(f"\naccuracy {acc}/{tot} (author, single rater)")


if __name__ == "__main__":
    main()
