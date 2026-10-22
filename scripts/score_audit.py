"""Score the Phase 3 blind audit against the pass criteria fixed in advance.

Gate: overall accuracy >= 85 %, no class recall below 70 %, false alarms on normal <= 10 %.
Severity bins are reported, not gated. Author, single rater.
Usage: python scripts/score_audit.py data/audit
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

from faultline.eval.stats import wilson
from faultline.sim import config as C

ACC_MIN, CLASS_MIN, FA_MAX = 0.85, 0.70, 0.10


def main() -> None:
    d = Path(sys.argv[1] if len(sys.argv) > 1 else "data/audit")
    key = {r["sheet"]: r for r in csv.DictReader(open(d / "key.csv", encoding="utf-8-sig"))}
    ans = {r["sheet"]: r["label"].strip().lower() for r in csv.DictReader(open(d / "answers.csv", encoding="utf-8-sig"))}
    classes = list(C.CLASSES)
    bad = [s for s in key if ans.get(s, "") not in classes]
    if bad:
        raise SystemExit(f"{len(bad)} sheets have a missing or invalid label (first: {bad[:5]}). Fill all {len(key)} first.")
    conf = {t: {p: 0 for p in classes} for t in classes}
    bins = defaultdict(lambda: [0, 0])
    misses = []
    for s, r in key.items():
        t, p = r["cls"], ans[s]
        conf[t][p] += 1
        if t != "normal":
            b = bins[(t, r["severity_bin"])]
            b[1] += 1
            b[0] += int(t == p)
        if t != p:
            misses.append((s, t, p, r["severity"], r["scene"]))
    n = len(key)
    acc_k = sum(conf[c][c] for c in classes)
    print("rows = true, columns = your label")
    print(" " * 14 + "".join(f"{c[:11]:>12s}" for c in classes))
    for t in classes:
        print(f"{t:14s}" + "".join(f"{conf[t][p]:12d}" for p in classes))
    print(f"\n{'class':14s}{'n':>4s}{'recall':>8s}  95% CI")
    recalls = {}
    for c in classes:
        nt = sum(conf[c].values()); lo, hi = wilson(conf[c][c], nt)
        recalls[c] = conf[c][c] / nt
        print(f"{c:14s}{nt:4d}{recalls[c]:8.2f}  [{lo:.2f},{hi:.2f}]")
    print("\nfault classes by severity bin (not gated):")
    for (c, b), (k, nn) in sorted(bins.items()):
        lo, hi = wilson(k, nn)
        print(f"  {c:14s}{b:>6s}  {k}/{nn} = {k/nn:.2f}  [{lo:.2f},{hi:.2f}]")
    fa = 1 - recalls["normal"]
    acc = acc_k / n
    lo, hi = wilson(acc_k, n)
    print(f"\noverall accuracy {acc_k}/{n} = {acc:.2f} [{lo:.2f},{hi:.2f}]  (author, single rater)")
    print(f"false alarms on normal: {fa:.2f}")
    if misses:
        print("\nmisses (sheet, true, your label, severity, scene):")
        for m in misses:
            print("  ", *m)
    checks = [("overall accuracy >= 85%", acc >= ACC_MIN), (f"every class recall >= 70% (min {min(recalls.values()):.2f})", min(recalls.values()) >= CLASS_MIN),
              ("false alarms on normal <= 10%", fa <= FA_MAX)]
    print("\nGATE:")
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print("RESULT:", "PASS" if all(ok for _, ok in checks) else "FAIL")


if __name__ == "__main__":
    main()
