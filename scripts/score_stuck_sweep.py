"""Score the stuck_item sweep: detection rate by stop duration, halted tiles and onset, plus false positives on normals."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

from faultline.eval.stats import wilson


def line(name: str, k: int, n: int) -> None:
    lo, hi = wilson(k, n)
    print(f"  {name:28s} {k:2d}/{n:2d} = {k / n:4.2f}   95% CI [{lo:.2f}, {hi:.2f}]" if n else f"  {name:28s} no clips")


def main() -> None:
    d = Path(sys.argv[1])
    key = {r["sheet"]: r for r in csv.DictReader(open(d / "key.csv", encoding="utf-8"))}
    ans = {r["sheet"]: r["label"].strip() for r in csv.DictReader(open(d / "answers.csv", encoding="utf-8"))}
    stuck = [(r, ans.get(s) == "stuck_item") for s, r in key.items() if r["cls"] == "stuck_item" and ans.get(s) in ("stuck_item", "normal")]
    norm = [(r, ans.get(s) == "stuck_item") for s, r in key.items() if r["cls"] == "normal" and ans.get(s) in ("stuck_item", "normal")]
    print("Detection of stuck_item (higher is better)")
    line("all stuck_item", sum(h for _, h in stuck), len(stuck))
    print("by stop duration")
    for lo, hi, nm in ((0, 2.6, "stop < 2.6 s"), (2.6, 3.0, "2.6 <= stop < 3.0 s"), (3.0, 99, "stop >= 3.0 s")):
        g = [h for r, h in stuck if lo <= float(r["stop_s"]) < hi]
        line(nm, sum(g), len(g))
    print("by halted tiles")
    for t in sorted({int(r["halted_tiles"]) for r, _ in stuck}):
        g = [h for r, h in stuck if int(r["halted_tiles"]) == t]
        line(f"{t} halted tiles", sum(g), len(g))
    print("by onset")
    for lo, hi, nm in ((0, 2.6, "early onset (<= 2.5 s)"), (2.6, 9, "late onset (>= 2.75 s)")):
        g = [h for r, h in stuck if lo <= float(r["onset_s"]) < hi]
        line(nm, sum(g), len(g))
    print("False alarms on normal clips (lower is better)")
    line("normal labelled stuck_item", sum(h for _, h in norm), len(norm))
    print("\nMisses (sheet, severity, stop_s, onset_s, halted_tiles):")
    for s, r in key.items():
        if r["cls"] == "stuck_item" and ans.get(s) != "stuck_item":
            print("  ", s, round(float(r["severity"]), 2), round(float(r["stop_s"]), 2), round(float(r["onset_s"]), 2), r["halted_tiles"], "label:", ans.get(s))


if __name__ == "__main__":
    main()
