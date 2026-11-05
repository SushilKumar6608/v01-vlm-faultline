"""Compare vision-language model predictions (zero-shot and fine-tuned) in one table.

  python scripts/compare_vlm.py --preds results/predictions/zeroshot_Qwen3-VL-4B-Instruct_v2.csv \
      "results/predictions/zeroshot_Qwen3-VL-4B-Instruct+adapter_sheet_v2.csv" \
      "results/predictions/zeroshot_Qwen3-VL-4B-Instruct+adapter_single_v2.csv" --out results/vlm_finetune

Reads the Phase 4 predictions format (model, input, condition, split, clip_id, pred). The true class is read from the clip id
(for example stuck_item_A3_200004). Writes summary.csv (accuracy with Wilson 95 % interval, detection recall, false-alarm rate),
per_class.csv (recall per class), confusion.csv, paired.csv (exact McNemar test between the sheet adapter on sheets and the single-frame
adapter on single frames, same clips) and results.md.
"""
from __future__ import annotations

import argparse
from math import comb, sqrt
from pathlib import Path

import pandas as pd

CLASSES = ["normal", "stuck_item", "gate_open", "spray", "blocked_lens"]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts b (A right, B wrong) and c (A wrong, B right)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, p)


def load(paths: list[str]) -> pd.DataFrame:
    d = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    d["true"] = d.clip_id.str.rsplit("_", n=2).str[0]
    bad = sorted(set(d.true) - set(CLASSES))
    if bad:
        raise ValueError(f"cannot read the class from clip ids: {bad}")
    d["ok"] = d.pred == d.true
    return d


def summary(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (m, i, c, s), g in d.groupby(["model", "input", "condition", "split"]):
        lo, hi = wilson(int(g.ok.sum()), len(g))
        faults, normal = g[g.true != "normal"], g[g.true == "normal"]
        rows.append(dict(model=m, input=i, condition=c, split=s, n=len(g), accuracy=g.ok.mean(), ci_lo=lo, ci_hi=hi,
                         detect_recall=(faults.pred != "normal").mean() if len(faults) else float("nan"),
                         false_alarm=(normal.pred != "normal").mean() if len(normal) else float("nan")))
    return pd.DataFrame(rows)


def per_class(d: pd.DataFrame) -> pd.DataFrame:
    t = d.groupby(["model", "input", "condition", "split", "true"]).ok.mean().unstack("true")
    return t.reindex(columns=CLASSES).reset_index()


def confusion(d: pd.DataFrame) -> pd.DataFrame:
    return d.groupby(["model", "input", "condition", "split", "true", "pred"]).size().rename("n").reset_index()


def paired(d: pd.DataFrame) -> pd.DataFrame:
    """Sheet adapter on sheets against single-frame adapter on single frames, per condition and split, same clips."""
    rows = []
    a_name = next((m for m in d.model.unique() if m.endswith("adapter_sheet")), None)
    b_name = next((m for m in d.model.unique() if m.endswith("adapter_single")), None)
    if not a_name or not b_name:
        return pd.DataFrame(columns=["condition", "split", "n", "sheet_only_right", "single_only_right", "p_value"])
    for (c, s), _ in d.groupby(["condition", "split"]):
        a = d[(d.model == a_name) & (d.input == "sheet") & (d.condition == c) & (d.split == s)].set_index("clip_id").ok
        b = d[(d.model == b_name) & (d.input == "single") & (d.condition == c) & (d.split == s)].set_index("clip_id").ok
        ids = a.index.intersection(b.index)
        if len(ids) == 0:
            continue
        x, y = a[ids], b[ids]
        n_sheet, n_single = int((x & ~y).sum()), int((~x & y).sum())
        rows.append(dict(condition=c, split=s, n=len(ids), sheet_only_right=n_sheet, single_only_right=n_single, p_value=mcnemar_exact(n_sheet, n_single)))
    return pd.DataFrame(rows)


def markdown(title: str, s: pd.DataFrame, pc: pd.DataFrame, pr: pd.DataFrame) -> str:
    out = [f"# {title}", "", "Accuracy with Wilson 95 % interval; chance is 20 %. False alarm = normal clips called a fault.", ""]
    for split in ("test", "heldout", "val"):
        t = s[s.split == split]
        if t.empty:
            continue
        out += [f"## {split}", "", "| model | input | condition | n | accuracy (95 % CI) | detect recall | false alarm |", "|---|---|---|---|---|---|---|"]
        for r in t.sort_values(["model", "condition", "input"]).itertuples():
            out.append(f"| {r.model} | {r.input} | {r.condition} | {r.n} | {r.accuracy:.1%} ({r.ci_lo:.0%}–{r.ci_hi:.0%}) | {r.detect_recall:.0%} | {r.false_alarm:.0%} |")
        out.append("")
    out += ["## Per-class recall (test)", "", "| model | input | condition | " + " | ".join(CLASSES) + " |", "|---|---|---|" + "---|" * len(CLASSES)]
    t = pc[pc.split == "test"].sort_values(["model", "condition", "input"])
    for _, r in t.iterrows():
        out.append(f"| {r['model']} | {r['input']} | {r['condition']} | " + " | ".join(f"{r[c]:.0%}" for c in CLASSES) + " |")
    out.append("")
    if not pr.empty:
        out += ["## Sheet adapter on sheets against single-frame adapter on single frames (paired, exact McNemar)", "",
                "| condition | split | n | only sheet right | only single right | p |", "|---|---|---|---|---|---|"]
        for r in pr.itertuples():
            out.append(f"| {r.condition} | {r.split} | {r.n} | {r.sheet_only_right} | {r.single_only_right} | {r.p_value:.3g} |")
        out.append("")
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", nargs="+", required=True)
    ap.add_argument("--out", default="results/vlm_finetune")
    ap.add_argument("--title", default="Vision-language models: zero-shot and fine-tuned")
    a = ap.parse_args()
    d = load(a.preds)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    s, pc, cf, pr = summary(d), per_class(d), confusion(d), paired(d)
    s.to_csv(out / "summary.csv", index=False)
    pc.to_csv(out / "per_class.csv", index=False)
    cf.to_csv(out / "confusion.csv", index=False)
    pr.to_csv(out / "paired.csv", index=False)
    md = markdown(a.title, s, pc, pr)
    (out / "results.md").write_text(md, encoding="utf-8")
    print(md)
    print("wrote", out)


if __name__ == "__main__":
    main()
