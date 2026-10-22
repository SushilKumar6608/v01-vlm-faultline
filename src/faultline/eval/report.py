"""Predictions file -> validated, scored, written as CSV tables and a markdown summary.

Predictions schema (one row per model answer):
  model, input (single|sheet), condition (clean|degraded), split (val|test|heldout), clip_id, pred
pred is one of normal, stuck_item, gate_open, spray, blocked_lens, or 'invalid' when the answer could not be parsed.
Every (model, input, condition, split) group must cover exactly the clips of that split unless partial results are allowed.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import metrics as M
from .inputs import CONDITIONS, KINDS

GROUP = ["model", "input", "condition", "split"]
REQUIRED = GROUP + ["clip_id", "pred"]
TRUTH_COLS = ["clip_id", "split", "cls", "severity_bin", "scene_config_id"]


def validate_predictions(preds: pd.DataFrame, manifest: pd.DataFrame, allow_partial: bool = False) -> list[str]:
    miss = [c for c in REQUIRED if c not in preds.columns]
    if miss:
        return [f"missing columns: {miss}"]
    errs = []
    for col, ok in (("input", KINDS), ("condition", CONDITIONS), ("split", sorted(manifest["split"].unique()))):
        bad = sorted(set(preds[col]) - set(ok))
        if bad:
            errs.append(f"{col}: unexpected values {bad} (allowed {list(ok)})")
    bad = sorted(set(preds["pred"]) - set(M.CLASSES + [M.INVALID]))
    if bad:
        errs.append(f"pred: unexpected labels {bad}")
    if errs:
        return errs
    expected = {s: set(g["clip_id"]) for s, g in manifest.groupby("split")}
    for key, g in preds.groupby(GROUP):
        name = "/".join(map(str, key))
        got = g["clip_id"]
        if got.duplicated().any():
            errs.append(f"{name}: {int(got.duplicated().sum())} duplicate clip ids")
        exp = expected[key[3]]
        extra = set(got) - exp
        if extra:
            errs.append(f"{name}: {len(extra)} clip ids not in split {key[3]} (e.g. {sorted(extra)[:3]})")
        missing = exp - set(got)
        if missing and not allow_partial:
            errs.append(f"{name}: {len(missing)} of {len(exp)} clips missing (e.g. {sorted(missing)[:3]})")
    return errs


def attach_truth(preds: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    df = preds.merge(manifest[TRUTH_COLS], on=["clip_id", "split"], how="left", validate="many_to_one")
    if df["cls"].isna().any():
        raise ValueError("some predictions do not match a manifest clip")
    return df


def _prefix(df: pd.DataFrame, key: tuple) -> pd.DataFrame:
    for name, val in reversed(list(zip(GROUP, key))):
        df.insert(0, name, val)
    return df


def sheet_vs_single(df: pd.DataFrame) -> pd.DataFrame:
    """Paired comparison on the same clips: does the sheet beat the single frame? Overall and per true class."""
    rows = []
    for (model, cond, split), g in df.groupby(["model", "condition", "split"]):
        s = g[g["input"] == "single"].set_index("clip_id")
        h = g[g["input"] == "sheet"].set_index("clip_id")
        common = s.index.intersection(h.index)
        if len(common) == 0:
            continue
        s, h = s.loc[common], h.loc[common]
        cs, ch = (s["pred"] == s["cls"]).to_numpy(), (h["pred"] == h["cls"]).to_numpy()
        for scope in ["all"] + M.CLASSES:
            m = (s["cls"] == scope).to_numpy() if scope != "all" else slice(None)
            a, b = cs[m], ch[m]
            if len(a) == 0:
                continue
            t = M.mcnemar_exact(a, b)
            rows.append(dict(model=model, condition=cond, split=split, scope=scope, n=len(a), acc_single=a.mean(), acc_sheet=b.mean(),
                             delta=b.mean() - a.mean(), only_single_right=t["only_a"], only_sheet_right=t["only_b"], p_mcnemar=t["p"]))
    return pd.DataFrame(rows)


def compute_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out: dict[str, list] = {k: [] for k in ("summary", "per_class", "by_severity", "by_scene", "confusion")}
    for key, g in df.groupby(GROUP):
        out["summary"].append(_prefix(pd.DataFrame([M.summary(g)]), key))
        out["per_class"].append(_prefix(M.per_class_table(g), key))
        out["by_severity"].append(_prefix(M.by_severity(g), key))
        out["by_scene"].append(_prefix(M.by_scene(g), key))
        out["confusion"].append(_prefix(M.confusion(g).reset_index(), key))
    tables = {k: pd.concat(v, ignore_index=True) for k, v in out.items()}
    tables["sheet_vs_single"] = sheet_vs_single(df)
    return tables


def _md(df: pd.DataFrame, cols: list[str], pct: tuple = ()) -> str:
    def f(c, v):
        if isinstance(v, float):
            if v != v:
                return "n/a"
            return f"{v:.2f}" if c not in pct else f"{100 * v:.0f}%"
        return str(v)

    head = "| " + " | ".join(cols) + " |\n|" + "|".join(["---"] * len(cols)) + "|\n"
    return head + "\n".join("| " + " | ".join(f(c, r[c]) for c in cols) + " |" for _, r in df.iterrows()) + "\n"


def write_report(tables: dict[str, pd.DataFrame], out_dir: Path, title: str = "Results") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, t in tables.items():
        t.to_csv(out_dir / f"{name}.csv", index=False, lineterminator="\n")
    s = tables["summary"].copy()
    s["accuracy (95% CI)"] = [f"{100*a:.0f}% [{100*lo:.0f}, {100*hi:.0f}]" for a, lo, hi in zip(s.accuracy, s.acc_lo, s.acc_hi)]
    s["fault detected (95% CI)"] = [f"{100*a:.0f}% [{100*lo:.0f}, {100*hi:.0f}]" for a, lo, hi in zip(s.detect_recall, s.detect_lo, s.detect_hi)]
    s["false alarm (95% CI)"] = [f"{100*a:.0f}% [{100*lo:.0f}, {100*hi:.0f}]" for a, lo, hi in zip(s.false_alarm, s.fa_lo, s.fa_hi)]
    pc = tables["per_class"].pivot_table(index=GROUP, columns="cls", values="recall").reset_index()
    pc = pc[GROUP + [c for c in M.CLASSES if c in pc.columns]]
    sv = tables["sheet_vs_single"]
    md = [f"# {title}\n",
          "Wilson 95 % intervals. All data are synthetic. Sheet versus single frame: exact McNemar test on paired clips; p-values are "
          "descriptive and not corrected for multiple comparisons.\n",
          "## Summary\n", _md(s, GROUP + ["n", "accuracy (95% CI)", "fault detected (95% CI)", "false alarm (95% CI)", "macro_recall", "invalid_rate"],
                              pct=("macro_recall", "invalid_rate")),
          "\n## Recall per class\n", _md(pc, list(pc.columns), pct=tuple(M.CLASSES)),
          "\n## Sheet versus single frame (paired)\n",
          _md(sv, ["model", "condition", "split", "scope", "n", "acc_single", "acc_sheet", "delta", "only_single_right", "only_sheet_right", "p_mcnemar"],
              pct=("acc_single", "acc_sheet", "delta")) if len(sv) else "No model has both inputs on the same split and condition.\n",
          "\nSee the CSV files in this folder for severity bins, scenes and confusion matrices. Held-out D1 has 40 clips per class "
          "(about 13 per severity bin), so its intervals are wide.\n"]
    p = out_dir / "results.md"
    p.write_text("\n".join(md), encoding="utf-8")
    return p
