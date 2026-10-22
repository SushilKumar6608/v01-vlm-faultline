"""Scoring. Every proportion comes with n and a Wilson 95 % interval.

Truth is column 'cls', the model's answer is 'pred' (one of the five classes, or 'invalid' when the answer could not be
parsed; invalid counts as wrong and as no class). Severity bins come from the manifest.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from ..sim import config as C
from .stats import wilson

CLASSES = list(C.CLASSES)
FAULTS = list(C.FAULT_CLASSES)
INVALID = "invalid"
BIN_ORDER = ["low", "mid", "high"]


def _rate(k: int, n: int) -> tuple[float, float, float]:
    lo, hi = wilson(int(k), int(n))
    return (k / n if n else math.nan, lo, hi)


def confusion(df: pd.DataFrame) -> pd.DataFrame:
    t = pd.crosstab(df["cls"], df["pred"]).reindex(index=CLASSES, columns=CLASSES + [INVALID], fill_value=0)
    t.index.name, t.columns.name = "true", "pred"
    return t


def per_class_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in CLASSES:
        true, pred = df["cls"] == c, df["pred"] == c
        n_true, n_pred, tp = int(true.sum()), int(pred.sum()), int((true & pred).sum())
        rec, rlo, rhi = _rate(tp, n_true)
        prec, plo, phi = _rate(tp, n_pred)
        if math.isnan(rec) or math.isnan(prec):
            f1 = math.nan
        else:
            f1 = 0.0 if rec + prec == 0 else 2 * rec * prec / (rec + prec)
        rows.append(dict(cls=c, n_true=n_true, recall=rec, recall_lo=rlo, recall_hi=rhi, n_pred=n_pred, precision=prec,
                         precision_lo=plo, precision_hi=phi, f1=f1))
    return pd.DataFrame(rows)


def summary(df: pd.DataFrame) -> dict:
    n = len(df)
    k = int((df["cls"] == df["pred"]).sum())
    acc, alo, ahi = _rate(k, n)
    pc = per_class_table(df)
    fault = df["cls"].isin(FAULTS)
    flagged = df["pred"].isin(FAULTS)
    det, dlo, dhi = _rate(int((fault & flagged).sum()), int(fault.sum()))
    fa, flo, fhi = _rate(int((~fault & flagged).sum()), int((~fault).sum()))
    return dict(n=n, accuracy=acc, acc_lo=alo, acc_hi=ahi, macro_recall=float(pc["recall"].mean()),
                invalid_rate=float((df["pred"] == INVALID).mean()) if n else math.nan,
                detect_recall=det, detect_lo=dlo, detect_hi=dhi, false_alarm=fa, fa_lo=flo, fa_hi=fhi)


def by_severity(df: pd.DataFrame) -> pd.DataFrame:
    """Recall of each fault class within each severity bin (normal has no severity)."""
    f = df[df["cls"].isin(FAULTS)]
    rows = []
    for c in FAULTS:
        for b in BIN_ORDER:
            g = f[(f["cls"] == c) & (f["severity_bin"] == b)]
            k = int((g["pred"] == g["cls"]).sum())
            r, lo, hi = _rate(k, len(g))
            rows.append(dict(cls=c, severity_bin=b, n=len(g), correct=k, recall=r, lo=lo, hi=hi))
    return pd.DataFrame(rows)


def by_scene(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for s, g in df.groupby("scene_config_id"):
        k = int((g["pred"] == g["cls"]).sum())
        r, lo, hi = _rate(k, len(g))
        rows.append(dict(scene=s, n=len(g), correct=k, accuracy=r, lo=lo, hi=hi))
    return pd.DataFrame(rows)


def mcnemar_exact(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """Exact two-sided McNemar test on paired outcomes. only_a: A right and B wrong; only_b: the reverse."""
    a, b = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    n = only_a + only_b
    p = 1.0 if n == 0 else float(binomtest(min(only_a, only_b), n, 0.5).pvalue)
    return dict(only_a=only_a, only_b=only_b, p=p)
