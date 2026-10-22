"""Score a predictions file and write the results tables.

  python scripts/evaluate.py --preds results/preds_cnn.csv [--out results/cnn] [--allow-partial] [--title "CNN baseline"]

Predictions columns: model, input (single|sheet), condition (clean|degraded), split (val|test|heldout), clip_id, pred.
Several --preds files are combined into one report (for a table that compares models).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from faultline import paths
from faultline.data import manifest
from faultline.eval import report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--title", default="Results")
    ap.add_argument("--allow-partial", action="store_true")
    a = ap.parse_args()

    m = manifest.read_manifest()
    preds = pd.concat([pd.read_csv(p) for p in a.preds], ignore_index=True)
    errs = report.validate_predictions(preds, m, allow_partial=a.allow_partial)
    if errs:
        print("\n".join(errs))
        sys.exit(1)
    tables = report.compute_tables(report.attach_truth(preds, m))
    out = Path(a.out) if a.out else paths.repo_root() / "results" / Path(a.preds[0]).stem
    md = report.write_report(tables, out, a.title)
    print(tables["summary"][report.GROUP + ["n", "accuracy", "detect_recall", "false_alarm"]].to_string(index=False))
    print("\nwrote", md)


if __name__ == "__main__":
    main()
