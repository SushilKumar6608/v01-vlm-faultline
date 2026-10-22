"""Check the scoring on the real manifest with reference models that need no images (about a minute).

  python scripts/selftest_eval.py

An oracle must score 100 %, always-normal must score exactly 20 % with every normal clip right, a random model must land near
20 %, and a noisy model that is better with sheets than with single frames must show that in the paired comparison.
Writes data/selftest/ (not tracked). Exit code 1 if any check fails.
"""
from __future__ import annotations

import sys

import pandas as pd

from faultline import paths
from faultline.data import manifest
from faultline.eval import report
from faultline.eval.models import ConstantModel, NoisyOracleModel, OracleModel, RandomModel, run_model


def main() -> None:
    m = manifest.read_manifest()
    parts = []
    for split in ("test", "heldout"):
        for kind in ("single", "sheet"):
            err = 0.30 if kind == "single" else 0.10
            models = [OracleModel(m), ConstantModel("normal"), RandomModel(0), NoisyOracleModel(m, err, seed=1, name="demo_noisy")]
            for mod in models:
                parts.append(run_model(mod, m, split, kind, "clean"))
    preds = pd.concat(parts, ignore_index=True)
    errs = report.validate_predictions(preds, m)
    if errs:
        sys.exit("\n".join(errs))
    tables = report.compute_tables(report.attach_truth(preds, m))
    out = paths.data_root() / "selftest"
    report.write_report(tables, out, "Scoring self-test (reference models, no images)")
    s = tables["summary"].set_index(report.GROUP)

    def acc(model, kind, split):
        return float(s.loc[(model, kind, "clean", split), "accuracy"])

    pc = tables["per_class"]
    normal_recall = float(pc[(pc.model == "constant_normal") & (pc.input == "sheet") & (pc.split == "test") & (pc.cls == "normal")].recall.iloc[0])
    sv = tables["sheet_vs_single"]
    noisy = sv[(sv.model == "demo_noisy") & (sv.split == "test") & (sv.scope == "all")].iloc[0]
    checks = [
        ("oracle scores 100% on every split and input", all(acc("oracle", k, sp) == 1.0 for k in ("single", "sheet") for sp in ("test", "heldout"))),
        ("always-normal scores exactly 20%", all(abs(acc("constant_normal", k, sp) - 0.2) < 1e-9 for k in ("single", "sheet") for sp in ("test", "heldout"))),
        ("always-normal gets every normal clip right", normal_recall == 1.0),
        ("random model is between 10% and 30%", all(0.1 < acc("random", k, sp) < 0.3 for k in ("single", "sheet") for sp in ("test", "heldout"))),
        ("noisy model: sheet better than single, p < 0.05 on test", noisy.delta > 0 and noisy.p_mcnemar < 0.05),
        ("results.md and CSVs written", (out / "results.md").exists() and (out / "summary.csv").exists()),
    ]
    for name, ok in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    print("\nwrote", out / "results.md")
    sys.exit(0 if all(ok for _, ok in checks) else 1)


if __name__ == "__main__":
    main()
