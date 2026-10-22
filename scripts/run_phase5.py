"""Phase 5: CNN baseline on the CPU, built to survive crashes. Run it, leave it, and if anything stops run the same command again.

  python scripts/run_phase5.py                    # tile cache, then 3 seeds x (single, sheet), then the results report
  python scripts/run_phase5.py --benchmark        # time a few training steps and estimate the total, then stop
  python scripts/run_phase5.py --seeds 0          # only seed 0 (quicker)

Safety: every finished step is kept (tile cache files, per-epoch checkpoints, prediction files), so re-running skips what is done
and resumes the unfinished training at its last epoch. Threads are capped (--threads, default 6), the process runs at below-normal
priority, and on Windows the computer is asked not to go to sleep while this runs (closing the lid can still sleep it: set
"when I close the lid" to "do nothing" in Power options). Progress: data/runs/phase5.log and data/runs/status.json.
"""
from __future__ import annotations

import argparse
import os
import sys
import time


def _early_env(argv: list[str]) -> None:
    n = "6"
    for i, a in enumerate(argv):
        if a == "--threads" and i + 1 < len(argv):
            n = argv[i + 1]
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = n


_early_env(sys.argv)

import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from faultline import paths  # noqa: E402
from faultline.data import manifest as mf  # noqa: E402
from faultline.eval import report  # noqa: E402
from faultline.models import cache  # noqa: E402
from faultline.models.cnn import TileCNN  # noqa: E402
from faultline.models.train import CLASSES, TrainConfig, load_best, predictions_table, select_tiles, to_tensor, train  # noqa: E402

EVAL_SPLITS = ("val", "test", "heldout")
CONDITIONS = ("clean", "degraded")


def keep_awake_and_low_priority() -> None:
    if sys.platform != "win32":
        return
    import ctypes

    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)  # BELOW_NORMAL


class Logger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()


def write_status(runs: Path, **kw) -> None:
    tmp = runs / "status.json.tmp"
    tmp.write_text(json.dumps(dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), **kw), indent=2), encoding="utf-8")
    os.replace(tmp, runs / "status.json")


def benchmark(m: pd.DataFrame, cache_root: Path, a, log) -> None:
    x = cache.load(m, "clean", "train", cache_root)
    y = cache.labels(m, "train", CLASSES)
    n_train = len(y)
    for kind in ("single", "sheet"):
        cfg = TrainConfig(kind=kind)
        model = TileCNN(cfg.n_tiles)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        xs = select_tiles(x, kind)
        times = []
        for i in range(6):
            idx = np.arange(i * cfg.batch, (i + 1) * cfg.batch)
            t0 = time.time()
            loss = torch.nn.functional.cross_entropy(model(to_tensor(xs[idx])), torch.from_numpy(y[idx]))
            opt.zero_grad()
            loss.backward()
            opt.step()
            times.append(time.time() - t0)
        step = float(np.median(times[1:]))
        epoch_s = step * (n_train // cfg.batch)
        log(f"benchmark {kind}: {step:.2f} s per batch of {cfg.batch} -> about {epoch_s / 60:.1f} min per epoch, "
            f"{epoch_s * a.epochs / 3600:.1f} h per run of {a.epochs} epochs")
        per_seed = epoch_s * a.epochs
        if kind == "single":
            tot = per_seed
        else:
            tot += per_seed
    log(f"estimate for {len(a.seeds)} seed(s), single + sheet: about {tot * len(a.seeds) / 3600:.1f} h (plus about 5-10 min for the cache)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--workers", type=int, default=4, help="processes for building the tile cache")
    ap.add_argument("--pause", type=float, default=0.0, help="seconds to rest after every epoch (lets the laptop cool)")
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--no-report", action="store_true")
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    keep_awake_and_low_priority()
    runs = paths.data_root() / "runs"
    log = Logger(runs / "phase5.log")
    cache_root = paths.data_root() / "cache_tiles"
    m = mf.read_manifest()
    log(f"Phase 5 start: seeds {a.seeds}, {a.epochs} epochs, {a.threads} threads, torch {torch.__version__}")

    # 1. tile cache (skips files that exist)
    jobs = [("clean", s) for s in ("train",) + EVAL_SPLITS] + [("degraded", s) for s in EVAL_SPLITS]
    for cond, split in jobs:
        if cache.cache_file(cache_root, cond, split).exists():
            continue
        t0 = time.time()
        write_status(runs, stage="cache", item=f"{cond}/{split}")
        cache.build(m, cond, split, cache_root, workers=a.workers, progress=True)
        log(f"cache {cond}/{split} built in {time.time() - t0:.0f}s")
    if a.benchmark:
        benchmark(m, cache_root, a, log)
        return

    x_train = cache.load(m, "clean", "train", cache_root)
    y_train, y_val = cache.labels(m, "train", CLASSES), cache.labels(m, "val", CLASSES)
    x_val = cache.load(m, "clean", "val", cache_root)
    eval_caches = {(c, s): cache.load(m, c, s, cache_root, mmap=True) for c in CONDITIONS for s in EVAL_SPLITS}

    # 2. train + predict, seed by seed (a partial result is usable after the first seed)
    pred_dir = paths.repo_root() / "results" / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    for seed in a.seeds:
        for kind in ("single", "sheet"):
            cfg = TrainConfig(kind=kind, seed=seed, epochs=a.epochs)
            run_dir = runs / f"{cfg.name}_{kind}"
            out = pred_dir / f"{cfg.name}_{kind}.csv"
            if out.exists() and (run_dir / "done.json").exists():
                continue
            write_status(runs, stage="train", seed=seed, kind=kind)
            info = train(cfg, x_train, y_train, x_val, y_val, run_dir, log=log, pause=a.pause)
            model = load_best(cfg, run_dir)
            table = predictions_table(model, cfg, m, eval_caches, log)
            tmp = out.with_suffix(".tmp")
            table.to_csv(tmp, index=False)
            os.replace(tmp, out)
            log(f"[{cfg.name}/{kind}] best epoch {info['best_epoch'] + 1}, val acc {info['best_val_acc']:.3f}; wrote {out.name}")

    # 3. report
    write_status(runs, stage="report")
    if not a.no_report:
        files = sorted(pred_dir.glob("cnn_s*_*.csv"))
        preds = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
        done_models = [mdl for mdl, g in preds.groupby("model") if g["input"].nunique() == 2]
        preds = preds[preds["model"].isin(done_models)]
        errs = report.validate_predictions(preds, m)
        if errs:
            log("prediction check failed:\n" + "\n".join(errs))
            sys.exit(1)
        tables = report.compute_tables(report.attach_truth(preds, m))
        out_dir = paths.repo_root() / "results" / "cnn_baseline"
        report.write_report(tables, out_dir, "CNN baseline (from scratch, trained on clean clips, CPU)")
        s = tables["summary"]
        seedtab = (s.groupby(["input", "condition", "split"])["accuracy"].agg(["mean", "std", "min", "max", "count"]).reset_index())
        seedtab.to_csv(out_dir / "seed_summary.csv", index=False)
        log("seed summary (accuracy across seeds):\n" + seedtab.to_string(index=False))
        log(f"report written to {out_dir}")
    write_status(runs, stage="finished")
    log("Phase 5 finished")


if __name__ == "__main__":
    main()
