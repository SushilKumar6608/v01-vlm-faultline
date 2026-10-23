"""Read-only audit of everything built so far (Phases 0-5). Changes nothing. About 1-2 minutes.

  python scripts/verify_progress.py            # all checks
  python scripts/verify_progress.py --quick    # skip the model re-prediction check

Each check prints PASS, FAIL or SKIP with the reason; the exit code is 1 if anything failed.
Run `git fetch` first so the check against GitHub is current.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import traceback

import numpy as np
import pandas as pd

from faultline import paths
from faultline.data import manifest as mf
from faultline.degrade.recipe_v1 import RECIPE_V1, recipe_hash
from faultline.eval import metrics as M

ROOT = paths.repo_root()
PINNED_RECIPE = "23356dcd8b203486df22394109e9008b50b18d1be55998168e6934208b9d528c"
RESULTS = []
CHECKS = []


def check(name):
    def deco(fn):
        def run(*a, **k):
            try:
                res = fn(*a, **k)
                status, msg = res if isinstance(res, tuple) else ("PASS", res or "")
            except Exception as e:  # noqa: BLE001
                status, msg = "FAIL", f"error: {e!r}"
                if "-v" in sys.argv:
                    traceback.print_exc()
            RESULTS.append(status)
            print(f"  {status:<5} {name}" + (f" - {msg}" if msg else ""))
        CHECKS.append(run)
        return run
    return deco


def git(*args) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, cwd=ROOT).stdout.strip()


def sha(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def ok(cond, msg_ok="", msg_bad=""):
    return ("PASS", msg_ok) if cond else ("FAIL", msg_bad)


# ---------------------------------------------------------------- repository
@check("git: working tree clean")
def _():
    s = git("status", "--porcelain")
    return ok(not s, "", f"uncommitted changes:\n{s}")


@check("git: local main equals origin/main (run `git fetch` first)")
def _():
    a, b = git("rev-parse", "HEAD"), git("rev-parse", "origin/main")
    return ok(a == b, a[:7], f"HEAD {a[:7]} vs origin/main {b[:7]}")


@check("git: milestone tags exist (dataset-v1, degradation-v1, m1-cnn-baseline)")
def _():
    have = set(git("tag").split())
    miss = {"dataset-v1", "degradation-v1", "m1-cnn-baseline"} - have
    return ok(not miss, "", f"missing {sorted(miss)}")


@check("repo text: no tool or vendor names in tracked files or commit messages")
def _():
    pat = "claude|anthropic|co-authored-by|(^|[^a-z])generated with"
    files = git("grep", "-IiEl", pat).split()
    msgs = git("log", "--all", "--format=%h", "-i", "-E", f"--grep={pat}")
    return ok(not files and not msgs, "", f"files {files[:5]} commits {msgs.split()[:5]}")


# ---------------------------------------------------------------- dataset
def manifest():
    return mf.read_manifest()


@check("dataset: manifest has 2,450 clips, split sizes 1500/250/500/200, 5 classes balanced")
def _():
    m = manifest()
    sizes = m.groupby("split").size().to_dict()
    per = m.groupby(["split", "cls"]).size().unstack()
    exp = {"train": 1500, "val": 250, "test": 500, "heldout": 200}
    bal = all((per.loc[s] == n // 5).all() for s, n in exp.items())
    return ok(len(m) == 2450 and sizes == exp and bal, f"{sizes}", f"sizes {sizes}, balanced={bal}")


@check("dataset: clip ids and seeds are unique; seed blocks per split do not overlap")
def _():
    m = manifest()
    blocks = {"train": 100000, "val": 200000, "test": 300000, "heldout": 400000}
    inb = all(((m[m.split == s].seed >= b) & (m[m.split == s].seed < b + 100000)).all() for s, b in blocks.items())
    return ok(m.clip_id.is_unique and m.seed.is_unique and inb, "", f"ids unique={m.clip_id.is_unique} seeds unique={m.seed.is_unique} in-block={inb}")


@check("dataset: held-out scene D1 appears only in held-out; A1-A3 only elsewhere")
def _():
    m = manifest()
    sc = m.groupby("split").scene_config_id.unique().to_dict()
    good = set(sc["heldout"]) == {"D1"} and all(set(sc[s]) <= {"A1", "A2", "A3"} for s in ("train", "val", "test"))
    return ok(good, str({k: sorted(v) for k, v in sc.items()}), str(sc))


@check("dataset: manifest and specs hashes equal the frozen record (configs/dataset_v1.json)")
def _():
    fz = json.loads((ROOT / "configs" / "dataset_v1.json").read_text(encoding="utf-8"))
    a, b = sha(paths.manifest_path()), sha(ROOT / "configs" / "specs_v1.csv")
    return ok(a == fz["manifest_sha256"] and b == fz["specs_sha256"], "", f"manifest match={a == fz['manifest_sha256']} specs match={b == fz['specs_sha256']}")


@check("dataset: all 2,450 clean clip files exist; 100 random files match their recorded sha256")
def _():
    m = manifest()
    from faultline.data.generate import clip_paths
    miss = [c for s, c in zip(m.split, m.clip_id) if not clip_paths(s, c)[0].exists()]
    if miss:
        return "FAIL", f"{len(miss)} missing, e.g. {miss[:3]}"
    bad = [r.clip_id for r in m.sample(100, random_state=0).itertuples() if sha(clip_paths(r.split, r.clip_id)[0]) != r.sha256]
    return ok(not bad, "", f"hash mismatch: {bad[:3]}")


# ---------------------------------------------------------------- degradation
@check("degradation: recipe hash equals the pinned value; recipe.json marker agrees")
def _():
    mk = json.loads((paths.degraded_dir() / "recipe.json").read_text(encoding="utf-8"))
    return ok(recipe_hash(RECIPE_V1) == PINNED_RECIPE == mk["hash"], PINNED_RECIPE[:12], f"code {recipe_hash(RECIPE_V1)[:12]} marker {mk['hash'][:12]}")


@check("degradation: 2,450 degraded files, all one recipe, 100 random files match their sha256")
def _():
    d = pd.read_csv(paths.data_root() / "degraded_manifest.csv")
    m = manifest()
    same = set(d.clip_id) == set(m.clip_id) and len(d) == 2450
    one = d.recipe_hash.nunique() == 1 and d.recipe_hash.iloc[0] == PINNED_RECIPE
    absent = [r.clip_id for r in d.itertuples() if not (paths.degraded_dir() / r.split / f"{r.clip_id}.mp4").exists()]
    sub = d.sample(100, random_state=0)
    bad = [r.clip_id for r in sub.itertuples() if sha(paths.degraded_dir() / r.split / f"{r.clip_id}.mp4") != r.sha256]
    return ok(same and one and not bad and not absent, "", f"ids match={same} one recipe={one} missing={absent[:3]} bad hashes={bad[:3]}")


# ---------------------------------------------------------------- baseline results
PRED_FILES = [f"cnn_s{s}_{k}.csv" for s in (0, 1, 2) for k in ("single", "sheet")]


def predictions():
    return pd.concat([pd.read_csv(ROOT / "results" / "predictions" / f) for f in PRED_FILES], ignore_index=True)


@check("results: 6 prediction files; every (model, input, condition, split) covers its split exactly, valid labels only")
def _():
    m, p = manifest(), predictions()
    errs = []
    for (mod, inp, cond, sp), g in p.groupby(["model", "input", "condition", "split"]):
        want = set(m[m.split == sp].clip_id)
        if set(g.clip_id) != want or len(g) != len(want):
            errs.append((mod, inp, cond, sp))
    n_groups = p.groupby(["model", "input", "condition", "split"]).ngroups
    valid = set(p.pred) <= set(M.CLASSES) | {M.INVALID}
    return ok(not errs and n_groups == 36 and valid, f"{n_groups} groups", f"incomplete {errs[:3]}, groups={n_groups}, valid={valid}")


@check("results: independent re-score of accuracy equals results/cnn_baseline/summary.csv in all 36 groups")
def _():
    m, p = manifest(), predictions()
    truth = dict(zip(m.clip_id, m.cls))
    p["ok"] = [truth[c] == x for c, x in zip(p.clip_id, p.pred)]
    mine = p.groupby(["model", "input", "condition", "split"]).ok.mean()
    s = pd.read_csv(ROOT / "results" / "cnn_baseline" / "summary.csv").set_index(["model", "input", "condition", "split"]).accuracy
    d = (mine - s.reindex(mine.index)).abs().max()
    return ok(d < 1e-9, f"max difference {d:.1e}", f"max difference {d}")


@check("results: README and findings headline numbers equal the recomputed means over 3 seeds (test split)")
def _():
    m, p = manifest(), predictions()
    truth = dict(zip(m.clip_id, m.cls))
    p["ok"] = [truth[c] == x for c, x in zip(p.clip_id, p.pred)]
    t = p[p.split == "test"].groupby(["input", "condition", "model"]).ok.mean().groupby(["input", "condition"]).agg(["mean", "min", "max"]) * 100
    print(t.round(1).to_string().replace("\n", "\n         "))
    r = open(ROOT / "README.md", encoding="utf-8").read()
    want = {("single", "clean"): "96.7", ("sheet", "clean"): "99.9"}
    bad = [k for k, v in want.items() if f"{t.loc[k, 'mean']:.1f}" != v or v not in r]
    wd = {("single", "degraded"): 57, ("sheet", "degraded"): 62}
    bad += [k for k, v in wd.items() if round(t.loc[k, "mean"]) != v]
    return ok(not bad, "", f"README disagrees for {bad}")


@check("results: six finished training runs (done.json) with validation accuracy above 0.9")
def _():
    out = []
    for s in (0, 1, 2):
        for k in ("single", "sheet"):
            j = json.loads((paths.data_root() / "runs" / f"cnn_s{s}_{k}" / "done.json").read_text(encoding="utf-8"))
            out.append(j["best_val_acc"])
    return ok(min(out) > 0.9 and len(out) == 6, f"min {min(out):.3f} max {max(out):.3f}", str(out))


@check("pipeline: tile cache matches a fresh decode of 3 clips per condition (same pixels the models saw)")
def _():
    from faultline.models import cache
    m, bad = manifest(), []
    root = paths.data_root() / "cache_tiles"
    for cond in ("clean", "degraded"):
        arr = cache.load(m, cond, "test", root, mmap=True)
        ids = m[m.split == "test"].clip_id.tolist()
        for i in (0, len(ids) // 2, len(ids) - 1):
            fresh = cache._load(("test", ids[i], cond, None, None))
            if not np.array_equal(arr[i], fresh):
                bad.append((cond, ids[i]))
    return ok(not bad, "", f"differs: {bad}")


@check("pipeline: re-running the saved seed-0 models on the cached test tiles reproduces the stored predictions")
def _():
    if "--quick" in sys.argv:
        return "SKIP", "--quick"
    import torch
    from faultline.models import cache
    from faultline.models.train import CLASSES, TrainConfig, load_best, predict_logits
    m, root = manifest(), paths.data_root() / "cache_tiles"
    p = predictions()
    torch.set_num_threads(4)
    bad = []
    for kind in ("single", "sheet"):
        cfg = TrainConfig(kind=kind, seed=0)
        model = load_best(cfg, paths.data_root() / "runs" / f"{cfg.name}_{kind}")
        for cond in ("clean", "degraded"):
            x = cache.load(m, cond, "test", root, mmap=True)
            pred = [CLASSES[k] for k in predict_logits(model, x, kind).argmax(1)]
            old = p[(p.model == cfg.name) & (p.input == kind) & (p.condition == cond) & (p.split == "test")].sort_values("clip_id")
            mine = pd.Series(pred, index=m[m.split == "test"].clip_id).sort_index()
            if not (old.pred.values == mine.values).all():
                bad.append((kind, cond))
    return ok(not bad, "", f"different: {bad}")


@check("no leakage: nothing in the test or held-out tiles is also in the training tiles (exact duplicate scan)")
def _():
    from faultline.models import cache
    m, root = manifest(), paths.data_root() / "cache_tiles"
    tr = cache.load(m, "clean", "train", root, mmap=True)
    hashes = {hashlib.md5(np.ascontiguousarray(tr[i, 7]).tobytes()).hexdigest() for i in range(len(tr))}
    dup = 0
    for sp in ("val", "test", "heldout"):
        a = cache.load(m, "clean", sp, root, mmap=True)
        dup += sum(hashlib.md5(np.ascontiguousarray(a[i, 7]).tobytes()).hexdigest() in hashes for i in range(len(a)))
    return ok(dup == 0, "0 duplicate last tiles", f"{dup} duplicates")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("-v", action="store_true")
    ap.parse_args()
    print(f"Audit of {ROOT}\n")
    for f in CHECKS:
        f()
    n_fail = RESULTS.count("FAIL")
    print(f"\n{RESULTS.count('PASS')} passed, {n_fail} failed, {RESULTS.count('SKIP')} skipped")
    sys.exit(1 if n_fail else 0)



if __name__ == "__main__":
    main()
