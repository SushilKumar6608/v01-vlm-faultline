"""Dataset spec: one row per clip (split, class, scene, seed, severity). No rendering here.

The spec is a pure function of the YAML config, so it can be rebuilt byte for byte. Replacement
clips for flagged ones are recorded in replacements.csv and applied on top (see regen.py).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .. import paths
from ..sim import config as C

SPEC_COLUMNS = ["clip_id", "split", "cls", "scene_config_id", "seed", "severity", "severity_bin"]
REPLACEMENT_COLUMNS = ["clip_id_old", "seed_old", "clip_id_new", "seed_new", "split", "cls", "scene_config_id", "severity", "reason"]


def load_config(path: str | Path | None = None) -> dict:
    p = Path(path) if path else paths.configs_dir() / "dataset_v1.yaml"
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def clip_id(cls: str, scene: str, seed: int) -> str:
    return f"{cls}_{scene}_{seed:06d}"


def _stratified_severities(rng: np.random.Generator, n: int) -> np.ndarray:
    """n severities spread evenly over [SEVERITY_MIN, SEVERITY_MAX): one random point per equal slice, shuffled."""
    u = (np.arange(n) + rng.uniform(0.0, 1.0, n)) / n
    sev = C.SEVERITY_MIN + (C.SEVERITY_MAX - C.SEVERITY_MIN) * u
    sev = np.minimum(np.round(sev, 4), C.SEVERITY_MAX)
    return rng.permutation(sev)


def build_specs(cfg: dict) -> pd.DataFrame:
    frames = []
    for si, (split, sc) in enumerate(cfg["splits"].items()):
        n, scenes = int(sc["n_per_class"]), list(sc["scenes"])
        rng = np.random.default_rng([int(cfg["master_seed"]), si])
        rows = []
        for cls in C.CLASSES:
            sev = None if cls == "normal" else _stratified_severities(rng, n)
            for i in range(n):
                rows.append(dict(split=split, cls=cls, scene_config_id=scenes[i % len(scenes)],
                                 severity=None if sev is None else float(sev[i])))
        order = rng.permutation(len(rows))  # shuffle so seeds are not correlated with class
        rows = [rows[j] for j in order]
        for k, r in enumerate(rows):
            r["seed"] = int(sc["seed_start"]) + k
            r["clip_id"] = clip_id(r["cls"], r["scene_config_id"], r["seed"])
            r["severity_bin"] = C.severity_bin(r["severity"])
        frames.append(pd.DataFrame(rows, columns=SPEC_COLUMNS))
    return pd.concat(frames, ignore_index=True)


def write_specs(df: pd.DataFrame, path: str | Path | None = None) -> Path:
    p = Path(path) if path else paths.data_root() / "specs.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False, lineterminator="\n", float_format="%.4f")
    return p


def spec_path_committed() -> Path:
    """specs.csv and replacements.csv are small and committed, so they live in configs/."""
    return paths.configs_dir() / "specs_v1.csv"


def replacements_path() -> Path:
    return paths.configs_dir() / "replacements_v1.csv"


def load_replacements() -> pd.DataFrame:
    p = replacements_path()
    if not p.exists():
        return pd.DataFrame(columns=REPLACEMENT_COLUMNS)
    return pd.read_csv(p, dtype={"clip_id_old": str, "clip_id_new": str})


def effective_specs(cfg: dict | None = None) -> pd.DataFrame:
    """Original specs with replacements applied (same split, class, scene and severity; only the seed changes)."""
    cfg = cfg or load_config()
    df = build_specs(cfg)
    rep = load_replacements()
    if rep.empty:
        return df
    new = {r.clip_id_old: r for r in rep.itertuples()}
    out = []
    for r in df.itertuples():
        if r.clip_id in new:
            n = new[r.clip_id]
            row = r._asdict()
            row.pop("Index", None)
            row["seed"], row["clip_id"] = int(n.seed_new), n.clip_id_new
            out.append(row)
        else:
            row = r._asdict(); row.pop("Index", None); out.append(row)
    return pd.DataFrame(out, columns=SPEC_COLUMNS)


def validate_specs(df: pd.DataFrame, cfg: dict) -> list[str]:
    """Returns a list of problems (empty if the spec is sound)."""
    errs: list[str] = []
    if df.clip_id.duplicated().any():
        errs.append(f"duplicate clip_id: {df.clip_id[df.clip_id.duplicated()].tolist()[:5]}")
    if df.seed.duplicated().any():
        errs.append("duplicate seed")
    for split, sc in cfg["splits"].items():
        d = df[df.split == split]
        n, scenes = int(sc["n_per_class"]), list(sc["scenes"])
        if len(d) != n * len(C.CLASSES):
            errs.append(f"{split}: {len(d)} rows, expected {n * len(C.CLASSES)}")
        lo, hi = int(sc["seed_start"]), int(sc["seed_start"]) + n * len(C.CLASSES)
        reserve = int(cfg["reserve_seed_start"])
        if not ((d.seed >= lo).all() and ((d.seed < hi) | (d.seed >= reserve)).all()):
            errs.append(f"{split}: seeds outside block [{lo},{hi}) (replacements aside)")
        for cls in C.CLASSES:
            dc = d[d.cls == cls]
            if len(dc) != n:
                errs.append(f"{split}/{cls}: {len(dc)} rows, expected {n}")
            if not set(dc.scene_config_id) <= set(scenes):
                errs.append(f"{split}/{cls}: scene outside {scenes}")
            for sc_id in scenes:
                k = int((dc.scene_config_id == sc_id).sum())
                if abs(k - n / len(scenes)) > 1:
                    errs.append(f"{split}/{cls}/{sc_id}: {k} clips, expected about {n / len(scenes):.1f}")
            if cls == "normal":
                if dc.severity.notna().any() or dc.severity_bin.notna().any():
                    errs.append(f"{split}/normal: severity must be null")
            else:
                if dc.severity.isna().any():
                    errs.append(f"{split}/{cls}: missing severity")
                    continue
                if not dc.severity.between(C.SEVERITY_MIN, C.SEVERITY_MAX).all():
                    errs.append(f"{split}/{cls}: severity outside range")
                bins = [C.severity_bin(s) for s in dc.severity]
                if bins != dc.severity_bin.tolist():
                    errs.append(f"{split}/{cls}: severity_bin does not match severity")
                for _, _, name in C.SEVERITY_BINS:
                    k = bins.count(name)
                    if abs(k - n / 3) > 1.5:
                        errs.append(f"{split}/{cls}/{name}: {k} clips, expected about {n / 3:.1f}")
    return errs
