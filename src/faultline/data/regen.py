"""Replace flagged clips: same split, class, scene and severity; only the seed changes (reserve block).

The flagged original is moved to data/quarantine/<split>/ (not deleted) and the replacement is logged
in configs/replacements_v1.csv, which is committed, so the effective spec stays reproducible.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from .. import paths
from .generate import clip_paths
from .spec import REPLACEMENT_COLUMNS, build_specs, clip_id, load_replacements, replacements_path


def quarantine_dir() -> Path:
    return paths.data_root() / "quarantine"


def regenerate(flagged: pd.DataFrame, cfg: dict, root: Path | None = None, dry_run: bool = False) -> pd.DataFrame:
    """flagged needs columns clip_id and reason (clip ids of the *effective* spec). Returns the rows added or updated."""
    rep = load_replacements()
    original = {r.clip_id: r for r in build_specs(cfg).itertuples()}
    next_seed = max([int(cfg["reserve_seed_start"])] + [int(s) + 1 for s in rep.seed_new])
    changed = []
    for f in flagged.itertuples():
        # a replacement can itself be flagged: keep one row per original clip
        hit = rep[rep.clip_id_new == f.clip_id]
        old_id = hit.clip_id_old.iloc[0] if len(hit) else f.clip_id
        o = original[old_id]
        new_seed = next_seed
        next_seed += 1
        new_id = clip_id(o.cls, o.scene_config_id, new_seed)
        reason = f.reason if not len(hit) else f"{hit.reason.iloc[0]} | then {f.reason}"
        row = dict(clip_id_old=old_id, seed_old=int(o.seed), clip_id_new=new_id, seed_new=new_seed, split=o.split, cls=o.cls,
                   scene_config_id=o.scene_config_id, severity=o.severity, reason=reason)
        changed.append(row)
        if dry_run:
            continue
        mp4, js = clip_paths(o.split, f.clip_id, root)
        qd = quarantine_dir() / o.split
        qd.mkdir(parents=True, exist_ok=True)
        for p in (mp4, js):
            if p.exists():
                shutil.move(str(p), str(qd / p.name))
        rep = rep[rep.clip_id_old != old_id]
        rep = pd.concat([rep, pd.DataFrame([row], columns=REPLACEMENT_COLUMNS)], ignore_index=True)
    if not dry_run and changed:
        p = replacements_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        rep.to_csv(p, index=False, lineterminator="\n", float_format="%.4f")
    return pd.DataFrame(changed, columns=REPLACEMENT_COLUMNS)
