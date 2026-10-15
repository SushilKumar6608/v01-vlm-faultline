"""Dataset gate: integrity, visibility flags, warnings."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..media import read_frames
from ..sim import config as C
from .generate import clip_paths, sha256_file


def integrity(specs: pd.DataFrame, manifest: pd.DataFrame, root: Path | None = None, partial: bool = False, decode_n: int = 0, seed: int = 0) -> list[str]:
    errs: list[str] = []
    m_ids, s_ids = set(manifest.clip_id), set(specs.clip_id)
    if not m_ids <= s_ids:
        errs.append(f"manifest has clips not in the spec: {sorted(m_ids - s_ids)[:5]}")
    if not partial and m_ids != s_ids:
        errs.append(f"{len(s_ids - m_ids)} spec clips missing from the manifest")
    if manifest.clip_id.duplicated().any():
        errs.append("duplicate clip_id in the manifest")
    merged = manifest.merge(specs, on="clip_id", suffixes=("", "_spec"))
    for col in ("split", "cls", "scene_config_id", "seed"):
        bad = merged[merged[col] != merged[col + "_spec"]]
        if len(bad):
            errs.append(f"{col} differs from spec for {len(bad)} clips, e.g. {bad.clip_id.iloc[0]}")
    sev_bad = merged[~((merged.severity.isna() & merged.severity_spec.isna()) | (merged.severity - merged.severity_spec).abs().lt(1e-3))]
    if len(sev_bad):
        errs.append(f"severity differs from spec for {len(sev_bad)} clips, e.g. {sev_bad.clip_id.iloc[0]}")
    for r in manifest.itertuples():
        mp4, _ = clip_paths(r.split, r.clip_id, root)
        if not mp4.exists():
            errs.append(f"missing file {mp4.name}")
        elif sha256_file(mp4) != r.sha256:
            errs.append(f"sha256 mismatch {mp4.name}")
    n = manifest[manifest.cls == "normal"]
    if n.severity.notna().any() or n.onset_s.notna().any():
        errs.append("normal clips must have null severity and onset")
    if decode_n and len(manifest):
        rng = np.random.default_rng(seed)
        for i in rng.choice(len(manifest), size=min(decode_n, len(manifest)), replace=False):
            r = manifest.iloc[int(i)]
            mp4, _ = clip_paths(r.split, r.clip_id, root)
            try:
                fr = read_frames(mp4)
                if fr.shape != (C.N_FRAMES, C.H, C.W, 3):
                    errs.append(f"{mp4.name} decodes to shape {fr.shape}")
            except Exception as e:  # noqa: BLE001
                errs.append(f"{mp4.name} does not decode: {e}")
    return errs


def visibility_flags(manifest: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Clips that fail their class's visibility rule (to be replaced from the reserve seed block)."""
    v = cfg["visibility"]
    rows = []
    for r in manifest.itertuples():
        if r.cls == "normal":
            continue
        if r.cls == "stuck_item":
            reasons = []
            if r.halted_tiles < v["stuck_min_halted_tiles"]:
                reasons.append(f"halted_tiles={r.halted_tiles}<{v['stuck_min_halted_tiles']}")
            if r.frozen_tiles < v["stuck_min_frozen_pairs"]:
                reasons.append(f"frozen_pairs={r.frozen_tiles}<{v['stuck_min_frozen_pairs']}")
            reason = "; ".join(reasons)
        else:
            reason = f"expected_visible_tiles={r.expected_visible_tiles}<{v['min_visible_tiles']}" if r.expected_visible_tiles < v["min_visible_tiles"] else ""
        if reason:
            rows.append(dict(clip_id=r.clip_id, split=r.split, cls=r.cls, severity=r.severity, reason=reason))
    return pd.DataFrame(rows, columns=["clip_id", "split", "cls", "severity", "reason"])


def warnings_list(manifest: pd.DataFrame, cfg: dict) -> list[str]:
    w = []
    gap = cfg["visibility"]["neighbor_gap_warn_px"]
    s = manifest[(manifest.cls == "stuck_item") & (manifest.neighbor_gap_at_onset_px < gap)]
    if len(s):
        w.append(f"{len(s)} stuck_item clips halt within {gap}px of a neighbour (labelability risk): {s.clip_id.tolist()[:5]}")
    return w


def count_tables(manifest: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "split x class": pd.crosstab(manifest.split, manifest.cls),
        "split x scene": pd.crosstab(manifest.split, manifest.scene_config_id),
        "fault class x severity bin (all splits)": pd.crosstab(manifest[manifest.cls != "normal"].cls, manifest[manifest.cls != "normal"].severity_bin),
    }
