"""Manifest: spec columns joined with the ground truth and provenance in each clip's sidecar JSON."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .. import paths
from .generate import clip_paths

MANIFEST_SCHEMA_VERSION = "1"

# Order of columns in manifest.csv. Ground-truth columns come from the sim (build independent);
# the provenance columns at the end depend on the machine that rendered the clip.
GROUND_TRUTH = ["clip_id", "split", "cls", "scene_config_id", "seed", "severity", "severity_bin", "severity_physical", "severity_unit",
                "onset_s", "halt_x", "fault_region_px", "visible_px_samples", "expected_visible_tiles", "visibility_rule",
                "post_onset_tiles", "halted_tiles", "frozen_tiles", "neighbor_gap_at_onset_px", "gate_schedule_id", "belt_speed",
                "brightness", "fps", "n_frames", "sample_times", "label_source", "sim_schema_version"]
PROVENANCE = ["path", "bytes", "sha256", "ffmpeg_version", "opencv_version", "numpy_version", "python_version", "git_commit",
              "manifest_schema_version"]
COLUMNS = GROUND_TRUTH + PROVENANCE


def _cell(v):
    if isinstance(v, (list, tuple)):
        return json.dumps(v)
    return v


def read_sidecars(specs: pd.DataFrame, root: Path | None = None) -> pd.DataFrame:
    rows = []
    for r in specs.itertuples():
        _, js = clip_paths(r.split, r.clip_id, root)
        if not js.exists():
            continue
        d = json.loads(js.read_text(encoding="utf-8"))
        d["path"] = f"clips/{r.split}/{r.clip_id}.mp4"
        d["manifest_schema_version"] = MANIFEST_SCHEMA_VERSION
        rows.append({c: _cell(d.get(c)) for c in COLUMNS})
    return pd.DataFrame(rows, columns=COLUMNS)


def write_manifest(df: pd.DataFrame, path: str | Path | None = None) -> Path:
    p = Path(path) if path else paths.manifest_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False, lineterminator="\n")
    return p


def read_manifest(path: str | Path | None = None) -> pd.DataFrame:
    return pd.read_csv(path or paths.manifest_path())
