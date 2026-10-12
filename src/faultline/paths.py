"""Single place that decides where files live.

Nothing else in the project may hard-code a drive letter or an absolute path.
By default everything sits inside the repository folder, so moving the project
to another drive needs no code change. To keep the (large) data elsewhere, set
the environment variable FAULTLINE_DATA to a folder.
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_DATA = "FAULTLINE_DATA"


def repo_root() -> Path:
    """Folder that contains pyproject.toml (src/faultline/paths.py -> two levels up)."""
    return Path(__file__).resolve().parents[2]


def data_root() -> Path:
    """Root for clips, degraded clips and manifest. Not tracked by git."""
    override = os.environ.get(ENV_DATA)
    return Path(override) if override else repo_root() / "data"


def clips_dir() -> Path:
    return data_root() / "clips"


def degraded_dir() -> Path:
    return data_root() / "degraded"


def manifest_path() -> Path:
    return data_root() / "manifest.csv"


def configs_dir() -> Path:
    return repo_root() / "configs"


def benchmarks_dir() -> Path:
    return repo_root() / "benchmarks"


def ensure_data_dirs() -> None:
    """Create the data folders if they do not exist yet."""
    for d in (clips_dir(), degraded_dir()):
        d.mkdir(parents=True, exist_ok=True)
