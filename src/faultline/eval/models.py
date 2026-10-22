"""Model interface and trivial reference models, plus the runner that produces a predictions table.

A model has .name, .needs_images and predict(clip_ids, images) -> list of labels (one of the five classes or 'invalid').
Reference models need no images and exist to test the scoring and to give floor and ceiling rows.
"""

from __future__ import annotations

import zlib
from typing import Protocol

import numpy as np
import pandas as pd

from ..sim import config as C
from .inputs import load_input

CLASSES = list(C.CLASSES)


class Model(Protocol):
    name: str
    needs_images: bool

    def predict(self, clip_ids: list[str], images: list) -> list[str]: ...


class ConstantModel:
    needs_images = False

    def __init__(self, label: str = "normal"):
        self.label, self.name = label, f"constant_{label}"

    def predict(self, clip_ids, images):
        return [self.label] * len(clip_ids)


class RandomModel:
    needs_images = False
    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def predict(self, clip_ids, images):
        return [str(x) for x in self.rng.choice(CLASSES, size=len(clip_ids))]


class OracleModel:
    """Reads the true label from the manifest. Must score 100 %: used to test the pipeline, never as a result."""

    needs_images = False
    name = "oracle"

    def __init__(self, manifest: pd.DataFrame):
        self.truth = dict(zip(manifest["clip_id"], manifest["cls"]))

    def predict(self, clip_ids, images):
        return [self.truth[c] for c in clip_ids]


class NoisyOracleModel(OracleModel):
    """Oracle that is wrong on a fixed fraction of clips (deterministic per clip id)."""

    def __init__(self, manifest: pd.DataFrame, error_rate: float, seed: int = 0, name: str | None = None):
        super().__init__(manifest)
        self.error_rate, self.seed = error_rate, seed
        self.name = name or f"oracle_err{int(round(100 * error_rate))}"

    def predict(self, clip_ids, images):
        out = []
        for c in clip_ids:
            rng = np.random.default_rng([self.seed, zlib.crc32(c.encode("utf-8"))])
            t = self.truth[c]
            out.append(str(rng.choice([k for k in CLASSES if k != t])) if rng.uniform() < self.error_rate else t)
        return out


def run_model(model: Model, manifest: pd.DataFrame, split: str, kind: str, condition: str, batch_size: int = 32,
              limit: int | None = None, progress: bool = False, clips_root=None, degraded_root=None) -> pd.DataFrame:
    """Run a model over one split with one input kind and condition; return a predictions table."""
    ids = manifest.loc[manifest["split"] == split, "clip_id"].tolist()
    if limit is not None:
        ids = ids[:limit]
    batches = range(0, len(ids), batch_size)
    if progress:
        from tqdm import tqdm

        batches = tqdm(batches, unit="batch")
    preds: list[str] = []
    for i in batches:
        chunk = ids[i:i + batch_size]
        imgs = [load_input(split, c, condition, kind, clips_root, degraded_root) for c in chunk] if model.needs_images else [None] * len(chunk)
        preds += model.predict(chunk, imgs)
    return pd.DataFrame(dict(model=model.name, input=kind, condition=condition, split=split, clip_id=ids, pred=preds))
