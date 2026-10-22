"""Resumable CPU training for the CNN baseline.

Everything that matters is saved after every epoch (model, optimiser, torch random state, history) with an atomic rename, so a
crash, a power cut or Ctrl+C costs at most one epoch; running the same command again continues where it stopped. The order of the
training clips and the augmentation in each epoch depend only on (seed, epoch), so a resumed run gives the same result as one that
was never interrupted. The checkpoint with the best validation accuracy is kept for the final predictions (validation is clean data).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from ..sim import config as C
from .cnn import TileCNN, n_params

CLASSES = list(C.CLASSES)


@dataclass(frozen=True)
class TrainConfig:
    kind: str = "sheet"  # single | sheet
    seed: int = 0
    epochs: int = 30
    batch: int = 32
    lr: float = 2e-3
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    warmup_epochs: int = 2
    train_condition: str = "clean"

    def key(self) -> str:
        return hashlib.sha256(json.dumps(dataclasses.asdict(self), sort_keys=True).encode()).hexdigest()[:16]

    @property
    def name(self) -> str:
        return f"cnn_s{self.seed}"

    @property
    def n_tiles(self) -> int:
        return 8 if self.kind == "sheet" else 1


def select_tiles(x: np.ndarray, kind: str) -> np.ndarray:
    """(N, 8, H, W, 3) tile cache -> sheet: all 8 tiles; single: the last tile only."""
    if kind == "sheet":
        return x
    if kind == "single":
        return x[:, 7:8]
    raise ValueError(kind)


def to_tensor(x: np.ndarray) -> torch.Tensor:
    """uint8 (B, T, H, W, 3) -> float (B, T, 3, H, W) in [0, 1]."""
    return torch.from_numpy(np.ascontiguousarray(x)).permute(0, 1, 4, 2, 3).float().div_(255.0)


def lr_factor(epoch: int, cfg: TrainConfig) -> float:
    if epoch < cfg.warmup_epochs:
        return (epoch + 1) / (cfg.warmup_epochs + 1)
    p = (epoch - cfg.warmup_epochs) / max(1, cfg.epochs - cfg.warmup_epochs)
    return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * p))


def augment(x: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    """Mild brightness and contrast change, the same for all tiles of a clip. No flips, no noise, no blur."""
    b = x.shape[0]
    c = torch.from_numpy(rng.uniform(0.9, 1.1, size=(b, 1, 1, 1, 1))).float()
    o = torch.from_numpy(rng.uniform(-0.05, 0.05, size=(b, 1, 1, 1, 1))).float()
    return ((x - 0.5) * c + 0.5 + o).clamp_(0, 1)


def _atomic_save(obj, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _atomic_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


@torch.no_grad()
def predict_logits(model: nn.Module, x: np.ndarray, kind: str, batch: int = 64) -> np.ndarray:
    model.eval()
    xs = select_tiles(x, kind)
    out = [model(to_tensor(xs[i:i + batch])).numpy() for i in range(0, len(xs), batch)]
    return np.concatenate(out)


def evaluate(model: nn.Module, x: np.ndarray, y: np.ndarray, kind: str) -> tuple[float, float]:
    lg = predict_logits(model, x, kind)
    loss = float(nn.functional.cross_entropy(torch.from_numpy(lg), torch.from_numpy(y)))
    return loss, float((lg.argmax(1) == y).mean())


def train(cfg: TrainConfig, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, run_dir: Path,
          log=print, pause: float = 0.0, stop_after_epoch: int | None = None) -> dict:
    """Train (or resume) and return the summary. `stop_after_epoch` exists for tests (simulates an interruption)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    done = run_dir / "done.json"
    if done.exists():
        info = json.loads(done.read_text(encoding="utf-8"))
        if info["config_key"] != cfg.key():
            raise RuntimeError(f"{run_dir} holds a finished run with a different configuration; use another folder")
        log(f"[{cfg.name}/{cfg.kind}] already finished (best epoch {info['best_epoch']}, val acc {info['best_val_acc']:.3f})")
        return info

    torch.manual_seed(cfg.seed)
    model = TileCNN(cfg.n_tiles)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    crit = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    xt = select_tiles(x_train, cfg.kind)
    n = len(xt)
    history: list[dict] = []
    best = dict(val_acc=-1.0, val_loss=1e9, epoch=-1)
    start = 0
    last = run_dir / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=False)
        if ck["config_key"] != cfg.key():
            raise RuntimeError(f"{last} was made with a different configuration; delete {run_dir} to start again")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        torch.set_rng_state(ck["torch_rng"])
        history, best, start = ck["history"], ck["best"], ck["epoch"] + 1
        log(f"[{cfg.name}/{cfg.kind}] resuming from epoch {start}/{cfg.epochs}")
    else:
        log(f"[{cfg.name}/{cfg.kind}] starting: {n_params(model):,} parameters, {n} training clips, {cfg.epochs} epochs")

    t_run = time.time()
    for epoch in range(start, cfg.epochs):
        t0 = time.time()
        for g in opt.param_groups:
            g["lr"] = cfg.lr * lr_factor(epoch, cfg)
        order = np.random.default_rng([cfg.seed, epoch]).permutation(n)
        aug_rng = np.random.default_rng([cfg.seed, epoch, 1])
        model.train()
        tot_loss, tot_ok, seen = 0.0, 0, 0
        for i in range(0, n - cfg.batch + 1 if n >= cfg.batch else 1, cfg.batch):  # drop the last partial batch (BatchNorm)
            idx = np.sort(order[i:i + cfg.batch])
            xb = augment(to_tensor(xt[idx]), aug_rng)
            yb = torch.from_numpy(y_train[idx])
            opt.zero_grad(set_to_none=True)
            lg = model(xb)
            loss = crit(lg, yb)
            loss.backward()
            opt.step()
            tot_loss += float(loss.detach()) * len(idx)
            tot_ok += int((lg.argmax(1) == yb).sum())
            seen += len(idx)
        v_loss, v_acc = evaluate(model, x_val, y_val, cfg.kind)
        row = dict(epoch=epoch, lr=opt.param_groups[0]["lr"], train_loss=tot_loss / seen, train_acc=tot_ok / seen,
                   val_loss=v_loss, val_acc=v_acc, seconds=time.time() - t0)
        history.append(row)
        improved = (v_acc > best["val_acc"]) or (v_acc == best["val_acc"] and v_loss < best["val_loss"])
        if improved:
            best = dict(val_acc=v_acc, val_loss=v_loss, epoch=epoch)
            _atomic_save(dict(model=model.state_dict(), epoch=epoch, config_key=cfg.key()), run_dir / "best.pt")
        _atomic_save(dict(model=model.state_dict(), opt=opt.state_dict(), torch_rng=torch.get_rng_state(), history=history,
                          best=best, epoch=epoch, config_key=cfg.key()), last)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        left = (cfg.epochs - epoch - 1) * (time.time() - t_run) / (epoch - start + 1)
        log(f"[{cfg.name}/{cfg.kind}] epoch {epoch + 1}/{cfg.epochs}  train {row['train_loss']:.3f}/{row['train_acc']:.3f}  "
            f"val {v_loss:.3f}/{v_acc:.3f}{'  *best*' if improved else ''}  {row['seconds']:.0f}s  about {left / 60:.0f} min left")
        if stop_after_epoch is not None and epoch + 1 >= stop_after_epoch:
            return dict(stopped=True, epoch=epoch)
        if pause > 0:
            time.sleep(pause)

    info = dict(config_key=cfg.key(), config=dataclasses.asdict(cfg), best_epoch=best["epoch"], best_val_acc=best["val_acc"],
                best_val_loss=best["val_loss"], parameters=n_params(model), train_seconds=sum(h["seconds"] for h in history),
                torch=torch.__version__, threads=torch.get_num_threads())
    _atomic_text(done, json.dumps(info, indent=2))
    return info


def load_best(cfg: TrainConfig, run_dir: Path) -> nn.Module:
    model = TileCNN(cfg.n_tiles)
    ck = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    if ck["config_key"] != cfg.key():
        raise RuntimeError("best.pt was made with a different configuration")
    model.load_state_dict(ck["model"])
    return model.eval()


def predictions_table(model: nn.Module, cfg: TrainConfig, manifest: pd.DataFrame, caches: dict, log=print) -> pd.DataFrame:
    """caches: {(condition, split): tile array}. Returns the Phase 4 predictions table for every cache given."""
    parts = []
    for (cond, split), x in caches.items():
        ids = manifest.loc[manifest["split"] == split, "clip_id"].tolist()
        pred = predict_logits(model, x, cfg.kind).argmax(1)
        parts.append(pd.DataFrame(dict(model=cfg.name, input=cfg.kind, condition=cond, split=split, clip_id=ids,
                                       pred=[CLASSES[k] for k in pred])))
    return pd.concat(parts, ignore_index=True)
