"""Make the upload folder for the zero-shot run on Kaggle (images the model will see, an index, the prompts, the runner).

  python scripts/export_vlm_inputs.py --workers 4

Writes data/kaggle_bundle/ (not tracked):
  images/<condition>/<single|sheet>/<split>/<clip_id>.png    lossless, same pipeline as the Phase 3 audit sheets
  index.csv       clip_id, split, condition, input, path, sha256
  prompts_v1.yaml, prompts_v2.yaml, run_zero_shot.py          copies, so one upload holds everything
Default: splits val, test and held-out, clean and degraded: 950 clips x 2 conditions x 2 inputs = 3,800 images (zero-shot needs no
training clips). For fine-tuning export the training clips into their own folder:

  python scripts/export_vlm_inputs.py --splits train --conditions clean --out data/kaggle_train --workers 4

1,500 clips x 1 condition x 2 inputs = 3,000 images. Resumable: existing images are kept. Upload the folder as one private Kaggle
dataset (see kaggle/README.md).
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
from multiprocessing import Pool
from pathlib import Path

import cv2
import pandas as pd

from faultline import paths
from faultline.data import manifest as mf
from faultline.eval.inputs import clip_file, make_input
from faultline.media import read_frames

SPLITS = ("val", "test", "heldout")
CONDITIONS = ("clean", "degraded")


def img_path(root: Path, cond: str, kind: str, split: str, clip_id: str) -> Path:
    return root / "images" / cond / kind / split / f"{clip_id}.png"


def _save(arr, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.png")
    if not cv2.imwrite(str(tmp), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 3]):
        raise RuntimeError(f"could not write {path}")
    os.replace(tmp, path)


def work(args: tuple) -> str:
    root, split, clip_id, cond = args
    root = Path(root)
    need = [k for k in ("single", "sheet") if not img_path(root, cond, k, split, clip_id).exists()]
    if need:
        frames = read_frames(clip_file(split, clip_id, cond))
        for k in need:
            _save(make_input(frames, k), img_path(root, cond, k, split, clip_id))
    return clip_id


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=str(paths.data_root() / "kaggle_bundle"))
    ap.add_argument("--limit", type=int, default=None, help="first N clips per split (testing)")
    ap.add_argument("--splits", nargs="+", default=list(SPLITS), choices=["train", "val", "test", "heldout"])
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=list(CONDITIONS))
    a = ap.parse_args()
    root = Path(a.out)
    m = mf.read_manifest()
    jobs = []
    for s in a.splits:
        ids = m.loc[m.split == s, "clip_id"].tolist()[: a.limit]
        jobs += [(str(root), s, c, cond) for c in ids for cond in a.conditions]
    print(f"{len(jobs)} clip decodes -> {2 * len(jobs)} images in {root}")
    from tqdm import tqdm

    if a.workers <= 1:
        list(tqdm(map(work, jobs), total=len(jobs), unit="clip"))
    else:
        with Pool(a.workers) as p:
            list(tqdm(p.imap_unordered(work, jobs, chunksize=4), total=len(jobs), unit="clip"))
    rows = []
    for (_, s, c, cond) in jobs:
        for k in ("single", "sheet"):
            p = img_path(root, cond, k, s, c)
            rows.append(dict(clip_id=c, split=s, condition=cond, input=k, path=p.relative_to(root).as_posix(), sha256=sha256(p)))
    pd.DataFrame(rows).to_csv(root / "index.csv", index=False)
    repo = paths.repo_root()
    for pf in ("prompts_v1.yaml", "prompts_v2.yaml"):
        shutil.copy2(repo / "configs" / pf, root / pf)
    for kf in ("run_zero_shot.py", "finetune_qlora.py"):
        if (repo / "kaggle" / kf).exists():
            shutil.copy2(repo / "kaggle" / kf, root / kf)
    mb = sum(f.stat().st_size for f in root.rglob("*.png")) / 1e6
    print(f"wrote {len(rows)} index rows, {mb:.0f} MB of images; upload the folder {root}")


if __name__ == "__main__":
    main()
