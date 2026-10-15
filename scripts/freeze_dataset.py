"""Freeze dataset v1: record hashes, counts and tool versions in configs/dataset_v1.json."""
import csv, hashlib, json, platform, subprocess, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(*cmd) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT).stdout.strip()
    except Exception as e:
        return f"unavailable ({e})"


def rows(p: Path):
    with open(p, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


manifest = ROOT / "data" / "manifest.csv"
specs = ROOT / "configs" / "specs_v1.csv"
repl = ROOT / "configs" / "replacements_v1.csv"
spot = ROOT / "data" / "spotcheck" / "spotcheck.csv"

m = rows(manifest)
counts = Counter((r.get("split", "?"), r.get("cls", "?")) for r in m)
by_split = Counter(r.get("split", "?") for r in m)

sp = rows(spot) if spot.exists() else []
labelable = Counter(r["labelable"].strip().lower() for r in sp)

import numpy, cv2  # noqa: E402

out = {
    "dataset_version": "v1",
    "n_clips": len(m),
    "clips_per_split": dict(by_split),
    "clips_per_split_class": {f"{s}/{c}": n for (s, c), n in sorted(counts.items())},
    "manifest_sha256": sha256(manifest),
    "specs_sha256": sha256(specs),
    "replacements_sha256": sha256(repl) if repl.exists() else None,
    "n_replacements": len(rows(repl)) if repl.exists() else 0,
    "spotcheck": {"n_checked": len(sp), "labelable": dict(labelable), "rater": "author, single rater"},
    "versions": {
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "opencv": cv2.__version__,
        "ffmpeg": (run("ffmpeg", "-version").splitlines() or ["unavailable"])[0],
    },
    "git_commit": run("git", "rev-parse", "HEAD"),
    "git_dirty": bool(run("git", "status", "--porcelain")),
    "note": "MP4 bytes are identical only on the same ffmpeg build; frame content and ground-truth columns regenerate exactly from seeds.",
}
dest = ROOT / "configs" / "dataset_v1.json"
dest.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
print(json.dumps(out, indent=2))
if len(m) != 2450:
    sys.exit("WARNING: expected 2450 manifest rows")
