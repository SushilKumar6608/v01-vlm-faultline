"""Cross-machine determinism check. Prints versions and hashes of the frames of six fixed clips.
Run it on two machines and compare the output lines starting with HASH.   python scripts/frame_hash.py
"""
from __future__ import annotations

import hashlib
import platform

import cv2
import numpy as np

from faultline.sample.sheet import build_sheet
from faultline.sim.clip import generate_clip

CASES = [("normal", "A1", None), ("stuck_item", "A2", 0.5), ("gate_open", "A3", 0.5), ("spray", "A1", 0.5), ("blocked_lens", "A2", 0.5), ("spray", "D1", 0.7)]


def h(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def main() -> None:
    print(f"python {platform.python_version()}  numpy {np.__version__}  opencv {cv2.__version__}  {platform.system()}")
    for cls, scene, sev in CASES:
        c = generate_clip(5, cls, scene, sev)
        print(f"HASH {cls:13s} {scene} frames={h(c.frames)} sheet={h(build_sheet(c.frames))} masks={h(c.sample_masks)} region_px={c.meta['fault_region_px']}")


if __name__ == "__main__":
    main()
