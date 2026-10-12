"""PROVISIONAL degradation recipe, used only for the Phase 1 visibility preview.

It is deliberately harsh: v0 (480x270, noise 14, CRF 32) was only about 35 dB PSNR at tile scale because the sheet
downsizing averages noise away; v1 targets about 28 dB measured on the final sheet tiles. Faults are tuned to survive it; the final recipe is frozen in Phase 3
and the visibility audit is repeated with it. The noise filter's seed defaults to -1 in the ffmpeg
build checked, so unseeded noise is not reproducible: always pass a seed derived from the clip seed.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from ..media import REPRO_FLAGS, ffmpeg_exe

PROVISIONAL_RECIPE = dict(name="provisional_worst_case_v1", scale=(320, 180), blur_sigma=1.0, noise_strength=36, crf=38, preset="medium")


def noise_seed(clip_seed: int) -> int:
    return (clip_seed * 7919 + 13) % 2_147_483_647


def degrade_mp4(src: str | Path, dst: str | Path, clip_seed: int, recipe: dict = PROVISIONAL_RECIPE) -> None:
    sw, sh = recipe["scale"]
    blur = f",gblur=sigma={recipe['blur_sigma']}" if recipe.get("blur_sigma") else ""
    vf = f"scale={sw}:{sh}:flags=bicubic{blur},noise=alls={recipe['noise_strength']}:allf=t+u:all_seed={noise_seed(clip_seed)}"
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(src), "-vf", vf, "-c:v", "libx264", "-preset", recipe["preset"],
           "-crf", str(recipe["crf"]), "-pix_fmt", "yuv420p", *REPRO_FLAGS, str(dst)]
    subprocess.run(cmd, check=True)
