"""Scene configurations.

A1-A3 are training scenes. D1 is the held-out configuration: dim, warm lighting and a
different belt texture (diagonal stripes). Item palette, geometry and dynamics are the same
as A1, so the shift is appearance only.
"""

from __future__ import annotations

from dataclasses import dataclass

RGB = tuple[int, int, int]


@dataclass(frozen=True)
class Scene:
    scene_id: str
    role: str  # "train" or "ood"
    floor: RGB
    belt: RGB
    stripe: RGB
    palette: tuple[RGB, ...]
    belt_speed: float  # px per second
    belt_texture: str  # "vertical" or "diagonal"
    lighting: float  # global brightness multiplier
    tint: tuple[float, float, float]  # per-channel multiplier (RGB)
    texture_seed: int


_A1_PALETTE = ((200, 70, 60), (60, 120, 200), (230, 180, 50), (80, 170, 100))

SCENES: dict[str, Scene] = {
    "A1": Scene("A1", "train", (172, 172, 166), (62, 62, 68), (84, 84, 92), _A1_PALETTE, 120.0, "vertical", 1.0, (1.0, 1.0, 1.0), 101),
    "A2": Scene("A2", "train", (150, 158, 150), (70, 66, 60), (94, 90, 82), ((210, 100, 40), (120, 60, 170), (60, 160, 170), (230, 230, 230)), 120.0, "vertical", 1.0, (1.0, 1.0, 1.0), 102),
    "A3": Scene("A3", "train", (182, 168, 152), (55, 60, 70), (76, 84, 98), ((190, 50, 90), (50, 100, 60), (240, 200, 80), (100, 140, 210)), 120.0, "vertical", 1.0, (1.0, 1.0, 1.0), 103),
    "D1": Scene("D1", "ood", (172, 172, 166), (62, 62, 68), (84, 84, 92), _A1_PALETTE, 120.0, "diagonal", 0.6, (1.12, 1.0, 0.82), 104),
}

TRAIN_SCENES = tuple(s for s, v in SCENES.items() if v.role == "train")
OOD_SCENES = tuple(s for s, v in SCENES.items() if v.role == "ood")
