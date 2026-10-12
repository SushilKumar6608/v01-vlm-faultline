"""Clip planning: items, motion (including the jam behind a stuck item), gate state, fault parameters.

Randomness is split into separate streams so that fault choices never change the world:
    stream 1: world (items, belt speed jitter, brightness)
    stream 2: fault parameters
    stream 100+k: spray droplets of frame k;  stream 1000+k: sensor noise of frame k
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import config as C
from .scenes import SCENES, Scene


@dataclass
class Item:
    x0: float  # centre x at frame 0
    lane: int
    size: int
    shape: str  # "box" or "disc"
    color: tuple[int, int, int]
    h: int = 0

    def __post_init__(self) -> None:
        if not self.h:
            self.h = self.size


@dataclass
class Plan:
    seed: int
    cls: str
    scene: Scene
    belt_speed: float
    brightness: float
    items: list[Item]
    pos_actual: np.ndarray  # (n_items, N_FRAMES) centre x
    pos_cf: np.ndarray  # counterfactual: same world without the fault
    gate_actual: np.ndarray  # (N_FRAMES,) open fraction 0..1
    gate_cf: np.ndarray
    severity: float | None = None
    severity_physical: float | None = None
    severity_unit: str | None = None
    onset_frame: int | None = None
    halt_x: float | None = None
    spray: dict | None = None
    lens: dict | None = None
    extra: dict = field(default_factory=dict)


# ------------------------------------------------------------------ gate schedule
def gate_open_fraction(t: float) -> float:
    """Open fraction of the gate at time t. Fixed schedule: no randomness, no seed."""
    r = C.GATE_RAMP_S
    for t0, t1 in C.GATE_WINDOWS:
        if t0 <= t < t0 + r:
            return (t - t0) / r
        if t0 + r <= t < t1:
            return 1.0
        if t1 <= t < t1 + r:
            return 1.0 - (t - t1) / r
    return 0.0


def gate_schedule_array() -> np.ndarray:
    return np.array([gate_open_fraction(k / C.FPS) for k in range(C.N_FRAMES)], dtype=np.float64)


# ------------------------------------------------------------------ items and motion
def _make_items(rng: np.random.Generator, scene: Scene, v: float) -> list[Item]:
    items: list[Item] = []
    cursor = C.W + rng.uniform(0.0, 60.0)
    lo = -v * C.DURATION_S - 60.0
    while cursor > lo:
        size = int(rng.integers(C.ITEM_SIZE_RANGE[0], C.ITEM_SIZE_RANGE[1] + 1))
        shape = "box" if rng.random() < 0.6 else "disc"
        color = scene.palette[int(rng.integers(0, len(scene.palette)))]
        h = int(round(size * rng.uniform(0.7, 1.0))) if shape == "box" else size
        items.append(Item(x0=float(cursor), lane=int(rng.integers(0, len(C.LANES_Y))), size=size, shape=shape, color=color, h=h))
        cursor -= rng.uniform(*C.ITEM_GAP_S) * v
    return items


def compute_positions(items: list[Item], v: float, hold: tuple[int, int, int, float] | None = None) -> np.ndarray:
    """Item centre x per frame. hold = (item_index, k_start, k_end, x): that item stays at x for
    frames k_start..k_end-1. Items behind it in the same lane queue up (a jam); other lanes are unaffected."""
    n = len(items)
    pos = np.zeros((n, C.N_FRAMES))
    dx = v / C.FPS
    for lane in range(len(C.LANES_Y)):
        idx = sorted((i for i in range(n) if items[i].lane == lane), key=lambda i: -items[i].x0)
        for rank, i in enumerate(idx):
            x = items[i].x0
            for k in range(C.N_FRAMES):
                if k > 0:
                    x += dx
                if hold is not None and i == hold[0] and hold[1] <= k < hold[2]:
                    x = hold[3]
                if rank > 0:
                    f = idx[rank - 1]
                    limit = pos[f, k] - (items[f].size / 2 + items[i].size / 2 + C.QUEUE_GAP_PX)
                    if x > limit:
                        x = limit
                pos[i, k] = x
    return pos


# ------------------------------------------------------------------ planning
def make_plan(seed: int, cls: str, scene_id: str, severity: float | None = None, onset_s: float | None = None) -> Plan:
    if cls not in C.CLASSES:
        raise ValueError(f"unknown class {cls!r}")
    scene = SCENES[scene_id]

    rw = np.random.default_rng([seed, 1])
    v = scene.belt_speed * rw.uniform(0.95, 1.05)
    brightness = rw.uniform(0.9, 1.1)
    items = _make_items(rw, scene, v)

    pos_cf = compute_positions(items, v)
    gate_cf = gate_schedule_array()
    plan = Plan(seed, cls, scene, v, brightness, items, pos_cf, pos_cf, gate_cf, gate_cf)
    if cls == "normal":
        return plan

    rf = np.random.default_rng([seed, 2])
    sev_draw = rf.uniform(C.SEVERITY_MIN, C.SEVERITY_MAX)
    onset_draw = rf.uniform(C.ONSET_MIN_S, C.ONSET_MAX_S)
    s = float(sev_draw if severity is None else severity)
    s = min(max(s, C.SEVERITY_MIN), C.SEVERITY_MAX)
    plan.severity = s

    if cls == "gate_open":
        k_on = int(round(C.GATE_ONSET_S * C.FPS))
        plan.onset_frame = k_on
        ramp = int(round(C.GATE_RAMP_S * C.FPS))
        act = gate_cf.copy()
        for k in range(k_on, C.N_FRAMES):
            act[k] = max(s, gate_cf[k]) if k <= k_on + ramp else s
        plan.gate_actual = act
        plan.severity_physical, plan.severity_unit = s, "open_fraction_at_rest"
        return plan

    t_on = float(onset_draw if onset_s is None else onset_s)
    t_on = min(max(t_on, C.ONSET_MIN_S), C.ONSET_MAX_S) if cls != "blocked_lens" else 0.0
    k_on = int(round(t_on * C.FPS))
    plan.onset_frame = k_on

    if cls == "stuck_item":
        lo, hi = C.STUCK_HALT_ZONE[0] * C.W, C.STUCK_HALT_ZONE[1] * C.W
        cand = [i for i in range(len(items)) if lo <= pos_cf[i, k_on] <= hi]
        if not cand:  # cannot happen with the item gaps used, but fail loudly rather than mislabel
            raise RuntimeError("no item in the halt zone at onset")
        i_stuck = int(cand[int(rf.integers(0, len(cand)))])
        d_max = (C.N_FRAMES - k_on) / C.FPS
        d = C.STUCK_MIN_STOP_S + (d_max - C.STUCK_MIN_STOP_S) * (s - C.SEVERITY_MIN) / (C.SEVERITY_MAX - C.SEVERITY_MIN)
        k_end = min(C.N_FRAMES, k_on + int(round(d * C.FPS)))
        x_hold = float(pos_cf[i_stuck, k_on])
        plan.pos_actual = compute_positions(items, v, hold=(i_stuck, k_on, k_end, x_hold))
        plan.halt_x = x_hold / C.W
        plan.severity_physical, plan.severity_unit = (k_end - k_on) / C.FPS, "stop_duration_s"
        plan.extra["stuck_item_index"] = i_stuck
        plan.extra["hold_frames"] = (k_on, k_end)
    elif cls == "spray":
        xj = float(C.FLANGE_X[int(rf.integers(0, len(C.FLANGE_X)))] + rf.uniform(-8, 8))
        angle = float(rf.uniform(-0.35, 0.35))
        w0 = C.SPRAY_WIDTH_PX[0] + (C.SPRAY_WIDTH_PX[1] - C.SPRAY_WIDTH_PX[0]) * s
        plan.spray = dict(xj=xj, angle=angle, w0=w0, reach=70.0 + 80.0 * s, n_drops=int(25 + 120 * s), drop_r=1.5 + 2.5 * s, severity=s)
        plan.severity_physical, plan.severity_unit = w0, "stream_width_px"
    elif cls == "blocked_lens":
        frac = C.LENS_FRACTION[0] + (C.LENS_FRACTION[1] - C.LENS_FRACTION[0]) * s
        ratio = float(rf.uniform(0.6, 1.0))
        ax = float(np.sqrt(frac * C.W * C.H * ratio / np.pi))
        ay = ax / ratio
        cx = float(rf.uniform(ax, C.W - ax)) if C.W - ax > ax else C.W / 2
        cy = float(rf.uniform(ay, C.H - ay)) if C.H - ay > ay else C.H / 2
        plan.lens = dict(cx=cx, cy=cy, ax=ax, ay=ay, angle=float(rf.uniform(-30, 30)), fraction=frac)
        plan.severity_physical, plan.severity_unit = frac, "occluded_fraction_target"
    return plan
