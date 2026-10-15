"""Generate one clip with its ground-truth metadata."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config as C
from .plan import Plan, make_plan
from .render import finalize, render_raw


@dataclass
class Clip:
    frames: np.ndarray  # (N_FRAMES, H, W, 3) uint8 RGB
    meta: dict
    sample_masks: np.ndarray  # (SAMPLE_COUNT, H, W) bool: fault region in the sampled frames (native resolution)


def _mask(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(a.astype(np.int16) - b.astype(np.int16)).max(axis=2) > C.MASK_DIFF_THRESHOLD


def generate_clip(seed: int, cls: str, scene_id: str, severity: float | None = None, onset_s: float | None = None) -> Clip:
    plan: Plan = make_plan(seed, cls, scene_id, severity, onset_s)
    idx = C.sample_frame_indices()
    frames = np.empty((C.N_FRAMES, C.H, C.W, 3), np.uint8)
    px = np.zeros(C.N_FRAMES)
    sample_masks = np.zeros((C.SAMPLE_COUNT, C.H, C.W), bool)

    for k in range(C.N_FRAMES):
        raw = render_raw(plan, k, faulty=True)
        frames[k] = finalize(raw, plan, k)
        if cls != "normal":
            m = _mask(raw, render_raw(plan, k, faulty=False))
            px[k] = m.sum()
            if k in idx:
                sample_masks[idx.index(k)] = m

    vis = [float(px[k] * C.TILE_AREA_SCALE) for k in idx]
    onset_frame = plan.onset_frame
    post_onset = None if onset_frame is None else int(sum(k >= onset_frame for k in idx))
    halted_tiles = frozen_tiles = gap_px = None
    if cls == "normal":
        rule, visible = "none", 0
    elif cls == "stuck_item":
        # the mask also covers displaced downstream items, so count tiles where the stuck item is actually halted
        k0, k1 = plan.extra["hold_frames"]
        halted = [k0 <= k < k1 for k in idx]
        halted_tiles = int(sum(halted))
        frozen_tiles = int(sum(a and b for a, b in zip(halted, halted[1:])))  # consecutive halted pairs
        gap_px = plan.extra.get("neighbor_gap_at_onset_px")
        rule, visible = "halted_tiles", halted_tiles
    else:
        rule = {"gate_open": "mask_gate_differs_from_schedule", "spray": "mask_area_ge_min_px", "blocked_lens": "mask_area_ge_min_px"}[cls]
        visible = int(sum(v >= C.MIN_VISIBLE_PX for v in vis))
    meta = dict(
        clip_id=f"{cls}_{scene_id}_{seed:06d}",
        seed=seed,
        scene_config_id=scene_id,
        cls=cls,
        severity=None if plan.severity is None else round(plan.severity, 4),
        severity_bin=C.severity_bin(plan.severity),
        severity_physical=None if plan.severity_physical is None else round(float(plan.severity_physical), 4),
        severity_unit=plan.severity_unit,
        onset_s=None if plan.onset_frame is None else plan.onset_frame / C.FPS,
        halt_x=None if plan.halt_x is None else round(plan.halt_x, 4),
        fault_region_px=round(float(px.max() * C.TILE_AREA_SCALE), 1),
        visible_px_samples=[round(v, 1) for v in vis],
        expected_visible_tiles=visible,
        visibility_rule=rule,
        post_onset_tiles=post_onset,
        halted_tiles=halted_tiles,
        frozen_tiles=frozen_tiles,
        neighbor_gap_at_onset_px=None if gap_px is None else round(gap_px, 2),
        gate_schedule_id=C.GATE_SCHEDULE_ID,
        belt_speed=round(plan.belt_speed, 3),
        brightness=round(plan.brightness, 4),
        fps=C.FPS,
        n_frames=C.N_FRAMES,
        sample_times=C.sample_times(),
        label_source="synthetic",
        sim_schema_version=C.SIM_SCHEMA_VERSION,
    )
    return Clip(frames, meta, sample_masks)
