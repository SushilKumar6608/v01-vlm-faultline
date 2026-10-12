"""Frame rendering. Images are RGB uint8 arrays of shape (H, W, 3).

render_raw() draws the scene before lighting and sensor noise. It can draw the world with or
without the fault, which gives a counterfactual pair; the pixels that differ are the fault region.
finalize() applies lighting and sensor noise to the actual frame only.
"""

from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np

from . import config as C
from .plan import Plan
from .scenes import Scene

STRIPE_PERIOD, STRIPE_WIDTH = 48, 14


@lru_cache(maxsize=16)
def _background(scene_id: str) -> np.ndarray:
    from .scenes import SCENES

    sc = SCENES[scene_id]
    rng = np.random.default_rng([sc.texture_seed, 7])
    coarse = rng.uniform(-7, 7, size=(6, 8)).astype(np.float32)
    tex = cv2.resize(coarse, (C.W, C.H), interpolation=cv2.INTER_CUBIC)
    img = np.empty((C.H, C.W, 3), np.float32)
    for c in range(3):
        img[..., c] = sc.floor[c] + tex
    img = np.clip(img, 0, 255).astype(np.uint8)
    # belt body and edges
    cv2.rectangle(img, (0, C.BELT_TOP), (C.W, C.BELT_BOTTOM), sc.belt, -1)
    edge = tuple(int(0.55 * c) for c in sc.belt)
    cv2.line(img, (0, C.BELT_TOP), (C.W, C.BELT_TOP), edge, 3)
    cv2.line(img, (0, C.BELT_BOTTOM), (C.W, C.BELT_BOTTOM), edge, 3)
    # pipe along the upper edge with flanges (leaks start at a flange)
    cv2.rectangle(img, (0, C.PIPE_Y0), (C.W, C.PIPE_Y1), (96, 106, 118), -1)
    cv2.line(img, (0, C.PIPE_Y0), (C.W, C.PIPE_Y0), (70, 78, 88), 2)
    cv2.line(img, (0, C.PIPE_Y1), (C.W, C.PIPE_Y1), (70, 78, 88), 2)
    for fx in C.FLANGE_X:
        cv2.rectangle(img, (fx - 10, C.PIPE_Y0 - 5), (fx + 10, C.PIPE_Y1 + 5), (70, 78, 88), -1)
    img.setflags(write=False)
    return img


@lru_cache(maxsize=1)
def _belt_grid() -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.mgrid[C.BELT_TOP + 2 : C.BELT_BOTTOM - 1, 0 : C.W]
    return xs.astype(np.float32), ys.astype(np.float32)


def _draw_belt_stripes(img: np.ndarray, scene: Scene, v: float, t: float) -> None:
    xs, ys = _belt_grid()
    phase = (xs - v * t) if scene.belt_texture == "vertical" else (xs + ys - v * t)
    mask = np.mod(phase, STRIPE_PERIOD) < STRIPE_WIDTH
    region = img[C.BELT_TOP + 2 : C.BELT_BOTTOM - 1]
    region[mask] = scene.stripe


def _draw_items(img: np.ndarray, plan: Plan, pos_col: np.ndarray) -> None:
    for i, it in enumerate(plan.items):
        x = float(pos_col[i])
        if x < -40 or x > C.W + 40:
            continue
        cy = C.LANES_Y[it.lane]
        dark = tuple(int(0.6 * c) for c in it.color)
        if it.shape == "box":
            p0 = (int(round(x - it.size / 2)), int(round(cy - it.h / 2)))
            p1 = (int(round(x + it.size / 2)), int(round(cy + it.h / 2)))
            cv2.rectangle(img, p0, p1, it.color, -1)
            cv2.rectangle(img, p0, p1, dark, 2)
        else:
            c = (int(round(x)), int(cy))
            cv2.circle(img, c, it.size // 2, it.color, -1, cv2.LINE_AA)
            cv2.circle(img, c, it.size // 2, dark, 2, cv2.LINE_AA)


def _draw_gate(img: np.ndarray, open_frac: float) -> None:
    px, py = C.GATE_PIVOT
    theta = (np.pi / 2) * (1.0 - float(open_frac))  # 90 deg across the belt (closed) .. 0 deg along it (open)
    d = np.array([np.cos(theta), np.sin(theta)])
    n = np.array([-d[1], d[0]])
    p = np.array([px, py], dtype=np.float64)
    half = C.GATE_THICK / 2
    pts = np.array([p + n * half, p + d * C.GATE_LEN + n * half, p + d * C.GATE_LEN - n * half, p - n * half])
    cv2.fillConvexPoly(img, np.round(pts).astype(np.int32), C.GATE_COLOR, cv2.LINE_AA)
    cv2.polylines(img, [np.round(pts).astype(np.int32)], True, (70, 15, 70), 1, cv2.LINE_AA)
    cv2.rectangle(img, (px - 12, py - 12), (px + 12, py + 12), (50, 52, 58), -1)


def _blend(img: np.ndarray, overlay: np.ndarray, alpha: float) -> np.ndarray:
    return cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0.0)


def _draw_spray(img: np.ndarray, plan: Plan, k: int) -> np.ndarray:
    sp = plan.spray
    t_since = (k - plan.onset_frame) / C.FPS
    if t_since < 0:
        return img
    xj, y0 = sp["xj"], float(C.PIPE_Y1)
    dvec = np.array([np.sin(sp["angle"]), np.cos(sp["angle"])])
    end = np.array([xj, y0]) + dvec * sp["reach"]
    # wet patch on the belt, growing for 1.5 s after onset
    g = min(1.0, t_since / 1.5)
    rx, ry = (10 + 40 * sp["severity"]) * g, (6 + 22 * sp["severity"]) * g
    if rx >= 1 and ry >= 1:
        ov = img.copy()
        cv2.ellipse(ov, (int(end[0]), int(end[1])), (int(rx), int(ry)), 0, 0, 360, (96, 140, 180), -1, cv2.LINE_AA)
        img = _blend(img, ov, 0.55)
    # tapered stream
    nrm = np.array([-dvec[1], dvec[0]])
    w0 = sp["w0"]
    p0 = np.array([xj, y0])
    poly = np.array([p0 + nrm * w0 / 2, end + nrm * w0 * 0.2, end - nrm * w0 * 0.2, p0 - nrm * w0 / 2])
    ov = img.copy()
    cv2.fillConvexPoly(ov, np.round(poly).astype(np.int32), (150, 214, 244), cv2.LINE_AA)
    img = _blend(img, ov, 0.8)
    # droplets, re-drawn every frame
    rd = np.random.default_rng([plan.seed, 100 + k])
    dist = rd.uniform(0, sp["reach"], sp["n_drops"])
    lat = rd.normal(0, 1.5 + 0.08 * dist)
    rad = np.maximum(1, np.round(sp["drop_r"] * rd.uniform(0.5, 1.0, sp["n_drops"]))).astype(int)
    for dd, ll, rr in zip(dist, lat, rad):
        c = p0 + dvec * dd + nrm * ll
        cv2.circle(img, (int(round(c[0])), int(round(c[1]))), int(rr), (196, 234, 252), -1, cv2.LINE_AA)
    return img


def _apply_lens(img: np.ndarray, plan: Plan) -> np.ndarray:
    """Semi-transparent smudge: soft edge, textured opacity, defocused background underneath."""
    ln = plan.lens
    mask = np.zeros((C.H, C.W), np.float32)
    cv2.ellipse(mask, (int(ln["cx"]), int(ln["cy"])), (int(ln["ax"]), int(ln["ay"])), ln["angle"], 0, 360, 1.0, -1)
    sigma = max(4.0, 0.18 * min(ln["ax"], ln["ay"]))
    mask = cv2.GaussianBlur(mask, (0, 0), sigma)
    rng = np.random.default_rng([plan.seed, 2, 77])  # fault stream: same smudge texture in every frame
    tex = cv2.resize(rng.uniform(0.0, 1.0, (9, 16)).astype(np.float32), (C.W, C.H), interpolation=cv2.INTER_CUBIC)
    alpha = np.clip(mask * (0.65 + 0.30 * tex), 0.0, 1.0)
    blurred = cv2.GaussianBlur(img, (0, 0), 7.0).astype(np.float32)
    smudge = np.array((70, 62, 52), np.float32) * (0.7 + 0.5 * tex[..., None])
    inner = 0.40 * blurred + 0.60 * smudge
    out = img.astype(np.float32) * (1 - alpha[..., None]) + inner * alpha[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def render_raw(plan: Plan, k: int, faulty: bool) -> np.ndarray:
    """Scene at frame k before lighting and noise. faulty=False draws the counterfactual world."""
    scene = plan.scene
    pos = plan.pos_actual if faulty else plan.pos_cf
    gate = plan.gate_actual if faulty else plan.gate_cf
    img = np.array(_background(scene.scene_id))  # writable copy
    _draw_belt_stripes(img, scene, plan.belt_speed, k / C.FPS)
    _draw_items(img, plan, pos[:, k])
    _draw_gate(img, gate[k])
    if faulty and plan.spray is not None:
        img = _draw_spray(img, plan, k)
    if faulty and plan.lens is not None:
        img = _apply_lens(img, plan)
    return img


def finalize(raw: np.ndarray, plan: Plan, k: int) -> np.ndarray:
    """Lighting and sensor noise. Deterministic per (seed, frame)."""
    sc = plan.scene
    gain = sc.lighting * plan.brightness
    out = raw.astype(np.float32) * np.array(sc.tint, np.float32) * gain
    rng = np.random.default_rng([plan.seed, 1000 + k])
    out += rng.standard_normal(out.shape, dtype=np.float32) * C.NOISE_SIGMA
    return np.clip(out, 0, 255).astype(np.uint8)
