"""Constants that define the simulated world, the sampling plan and the fault parameters.

Everything here is part of the scene definition. Changing a value changes the dataset,
so bump SIM_SCHEMA_VERSION when you do.
"""

from __future__ import annotations

SIM_SCHEMA_VERSION = "1.1"

# --- video geometry and timing -------------------------------------------------
W, H = 640, 360
FPS = 16
DURATION_S = 6.0
N_FRAMES = int(round(DURATION_S * FPS))  # 96

# --- sampling for the vision models (8 frames, fixed spacing) -----------------
SAMPLE_COUNT = 8
SAMPLE_STEP_FRAMES = 12  # 0.75 s at 16 fps, an exact number of frames
SAMPLE_STEP_S = SAMPLE_STEP_FRAMES / FPS
TILE_W, TILE_H = 320, 180
SHEET_COLS, SHEET_ROWS = 4, 2  # reading order: time left to right, top row first

# --- classes -------------------------------------------------------------------
CLASSES = ("normal", "stuck_item", "gate_open", "spray", "blocked_lens")
FAULT_CLASSES = CLASSES[1:]
TEMPORAL_FAULTS = ("stuck_item",)
SCHEDULE_FAULTS = ("gate_open",)  # state versus a fixed schedule
STATIC_FAULTS = ("spray", "blocked_lens")

# --- severity and onset --------------------------------------------------------
SEVERITY_MIN, SEVERITY_MAX = 0.1, 1.0
SEVERITY_BINS = ((0.1, 0.4, "low"), (0.4, 0.7, "mid"), (0.7, 1.0001, "high"))
ONSET_MIN_S, ONSET_MAX_S = 1.0, 3.5  # cap 3.5 s: at least 3 sampled frames after onset

# --- belt, pipe, gate geometry (pixels) ---------------------------------------
BELT_TOP, BELT_BOTTOM = 130, 250
LANES_Y = (155, 190, 225)
ITEM_SIZE_RANGE = (22, 32)  # lane spacing is 35, so items in different lanes never touch
ITEM_GAP_S = (0.4, 1.2)  # time between consecutive items entering
QUEUE_GAP_PX = 6
PIPE_Y0, PIPE_Y1 = 40, 62
FLANGE_X = (160, 320, 480)

# gate: deterministic schedule, identical in every clip and every split
GATE_PIVOT = (512, 112)
GATE_LEN = 140
GATE_THICK = 14
GATE_COLOR = (225, 40, 225)  # magenta: outside every item palette (tests enforce the distance)
GATE_WINDOWS = ((1.0, 2.0), (4.0, 5.0))  # seconds during which the gate is open
GATE_RAMP_S = 0.25
GATE_ONSET_S = 2.0  # the first scheduled close fails
GATE_SCHEDULE_ID = "fixed_v1"

# --- fault specifics -----------------------------------------------------------
STUCK_MIN_STOP_S = 2.25  # at least three sampled frames show the halted item (two consecutive pairs)
STUCK_HALT_ZONE = (0.25, 0.60)  # fraction of frame width; gate sits at x = 0.8 W
SPRAY_WIDTH_PX = (6.0, 16.0)  # stream width at the joint, severity 0.1 -> 1.0 (mapped linearly from 0)
LENS_FRACTION = (0.02, 0.30)  # occluded fraction of the frame

# --- visibility accounting -----------------------------------------------------
MASK_DIFF_THRESHOLD = 10  # grey levels, on the noise-free counterfactual difference
MIN_VISIBLE_PX = 40.0  # fault area at tile scale for a tile to count as showing the fault (provisional)
TILE_AREA_SCALE = (TILE_W / W) * (TILE_H / H)

# --- camera --------------------------------------------------------------------
NOISE_SIGMA = 2.0


def severity_bin(severity: float | None) -> str | None:
    if severity is None:
        return None
    for lo, hi, name in SEVERITY_BINS:
        if lo <= severity < hi:
            return name
    return "high"


def sample_frame_indices() -> list[int]:
    return [i * SAMPLE_STEP_FRAMES for i in range(SAMPLE_COUNT)]


def sample_times() -> list[float]:
    return [i * SAMPLE_STEP_S for i in range(SAMPLE_COUNT)]
