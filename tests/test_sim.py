import hashlib

import numpy as np
import pytest

from faultline.sim import config as C
from faultline.sim.clip import generate_clip
from faultline.sim.plan import gate_open_fraction


def test_determinism():
    a, b = generate_clip(5, "spray", "A1", 0.5), generate_clip(5, "spray", "A1", 0.5)
    assert np.array_equal(a.frames, b.frames) and a.meta == b.meta


def test_seeds_differ():
    assert not np.array_equal(generate_clip(1, "normal", "A1").frames, generate_clip(2, "normal", "A1").frames)


def test_normal_has_no_fault_area():
    m = generate_clip(3, "normal", "A2").meta
    assert m["fault_region_px"] == 0 and m["expected_visible_tiles"] == 0 and m["severity"] is None


@pytest.mark.parametrize("cls", ["stuck_item", "gate_open", "spray", "blocked_lens"])
def test_faults_have_area(cls):
    m = generate_clip(7, cls, "A1", 0.5).meta
    assert m["fault_region_px"] > 0 and m["expected_visible_tiles"] >= 3


@pytest.mark.parametrize("cls", ["spray", "blocked_lens"])
def test_area_grows_with_severity(cls):
    lo = generate_clip(9, cls, "A1", 0.15).meta["fault_region_px"]
    hi = generate_clip(9, cls, "A1", 0.95).meta["fault_region_px"]
    assert hi > lo


def test_gate_schedule_fixed():
    assert [round(gate_open_fraction(t)) for t in C.sample_times()] == [0, 0, 1, 0, 0, 0, 1, 0]
    for s in (1, 2):
        assert generate_clip(s, "gate_open", "A1", 0.6).meta["onset_s"] == 2.0


def test_stuck_halt_zone_and_onset():
    for s in range(20, 26):
        m = generate_clip(s, "stuck_item", "A1").meta
        assert C.STUCK_HALT_ZONE[0] <= m["halt_x"] <= C.STUCK_HALT_ZONE[1]
        assert C.ONSET_MIN_S <= m["onset_s"] <= C.ONSET_MAX_S
        assert m["expected_visible_tiles"] >= 3


def test_lens_onset_zero():
    assert generate_clip(4, "blocked_lens", "A1", 0.5).meta["onset_s"] == 0.0


def test_severity_bin():
    assert [C.severity_bin(x) for x in (0.1, 0.39, 0.5, 0.69, 0.9, 1.0)] == ["low", "low", "mid", "mid", "high", "high"]
    assert C.severity_bin(None) is None


def test_mp4_roundtrip_and_repro(tmp_path):
    from faultline.media import ffmpeg_exe, read_frames, write_mp4
    try:
        ffmpeg_exe()
    except RuntimeError:
        pytest.skip("no ffmpeg")
    f = generate_clip(6, "normal", "A1").frames
    write_mp4(f, tmp_path / "a.mp4"); write_mp4(f, tmp_path / "b.mp4")
    h = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    assert h(tmp_path / "a.mp4") == h(tmp_path / "b.mp4")
    assert read_frames(tmp_path / "a.mp4").shape == (C.N_FRAMES, C.H, C.W, 3)


def test_stuck_minimum_visible_tiles():
    for sev in (0.1, 0.2):
        for s in range(30, 40):
            m = generate_clip(s, "stuck_item", "A1", sev).meta
            assert m["severity_physical"] >= C.STUCK_MIN_STOP_S - 1e-9
            assert m["expected_visible_tiles"] >= 3


def test_gate_colour_distinct_from_item_palettes():
    from faultline.sim.scenes import SCENES
    g = np.array(C.GATE_COLOR, float)
    for sc in SCENES.values():
        for c in sc.palette:
            assert np.linalg.norm(g - np.array(c, float)) > 100, (sc.scene_id, c)


def test_lens_is_semi_transparent():
    c = generate_clip(4, "blocked_lens", "A1", 0.9)
    m = c.sample_masks[0]
    inside = c.frames[0][m].astype(float).mean()
    assert inside > 40  # not an opaque black blob: scene detail still shows through
