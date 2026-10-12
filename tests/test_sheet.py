import numpy as np

from faultline.sample.sheet import build_sheet, single_frame_tile, tiles
from faultline.sim import config as C


def _frames():
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (C.N_FRAMES, C.H, C.W, 3), dtype=np.uint8)


def test_sheet_shape():
    assert build_sheet(_frames()).shape == (360, 1280, 3)


def test_single_frame_is_last_tile():
    f = _frames()
    assert np.array_equal(single_frame_tile(f), tiles(f)[-1])
    s = build_sheet(f)
    assert np.array_equal(s[180:, 960:], tiles(f)[-1])


def test_sample_indices():
    assert C.sample_frame_indices() == [0, 12, 24, 36, 48, 60, 72, 84]
