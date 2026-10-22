import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from faultline.models import cache  # noqa: E402
from faultline.models.cnn import TileCNN  # noqa: E402
from faultline.models.train import CLASSES, TrainConfig, load_best, lr_factor, predictions_table, train  # noqa: E402
from faultline.sample.sheet import single_frame_tile  # noqa: E402


def _data(n=40, seed=0):
    rng = np.random.default_rng(seed)
    y = np.arange(n) % 5
    x = rng.integers(0, 40, size=(n, 8, 90, 160, 3), dtype=np.uint8)
    for i, c in enumerate(y):  # a learnable cue: brightness of a patch depends on the class
        x[i, :, 10:30, 10:40, :] += np.uint8(20 * c)
    return x, y.astype(np.int64)


def test_forward_shapes():
    for t in (1, 8):
        out = TileCNN(t)(torch.rand(3, t, 3, 90, 160))
        assert out.shape == (3, 5)
    with pytest.raises(ValueError):
        TileCNN(8)(torch.rand(2, 1, 3, 90, 160))


def test_tiles_from_frames_and_single_is_last_tile():
    import cv2

    frames = np.random.default_rng(0).integers(0, 255, size=(96, 360, 640, 3), dtype=np.uint8)
    t = cache.tiles_for_frames(frames)
    assert t.shape == (8, 90, 160, 3) and t.dtype == np.uint8
    want = cv2.resize(single_frame_tile(frames), (160, 90), interpolation=cv2.INTER_AREA)
    assert np.array_equal(t[7], want)


def test_cache_roundtrip_and_manifest_check(tmp_path, monkeypatch):
    m = pd.DataFrame(dict(clip_id=["a", "b", "c"], split=["val", "val", "test"], cls=["normal", "spray", "normal"]))
    monkeypatch.setattr(cache, "_load", lambda args: np.full((8, 90, 160, 3), ord(args[1]) % 255, dtype=np.uint8))
    p = cache.build(m, "clean", "val", tmp_path)
    x = cache.load(m, "clean", "val", tmp_path)
    assert x.shape == (2, 8, 90, 160, 3) and x[1, 0, 0, 0, 0] == ord("b")
    assert cache.build(m, "clean", "val", tmp_path) == p  # second call keeps the file
    m2 = m.iloc[[1, 0, 2]].reset_index(drop=True)
    with pytest.raises(RuntimeError):
        cache.load(m2, "clean", "val", tmp_path)
    assert list(cache.labels(m, "val", CLASSES)) == [0, 3]


def test_lr_schedule():
    cfg = TrainConfig(epochs=10)
    f = [lr_factor(e, cfg) for e in range(10)]
    assert f[0] < f[1] < f[2] and f[2] >= max(f[3:]) and min(f) > 0


@pytest.mark.parametrize("kind", ["single", "sheet"])
def test_resume_gives_the_same_model_as_an_uninterrupted_run(tmp_path, kind):
    x, y = _data(40)
    xv, yv = _data(10, seed=1)
    cfg = TrainConfig(kind=kind, epochs=3, batch=8, seed=3)
    torch.set_num_threads(2)
    straight = train(cfg, x, y, xv, yv, tmp_path / "a", log=lambda s: None)
    r = train(cfg, x, y, xv, yv, tmp_path / "b", log=lambda s: None, stop_after_epoch=1)
    assert r["stopped"] and not (tmp_path / "b" / "done.json").exists()
    resumed = train(cfg, x, y, xv, yv, tmp_path / "b", log=lambda s: None)
    sa = torch.load(tmp_path / "a" / "last.pt", weights_only=False)["model"]
    sb = torch.load(tmp_path / "b" / "last.pt", weights_only=False)["model"]
    assert all(torch.allclose(sa[k].float(), sb[k].float(), atol=1e-5) for k in sa)
    assert straight["best_epoch"] == resumed["best_epoch"]
    again = train(cfg, x, y, xv, yv, tmp_path / "b", log=lambda s: None)  # finished run is skipped
    assert again["best_epoch"] == resumed["best_epoch"]


def test_config_mismatch_is_refused(tmp_path):
    x, y = _data(16)
    cfg = TrainConfig(kind="single", epochs=2, batch=8)
    train(cfg, x, y, x, y, tmp_path, log=lambda s: None, stop_after_epoch=1)
    with pytest.raises(RuntimeError):
        train(TrainConfig(kind="single", epochs=2, batch=8, lr=0.1), x, y, x, y, tmp_path, log=lambda s: None)


def test_learns_and_predictions_table(tmp_path):
    x, y = _data(60)
    cfg = TrainConfig(kind="sheet", epochs=6, batch=10, lr=3e-3)
    torch.set_num_threads(2)
    info = train(cfg, x, y, x, y, tmp_path, log=lambda s: None)
    assert info["best_val_acc"] > 0.5
    m = pd.DataFrame(dict(clip_id=[f"c{i}" for i in range(60)], split="val", cls=[CLASSES[k] for k in y]))
    t = predictions_table(load_best(cfg, tmp_path), cfg, m, {("clean", "val"): x})
    assert list(t.columns) == ["model", "input", "condition", "split", "clip_id", "pred"]
    assert len(t) == 60 and set(t.pred) <= set(CLASSES) and t.model.iloc[0] == "cnn_s0"
