import json

import pandas as pd
import pytest

from faultline.data import checks, generate, manifest, regen, spec
from faultline.sim import config as C
from faultline.sim.clip import generate_clip


@pytest.fixture(scope="module")
def tiny():
    cfg = spec.load_config()
    cfg["splits"] = {"train": dict(n_per_class=1, scenes=["A1"], seed_start=100000)}
    return cfg


@pytest.fixture()
def built(tiny, tmp_path):
    specs = spec.build_specs(tiny)
    root = tmp_path / "clips"
    res = generate.run(specs, root=root, workers=2, progress=False)
    return tiny, specs, root, res


def test_render_resume_and_manifest(built):
    cfg, specs, root, res = built
    assert res == dict(rendered=5, skipped=0, remaining=0)
    again = generate.run(specs, root=root, workers=1, progress=False)
    assert again["rendered"] == 0 and again["skipped"] == 5
    m = manifest.read_sidecars(specs, root)
    assert list(m.columns) == manifest.COLUMNS and len(m) == 5
    assert checks.integrity(specs, m, root=root, decode_n=5) == []


def test_interrupted_and_corrupt_clips_are_redone(built):
    cfg, specs, root, _ = built
    r = specs.iloc[0]
    mp4, js = generate.clip_paths(r.split, r.clip_id, root)
    mp4.write_bytes(b"truncated")  # corrupt file, sidecar says otherwise
    stale = mp4.with_name("zzz.tmp.mp4")
    stale.write_bytes(b"half written")
    assert not generate.is_done(r.split, r.clip_id, root)
    res = generate.run(specs, root=root, workers=1, progress=False)
    assert res["rendered"] == 1 and not stale.exists()
    assert generate.is_done(r.split, r.clip_id, root)


def test_ground_truth_columns_are_deterministic(built):
    cfg, specs, root, _ = built
    m = manifest.read_sidecars(specs, root)
    for r in m.itertuples():
        sev = None if pd.isna(r.severity) else float(r.severity)
        meta = generate_clip(int(r.seed), r.cls, r.scene_config_id, sev).meta
        assert meta["fault_region_px"] == r.fault_region_px
        assert meta["expected_visible_tiles"] == r.expected_visible_tiles


def test_regenerate_inherits_spec_and_quarantines(built, tmp_path, monkeypatch):
    cfg, specs, root, _ = built
    monkeypatch.setattr("faultline.data.regen.paths.data_root", lambda: tmp_path)
    monkeypatch.setattr("faultline.data.regen.replacements_path", lambda: tmp_path / "rep.csv")
    monkeypatch.setattr("faultline.data.regen.load_replacements", lambda: pd.DataFrame(columns=spec.REPLACEMENT_COLUMNS))
    r = specs[specs.cls == "spray"].iloc[0]
    flagged = pd.DataFrame([dict(clip_id=r.clip_id, reason="test")])
    ch = regen.regenerate(flagged, cfg, root=root)
    n = ch.iloc[0]
    assert n.seed_new >= cfg["reserve_seed_start"] and n.cls == r.cls and n.scene_config_id == r.scene_config_id
    assert n.severity == pytest.approx(r.severity)
    assert (tmp_path / "quarantine" / r.split / f"{r.clip_id}.mp4").exists()
    assert not generate.clip_paths(r.split, r.clip_id, root)[0].exists()
    assert (tmp_path / "rep.csv").exists()


def test_visibility_flags_use_class_rules(tiny):
    m = pd.DataFrame([
        dict(clip_id="a", split="train", cls="stuck_item", severity=0.3, expected_visible_tiles=6, halted_tiles=2, frozen_tiles=1),
        dict(clip_id="b", split="train", cls="spray", severity=0.3, expected_visible_tiles=2, halted_tiles=None, frozen_tiles=None),
        dict(clip_id="c", split="train", cls="blocked_lens", severity=0.3, expected_visible_tiles=8, halted_tiles=None, frozen_tiles=None),
    ])
    f = checks.visibility_flags(m, tiny)
    assert set(f.clip_id) == {"a", "b"}  # stuck_item judged on halted tiles, not on the mask-based count


@pytest.mark.slow
def test_full_ground_truth_sample_reproduces():
    """Sample of the real spec: ground-truth columns regenerate exactly (no encoding). Run with: pytest -m slow"""
    cfg = spec.load_config()
    df = spec.build_specs(cfg).sample(20, random_state=0)
    for r in df.itertuples():
        sev = None if pd.isna(r.severity) else float(r.severity)
        a = generate_clip(int(r.seed), r.cls, r.scene_config_id, sev).meta
        b = generate_clip(int(r.seed), r.cls, r.scene_config_id, sev).meta
        assert a == b
