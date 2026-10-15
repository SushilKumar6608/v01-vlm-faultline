import pandas as pd
import pytest

from faultline.data import spec
from faultline.sim import config as C


@pytest.fixture(scope="module")
def cfg():
    return spec.load_config()


@pytest.fixture(scope="module")
def df(cfg):
    return spec.build_specs(cfg)


def test_sizes(df):
    assert len(df) == 2450
    t = pd.crosstab(df.split, df.cls)
    assert t.loc["train"].eq(300).all() and t.loc["val"].eq(50).all()
    assert t.loc["test"].eq(100).all() and t.loc["heldout"].eq(40).all()


def test_validates_clean(df, cfg):
    assert spec.validate_specs(df, cfg) == []


def test_heldout_only_D1_and_train_never_D1(df):
    assert set(df[df.split == "heldout"].scene_config_id) == {"D1"}
    assert "D1" not in set(df[df.split != "heldout"].scene_config_id)


def test_seed_blocks_disjoint(df, cfg):
    spans = {s: (c["seed_start"], c["seed_start"] + c["n_per_class"] * 5) for s, c in cfg["splits"].items()}
    for s, (lo, hi) in spans.items():
        assert df[df.split == s].seed.between(lo, hi - 1).all()
    assert df.seed.is_unique and df.clip_id.is_unique


def test_severity_stratified_bins(df):
    t = df[(df.split == "test") & (df.cls == "spray")]
    counts = t.severity_bin.value_counts()
    assert counts.between(32, 34).all() and counts.sum() == 100
    assert df[df.cls == "normal"].severity.isna().all()


def test_spec_is_byte_reproducible(cfg, tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    spec.write_specs(spec.build_specs(cfg), a)
    spec.write_specs(spec.build_specs(cfg), b)
    assert a.read_bytes() == b.read_bytes()


def test_committed_spec_matches_config(cfg, tmp_path):
    committed = spec.spec_path_committed()
    if not committed.exists():
        pytest.skip("run scripts/generate_dataset.py --dry-run to create configs/specs_v1.csv")
    p = tmp_path / "x.csv"
    spec.write_specs(spec.build_specs(cfg), p)
    assert committed.read_bytes() == p.read_bytes()


def test_validator_catches_problems(df, cfg):
    bad = df.copy()
    bad.loc[0, "clip_id"] = bad.loc[1, "clip_id"]
    assert any("duplicate clip_id" in e for e in spec.validate_specs(bad, cfg))
    bad = df.copy()
    i = bad[bad.cls == "spray"].index[0]
    bad.loc[i, "severity"] = 0.95  # not in its declared bin
    assert any("severity_bin" in e for e in spec.validate_specs(bad, cfg))
    bad = df.copy()
    j = bad[bad.cls == "normal"].index[0]
    bad.loc[j, "severity"] = 0.5
    assert any("normal" in e for e in spec.validate_specs(bad, cfg))
