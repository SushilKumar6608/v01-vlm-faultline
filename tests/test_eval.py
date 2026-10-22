import numpy as np
import pandas as pd
import pytest

from faultline.data import degrade, generate
from faultline.data.spec import clip_id as make_id
from faultline.eval import metrics as M
from faultline.eval import report
from faultline.eval.inputs import load_input
from faultline.eval.models import ConstantModel, NoisyOracleModel, OracleModel, run_model
from faultline.sim import config as C


def fake_manifest(per_class=6, splits=("test",)):
    rows, i = [], 0
    for sp in splits:
        for c in C.CLASSES:
            for j in range(per_class):
                rows.append(dict(clip_id=f"{c}_{sp}_{j}", split=sp, cls=c, scene_config_id=["A1", "A2", "A3"][j % 3],
                                 severity_bin=None if c == "normal" else ["low", "mid", "high"][j % 3], seed=i, n_frames=96))
                i += 1
    return pd.DataFrame(rows)


def preds_for(m, model, kind="sheet", cond="clean"):
    return run_model(model, m, "test", kind, cond)


def test_mcnemar_exact():
    r = M.mcnemar_exact(np.zeros(5, bool), np.ones(5, bool))
    assert (r["only_a"], r["only_b"]) == (0, 5) and r["p"] == pytest.approx(0.0625)
    assert M.mcnemar_exact(np.ones(4, bool), np.ones(4, bool))["p"] == 1.0
    assert M.mcnemar_exact(np.array([1, 0, 1, 0], bool), np.array([0, 1, 0, 1], bool))["p"] == 1.0


def test_per_class_and_summary_values():
    m = fake_manifest()
    df = report.attach_truth(preds_for(m, OracleModel(m)).assign(model="x"), m)
    df.loc[df.index[0], "pred"] = "normal"          # first normal stays normal: no change; pick a fault clip instead
    i = df.index[df.cls == "spray"][0]
    df.loc[i, "pred"] = "normal"                    # one spray clip missed as normal
    pc = M.per_class_table(df).set_index("cls")
    assert pc.loc["spray", "n_true"] == 6 and pc.loc["spray", "recall"] == pytest.approx(5 / 6)
    assert pc.loc["normal", "n_pred"] == 7 and pc.loc["normal", "precision"] == pytest.approx(6 / 7)
    s = M.summary(df)
    assert s["n"] == 30 and s["accuracy"] == pytest.approx(29 / 30)
    assert s["detect_recall"] == pytest.approx(23 / 24) and s["false_alarm"] == 0.0
    assert M.confusion(df).loc["spray", "normal"] == 1


def test_invalid_counts_as_wrong_and_as_no_class():
    m = fake_manifest()
    df = report.attach_truth(preds_for(m, OracleModel(m)).assign(model="x"), m)
    i = df.index[df.cls == "gate_open"][0]
    df.loc[i, "pred"] = M.INVALID
    s = M.summary(df)
    assert s["accuracy"] == pytest.approx(29 / 30) and s["invalid_rate"] == pytest.approx(1 / 30)
    assert M.per_class_table(df).set_index("cls").loc["gate_open", "recall"] == pytest.approx(5 / 6)


def test_by_severity_bins():
    m = fake_manifest()
    df = report.attach_truth(preds_for(m, OracleModel(m)).assign(model="x"), m)
    t = M.by_severity(df)
    assert len(t) == 12 and (t["n"] == 2).all() and (t["recall"] == 1.0).all()


def test_validate_predictions():
    m = fake_manifest()
    good = preds_for(m, ConstantModel("normal"))
    assert report.validate_predictions(good, m) == []
    assert any("missing" in e for e in report.validate_predictions(good.iloc[1:], m))
    assert report.validate_predictions(good.iloc[1:], m, allow_partial=True) == []
    assert any("duplicate" in e for e in report.validate_predictions(pd.concat([good, good.iloc[:1]]), m))
    assert any("not in split" in e for e in report.validate_predictions(good.assign(clip_id=good.clip_id + "x"), m, allow_partial=True))
    assert any("unexpected labels" in e for e in report.validate_predictions(good.assign(pred="dog"), m))
    assert any("missing columns" in e for e in report.validate_predictions(good.drop(columns="pred"), m))
    assert any("input" in e for e in report.validate_predictions(good.assign(input="video"), m))


def test_sheet_vs_single_pairs_by_clip():
    m = fake_manifest(per_class=20)
    single = preds_for(m, NoisyOracleModel(m, 0.4, seed=1, name="n"), "single")
    sheet = preds_for(m, NoisyOracleModel(m, 0.05, seed=1, name="n"), "sheet")
    df = report.attach_truth(pd.concat([single, sheet.sample(frac=1, random_state=0)]), m)  # row order must not matter
    sv = report.sheet_vs_single(df)
    allrow = sv[sv.scope == "all"].iloc[0]
    assert allrow.n == 100 and allrow.delta > 0.2 and allrow.p_mcnemar < 0.001
    assert set(sv.scope) == {"all", *C.CLASSES}


def test_reference_models():
    m = fake_manifest(per_class=50)
    assert (preds_for(m, OracleModel(m)).pred.values == m.cls.values).all()
    assert (preds_for(m, ConstantModel("spray")).pred == "spray").all()
    a = preds_for(m, NoisyOracleModel(m, 0.2, seed=3))
    b = preds_for(m, NoisyOracleModel(m, 0.2, seed=3))
    assert a.pred.tolist() == b.pred.tolist()                                  # deterministic per clip
    assert abs((a.pred.values != m.cls.values).mean() - 0.2) < 0.08


def test_report_files(tmp_path):
    m = fake_manifest()
    p = pd.concat([preds_for(m, OracleModel(m), k) for k in ("single", "sheet")])
    tables = report.compute_tables(report.attach_truth(p, m))
    md = report.write_report(tables, tmp_path / "r", "t")
    for name in ("summary", "per_class", "by_severity", "by_scene", "confusion", "sheet_vs_single"):
        assert (tmp_path / "r" / f"{name}.csv").exists()
    assert "Sheet versus single frame" in md.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def two_clips(tmp_path_factory):
    root = tmp_path_factory.mktemp("clips")
    rows = []
    for cls, sev in (("normal", None), ("spray", 0.7)):
        seed = 100001 if cls == "normal" else 100002
        row = dict(split="train", cls=cls, scene_config_id="A1", seed=seed, severity=sev, clip_id=make_id(cls, "A1", seed))
        generate.render_one((row, str(root)))
        rows.append({**row, "n_frames": 96})
    return root, pd.DataFrame(rows)


def test_degrade_end_to_end(two_clips, tmp_path):
    src, m = two_clips
    dst = tmp_path / "degraded"
    assert degrade.run(m, src_root=src, dst_root=dst, workers=1, progress=False) == dict(made=2, skipped=0, remaining=0)
    assert degrade.run(m, src_root=src, dst_root=dst, workers=1, progress=False) == dict(made=0, skipped=2, remaining=0)
    assert degrade.verify(m, dst, decode_n=2) == []
    out = degrade.write_degraded_manifest(m, dst, tmp_path / "dm.csv")
    dm = pd.read_csv(out)
    assert len(dm) == 2 and dm.recipe_name.eq("degradation_v1").all()
    # same input and seed give the same bytes
    other = tmp_path / "again"
    degrade.run(m, src_root=src, dst_root=other, workers=1, progress=False)
    a = pd.read_csv(degrade.write_degraded_manifest(m, other, tmp_path / "dm2.csv"))
    assert a.sha256.tolist() == dm.sha256.tolist()
    # a folder made with another recipe is refused
    (dst / "recipe.json").write_text('{"hash": "other", "name": "x"}', encoding="utf-8")
    with pytest.raises(RuntimeError):
        degrade.run(m, src_root=src, dst_root=dst, workers=1, progress=False)


def test_load_input_shapes_and_degradation(two_clips, tmp_path):
    src, m = two_clips
    dst = tmp_path / "degraded"
    degrade.run(m, src_root=src, dst_root=dst, workers=1, progress=False)
    r = m.iloc[1]
    sheet = load_input(r.split, r.clip_id, "clean", "sheet", src, dst)
    single = load_input(r.split, r.clip_id, "clean", "single", src, dst)
    deg = load_input(r.split, r.clip_id, "degraded", "sheet", src, dst)
    assert sheet.shape == (360, 1280, 3) and single.shape == (180, 320, 3) and sheet.dtype == np.uint8
    assert deg.shape == sheet.shape and np.abs(deg.astype(int) - sheet.astype(int)).mean() > 2
    assert np.array_equal(single, sheet[180:, 960:])                           # single frame is the last tile of the sheet
    with pytest.raises(ValueError):
        load_input(r.split, r.clip_id, "blurry", "sheet", src, dst)
