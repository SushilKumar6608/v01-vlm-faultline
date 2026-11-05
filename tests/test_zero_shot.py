import importlib.util
import math
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_zero_shot", ROOT / "kaggle" / "run_zero_shot.py")
zs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(zs)
PROMPTS = yaml.safe_load((ROOT / "configs" / "prompts_v1.yaml").read_text(encoding="utf-8"))


def test_classes_match_the_project():
    from faultline.sim import config as C

    assert tuple(C.CLASSES) == zs.CLASSES and sorted(PROMPTS["order"]) == sorted(C.CLASSES)


@pytest.mark.parametrize("kind", ["single", "sheet"])
@pytest.mark.parametrize("reverse", [False, True])
def test_prompt_lists_every_option_once_in_order(kind, reverse):
    text, order = zs.build_prompt(PROMPTS, kind, reverse)
    assert "{options}" not in text and text.endswith("Answer with only the letter.")
    for i, c in enumerate(order):
        assert f"{'ABCDE'[i]}. {PROMPTS['options'][c]}" in text
    assert order == (PROMPTS["order"][::-1] if reverse else PROMPTS["order"])


def test_prompts_state_the_gate_schedule():
    from faultline.sim import config as C

    for kind in ("single", "sheet"):
        t = PROMPTS[kind]
        assert "1.0 s to 2.0 s" in t and "4.0 s to 5.0 s" in t
    assert C.GATE_WINDOWS == ((1.0, 2.0), (4.0, 5.0)) and C.GATE_SCHEDULE_ID == "fixed_v1"
    assert "1.50 s and 4.50 s" in PROMPTS["sheet"] and "5.25 s" in PROMPTS["single"]


def test_letter_probs_and_decision():
    p = zs.letter_probs([0.0, 3.0, 0.0, 0.0, 0.0])
    assert math.isclose(sum(p), 1.0) and zs.decide(p, zs.CLASSES) == "stuck_item"
    assert zs.decide(p, list(zs.CLASSES)[::-1]) == "spray"  # reversed letters: B is a different class
    bad = zs.letter_probs([float("nan"), 0, 0, 0, 0])
    assert all(math.isnan(x) for x in bad) and zs.decide(bad, zs.CLASSES) == "invalid"
    assert zs.letter_probs([1e4, 0, 0, 0, 0])[0] == pytest.approx(1.0)  # no overflow


def test_resume_and_merge(tmp_path):
    jf = zs.job_file(tmp_path, "m", "sheet", "clean", "test", False)
    zs.append_rows(jf, [["c1", "normal", 0.9, 0.1, 0, 0, 0, 0.99, 1.0], ["c2", "spray", 0, 0, 0, 1, 0, 0.98, 1.0]])
    assert zs.done_ids(jf) == {"c1", "c2"}
    zs.append_rows(jf, [["c3", "invalid", "nan", "nan", "nan", "nan", "nan", 0.0, 1.0]])
    jf2 = zs.job_file(tmp_path, "m", "single", "degraded", "heldout", False)
    zs.append_rows(jf2, [["c9", "gate_open", 0, 0, 1, 0, 0, 0.97, 1.0]])
    dest = zs.merge_predictions(tmp_path, "m", "myname", False)
    import pandas as pd

    d = pd.read_csv(dest)
    assert list(d.columns) == ["model", "input", "condition", "split", "clip_id", "pred"]
    assert len(d) == 4 and set(d.model) == {"myname"}
    assert d[(d.clip_id == "c9")].iloc[0].to_dict() == dict(model="myname", input="single", condition="degraded", split="heldout", clip_id="c9", pred="gate_open")
    assert (tmp_path / "zeroshot_m_probs.csv").exists()


def test_predictions_file_passes_the_phase4_validation(tmp_path):
    """A zero-shot file with an 'invalid' answer is accepted by the evaluation harness and counted wrong."""
    import pandas as pd

    from faultline.eval import report

    m = pd.DataFrame(dict(clip_id=["a", "b"], split="test", cls=["normal", "spray"], severity_bin=[None, "low"], scene_config_id="A1"))
    p = pd.DataFrame(dict(model="z", input="sheet", condition="clean", split="test", clip_id=["a", "b"], pred=["normal", "invalid"]))
    assert report.validate_predictions(p, m) == []


def test_export_writes_lossless_images_identical_to_the_model_input(tmp_path, monkeypatch):
    import cv2
    import numpy as np

    from faultline.data.generate import clip_paths
    from faultline.eval.inputs import make_input
    from faultline.media import read_frames, write_mp4

    monkeypatch.setenv("FAULTLINE_DATA", str(tmp_path / "data"))
    sp = importlib.util.spec_from_file_location("export_vlm_inputs", ROOT / "scripts" / "export_vlm_inputs.py")
    ex = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(ex)
    frames = np.random.default_rng(0).integers(0, 255, size=(96, 360, 640, 3), dtype=np.uint8)
    src = clip_paths("val", "clipx")[0]
    src.parent.mkdir(parents=True)
    write_mp4(frames, src)
    root = tmp_path / "bundle"
    ex.work((str(root), "val", "clipx", "clean"))
    for kind, shape in (("sheet", (360, 1280, 3)), ("single", (180, 320, 3))):
        img = cv2.cvtColor(cv2.imread(str(ex.img_path(root, "clean", kind, "val", "clipx"))), cv2.COLOR_BGR2RGB)
        assert img.shape == shape
        assert np.array_equal(img, make_input(read_frames(src), kind))
    ex.work((str(root), "val", "clipx", "clean"))  # second call keeps the files


# ----------------------------------------------------------------------------- prompts_v2 (one-word answers, two option orders)
P2 = yaml.safe_load((ROOT / "configs" / "prompts_v2.yaml").read_text(encoding="utf-8"))


def test_v2_orders_move_every_class_and_cover_all_classes():
    o1, o2 = P2["orders"]
    assert sorted(o1) == sorted(zs.CLASSES) == sorted(o2)
    assert all(o1.index(c) != o2.index(c) for c in zs.CLASSES)
    assert set(P2["labels"]) == set(zs.CLASSES) == set(P2["descriptions"]) and P2["scoring"] == "names"


@pytest.mark.parametrize("kind", ["single", "sheet"])
def test_v2_prompt_lists_each_option_once_and_names_the_answer_words(kind):
    words = [P2["labels"][c] for c in zs.CLASSES]
    for order in P2["orders"]:
        text = zs.build_name_prompt(P2, kind, order)
        assert "{options}" not in text
        lines = [ln for ln in text.splitlines() if ln.startswith("- ") and ln[2:].split(":")[0] in words]
        assert [ln[2:].split(":")[0] for ln in lines] == [P2["labels"][c] for c in order]
        assert text.endswith("Answer with exactly one word: normal, stuck, gate, spray or blocked.")
    assert len(set(words)) == 5


def test_v2_keeps_the_scene_facts_of_v1():
    for kind in ("single", "sheet"):
        assert "1.0 s to 2.0 s" in P2[kind] and "4.0 s to 5.0 s" in P2[kind]
    assert "1.50 s and 4.50 s" in P2["sheet"] and "5.25 s" in P2["single"]
    for c in zs.CLASSES:  # option wording is v1's wording
        assert PROMPTS["options"][c].split(": ", 1)[1] == P2["descriptions"][c]


def test_v2_name_probabilities_and_averaging():
    p = zs.name_probs({"normal": 0.2, "stuck_item": 0.2, "gate_open": 0.4, "spray": 0.1, "blocked_lens": 0.1})
    assert math.isclose(sum(p.values()), 1.0) and zs.decide_class(p) == "gate_open"
    # two orders that disagree: averaging decides
    a = {"normal": 0.7, "stuck_item": 0.0, "gate_open": 0.3, "spray": 0.0, "blocked_lens": 0.0}
    b = {"normal": 0.3, "stuck_item": 0.0, "gate_open": 0.7, "spray": 0.0, "blocked_lens": 0.0}
    avg = zs.average_over_orders([a, b])
    assert avg["normal"] == pytest.approx(0.5) and avg["gate_open"] == pytest.approx(0.5)
    bad = zs.name_probs({"normal": float("nan"), "stuck_item": 0, "gate_open": 0, "spray": 0, "blocked_lens": 0})
    assert zs.decide_class(bad) == "invalid"
    assert zs.decide_class(zs.name_probs({c: 0.0 for c in zs.CLASSES})) == "invalid"


def test_v2_answer_tokens_must_not_overlap():
    class Tok:
        def __init__(self, table):
            self.table = table

        def encode(self, s, add_special_tokens=False):
            return self.table[s]

    labels = P2["labels"]
    table = {}
    n = 10
    for c in zs.CLASSES:
        w = labels[c]
        table[w.lower()] = [n, 99]
        table[w.capitalize()] = [n + 1, 99]
        n += 2
    ids = zs.name_token_ids(Tok(table), labels)
    assert ids["normal"] == [10, 11] and ids["blocked_lens"] == [18, 19]
    table["spray"] = [10]  # collides with 'normal'
    with pytest.raises(RuntimeError):
        zs.name_token_ids(Tok(table), labels)


def test_v2_files_do_not_mix_with_v1_files(tmp_path):
    v1 = zs.job_file(tmp_path, "m", "sheet", "clean", "val", False)
    v2 = zs.job_file(tmp_path, "m", "sheet", "clean", "val", False, "_v2")
    assert v1 != v2
    zs.append_rows(v1, [["c1", "normal", 1, 0, 0, 0, 0, 1.0, 1.0]])
    zs.append_rows(v2, [["c1", "spray", 0, 0, 0, 1, 0, 0.9, 2.0, 1, "spray", "spray"]], zs.HEADER_V2)
    import pandas as pd

    d1 = pd.read_csv(zs.merge_predictions(tmp_path, "m", "n", False))
    d2 = pd.read_csv(zs.merge_predictions(tmp_path, "m", "n", False, "_v2"))
    assert list(d1.pred) == ["normal"] and list(d2.pred) == ["spray"]
    assert {"agree", "pred_o1", "pred_o2", "answer_mass"} <= set(pd.read_csv(tmp_path / "zeroshot_m_v2_probs.csv").columns)


def test_v2_run_loop_with_a_fake_model_resumes_and_averages(tmp_path, monkeypatch):
    import argparse

    import pandas as pd
    from PIL import Image

    data = tmp_path / "bundle"
    rows = []
    for i in range(3):
        rel = f"images/clean/sheet/val/c{i}.png"
        (data / rel).parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8)).save(data / rel)
        rows.append(dict(clip_id=f"c{i}", split="val", condition="clean", input="sheet", path=rel, sha256=""))
    index = pd.DataFrame(rows)

    class Tok:
        def encode(self, s, add_special_tokens=False):
            return [abs(hash(s)) % 10_000_019]

    class Proc:
        tokenizer = Tok()

    calls = []

    def fake_score(model, processor, image, prompt, ids, max_pixels=None):
        calls.append(prompt)
        words = [P2["labels"][c] for c in zs.CLASSES]
        first = next(ln[2:].split(":")[0] for ln in prompt.splitlines() if ln.startswith("- ") and ln[2:].split(":")[0] in words)  # first listed option: the fake model always answers it
        mass = {c: 0.01 for c in zs.CLASSES}
        mass[next(c for c in zs.CLASSES if P2["labels"][c] == first)] = 0.6
        return mass, sum(mass.values())

    monkeypatch.setattr(zs, "score_names", fake_score)
    a = argparse.Namespace(splits=["val"], conditions=["clean"], kinds=["sheet"], limit=None, n_orders=2, max_pixels=None, upscale=1.0)
    out = tmp_path / "out"
    out.mkdir()
    dest = zs.run_v2(a, data, index, P2, out, "m", "mname", None, Proc())
    d = pd.read_csv(dest.replace(".csv", "_probs.csv"))
    assert len(d) == 3 and len(calls) == 6  # two orders per clip
    # order 1 lists 'normal' first, order 2 lists 'spray' first: the two answers disagree, so averaging gives a tie broken by class order
    assert set(d.pred_o1) == {"normal"} and set(d.pred_o2) == {"spray"} and (d.agree == 0).all()
    expect = 0.5 * (0.6 / 0.64) + 0.5 * (0.01 / 0.64)
    assert (d.p_normal - expect).abs().max() < 1e-4 and (d.p_spray - expect).abs().max() < 1e-4
    assert (d.answer_mass > 0.6).all() and (d.pred == "normal").all()  # tie broken by class order
    zs.run_v2(a, data, index, P2, out, "m", "mname", None, Proc())
    assert len(calls) == 6  # nothing is redone on a second start


def test_v2_upscale_runs_go_to_their_own_files_and_enlarge_the_image(tmp_path, monkeypatch):
    import argparse

    import pandas as pd
    from PIL import Image

    data = tmp_path / "b"
    rel = "images/clean/single/val/c0.png"
    (data / rel).parent.mkdir(parents=True)
    Image.new("RGB", (32, 18)).save(data / rel)
    index = pd.DataFrame([dict(clip_id="c0", split="val", condition="clean", input="single", path=rel, sha256="")])
    sizes = []

    class Tok:
        def encode(self, s, add_special_tokens=False):
            return [abs(hash(s)) % 10_000_019]

    class Proc:
        tokenizer = Tok()

    def fake(model, processor, image, prompt, ids, max_pixels=None):
        sizes.append(image.size)
        return {c: 0.1 for c in zs.CLASSES}, 0.5

    monkeypatch.setattr(zs, "score_names", fake)
    out = tmp_path / "o"
    out.mkdir()
    base = dict(splits=["val"], conditions=["clean"], kinds=["single"], limit=None, n_orders=1, max_pixels=None)
    zs.run_v2(argparse.Namespace(**base, upscale=2.0), data, index, P2, out, "m", "n", None, Proc())
    assert sizes == [(64, 36)] and (out / "raw_m_v2x2_single_clean_val.csv").exists()
    zs.run_v2(argparse.Namespace(**base, upscale=1.0), data, index, P2, out, "m", "n", None, Proc())
    assert (out / "raw_m_v2_single_clean_val.csv").exists()
    assert len(pd.read_csv(out / "zeroshot_m_v2.csv")) == 1 and len(pd.read_csv(out / "zeroshot_m_v2x2.csv")) == 1


# ----------------------------------------------------------------------------- fine-tuning helpers (no GPU needed)
ft_spec = importlib.util.spec_from_file_location("finetune_qlora", ROOT / "kaggle" / "finetune_qlora.py")
ft = importlib.util.module_from_spec(ft_spec)
ft_spec.loader.exec_module(ft)


def _index(n_per_class=10):
    import pandas as pd

    rows = []
    for c in zs.CLASSES:
        for i in range(n_per_class):
            cid = f"{c}_A1_{len(rows):06d}"
            for kind in ("single", "sheet"):
                rows.append(dict(clip_id=cid, split="train", condition="clean", input=kind, path=f"images/clean/{kind}/train/{cid}.png", sha256=""))
    rows.append(dict(clip_id="normal_A1_999999", split="val", condition="clean", input="sheet", path="x.png", sha256=""))
    return pd.DataFrame(rows)


def test_ft_label_from_clip_id_and_train_samples():
    assert ft.label_of("stuck_item_A3_200004") == "stuck_item" and ft.label_of("blocked_lens_D1_1") == "blocked_lens"
    with pytest.raises(ValueError):
        ft.label_of("unknown_A1_1")
    idx = _index()
    s = ft.train_samples(idx, "sheet", ["clean"])
    assert len(s) == 50 and all("/sheet/" in p for _, p, _ in s) and all(c in zs.CLASSES for *_, c in s)  # train split only
    sub = ft.train_samples(idx, "sheet", ["clean"], per_class=3, seed=1)
    assert len(sub) == 15 and {c: sum(1 for *_, k in sub if k == c) for c in zs.CLASSES} == {c: 3 for c in zs.CLASSES}
    assert sub == ft.train_samples(idx, "sheet", ["clean"], per_class=3, seed=1) and sub != ft.train_samples(idx, "sheet", ["clean"], per_class=3, seed=2)


def test_ft_order_is_reproducible_and_covers_every_sample():
    assert ft.epoch_order(20, 0, 3) == ft.epoch_order(20, 0, 3) and ft.epoch_order(20, 0, 3) != ft.epoch_order(20, 0, 4)
    assert sorted(ft.epoch_order(20, 0, 0)) == list(range(20))
    o = ft.sample_option_order(0, 1, 5)
    assert sorted(o) == sorted(zs.CLASSES) and o == ft.sample_option_order(0, 1, 5)
    seen = {tuple(ft.sample_option_order(0, 0, i)) for i in range(200)}
    assert len(seen) > 40  # many different orders, so no position shortcut can be learned
    assert ft.sequence_position(25, 10) == (2, 5)


def test_ft_schedule_warms_up_then_decays():
    lrs = [ft.lr_at(s, 100, 1e-4) for s in range(100)]
    assert lrs[0] < lrs[4] and max(lrs) == pytest.approx(1e-4, rel=0.05) and lrs[-1] < 0.05e-4
    assert all(a >= b for a, b in zip(lrs[6:], lrs[7:]))


def test_ft_lora_regex_hits_language_layers_and_only_optionally_the_vision_tower():
    import re

    lang = ["model.language_model.layers.3.self_attn.q_proj", "model.language_model.layers.3.mlp.down_proj"]
    vis = ["model.visual.blocks.2.attn.qkv", "model.visual.blocks.2.mlp.linear_fc1"]
    other = ["lm_head", "model.language_model.embed_tokens", "model.language_model.layers.3.input_layernorm"]
    r0, r1 = re.compile(ft.lora_target_regex(False)), re.compile(ft.lora_target_regex(True))
    assert all(r0.fullmatch(n) for n in lang) and not any(r0.fullmatch(n) for n in vis + other)
    assert all(r1.fullmatch(n) for n in lang + vis) and not any(r1.fullmatch(n) for n in other)


def test_runner_tags_adapter_runs_by_the_adapter_folder_name(tmp_path):
    jf = zs.job_file(tmp_path, "Q+adapter_sheet", "sheet", "clean", "val", False, "_v2")
    zs.append_rows(jf, [["c1", "spray", 0, 0, 0, 1, 0, 0.9, 2.0, 1, "spray", "spray"]], zs.HEADER_V2)
    d = zs.merge_predictions(tmp_path, "Q+adapter_sheet", "n", False, "_v2")
    assert d.name == "zeroshot_Q+adapter_sheet_v2.csv"


def test_adapter_info_is_found_next_to_or_above_the_adapter(tmp_path):
    import json

    (tmp_path / "adapter_sheet" / "final").mkdir(parents=True)
    assert zs.adapter_info(tmp_path / "adapter_sheet" / "final") == {}
    (tmp_path / "adapter_sheet" / "run_info.json").write_text(json.dumps(dict(model="m", load_4bit=True, kind="sheet")))
    assert zs.adapter_info(tmp_path / "adapter_sheet" / "final")["kind"] == "sheet"
    assert zs.adapter_info(tmp_path / "adapter_sheet")["load_4bit"] is True
