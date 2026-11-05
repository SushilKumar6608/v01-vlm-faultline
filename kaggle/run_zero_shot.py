"""Zero-shot vision-language model run on Kaggle (self-contained: needs only the dataset folder this file sits in).

  python run_zero_shot.py --model Qwen/Qwen3-VL-4B-Instruct --splits val --limit 40            # smoke test
  python run_zero_shot.py --model Qwen/Qwen3-VL-4B-Instruct --splits test heldout                # full run

Inputs (made on the laptop by scripts/export_vlm_inputs.py): index.csv and images/<condition>/<input>/<split>/<clip_id>.png.
The model sees exactly the images a human saw in the Phase 3 audit. It is shown one image and the lettered options of
the options of a prompt file and is not allowed to write text. prompts_v1 (scoring: letters): the answer is the letter with the
highest probability at the first answer position. prompts_v2 (scoring: names, the default): the answer is a one-word label read
from the first answer token, every clip is scored under two option orders and the class probabilities are averaged. The
probabilities of all five classes are saved (they are the teacher's soft labels in the distillation phase).

Safe to stop and restart: finished clips are kept (one append-only CSV per job) and skipped on the next start.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import os
import sys
import time
from pathlib import Path

LETTERS = "ABCDE"
CLASSES = ("normal", "stuck_item", "gate_open", "spray", "blocked_lens")


# ----------------------------------------------------------------------------- prompts and scoring (no heavy imports)
def load_prompts(path: Path) -> dict:
    import yaml

    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def option_order(prompts: dict, reverse: bool = False) -> list[str]:
    order = list(prompts["order"])
    if sorted(order) != sorted(CLASSES):
        raise ValueError(f"prompt order must contain exactly {CLASSES}")
    return order[::-1] if reverse else order


def build_prompt(prompts: dict, kind: str, reverse: bool = False) -> tuple[str, list[str]]:
    """Return (prompt text, class for each letter A..E)."""
    order = option_order(prompts, reverse)
    lines = [f"{LETTERS[i]}. {prompts['options'][c]}" for i, c in enumerate(order)]
    return prompts[kind].replace("{options}", "\n".join(lines)).strip(), order


def letter_probs(logits_by_letter: list[float]) -> list[float]:
    """Softmax over the five letter logits. Non-finite input gives [nan]*5."""
    if len(logits_by_letter) != 5 or any(not math.isfinite(x) for x in logits_by_letter):
        return [float("nan")] * 5
    m = max(logits_by_letter)
    e = [math.exp(x - m) for x in logits_by_letter]
    s = sum(e)
    return [x / s for x in e]


def decide(probs_by_letter: list[float], order: list[str]) -> str:
    """Class with the highest letter probability; 'invalid' if the logits were not finite."""
    if any(not math.isfinite(p) for p in probs_by_letter):
        return "invalid"
    return order[max(range(5), key=lambda i: probs_by_letter[i])]


def probs_by_class(probs_by_letter: list[float], order: list[str]) -> dict[str, float]:
    return {c: probs_by_letter[i] for i, c in enumerate(order)}


# ----------------------------------------------------------------------------- scoring by one-word labels (prompts_v2)
def build_name_prompt(prompts: dict, kind: str, order: list[str]) -> str:
    """Prompt with the options as a bulleted list in the given class order (no letters)."""
    if sorted(order) != sorted(CLASSES):
        raise ValueError(f"order must contain exactly {CLASSES}")
    lines = [f"- {prompts['labels'][c]}: {prompts['descriptions'][c]}" for c in order]
    return prompts[kind].replace("{options}", "\n".join(lines)).strip()


def name_probs(mass_by_class: dict[str, float]) -> dict[str, float]:
    """Normalise the probability mass of each class's answer token(s) over the five classes (nan if not finite or zero)."""
    vals = [mass_by_class[c] for c in CLASSES]
    tot = sum(vals) if all(math.isfinite(v) for v in vals) else float("nan")
    if not math.isfinite(tot) or tot <= 0:
        return {c: float("nan") for c in CLASSES}
    return {c: mass_by_class[c] / tot for c in CLASSES}


def average_over_orders(per_order: list[dict[str, float]]) -> dict[str, float]:
    return {c: sum(p[c] for p in per_order) / len(per_order) for c in CLASSES}


def decide_class(probs: dict[str, float]) -> str:
    if any(not math.isfinite(probs[c]) for c in CLASSES):
        return "invalid"
    return max(CLASSES, key=lambda c: probs[c])


def name_token_ids(tokenizer, labels: dict[str, str]) -> dict[str, list[int]]:
    """First token of each one-word label, in lower case and capitalised. The sets must not overlap between classes."""
    ids: dict[str, list[int]] = {}
    for c in CLASSES:
        w = labels[c]
        found = []
        for variant in (w.lower(), w.capitalize()):
            enc = tokenizer.encode(variant, add_special_tokens=False)
            if not enc:
                raise RuntimeError(f"label {variant!r} has no tokens")
            if enc[0] not in found:
                found.append(enc[0])
        ids[c] = found
    flat = [i for v in ids.values() for i in v]
    if len(set(flat)) != len(flat):
        raise RuntimeError(f"answer tokens of different classes overlap: {ids}")
    return ids


# ----------------------------------------------------------------------------- job bookkeeping
HEADER = ["clip_id", "pred", "p_normal", "p_stuck_item", "p_gate_open", "p_spray", "p_blocked_lens", "letter_mass", "seconds"]
HEADER_V2 = ["clip_id", "pred", "p_normal", "p_stuck_item", "p_gate_open", "p_spray", "p_blocked_lens", "answer_mass", "seconds", "agree", "pred_o1", "pred_o2"]


def job_file(out: Path, model_tag: str, kind: str, cond: str, split: str, reverse: bool, variant: str = "") -> Path:
    return out / f"raw_{model_tag}{'_rev' if reverse else ''}{variant}_{kind}_{cond}_{split}.csv"


def done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    lines = path.read_text(encoding="utf-8").splitlines()[1:]
    return {ln.split(",", 1)[0] for ln in lines if ln.strip()}


def append_rows(path: Path, rows: list[list], header: list[str] | None = None) -> None:
    new = not path.exists()
    with open(path, "a", encoding="utf-8", newline="") as f:
        if new:
            f.write(",".join(header or HEADER) + "\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
        f.flush()
        os.fsync(f.fileno())


def merge_predictions(out: Path, model_tag: str, name: str, reverse: bool, variant: str = "") -> Path:
    """Combine the raw job files into the Phase 4 predictions format: model, input, condition, split, clip_id, pred."""
    import pandas as pd

    parts = []
    pre = f"raw_{model_tag}{'_rev' if reverse else ''}{variant}_"
    for p in sorted(out.glob(f"{pre}*.csv")):
        stem = p.stem.removeprefix(pre)
        if len(stem.split("_")) != 3:  # another variant's file (e.g. v2 or an upscaled v2 run)
            continue
        kind, cond, split = stem.split("_")
        d = pd.read_csv(p)
        d.insert(0, "split", split)
        d.insert(0, "condition", cond)
        d.insert(0, "input", kind)
        d.insert(0, "model", name)
        parts.append(d)
    allp = pd.concat(parts, ignore_index=True)
    dest = out / f"zeroshot_{model_tag}{'_rev' if reverse else ''}{variant}.csv"
    allp[["model", "input", "condition", "split", "clip_id", "pred"]].to_csv(dest, index=False)
    allp.to_csv(dest.with_name(dest.stem + "_probs.csv"), index=False)
    return dest


# ----------------------------------------------------------------------------- model (imported lazily)
def adapter_info(adapter: str | Path) -> dict:
    """run_info.json written next to a fine-tuned adapter (base model, 4-bit or not, input kind); {} if absent."""
    import json

    for p in (Path(adapter) / "run_info.json", Path(adapter).parent / "run_info.json"):
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    return {}


def load_model(model_id: str, dtype: str, load_4bit: bool):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    kw = {}
    if load_4bit:
        from transformers import BitsAndBytesConfig

        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
    else:
        kw["torch_dtype"] = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32, "auto": "auto"}[dtype]
    model = AutoModelForImageTextToText.from_pretrained(model_id, device_map="auto", **kw).eval()
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


def letter_token_ids(processor) -> list[int]:
    tok = processor.tokenizer
    ids = []
    for L in LETTERS:
        enc = tok.encode(L, add_special_tokens=False)
        if len(enc) != 1:
            raise RuntimeError(f"letter {L!r} is not a single token for this tokenizer: {enc}")
        ids.append(enc[0])
    if len(set(ids)) != 5:
        raise RuntimeError("letter tokens are not distinct")
    return ids


def score_image(model, processor, image, prompt: str, ids: list[int], max_pixels: int | None = None):
    """Return (letter logits, letter_mass) for one image and prompt."""
    import torch

    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    kw = {}
    if max_pixels:
        kw["images_kwargs"] = {"max_pixels": max_pixels}
    inputs = processor(text=[text], images=[image], return_tensors="pt", **kw).to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1].float()
    logp = torch.log_softmax(logits, dim=-1)
    sel = logits[ids].tolist()
    mass = float(logp[ids].exp().sum())
    return sel, mass


def last_logits(model, processor, image, prompt: str, max_pixels: int | None = None):
    import torch

    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    kw = {}
    if max_pixels:
        kw["images_kwargs"] = {"max_pixels": max_pixels}
    inputs = processor(text=[text], images=[image], return_tensors="pt", **kw).to(model.device)
    with torch.no_grad():
        return model(**inputs).logits[0, -1].float()


def score_names(model, processor, image, prompt: str, ids: dict[str, list[int]], max_pixels: int | None = None):
    """Return (probability mass on each class's answer tokens, total mass) for one image and prompt."""
    import torch

    logp = torch.log_softmax(last_logits(model, processor, image, prompt, max_pixels), dim=-1)
    mass = {c: float(logp[ids[c]].exp().sum()) for c in CLASSES}
    return mass, sum(mass.values())


def run_v2(a, data, index, prompts, out, tag, name, model, processor) -> str:
    from PIL import Image

    ids = name_token_ids(processor.tokenizer, prompts["labels"])
    orders = prompts["orders"][: a.n_orders]
    variant = "_v2" + (f"x{a.upscale:g}" if a.upscale != 1 else "")
    n_bad = n_done = 0
    for split in a.splits:
        for cond in a.conditions:
            for kind in a.kinds:
                rows = index[(index.split == split) & (index.condition == cond) & (index["input"] == kind)]
                if a.limit:
                    rows = rows.head(a.limit)
                jf = job_file(out, tag, kind, cond, split, False, variant)
                have = done_ids(jf)
                todo = rows[~rows.clip_id.isin(have)]
                texts = [build_name_prompt(prompts, kind, o) for o in orders]
                print(f"[{kind}/{cond}/{split}] {len(have)} done, {len(todo)} to do", flush=True)
                buf, t_job = [], time.time()
                for k, r in enumerate(todo.itertuples()):
                    t0 = time.time()
                    img = Image.open(data / r.path).convert("RGB")
                    if a.upscale != 1:
                        img = img.resize((round(img.width * a.upscale), round(img.height * a.upscale)), Image.BICUBIC)
                    per, preds, masses = [], [], []
                    for text in texts:
                        mass, tot = score_names(model, processor, img, text, ids, a.max_pixels)
                        p = name_probs(mass)
                        per.append(p)
                        preds.append(decide_class(p))
                        masses.append(tot)
                    avg = average_over_orders(per)
                    pred = decide_class(avg)
                    n_bad += pred == "invalid" or min(masses) < 0.5
                    n_done += 1
                    p1, p2 = (preds + [""])[:2]
                    buf.append([r.clip_id, pred, *[f"{avg[c]:.6f}" for c in CLASSES], f"{min(masses):.4f}", f"{time.time() - t0:.2f}", int(len(set(preds)) == 1), p1, p2])
                    if len(buf) >= 20:
                        append_rows(jf, buf, HEADER_V2)
                        buf = []
                        print(f"  {k + 1}/{len(todo)}  {(time.time() - t_job) / (k + 1):.2f} s/clip", flush=True)
                    if n_done >= 20 and n_bad / n_done > 0.2:
                        append_rows(jf, buf, HEADER_V2)
                        sys.exit(f"STOP: {n_bad} of {n_done} answers were non-finite or put under half of the probability on the answer words (format failure or fp16 instability). Try --dtype float32.")
                if buf:
                    append_rows(jf, buf, HEADER_V2)
    return str(merge_predictions(out, tag, name, False, variant))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Hugging Face model id")
    ap.add_argument("--name", default=None, help="name in the predictions file (default: derived from the model id)")
    ap.add_argument("--data", default=None, help="folder with index.csv, images/ and the prompts file (default: this file's folder, or /kaggle/input/*)")
    ap.add_argument("--prompts", default="prompts_v2.yaml", help="prompt file name inside the data folder (prompts_v1.yaml = letter scoring)")
    ap.add_argument("--upscale", type=float, default=1.0, help="prompts_v2: enlarge each image by this factor before the model sees it (the 320x180 single frame becomes only about 60 visual tokens at native size)")
    ap.add_argument("--adapter", default=None, help="folder with a LoRA adapter from finetune_qlora.py; the predictions are tagged with its name")
    ap.add_argument("--n-orders", type=int, default=2, choices=[1, 2], help="prompts_v2: option orders averaged per clip")
    ap.add_argument("--out", default="/kaggle/working" if Path("/kaggle/working").exists() else "out")
    ap.add_argument("--splits", nargs="+", default=["test", "heldout"])
    ap.add_argument("--conditions", nargs="+", default=["clean", "degraded"])
    ap.add_argument("--kinds", nargs="+", default=["single", "sheet"])
    ap.add_argument("--limit", type=int, default=None, help="first N clips per job (smoke test)")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32", "auto"])
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--reverse-options", action="store_true", help="letters in reverse class order (position-bias check)")
    ap.add_argument("--max-pixels", type=int, default=None)
    a = ap.parse_args()

    here = Path(__file__).resolve().parent
    data = Path(a.data) if a.data else next((p.parent for p in [here / "index.csv", *Path("/kaggle/input").rglob("index.csv")] if p.exists()), None)
    if data is None or not (data / "index.csv").exists():
        sys.exit("index.csv not found: pass --data <folder>")
    import pandas as pd
    from PIL import Image

    index = pd.read_csv(data / "index.csv")
    prompt_path = data / a.prompts if (data / a.prompts).exists() else here / a.prompts
    prompts = load_prompts(prompt_path)
    prompt_sha = hashlib.sha256(prompt_path.read_bytes()).hexdigest()[:12]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = a.model.split("/")[-1]
    if a.adapter:
        info = adapter_info(a.adapter)
        if info:
            if info.get("model") and info["model"] != a.model:
                sys.exit(f"the adapter was trained on {info['model']}, not {a.model}")
            a.load_4bit = bool(info.get("load_4bit", a.load_4bit))  # score with the same base weights the adapter was trained on
            if a.kinds == ["single", "sheet"] and info.get("kind"):
                a.kinds = [info["kind"]]
            print(f"adapter info: {info}", flush=True)
        label = Path(a.adapter).parent.name if Path(a.adapter).name == "final" else Path(a.adapter).name
        tag = f"{tag}+{label}"
    name = a.name or tag
    print(f"model {a.model} | dtype {a.dtype} | 4bit {a.load_4bit} | prompts {prompts['name']} ({prompt_sha}) | data {data}", flush=True)

    model, processor = load_model(a.model, a.dtype, a.load_4bit)
    if a.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, a.adapter).eval()
        print("adapter", a.adapter, flush=True)
    if prompts.get("scoring") == "names":
        print("wrote", run_v2(a, data, index, prompts, out, tag, name, model, processor), flush=True)
        return
    ids = letter_token_ids(processor)
    n_nan = n_done = 0
    for split in a.splits:
        for cond in a.conditions:
            for kind in a.kinds:
                rows = index[(index.split == split) & (index.condition == cond) & (index["input"] == kind)]
                if a.limit:
                    rows = rows.head(a.limit)
                jf = job_file(out, tag, kind, cond, split, a.reverse_options)
                have = done_ids(jf)
                todo = rows[~rows.clip_id.isin(have)]
                prompt, order = build_prompt(prompts, kind, a.reverse_options)
                print(f"[{kind}/{cond}/{split}] {len(have)} done, {len(todo)} to do", flush=True)
                buf, t_job = [], time.time()
                for k, r in enumerate(todo.itertuples()):
                    t0 = time.time()
                    img = Image.open(data / r.path).convert("RGB")
                    sel, mass = score_image(model, processor, img, prompt, ids, a.max_pixels)
                    pl = letter_probs(sel)
                    pred = decide(pl, order)
                    pc = probs_by_class(pl, order)
                    n_nan += pred == "invalid"
                    n_done += 1
                    buf.append([r.clip_id, pred, *[f"{pc[c]:.6f}" for c in CLASSES], f"{mass:.4f}", f"{time.time() - t0:.2f}"])
                    if len(buf) >= 20:
                        append_rows(jf, buf)
                        buf = []
                        print(f"  {k + 1}/{len(todo)}  {(time.time() - t_job) / (k + 1):.2f} s/clip", flush=True)
                    if n_done >= 20 and n_nan / n_done > 0.2:
                        append_rows(jf, buf)
                        sys.exit(f"STOP: {n_nan} of {n_done} answers had non-finite logits (fp16 instability?). Try --dtype float32 or --load-4bit.")
                if buf:
                    append_rows(jf, buf)
    dest = merge_predictions(out, tag, name, a.reverse_options)
    print("wrote", dest, flush=True)


if __name__ == "__main__":
    main()
