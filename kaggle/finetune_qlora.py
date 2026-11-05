"""QLoRA fine-tuning of a vision-language model on the training clips (Kaggle, one free T4 or two).

  python finetune_qlora.py --fit-test --kind sheet --train-data <train bundle>      # 5-10 minutes: does it fit, is the loss finite and falling
  python finetune_qlora.py --kind sheet  --train-data <train bundle> --epochs 1      # real run, one adapter per input kind
  python run_zero_shot.py --adapter /kaggle/working/adapter_sheet/final ...         # score it with the zero-shot runner (same prompts_v2 scoring)

What is trained: LoRA adapters on the language model's attention and MLP layers (the vision tower stays frozen unless
--lora-vision). The base model is loaded in 4 bit (nf4, fp16 compute) unless --no-4bit. The training target is the one-word answer
of prompts_v2 (normal, stuck, gate, spray, blocked): ordinary next-token cross-entropy on the first answer token. The option
order in the prompt is a random permutation for every sample, so the model cannot learn a letter or position shortcut.
Clean training clips only by default (same protocol as the CNN baseline). One adapter per input kind (single, sheet), so the
sheet-versus-single comparison is made with the same recipe.

Safe to stop and restart: a checkpoint (adapter weights, optimizer, step) is written every --save-every optimizer steps; a restart
with the same --out continues from it. Data order depends only on (seed, epoch).
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_zero_shot as zs  # noqa: E402  (same folder; shares prompts, token ids and model loading with the evaluation runner)

CLASSES = zs.CLASSES
LANGUAGE_MODULES = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
VISION_MODULES = r".*visual.*\.(qkv|proj|linear_fc1|linear_fc2)"


# ----------------------------------------------------------------------------- pure helpers (no torch)
def lora_target_regex(vision: bool = False) -> str:
    return f"(?:{LANGUAGE_MODULES})|(?:{VISION_MODULES})" if vision else LANGUAGE_MODULES


def label_of(clip_id: str) -> str:
    """Class from a clip id such as stuck_item_A3_200004."""
    c = clip_id.rsplit("_", 2)[0]
    if c not in CLASSES:
        raise ValueError(f"cannot read the class from clip id {clip_id!r}")
    return c


def train_samples(index, kind: str, conditions: list[str], per_class: int | None = None, seed: int = 0, split: str = "train") -> list[tuple[str, str, str]]:
    """List of (clip_id, relative image path, class), optionally the same number of clips per class (random, seeded)."""
    rows = index[(index.split == split) & (index["input"] == kind) & (index.condition.isin(conditions))]
    out = [(r.clip_id, r.path, label_of(r.clip_id)) for r in rows.sort_values(["condition", "clip_id"]).itertuples()]
    if per_class is not None:
        rng = random.Random(seed)
        keep = []
        for c in CLASSES:
            group = [s for s in out if s[2] == c]
            rng.shuffle(group)
            keep += group[:per_class]
        out = sorted(keep, key=lambda s: (s[1], s[0]))
    return out


def epoch_order(n: int, seed: int, epoch: int) -> list[int]:
    order = list(range(n))
    random.Random(seed * 1_000_003 + epoch).shuffle(order)
    return order


def sample_option_order(seed: int, epoch: int, i: int) -> list[str]:
    """Random permutation of the five classes for the prompt of sample i (depends only on seed, epoch and i)."""
    order = list(CLASSES)
    random.Random((seed * 1_000_003 + epoch) * 1_000_033 + i).shuffle(order)
    return order


def lr_at(step: int, total: int, base_lr: float, warmup_frac: float = 0.05) -> float:
    warm = max(1, int(total * warmup_frac))
    if step < warm:
        return base_lr * (step + 1) / warm
    prog = (step - warm) / max(1, total - warm)
    return base_lr * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))


def sequence_position(global_sample: int, n: int) -> tuple[int, int]:
    """(epoch, position within the epoch) of the global sample counter."""
    return divmod(global_sample, n)


# ----------------------------------------------------------------------------- model
def attach_lora(model, rank: int, alpha: int, dropout: float, vision: bool, four_bit: bool, grad_ckpt: bool):
    import torch
    from peft import LoraConfig, get_peft_model

    if four_bit:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=grad_ckpt)
    elif grad_ckpt:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    cfg = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout, target_modules=lora_target_regex(vision), task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    n_train = 0
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()  # adapter weights in fp32; the forward pass runs under fp16 autocast
            n_train += p.numel()
    if n_train == 0:
        names = [n for n, _ in model.named_modules() if "q_proj" in n or "qkv" in n][:6]
        raise RuntimeError(f"no LoRA module matched {lora_target_regex(vision)!r}; example module names: {names}")
    n_all = sum(p.numel() for p in model.parameters())
    print(f"trainable parameters {n_train / 1e6:.1f} M of {n_all / 1e6:.0f} M", flush=True)
    return model


def forward_logits(model, processor, image, prompt: str):
    """Logits at the first answer position (float32, grad enabled)."""
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
    try:
        out = model(**inputs, logits_to_keep=1)
    except TypeError:
        out = model(**inputs)
    return out.logits[0, -1].float()


def save_state(path: Path, model, opt, scaler, step: int) -> None:
    import torch

    tmp = path.with_suffix(".tmp")
    trainable = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
    torch.save({"trainable": trainable, "opt": opt.state_dict(), "scaler": scaler.state_dict(), "step": step}, tmp)
    os.replace(tmp, path)


def load_state(path: Path, model, opt, scaler) -> int:
    import torch

    st = torch.load(path, map_location="cpu")
    params = dict(model.named_parameters())
    for n, v in st["trainable"].items():
        params[n].data.copy_(v.to(params[n].device, params[n].dtype))
    opt.load_state_dict(st["opt"])
    scaler.load_state_dict(st["scaler"])
    return int(st["step"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--train-data", required=True, help="folder with index.csv and images/ of the training clips (scripts/export_vlm_inputs.py --splits train)")
    ap.add_argument("--prompts", default="prompts_v2.yaml")
    ap.add_argument("--kind", required=True, choices=["single", "sheet"])
    ap.add_argument("--conditions", nargs="+", default=["clean"], choices=["clean", "degraded"])
    ap.add_argument("--out", default=None, help="default /kaggle/working/adapter_<kind>")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--per-class", type=int, default=None, help="use only this many training clips per class (random, seeded)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--accum", type=int, default=8, help="samples per optimizer step (batch size is 1 per forward pass)")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--lora-vision", action="store_true", help="also adapt the vision tower")
    ap.add_argument("--no-4bit", action="store_true", help="fp16 base weights instead of 4-bit (needs about 8 GB more)")
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=25, help="optimizer steps between checkpoints")
    ap.add_argument("--max-pixels", type=int, default=None)
    ap.add_argument("--fit-test", action="store_true", help="short run on 4 clips per class, 3 epochs: checks memory, speed, finite and falling loss")
    a = ap.parse_args()

    import pandas as pd
    import torch
    from PIL import Image

    torch.manual_seed(a.seed)
    data = Path(a.train_data)
    index = pd.read_csv(data / "index.csv")
    pp = data / a.prompts if (data / a.prompts).exists() else HERE / a.prompts
    prompts = zs.load_prompts(pp)
    if a.fit_test:
        a.per_class, a.epochs, a.accum, a.lr, a.save_every = 4, 3.0, 4, 2e-4, 10_000
    samples = train_samples(index, a.kind, a.conditions, a.per_class, a.seed)
    if not samples:
        sys.exit(f"no training images for {a.kind}/{a.conditions} in {data}")
    n = len(samples)
    total_samples = int(round(a.epochs * n))
    total_steps = max(1, total_samples // a.accum)
    out = Path(a.out or ("/kaggle/working" if Path("/kaggle/working").exists() else "out") + f"/adapter_{a.kind}" + ("_fit" if a.fit_test else ""))
    out.mkdir(parents=True, exist_ok=True)
    print(f"{a.model} | {a.kind} | {n} clips | {a.epochs} epoch(s) = {total_samples} samples = {total_steps} optimizer steps | 4bit {not a.no_4bit} | out {out}", flush=True)

    model, processor = zs.load_model(a.model, "float16", not a.no_4bit)
    model = attach_lora(model, a.rank, a.alpha, a.dropout, a.lora_vision, not a.no_4bit, not a.no_grad_ckpt)
    model.train()
    ids = zs.name_token_ids(processor.tokenizer, prompts["labels"])
    target_tok = {c: processor.tokenizer.encode(prompts["labels"][c].lower(), add_special_tokens=False)[0] for c in CLASSES}
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda")
    state_file = out / "state.pt"
    step = load_state(state_file, model, opt, scaler) if state_file.exists() else 0
    if step:
        print(f"resumed at optimizer step {step}", flush=True)

    log_path = out / "train_log.csv"
    if not log_path.exists():
        log_path.write_text("step,loss,acc,lr,seconds_per_sample,nonfinite\n")
    g = step * a.accum
    nonfinite = seen = 0
    roll_loss, roll_acc, roll_n = 0.0, 0, 0
    first_epoch_losses: list[float] = []
    last_epoch_losses: list[float] = []
    t_block = time.time()
    opt.zero_grad(set_to_none=True)
    accum_ok = 0
    while g < total_samples:
        epoch, pos = sequence_position(g, n)
        i = epoch_order(n, a.seed, epoch)[pos]
        clip_id, rel, label = samples[i]
        img = Image.open(data / rel).convert("RGB")
        prompt = zs.build_name_prompt(prompts, a.kind, sample_option_order(a.seed, epoch, i))
        with torch.autocast("cuda", dtype=torch.float16):
            logits = forward_logits(model, processor, img, prompt)
        loss = torch.nn.functional.cross_entropy(logits[None], torch.tensor([target_tok[label]], device=logits.device))
        seen += 1
        if not torch.isfinite(loss):
            nonfinite += 1
            if seen >= 20 and nonfinite / seen > 0.05:
                sys.exit(f"STOP: {nonfinite} of {seen} losses were not finite (fp16 instability). Lower --lr, or try --no-4bit.")
        else:
            scaler.scale(loss / a.accum).backward()
            accum_ok += 1
            lv = float(loss.detach())
            pred = max(CLASSES, key=lambda c: float(torch.logsumexp(logits[ids[c]], 0)))
            roll_loss += lv
            roll_acc += pred == label
            roll_n += 1
            (first_epoch_losses if epoch == 0 else last_epoch_losses).append(lv)
        g += 1
        if g % a.accum == 0:
            for grp in opt.param_groups:
                grp["lr"] = lr_at(step, total_steps, a.lr)
            if accum_ok:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt)
                scaler.update()
            opt.zero_grad(set_to_none=True)
            accum_ok = 0
            step += 1
            if step % 5 == 0 and roll_n:
                sps = (time.time() - t_block) / max(1, 5 * a.accum)
                with open(log_path, "a") as f:
                    f.write(f"{step},{roll_loss / roll_n:.4f},{roll_acc / roll_n:.3f},{opt.param_groups[0]['lr']:.2e},{sps:.2f},{nonfinite}\n")
                mem = " ".join(f"{torch.cuda.max_memory_allocated(d) / 1e9:.1f}" for d in range(torch.cuda.device_count()))
                print(f"step {step}/{total_steps} loss {roll_loss / roll_n:.3f} acc {roll_acc / roll_n:.2f} {sps:.2f} s/sample peak GB {mem}", flush=True)
                roll_loss, roll_acc, roll_n, t_block = 0.0, 0, 0, time.time()
            if step % a.save_every == 0:
                save_state(state_file, model, opt, scaler, step)
    import json

    model.save_pretrained(out / "final")
    (out / "run_info.json").write_text(json.dumps(dict(model=a.model, load_4bit=not a.no_4bit, kind=a.kind, conditions=a.conditions, epochs=a.epochs, samples=n, steps=step, lr=a.lr, rank=a.rank, lora_vision=a.lora_vision, seed=a.seed, prompts=prompts["name"]), indent=1), encoding="utf-8")
    save_state(state_file, model, opt, scaler, step)
    print(f"saved adapter {out / 'final'}", flush=True)
    if a.fit_test:
        f0 = sum(first_epoch_losses) / max(1, len(first_epoch_losses))
        fl = sum(last_epoch_losses[-n:]) / max(1, len(last_epoch_losses[-n:]))
        peaks = [torch.cuda.max_memory_allocated(d) / 1e9 for d in range(torch.cuda.device_count())]
        ok = nonfinite == 0 and fl < f0 and max(peaks) < 14.5
        print(f"FIT TEST {'OK' if ok else 'CHECK'}: first-epoch loss {f0:.3f} -> last-epoch loss {fl:.3f}, non-finite {nonfinite}/{seen}, peak memory per GPU (GB) {[round(x, 1) for x in peaks]}")


if __name__ == "__main__":
    main()
