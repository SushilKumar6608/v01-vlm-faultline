# V01 – VLM Faultline

Synthetic conveyor / machine-hall simulation with four fault types, used to measure how well vision models detect each fault from short clips, and what breaks first.

**Status:** Phase 5 complete (M1: CNN baseline, clean and degraded, 3 seeds). Phase 6 complete (zero-shot Qwen3-VL-4B baseline). Phase 7 (QLoRA fine-tuning of the VLM, one adapter for single frames and one for sheets) in progress; fine-tuned results are not in yet.

**Central question:** do short clips (8 frames on a contact sheet) beat a single frame, and for which faults? Prediction to test: static faults need no sheet, temporal faults do.

## Summary (test split, 500 clips per cell, chance 20 %)

| Model | Single frame, clean | Sheet, clean | Single, degraded | Sheet, degraded | Status |
|---|---|---|---|---|---|
| CNN from scratch (95–192 k parameters) | 96.7 % | 99.9 % | 57 % (36–76) | 62 % (31–81) | done, 3 seeds |
| Qwen3-VL-4B zero-shot | 57.2 % | 51.2 % | 35.6 % | 33.6 % | done |
| Qwen3-VL-4B + QLoRA adapter | – | – | – | – | training and scoring in progress |

What the numbers say so far: the clean test is solved by a tiny CNN, so the informative conditions are degraded and held-out data. A zero-shot VLM with a written task description finds spray and the dark lens smudge and misses gate-open and stuck-item clips. The fine-tuned row stays empty until the scored predictions exist; nothing is estimated.

## Results

M1 – small CNN trained from scratch on clean clips, test split, mean over 3 seeds (range). Full tables in `results/cnn_baseline/results.md`, interpretation in `findings.md` (011).

| Input | Clean | Degraded | Held-out D1 (clean) |
|---|---|---|---|
| Single frame | 96.7 % (96.0–97.2) | 57 % (36–76) | 20 % |
| 8-frame sheet | 99.9 % (99.8–100) | 62 % (31–81) | 23 % (20–30) |

- On clean data the sheet fixes exactly the confusions the single frame has: normal versus stuck_item. Gate, spray and blocked-lens faults are found from one frame.
- A CNN trained on clean clips does not survive the degradation (it calls everything blocked_lens; false alarms 73–100 %) and does not transfer to the held-out scene (chance level). Seeds disagree strongly on degraded data, so no sheet-versus-single claim is made there.

Zero-shot Qwen3-VL-4B-Instruct (fp16, T4), test split, 500 clips per cell (chance is 20 %). The prompt states the scene and the gate schedule; the answer is a one-word label read from the first answer token and averaged over two option orders (`findings.md` 013, `results/zero_shot/`).

| Input | Clean | Degraded | Held-out D1 (clean) |
|---|---|---|---|
| Single frame | 57.2 % (53–61) | 35.6 % (32–40) | 52.5 % |
| 8-frame sheet | 51.2 % (47–56) | 33.6 % (30–38) | 43.0 % |

- The model finds spray (97–100 %) and the dark smudge (59–86 %) on clean images and finds none of the gate-open or stuck-item clips. On sheets it calls every gate_open and stuck_item clip normal; on single frames it calls every normal, gate_open and stuck_item clip gate_open. The single-frame accuracy comes from that constant answer, not from reading the gate, so it is not evidence that single frames beat sheets.
- A first version of the prompt with lettered options was dominated by answer-position bias (49 % to 75 % accuracy when only the letter order changed); the final scoring removes that. Enlarging the images 2x did not change the result. Fine-tuned results will be compared against this baseline.

## Repository map

- `src/faultline/` – simulator, dataset generation, degradation recipe, evaluation code.
- `configs/` – frozen dataset spec, degradation recipe and the two zero-shot prompt files (`prompts_v1` letters, `prompts_v2` word labels).
- `scripts/` – dataset generation and checks, CNN baseline runner, VLM export, `compare_vlm.py` (accuracy with Wilson intervals, per-class tables, confusion matrices, paired McNemar test).
- `kaggle/` – zero-shot scoring (`run_zero_shot.py`), QLoRA fine-tuning (`finetune_qlora.py`) and validation scoring for a free Kaggle T4 x2 session; see `kaggle/README.md`.
- `findings.md` – dated log of every result and decision, including the ones that went against the prediction. `docs/design.md` – the design and its addenda.
- `results/` – prediction CSVs and summary tables.

## Honest framing

- All data is synthetic and labelled `synthetic` in the manifest. Synthetic faults are easier than real ones; nothing here claims real-world accuracy.
- Single camera view only. Out of scope: real footage, multi-camera, 3D, tracking, real-time VLM inference.
- Human baseline (when added) is one rater, the author.
- Held-out D1 has 40 clips per class (about 13 per severity bin), so per-bin intervals there are wide.

## Setup (Windows, Anaconda Prompt)

See `docs/design.md` for the full design. Quick start:

```
conda env create -p <DRIVE>:\conda-envs\v01-vlm-faultline -f environment.yml
conda activate <DRIVE>:\conda-envs\v01-vlm-faultline
pip install -e .
python scripts\check_env.py
pytest -q
```

Data location: by default `data/` inside this folder. To keep data elsewhere, set the environment variable `FAULTLINE_DATA`.
