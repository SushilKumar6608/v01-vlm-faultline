# V01 – VLM Faultline

Synthetic conveyor / machine-hall simulation with four fault types, used to measure how well vision models detect each fault from short clips, and what breaks first.

**Status:** Phase 5 complete (M1: CNN baseline, clean and degraded, 3 seeds). Next: zero-shot vision-language models.

**Central question:** do short clips (8 frames on a contact sheet) beat a single frame, and for which faults? Prediction to test: static faults need no sheet, temporal faults do.

## Results

M1 – small CNN trained from scratch on clean clips, test split, mean over 3 seeds (range). Full tables in `results/cnn_baseline/results.md`, interpretation in `findings.md` (011).

| Input | Clean | Degraded | Held-out D1 (clean) |
|---|---|---|---|
| Single frame | 96.7 % (96.0–97.2) | 57 % (36–76) | 20 % |
| 8-frame sheet | 99.9 % (99.8–100) | 62 % (31–81) | 23 % (20–30) |

- On clean data the sheet fixes exactly the confusions the single frame has: normal versus stuck_item. Gate, spray and blocked-lens faults are found from one frame.
- A CNN trained on clean clips does not survive the degradation (it calls everything blocked_lens; false alarms 73–100 %) and does not transfer to the held-out scene (chance level). Seeds disagree strongly on degraded data, so no sheet-versus-single claim is made there.
- Vision-language model results follow in the next phases.

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
