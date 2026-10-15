# V01 – VLM Faultline

Synthetic conveyor / machine-hall simulation with four fault types, used to measure how well vision models detect each fault from short clips, and what breaks first.

**Status:** Phase 2 complete (dataset v1 generated and checked). No model results yet.

**Central question:** do short clips (8 frames on a contact sheet) beat a single frame, and for which faults? Prediction to test: static faults need no sheet, temporal faults do.

## Results

_To be filled after M1. This section will lead with the per-class results tables and the known-failure cases, not with the tool list._

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
