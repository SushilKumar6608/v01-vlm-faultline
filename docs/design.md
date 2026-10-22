# V01 – VLM Faultline: Design v1 (approved 2026-10-12)

Markdown copy of the approved Word design document, for use inside the repo.

## 1. Purpose

A fully synthetic conveyor / machine-hall simulation with four fault types, used to measure how well vision models detect each fault from short clips, and what breaks first. Built to show careful evaluation on a camera-based fault-detection problem, not to claim real-world performance.

Central question: do short clips (8 frames on a contact sheet) beat a single frame, and for which faults?

Headline result: per-class table (precision, recall, n, Wilson intervals) for four models, clean vs degraded, on the standard test set and a held-out scene configuration; accuracy-versus-severity curves; known failure cases.

## 2. Locked decisions

- Single camera view, fully synthetic, seeded (OpenCV 2D top-down).
- Classes: normal, stuck item, gate stuck open, leak/spray, blocked lens.
- Input: 8 frames uniform, 2×4 contact sheet, timestamps drawn at sheet-build time; single frame is the control.
- VLM: Qwen2.5-VL-3B-Instruct primary, SmolVLM2-2.2B fallback; fit check in Phase 5.
- Student and supervised baseline: same small CNN; hard labels vs VLM soft probabilities with temperature.
- Splits by seed; one scene configuration held out (domain shift).
- $0 spend. vLLM serving and GGUF moved to L01.
- Frigate recorder only (2-day timebox); fallback MediaMTX + ffmpeg segments.

## 3. Classes, severity, onset

| Class | Type | Severity | Onset |
|---|---|---|---|
| normal | – | none (nuisance variation still present) | – |
| stuck item | temporal | fraction of remaining clip the item stays stopped | item halts while others move |
| gate stuck open | temporal | fraction of closing travel that fails | when the gate should have closed |
| leak / spray | partly temporal | spray area × duration (px at tile size) | first spray frame |
| blocked lens | static | fraction of frame occluded | t = 0 |

Severity is continuous plus a low/mid/high bin. Onset time is stratified. Severity below the human-detectable floor is reported separately.

## 4. Manifest fields

clip_id, source_clip_id, seed, scene_config_id, class, severity, severity_bin, onset_s, fault_region_px, split (train/val/test/ood_config), degradation, ffmpeg_flag_id, sha256, sim_version, label_source ("synthetic"), visibility_ok, human_checked.

## 5. Splits

train/val/test: disjoint seeds within training configurations. ood_config: one scene configuration never seen in training. Targets: test 300–400 clips, ood_config ≥ 100, fault classes stratified over severity bins. Test and ood sets are frozen with checksums before any model runs.

## 6. Experiments

| Model | Inputs | Conditions |
|---|---|---|
| Supervised CNN baseline | single frame; sheet | clean/degraded; test/ood |
| Zero-shot VLM | single frame; sheet | clean/degraded; test/ood |
| QLoRA VLM | sheet (single frame if time) | clean/degraded; test/ood |
| Distilled student | sheet | clean/degraded; test/ood |

Reporting: per-class precision/recall with n and Wilson intervals (primary), confusion matrix, macro-F1, accuracy vs severity, static-vs-temporal delta, blind human baseline, student vs VLM latency, Triton benchmark (Phase 8). Time-to-detect optional after M2.

## 7. Data gates

- Phase 2 (label correctness, label-aware): automated checks on all 2,450 clips, worst-case montages, and a random 10 % label-aware spot check of the test split (50 sheets; stopping rule: 3 or more not labelable means review the whole test split). Clean dataset frozen after the gate.
- Phase 3 (human detectability, blind): the blind audit. It is a different gate from the Phase 2 spot check: the first checks that labels are right, the second that faults can be seen.
- Phase 3: visibility audit on clean and degraded sheets at final resolution; blind human baseline (see above).
- Degradation: two levels (clean, one degraded); ffmpeg flags in a table; noise seed from the clip seed; `-threads 1`; record ffmpeg version.

## 8. Phases

| Phase | Content | Gate |
|---|---|---|
| 0 | Scaffold, env, findings.md, design.md | |
| 1 | Simulator, four fault injectors, severity/onset, determinism tests | fix clip length, fps, resolution, tile size |
| 2 | Seeded generation, manifest, splits, held-out config, test-sheet check | clean dataset frozen |
| 3 | ffmpeg degradation, visibility audit, blind human baseline | data gate |
| 4 | Eval harness and CNN baseline | |
| 5 | Zero-shot VLM on Kaggle; QLoRA fit check | M1 – publish |
| 6 | QLoRA fine-tune; archive adapter on Hugging Face Hub | |
| 7 | Teacher soft probabilities; student distillation | |
| 8 | ONNX and Triton benchmark (needs Docker) | M2 |
| 9 | MediaMTX, Frigate or fallback; plan the 60–90 s demo video | |
| 10 | Terraform, budget alert, destroy | |
| 11 | README, dashboard, thumbnail, write-up | M3 |

Cut order: time-to-detect, Terraform polish, Frigate (keep fallback), QLoRA single-frame row, shrink ood_config. Never cut: visibility audit, single-frame control, per-class tables.

## 9. VERIFY

Qwen2.5-VL-3B licence; QLoRA fit and fp16 stability on Kaggle T4×2; answer-token scoring; Frigate config (current version); ffmpeg noise seed; Triton ONNX CPU backend and perf_analyzer flags; GCS free tier.

## 10. Out of scope

Real footage, multiple camera angles, 3D, tracking, real-time VLM inference, GGUF conversion and vLLM serving (L01), Vespa/Feast, Flyte/Ray.

## Phase 1 addendum (simulator spec)

- Clip: 6 s, 16 fps, 96 frames, 640×360. Samples at t = 0, 0.75, …, 5.25 s (frames 0, 12, …, 84). Tile 320×180, sheet 1280×360.
- Counterfactual rendering gives a noise-free fault mask per sampled frame. `fault_region_px` = max mask area at tile scale; `expected_visible_tiles` = sampled frames with ≥ 40 px (provisional).
- Random streams: [seed,1] world, [seed,2] fault, [seed,100+k] spray droplets, [seed,1000+k] sensor noise.
- Scenes: A1–A3 train, D1 (dim warm light, diagonal belt texture) held out.
- Planned source clips: train 1,500, val 250, test 500, held-out 200. One clip per seed; degraded copies inherit the split.
- Gate: `fixed_v1` schedule. Stuck item: halt zone and 2.25 s minimum stop (see findings 005). Severity bins: three equal bins over 0.1–1.0.
- Code: `faultline.sim` (config, scenes, plan, render, clip), `faultline.media`, `faultline.sample.sheet`, `faultline.degrade.provisional`; `scripts/preview_sim.py`.

## Phase 2 addendum (dataset v1)

- Pipeline: `configs/dataset_v1.yaml` → `specs` (one row per clip, no rendering) → validation → render (parallel, resumable, atomic) → `manifest.csv` → gate → freeze (`dataset-v1` tag).
- Sizes: train 1,500 / val 250 / test 500 / held-out D1 200 (300 / 50 / 100 / 40 per class). Held-out intervals are wide (about 13 clips per severity bin); the README says so. It is a qualitative domain-shift check.
- Seeds: disjoint blocks per split (100000, 200000, 300000, 400000); reserve block 500000+ for replacements; blind audits 900000+. Severity is stratified within each class and split (equal counts per bin, within ±1), scenes A1–A3 in equal thirds.
- Replacement rule: a clip that fails its visibility rule is replaced by a clip with the same split, class, scene and exact severity and a new seed from the reserve block. The original moves to `data/quarantine/`. Replacements are logged in `configs/replacements_v1.csv` (committed), so the effective spec is reproducible.
- Visibility rule per class (column `visibility_rule`): `blocked_lens`, `spray`: tiles with mask area ≥ 40 px; `gate_open`: tiles where the arm differs from the schedule; `stuck_item`: halted tiles and consecutive halted pairs (`frozen_tiles`). The tile count is a spatial proxy only. For stuck_item the floor (minimum stop duration, optional onset cap `STUCK_ONSET_MAX_S`) is set from the targeted blind sweep (severity 0.15–0.45, early and late onset) before the full render.
- Resume: each clip is written as `<id>.tmp.mp4` and renamed; the sidecar JSON (ground truth, sha256, build versions) is written last and marks the clip as done. A clip counts as done only if the sha256 in the sidecar matches the file.
- Reproducibility claim: frame content is deterministic from the seed given the same sim code and library versions. The manifest records ffmpeg, OpenCV, numpy and Python versions and the git commit per clip. MP4 bytes are only identical on the same ffmpeg build. `scripts/frame_hash.py` checks frame determinism across machines.
- Tests: spec built byte for byte from the config; committed `configs/specs_v1.csv` equals the rebuilt spec; ground-truth columns of a sample regenerate exactly (slow-marked test, `pytest -m slow`).

## Phase 3 addendum (audit and frozen recipe)

- Blind audit: 100 fresh degraded sheets (seed block 950000+, 20 per class, severities stratified), author as single rater. Pass criteria fixed beforehand: accuracy at least 85 %, no class recall below 70 %, false alarms on normal at most 10 %. Result in findings 010.
- Degradation recipe v1 is frozen in `configs/degradation_v1.yaml` (320×180, blur σ 1.0, noise 36, CRF 38, noise seed from the clip seed); `tests/test_degrade_recipe.py` pins it by hash. A change means a new recipe file and re-evaluating every model that used v1.

## Phase 4 addendum (evaluation harness)

- Degraded copies: `scripts/degrade_dataset.py` makes `data/degraded/<split>/<clip_id>.mp4` from the clean clips with recipe v1. Resumable and atomic; `data/degraded/recipe.json` refuses a mix of recipes; `data/degraded_manifest.csv` holds the sha256 of every file. Models train on clean clips only and are evaluated on clean and degraded.
- Inputs (`faultline.eval.inputs`): `sheet` (360×1280) or `single` (180×320, the last tile with its timestamp), from clean or degraded clips. Same pipeline as the Phase 3 audit sheets.
- Predictions file, one row per answer: `model, input, condition, split, clip_id, pred`. `pred` is one of the five classes or `invalid` (unparseable answer; counts as wrong and as no class). Each (model, input, condition, split) group must cover its split exactly unless partial results are explicitly allowed.
- Metrics (`faultline.eval.metrics`): per-class recall and precision with n and Wilson 95 % intervals; accuracy; macro recall; fault-detected rate and false-alarm rate (any fault class versus normal); recall per severity bin; accuracy per scene; confusion matrices; invalid rate.
- Sheet versus single frame: paired on the same clips, exact McNemar test, overall and per true class. The p-values are descriptive and not corrected for multiple comparisons.
- Reference models (`faultline.eval.models`): oracle (must score 100 %), always-normal (must score exactly 20 %), random, and a noisy oracle. `scripts/selftest_eval.py` runs them on the real manifest to check the scoring before any real model is scored.
- Output: `scripts/evaluate.py` writes CSV tables and `results.md` to `results/<name>/`; these are committed. Several predictions files can be combined into one comparison table.

## Phase 5 addendum (CNN baseline)

- Question answered: how far does a small CNN trained from scratch on the clean training clips get, with one tile versus all eight, on clean and degraded test and held-out clips. It is the floor the vision-language models have to beat, not a tuned model.
- Input: the same 8 tiles as the sheets, halved to 160×90 with area interpolation (`faultline.models.cache`). Single = last tile; sheet = all 8 in time order. The timestamp is in the pixels in both cases. Cached once per (condition, split) in `data/cache_tiles/` (not tracked).
- Model (`faultline.models.cnn`): four conv blocks applied to each tile with shared weights, a 1×1 reduction to 16 channels, tiles stacked along the channel axis in time order, a 3×3 conv, pooling, dropout, a linear layer. The single-frame model is the same network with one tile. About 0.09 M parameters (single) and 0.19 M (sheet).
- Training (`faultline.models.train`): AdamW, 30 epochs, batch 32, cosine schedule with 2 warm-up epochs, label smoothing 0.05, mild brightness and contrast jitter only (no noise, blur or flips, so robustness to degradation is not trained in). Clean training clips only. The checkpoint with the best clean validation accuracy is used. Three seeds (0, 1, 2); the report shows each seed and the spread.
- Crash safety: checkpoint after every epoch with atomic rename; data order and augmentation depend only on (seed, epoch); a resumed run gives the same weights as an uninterrupted one (tested). Finished steps (cache files, runs, prediction files) are skipped on re-run. CPU threads are capped (6), priority is below normal, sleep is blocked on Windows.
- Outputs: `results/predictions/cnn_s<seed>_<kind>.csv` (committed) and `results/cnn_baseline/` (tables, `results.md`, `seed_summary.csv`). Checkpoints stay in `data/runs/` (not tracked).
- Limits to state with the results: validation data chooses the checkpoint; three seeds is a small sample of training randomness; the CNN has no pretrained weights; held-out D1 intervals are wide.
