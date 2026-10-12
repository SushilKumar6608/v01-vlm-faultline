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

- Phase 2: every test clip checked on its contact sheet; fault_region_px recorded; clean dataset frozen.
- Phase 3: visibility audit on clean and degraded sheets at final resolution; blind human baseline.
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
