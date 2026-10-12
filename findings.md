# findings.md – living decision and failure log

Every design decision and failure mode goes here, newest at the bottom. Format: number, date, short title, what was decided or found, why, and what it means for the results.

---

## 001 – 2026-10-12 – Design v1 locked

- Name: V01 – VLM Faultline (repo and folder slug v01-vlm-faultline; Python package stays `faultline`). Single synthetic camera view, five classes (normal + stuck item, gate stuck open, leak/spray, blocked lens).
- Vision input: 8 frames sampled uniformly, tiled 2×4 into a contact sheet, timestamps drawn at sheet-build time (after degradation). Single frame is the control condition.
- Student CNN and supervised baseline share one architecture; baseline trains on hard labels, student on VLM soft probabilities with temperature (distillation).
- Splits by simulation seed. One scene configuration is held out of all training (domain-shift test). A held-out seed alone was rejected: it is only another draw from the same distribution.
- Manifest records severity, onset_s and fault_region_px from day one. The time-to-detect table is optional and decided after M2.
- Human baseline: blind pass on shuffled contact sheets before the label-aware check. Reported as "author, single rater".
- Cost target $0. vLLM serving and GGUF conversion moved to L01.

## 002 – 2026-10-12 – Environment decisions

- Project and conda env live on an external 2.5" spinning disk (WD My Passport). Expect slow small-file operations (env install, first imports). Sequential reads are fine, so frames are packed into large files before training (Phase 4).
- Paths: no hard-coded drive letters. Everything resolves relative to the repo; optional override via FAULTLINE_DATA.
- Docker is skipped for Phases 0–7. It is only needed for Triton, Frigate and the Terraform Docker provider (Phases 8–10). Decide at Phase 8 once free space is known.
- Open for Phase 1: clip length, fps, resolution, tile size, severity/onset distributions, held-out configuration, student architecture, final set sizes.
- To verify: Qwen2.5-VL-3B licence; QLoRA fit and fp16 stability on Kaggle T4×2; answer-token scoring for soft targets; ffmpeg noise seed behaviour.

## 003 – 2026-10-12 – Renamed to v01-vlm-faultline

- Slug follows the m01-navi-opt pattern (code + short name): folder, conda env and GitHub repo are v01-vlm-faultline. Title: V01 – VLM Faultline. The Python package stays `faultline`.
- Done before the conda env was created, so no environment had to be rebuilt.
