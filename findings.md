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

## 004 – 2026-10-12 – Phase 1 review decisions

- Gate fault redefined: the normal gate follows one fixed open/close schedule (open 1.0–2.0 s and 4.0–5.0 s, schedule id `fixed_v1`). `gate_open` stays open from 2.0 s on. Without a fixed schedule the fault is undetectable from the clip alone.
- Stuck item halts in a zone (25–60 % of belt width) upstream of the gate, minimum stop 1.5 s, so the halt is unambiguous and survives 0.75 s sampling.
- Onset is random in [1.0, 3.5] s so at least 3 sampled frames follow it. Gate onset is fixed 2.0 s, lens onset 0.0.
- Phase 1 preview is a visibility gate: the sim is tuned until faults survive tile scale and a deliberately harsh provisional degradation. The final recipe is frozen in Phase 3 and the audit is repeated.
- Models train on clean clips only and are evaluated on clean and degraded.
- Contact sheet reading order: time left to right, top row first. Single-frame control is the last tile (t = 5.25 s).
- Human audit load reduced. Held-out and per-class tables report n and Wilson intervals.
- fps changed 15 → 16 so sample times fall on whole frames (every 12th frame, 0.75 s).
- ffmpeg finding: the `noise` filter's `all_seed` defaults to -1 (random) in the build checked, so unseeded noise is not reproducible. The seed is derived from the clip seed.
- Phase 1 observations: `fault_region_px` for `gate_open` is the union of actual and scheduled arm, so it does not grow with severity (tests cover spray and lens only). Stuck-item faults are subtle in a single tile by design; they need the sheet. Low-severity spray is small (~250–400 px at tile scale) and is the one to watch in the human audit.

## 005 – 2026-10-12 – Phase 1 preview review: fixes

Found in the first preview review (outside reviewer plus my own numerical checks). Changes:

- **Stuck item floor.** Counted tiles where the item is actually halted (100 seeds per level): severity 0.1 gave exactly 2 tiles (one frozen pair), mean 2.0; 0.2 gave 2.4; 0.5 gave 3.3; 0.9 gave 4.5. The floor was perceptually borderline. Minimum stop raised 1.5 s → 2.25 s (≥3 halted tiles). The blind audit decides if that is enough.
- **Metric bug.** For `stuck_item`, `expected_visible_tiles` came from the pixel mask, which also covers displaced downstream items, so it overstated visibility (a clip with 2 halted tiles reported 6). It now counts halted tiles for this fault. `fault_region_px` is unchanged.
- **Blocked lens** was an ~80 % opaque near-black ellipse (a shortcut a model can learn). Now semi-transparent with a soft edge, textured opacity and defocused background underneath. Severity is still the covered area.
- **Degradation was not harsh.** Recipe v0 gave ~35 dB PSNR at tile scale, because sheet downsizing averages the noise away. Recipe v1 (320×180, blur σ=1.0, noise 36, CRF 38) gives ~28 dB measured on the sheet. Low-severity spray still shows. The final recipe is still frozen in Phase 3.
- **Gate arm colour** was nearly identical to the yellow item colour (232,200,40 vs 230,180,50). It is now magenta, and a test enforces a minimum RGB distance (>100) from every scene palette.
- **Gate schedule check (no change).** Normal clips are open at tiles 2 and 6 only (t = 1.5, 4.5 s). Gate-fault masks are zero at tiles 0–2, so no fault shows before onset.
- **Blind baseline** moved after these fixes and onto fresh unlabelled clips (seeds 900000+): `scripts/make_blind_sheets.py` and `scripts/score_blind.py`. The preview sheets cannot be blind (labels in filenames, already seen).
- Still open: low-severity lens and low-severity spray are the faintest signals; decide after the blind pass.

## 006 – 2026-10-12 – Blind baseline on the Phase 1 sim (author, single rater)

- 30 unlabelled degraded sheets (v1 recipe), 6 per class, shuffled, scenes A1–A3, seeds 900000+. Result: 29/30 correct. Per-class recall: normal 1.00 (n=6), gate_open 1.00, spray 1.00, blocked_lens 1.00, stuck_item 0.83 (5/6). Wilson 95 % intervals are wide (about 0.44–1.00 at best) with n=6 per class.
- The one miss: `stuck_item` labelled `normal`, severity 0.26 (low bin), scene A2. It is consistent with the measured floor: low-severity stuck items give about 2–3 halted tiles.
- Caveats: one rater who is also the author, knew the classes were balanced, and had the decision checklist; one stuck_item sheet (sheet_001, assisted) was labelled with help from Claude and counts as correct. This shows the faults are visible, not how hard the task is.
- Decision: no sim change. The low-severity stuck item stays as the faintest case on the severity axis and is reported per severity bin. Revisit if the VLM or CNN results show it is at chance in the high bin too.
- Next: Phase 2 (bulk generation, manifest, seed-based splits). Final degradation recipe frozen in Phase 3, followed by a shorter repeat of the blind audit.
