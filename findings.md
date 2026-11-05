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
- Caveats: one rater who is also the author, knew the classes were balanced, and had the decision checklist; one stuck_item sheet (sheet_001, assisted) was labelled with help and counts as correct. This shows the faults are visible, not how hard the task is.
- Decision: no sim change. The low-severity stuck item stays as the faintest case on the severity axis and is reported per severity bin. Revisit if the VLM or CNN results show it is at chance in the high bin too.
- Next: Phase 2 (bulk generation, manifest, seed-based splits). Final degradation recipe frozen in Phase 3, followed by a shorter repeat of the blind audit.

## 007 – 2026-10-12 – Phase 2 design decisions (dataset v1)

- Sizes kept: train 1,500 / val 250 / test 500 / held-out 200. Held-out is not enlarged; the README states "wide intervals, about 13 clips per bin" and treats it as a qualitative domain-shift result.
- Flagged clips are regenerated with the same split, class, scene and exact severity, only the seed changes (reserve block 500000+), so the stratification does not drift. Originals go to `quarantine/` with the reason logged in `configs/replacements_v1.csv`.
- The tile count catches spatial invisibility only. The sheet_006 miss was generated after the 2.25 s floor and had 3–4 halted tiles, so tile counts are not a sufficient floor for stuck_item. A targeted blind sweep (24 stuck_item sheets at severity 0.15–0.45, early and late onset, mixed with 24 normals) sets the stop-duration floor and an optional stuck-specific onset cap before the full render. Detection by halted tiles and by onset is reported with Wilson intervals.
- Human review in Phase 2 is a middle option: automated checks on all clips, worst-case montages, and a random 10 % label-aware spot check of the test split (50 sheets). Stopping rule: 3 or more not labelable means review the whole test split; 0 of 50 gives an upper 95 % bound of about 6 %. The label-aware check (are labels right) and the Phase 3 blind audit (can a human see the fault) are separate gates.
- Reproducibility: the manifest records ffmpeg, OpenCV, numpy, Python versions and git commit per clip. Frame determinism across machines is tested with `scripts/frame_hash.py`.
- Pipeline robustness: atomic tmp-file rename and sidecar-last completion marker; resume trusts a clip only if the sidecar sha256 matches. Spec validation runs before any rendering (duplicates, counts, null severity for normals, severity inside its bin, scene sets, seed blocks).
- New manifest fields: `visibility_rule`, `post_onset_tiles`, `halted_tiles`, `frozen_tiles`, `neighbor_gap_at_onset_px` (stuck_item halting against a neighbour; report-only warning).

## 008 – 2026-10-15 – Stuck-item sweep: floor kept

- Targeted blind pass (author, single rater, degraded v1): 24 stuck_item sheets at severity 0.15–0.45 (8 per sub-range, half early onset ≤2.5 s, half late ≥2.75 s) mixed with 24 normals. Detection 22/24 = 0.92 (95 % CI 0.74–0.98); false alarms 0/24 (CI 0–0.14).
- By stop duration: <2.6 s 16/18 = 0.89 (CI 0.67–0.97); 2.6–3.0 s 3/3; ≥3.0 s 3/3. By halted tiles: 3 tiles 13/14, 4 tiles 9/10. By onset: early 11/12, late 11/12.
- Misses: stop 2.44 s (4 halted tiles, onset 3.0 s) and stop 2.50 s (3 halted tiles, onset 2.38 s). With the earlier sheet_006 miss (severity 0.26) the faint end of the axis is where a human occasionally fails.
- Decision: keep STUCK_MIN_STOP_S = 2.25 s and STUCK_ONSET_MAX_S = 3.5 s (no config change). The 80 % criterion is met overall and in the lowest bin, onset made no difference, and keeping the faint cases keeps the severity axis honest. Results are reported per severity bin; low-severity stuck items are expected to be the hardest class and the README says so.
- Caveats: 18 sheets in the lowest bin (interval 0.67–0.97), so no measured 80 % claim there; rater is the author and knows the signature; single rater.
- Next: 40-clip pilot render on the laptop (timing, manifest, integrity check), then the full render.

## 009 – 2026-10-15 – Phase 2 gate: dataset v1 rendered and checked

- 2,450 clips rendered (train 1,500 / val 250 / test 500 / held-out 200), about 0.9 s per clip wall with 16 workers.
- Automated checks on all clips: integrity OK, 0 visibility flags, 0 replacements (`configs/replacements_v1.csv` empty). 50 clips fully decoded.
- Worst-case montages (one per fault class) reviewed by the author: the faintest cases still show their fault.
- Label-aware spot check, 50 random test sheets: 0 not labelable; upper 95 % bound on the failure rate about 5.8 %. Author, single rater.
- This is not a detectability test: the Phase 3 blind audit is the separate gate.

## 010 – 2026-10-22 – Phase 3: blind audit passed, degradation recipe frozen

- Blind audit (author, single rater): 100 fresh degraded sheets (seed block 950000+, never in train/val/test/held-out), 20 per class, severities stratified per class, scenes A1–A3. Pass criteria were fixed before labelling.
- Result: accuracy 96/100 (Wilson 95 % CI 0.90–0.98). Recall: normal 20/20, spray 20/20, blocked_lens 20/20 (CI 0.84–1.00 each); stuck_item 18/20 and gate_open 18/20 (0.90, CI 0.70–0.97). False alarms on normal 0/20. All three gates passed (accuracy at least 85 %, no class below 70 %, false alarms at most 10 %).
- Four misses, all labelled normal: stuck_item at severity 0.23 and 0.76 (A3); gate_open at severity 0.95 (A2), twice. The two gate misses were re-rendered from their seeds: the labels are correct. At high severity the arm stays open from t = 2.25 s onward, which looks like the scheduled opening at t = 1.5 s, so it reads as normal unless the schedule is applied (arm open in 6 of 8 tiles instead of the scheduled 2). This is the state-versus-schedule design working as intended, not a generator error. Consequence: gate_open needs schedule knowledge, so zero-shot prompts must state the schedule (fixed_v1). The two stuck_item misses were not re-examined individually and are consistent with the faint or late cases in finding 008.
- Decision: recipe frozen as degradation_v1 (`configs/degradation_v1.yaml`, parameters unchanged from the provisional recipe, pinned by `tests/test_degrade_recipe.py`). No tuning after a pass, so the benchmark is not made easier after the fact. Degraded copies are derived from the clean clips, so dataset v1 is unchanged.
- Caveats: the rater is the author and knows the fault signatures; n = 20 per class gives wide intervals; severity bins hold 6–7 sheets each.

## 011 – 2026-10-23 – Phase 5: CNN baseline (M1)

- Setup: small CNN from scratch (95 k parameters for one tile, 192 k for the 8-tile sheet), trained on clean training clips only, 30 epochs, 3 seeds, checkpoint chosen on clean validation. Evaluated on clean and degraded val, test and held-out D1. Details in `docs/design.md`, numbers in `results/cnn_baseline/`. Each seed is shown separately; the table below is the mean over seeds with the range.
- Clean test accuracy: sheet 99.9 % (99.8–100), single frame 96.7 % (96.0–97.2). In every seed the sheet is right on clips where the single frame is wrong (14, 15, 20 clips) and never the reverse (paired McNemar, descriptive). The gain is entirely in normal and stuck_item: the single frame confuses them (stuck_item 88–98 % recall, normal 89–93 %), the sheet does not. gate_open, spray and blocked_lens are 99–100 % from one frame. This matches the prediction (static faults need no sheet, temporal faults do), with one caveat: the single-frame control is the last tile, t = 5.25 s, when the scheduled gate is vertical, so any tilted arm there is a fault. gate_open is therefore easy from one frame in this benchmark; a different single tile would change that class.
- Clean data is close to saturated for a trained model, so the clean results do not separate good models. The informative conditions are degraded and held-out.
- Degraded test accuracy (models trained on clean only): sheet 62 % (31–81), single 57 % (36–76). The seeds disagree by up to 50 points and which input wins flips between seeds (single better in seed 0, sheet better in seeds 1 and 2), so no sheet-versus-single claim is made for degraded data. The McNemar p-values within one seed do not account for this seed-to-seed variation.
- Failure pattern on degraded data: every seed calls the noisy, blurred frames blocked_lens (blocked_lens recall 100 %, precision 22–55 %). Normal clips are almost never recognised (recall 0–27 % sheet, 1–14 % single), so false alarms are 73–100 % while fault-detected stays near 100 %. A clean-trained CNN detects "something is wrong" on every degraded clip and is useless as an alarm.
- Held-out D1 (dim warm light, diagonal belt): about 20 % everywhere (chance level) except one seed's sheet model on clean clips (29.5 %). The models predict blocked_lens for the whole domain. From-scratch CNNs with only mild brightness jitter do not transfer to D1 at all; this is the floor the vision-language models are compared against.
- What this fixes for the rest of the project: the in-distribution clean test is solved by a tiny model, so a zero-shot or fine-tuned VLM cannot be compared to it on clean accuracy alone. The comparison that matters is degraded and D1 performance, false-alarm rate, and per-class behaviour.
- Caveats: three seeds is a small sample of training randomness; the checkpoint was chosen on clean validation data; no pretrained weights and no photometric augmentation beyond ±10 % brightness and contrast; held-out intervals are wide (40 clips per class). A CNN trained on degraded (or mixed) clips was not run and would be the fair "trained for this condition" reference; it is an optional extra, not part of M1.

## 012 – 2026-10-23 – Phase 6 start: model licence check changes the primary model

- Read the model cards and the LICENSE file. Qwen2.5-VL-3B-Instruct (the primary model in the approved design) is under the Qwen Research License: non-commercial, "research or evaluation purposes only", and derived models that are redistributed need the notice "Built with Qwen" or "Improved using Qwen". The design plans to publish a fine-tuned adapter, so that licence would apply to the published artefact.
- Qwen3-VL-4B-Instruct, Qwen3-VL-2B-Instruct, Qwen2.5-VL-7B-Instruct and SmolVLM2-2.2B-Instruct are listed as Apache 2.0.
- Decision: primary model Qwen3-VL-4B-Instruct; Qwen3-VL-2B and SmolVLM2-2.2B as comparison models (a size comparison inside one family, and a second family). Design section 2 and the VERIFY list updated in the Phase 6 addendum.
- Caveats: licence terms are taken from the model cards at this date and are not legal advice; Qwen3-VL needs a recent transformers release (4.57 or later); fp16 on a T4, answer-letter scoring and speed are still to be verified by the smoke test.

## 013 – 2026-10-23 – Phase 6: first zero-shot validation run shows answer-letter position bias, scoring changed (prompts_v2)

- Setup: Qwen3-VL-4B-Instruct, fp16 on a Kaggle T4, prompts_v1 (options lettered A–E, answer read from the first token), validation split only (250 clips per condition and input). No test or held-out clips were run.
- Result: sheet clean 48.8 %, sheet degraded 36.8 %, single clean 57.2 %, single degraded 40.4 %. In all 1,000 answers the model never chose A (normal) or B (stuck_item); on the 50 normal clips the mean probability of "normal" was 0.004. Everything that was not clearly spray or blocked_lens became C (gate_open). Letter token ids were checked (A–E are the distinct tokens 32–36), so this is not a lookup error.
- Position check: with the letters in reverse order (first 40 clean-sheet validation clips) the model chose "normal" for 10 of 12 normal clips and was 30/40 (75 %) correct overall. Only the letters moved, so the v1 numbers mostly measure letter position, not what the model sees. Reversing five options leaves C in place, so this check shows the effect for normal clips but does not isolate every class.
- Also visible in the reversed run: 6 of 7 stuck_item clips were called normal. That may be real difficulty (one stopped item among moving ones) and is not explained by letter bias alone.
- Decision: v1 results are kept as a recorded finding and are not used for the sheet-versus-single comparison. prompts_v2 (`configs/prompts_v2.yaml`): the answer is a one-word label read from the first answer token (lower case and capitalised variants pooled), each clip is scored under two option orders in which every class changes position, and the class probabilities are averaged. Scene description, gate schedule and option wording are unchanged. The change was made after seeing validation results only; v2 is checked on validation before it is frozen and run once on test and held-out.
- Caveats: the reversed check used 40 clips of one input and condition; averaging two orders reduces but does not remove position effects; model ranking should not rest on a prompt that was adjusted after seeing the model fail.
- prompts_v2 validation result (same model, 250 clips per condition and input): sheet clean 51.6 %, sheet degraded 34.0 %, single clean 57.6 %, single degraded 34.4 %. The two option orders agree on 93–97 % of clips, so position bias is no longer the explanation; the model now answers consistently, and consistently wrong on some classes. Sheets: stuck_item and gate_open clips are called normal in 100 % of cases (50/50 each, clean and degraded); spray and blocked_lens are found (spray 48/50 clean, blocked_lens 31/50 clean). Single frames: normal, stuck_item and gate_open clips are all called gate_open (150/150 on clean). On degraded data spray and blocked_lens recognition falls and the model calls most of them normal (sheet) or gate_open (single).
- Reading: at native size the model finds faults that have a distinctive appearance (spray, a dark smudge) on clean images and does not find faults that need a closer look at the gate arm or a comparison across tiles. A single 320x180 frame reaches the model as only about 60 visual tokens, which may be too coarse to tell a closed gate from an open one; this has not been tested. Next check, on validation only: the same prompts with the images enlarged (`--upscale`). If that does not help, the zero-shot result is reported as it stands: a floor that the fine-tuned model has to beat.
- Resolution check (validation, first 60 clean clips per input, images enlarged 2x, one option order): single frame 56.7 %, sheet 40.0 %. Same failure pattern as at native size (single: normal, stuck_item and gate_open all called gate_open; sheet: everything called normal, blocked_lens 0/7). Enlarging the image does not help, so image size is not the cause. No further prompt or input changes were made after this.
- Frozen run, prompts_v2, native size, two option orders averaged, test (500 clips) and held-out D1 (200 clips), 2,800 answers, no invalid answers, answer-word probability mass at least 0.999 on every answer. Test accuracy (Wilson 95 % CI): clean sheet 51.2 % (47–56), clean single 57.2 % (53–61), degraded sheet 33.6 % (30–38), degraded single 35.6 % (32–40). Held-out D1: clean sheet 43.0 %, clean single 52.5 %, degraded sheet 29.0 %, degraded single 33.5 %. Chance is 20 %.
- Per-class (test, clean): spray found 97 % (sheet) and 100 % (single), blocked_lens 59 % and 86 %; gate_open and stuck_item 0 % in both inputs. Sheet answers are "normal" for every gate_open and stuck_item clip; single-frame answers are "gate_open" for every normal, gate_open and stuck_item clip (so single-frame normal recall is 0 %). Degraded: spray 58–67 %, blocked_lens about 10 %.
- Reading: zero-shot Qwen3-VL-4B with a written task description finds the faults that have a distinctive appearance on clean images and nothing that needs a close look at the gate arm or a comparison across frames. The sheet does not help it. The single-frame score is not a sheet-versus-single result: its accuracy comes from always answering gate_open plus finding spray and blocked_lens, and it flags every normal clip as a fault. The zero-shot numbers are the baseline a fine-tuned model has to beat.
- Caveats: prompts_v2 was written after seeing validation failures of prompts_v1 (position bias); one model, one prompt wording; fp16 on a T4; the held-out scene is a single scene.

