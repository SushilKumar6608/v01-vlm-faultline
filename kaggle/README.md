# Zero-shot run on Kaggle (Phase 6)

Kaggle gives a free GPU (T4 x2 or P100) for a limited number of hours per week; check the current quota on your Kaggle profile before long runs. Everything here costs $0.

## 1. On the laptop: make the upload folder (about 10-15 minutes)

    python scripts\export_vlm_inputs.py --workers 4

This writes `data\kaggle_bundle\` (images, `index.csv`, `prompts_v1.yaml`, `run_zero_shot.py`). Re-running keeps finished images.

## 2. On kaggle.com: upload it as a private dataset

1. Datasets, New Dataset. Drag the contents of `data\kaggle_bundle` into the upload box (if the browser struggles with thousands of files, zip the folder first and upload the single zip: Kaggle unpacks zips).
2. Name it `faultline-vlm-inputs`, visibility Private, Create.

## 3. New notebook

1. Notebooks, New Notebook. Add Data: your `faultline-vlm-inputs` dataset.
2. Right-hand Settings: Accelerator = GPU T4 x2 (or P100), Internet = On.
3. Cell 1:

        !pip install -q -U "transformers>=4.57" accelerate pyyaml pillow

4. Cell 2, smoke test (a few minutes). It scores 40 validation sheets and 40 single frames:

        !python /kaggle/input/faultline-vlm-inputs/run_zero_shot.py --model Qwen/Qwen3-VL-4B-Instruct --splits val --conditions clean --limit 40

   Look for: the model loads, about 1-3 s per clip, no `STOP` line, and `letter_mass` close to 1 in `/kaggle/working/zeroshot_*_probs.csv` (the model really answered with a letter).
   If you see non-finite logits, rerun with `--dtype float32` or `--load-4bit` (needs `bitsandbytes`).

Send me the printed output and the first lines of the `_probs.csv` file. The full run (test and held-out, clean and degraded, single and sheet, 3,400 images per model) comes after the smoke test passes.

## Notes

- The runner is safe to stop and restart. Output goes to `/kaggle/working`; download it before the session ends, or use Save Version (Save & Run All) so it runs without the browser open and keeps its output.
- Models: Qwen/Qwen3-VL-4B-Instruct (Apache 2.0) first; Qwen/Qwen3-VL-2B-Instruct and HuggingFaceTB/SmolVLM2-2.2B-Instruct for comparison.
- `--reverse-options` runs the same prompt with the letters in the opposite order, to measure position bias.

## Second run: prompts_v2 (one-word answers, two option orders)

The images are already in the Kaggle dataset `faultline-vlm-inputs`. Only the small code files changed, so upload them as a second, tiny private dataset:

1. Make a folder `kaggle_code` containing `run_zero_shot.py` (from `kaggle\`) and `prompts_v2.yaml` (from `configs\`).
2. On Kaggle: Create > New Dataset, drag the two files, title `faultline-code`, Private, Create.
3. In the notebook: Add Input > Your Work/Datasets > `faultline-code`. Then run, with the path Kaggle shows for it:

        !python /kaggle/input/datasets/<your-user>/faultline-code/run_zero_shot.py --data /kaggle/input/datasets/bergen6608/faultline-vlm-inputs/kaggle_bundle --model Qwen/Qwen3-VL-4B-Instruct --splits val

4. Score it with `kaggle/score_val.py` (paste it into a cell or run it from the same dataset):

        !python /kaggle/input/datasets/<your-user>/faultline-code/score_val.py /kaggle/working/zeroshot_Qwen3-VL-4B-Instruct_v2_probs.csv

## Fine-tuning (QLoRA)

Training needs the training clips, which the first dataset does not hold. On your PC:

    python scripts\export_vlm_inputs.py --splits train --conditions clean --out data\kaggle_train --workers 4

(3,000 images; the folder is about 600 MB.) Upload `data\kaggle_train` as a second private dataset, `faultline-train-images`. Put the new `finetune_qlora.py` and `run_zero_shot.py` into `faultline-code2` (New Version) or a new code dataset.

In the notebook (T4 x2, Internet On, "Pin to original environment"), add the inputs `faultline-train-images`, `faultline-vlm-inputs` and the code dataset. Install:

    !pip install -q -U "transformers>=4.57" accelerate pyyaml pillow peft bitsandbytes

1. Fit test (5-10 minutes; prints FIT TEST OK or CHECK):

        !python <code>/finetune_qlora.py --fit-test --kind sheet --train-data <train images>/kaggle_train

2. Real run, one adapter per input kind (add `--epochs`, `--per-class` after reading the fit-test speed):

        !python <code>/finetune_qlora.py --kind sheet --train-data <train images>/kaggle_train --epochs 1

3. Score the adapter on validation, then test and held-out (the runner reads `run_info.json`, so it uses the same 4-bit base and the right input kind):

        !python <code>/run_zero_shot.py --data <vlm inputs>/kaggle_bundle --model Qwen/Qwen3-VL-4B-Instruct --adapter /kaggle/working/adapter_sheet/final --splits val

