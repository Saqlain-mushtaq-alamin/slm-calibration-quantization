# Phase 5 — English Control Fine-Tuning: Run Guide

How to run, verify, and interpret Phase 5 of
*Quantization × Calibration × Selective Prediction for Sub-3B SLMs on Bengali
Hate-Speech Classification under Domain Shift*.

**Status: code complete and tested (208 passing tests across the repo). Ready to run.**
Phase 5 mirrors Phase 4 across all 16 cells (same models, seeds, and training hyperparameters), but with `lang=en` and the matched English control dataset (`data/processed/english_control_matched.csv`, split `loco_lgbtq`).

---

## 1. What Phase 5 produces

Every `(model, seed)` cell of the matrix produces three artifacts:

| Artifact | Path | Why it matters |
|---|---|---|
| LoRA adapter | `outputs/models/phase5/<run_name>/adapter` | Reproducible delta from the base model for English control |
| Merged BF16 checkpoint | `outputs/models/phase5/<run_name>/merged` | **English FP reference for Phase 6 quantization** — used for language × quantization interaction test (RQ3, H3) |
| Per-example logits | `outputs/logits/phase5/<run_name>/bf16__{id_val,id_test,ood_test}.parquet` | Stored logits on all 3 partitions enabling delta metrics without re-inference |

Plus:
- `outputs/runs/phase5/<run_name>.json` — run manifest for each cell with provenance (commit, config hash, seed, split SHA-256, timing, VRAM, library versions, gates).
- `outputs/results/english_control_matching.json` — matching verification report (T5.1).
- `outputs/results/phase5_summary.json` — aggregate gate verdicts (T5.1, T5.2, T5.3, T5.4) + seed variance (T4.9).
- `outputs/results/phase5_metrics.csv` — full task, calibration, and selective metrics per (model, seed, eval set).

---

## 2. Prerequisites & Verification

Confirm environment health and unit tests:

```powershell
& .venv\Scripts\python.exe -m pytest tests/test_phase5.py -v
```

Required input files (already generated and verified):
- `data/processed/english_control_matched.csv` (50,281 rows, size within 10% of Bengali, class prior 0.4804 matching Bengali 0.4800).
- `data/splits/english_control_loco_splits.json` + `.sha256` (13 LOCO folds, primary fold: `loco_lgbtq`).
- `outputs/results/shift_magnitudes.csv` (contains MMD domain shift documentation for T5.4).

---

## 3. The Run Matrix

Config: `configs/baseline/english_control.yaml`. Language: `en`, dataset: `toxigen`, primary split: `loco_lgbtq`.

| Model key | Canonical HF id | Mirror used | Params | Seeds | Cells |
|---|---|---|---|---|---|
| `qwen2_5_0_5b_instruct` | `Qwen/Qwen2.5-0.5B-Instruct` | — | 0.50B | 13, 42, 1234, 2024, 7 | 5 |
| `qwen2_5_1_5b_instruct` | `Qwen/Qwen2.5-1.5B-Instruct` | — | 1.56B | 13, 42, 1234, 2024, 7 | 5 |
| `llama3_2_1b_instruct` | `meta-llama/Llama-3.2-1B-Instruct` | `unsloth/Llama-3.2-1B-Instruct` | 1.25B | 13, 42, 1234 | 3 |
| `gemma2_2b_it` | `google/gemma-2-2b-it` | `unsloth/gemma-2-2b-it` | 2.64B | 13, 42, 1234 | 3 |
| | | | | **Total** | **16** |

All runs use canonical run names prefixed with `en_`:
e.g. `en_qwen2_5_0_5b_instruct_lora_bf16_s13`, `en_llama3_2_1b_instruct_lora_bf16_s42`, etc.

---

## 4. How to Run Phase 5

### Step 4.1: Inspect Matrix Status (Zero Cost)
To see the matrix and status of all 16 cells:
```powershell
& .venv\Scripts\python.exe scripts/03_finetune.py --english --status
```
or dry-run:
```powershell
& .venv\Scripts\python.exe scripts/03_finetune.py --english --dry-run
```

### Step 4.2: Fast Smoke Rehearsal (Recommended First Run)
Tests the complete training, merging, logit extraction, and validation pipeline on capped data (64 train / 32 eval samples).
Isolates artifacts to `outputs/models/phase5_smoke`, leaving production paths clean:
```powershell
& .venv\Scripts\python.exe scripts/03_finetune.py --english --smoke --models qwen2_5_0_5b_instruct --seeds 13
```

### Step 4.3: Run a Single Cell (e.g. Qwen 0.5B seed 13)
```powershell
& .venv\Scripts\python.exe scripts/03_finetune.py --english --models qwen2_5_0_5b_instruct --seeds 13
```

### Step 4.4: Run a Single Model Across All Seeds
```powershell
& .venv\Scripts\python.exe scripts/03_finetune.py --english --models qwen2_5_0_5b_instruct
```

### Step 4.5: Run the Full 16-Cell Matrix
```powershell
& .venv\Scripts\python.exe scripts/03_finetune.py --english
```
*Note: You can also explicitly specify `--config configs/baseline/english_control.yaml`.*

---

## 5. Resumability & Interruption Handling

The orchestrator includes robust resumability:
- **Ctrl+C / Power Loss safe**: Checkpoints and Parquet files are written atomically. Interrupted runs can be resumed simply by re-running the exact same command.
- **Completed runs are automatically skipped**: If a cell succeeded with matching config hash and split SHA-256, it is skipped without re-running training or inference.
- To force re-running completed runs: add `--force`.

---

## 6. Acceptance Tests & Validation Conditions

Phase 5 evaluates the following gates:

| Test ID | Test | Pass Condition | Implementation |
|---|---|---|---|
| **T5.1** | English control matching | Dataset size within 10% of Bengali, class prior within 2%, ≥3 shift domains | `verify_english_control_matching()` (passes with 0.68% size diff, 0.035% prior diff, 13 domains) |
| **T5.2** | All English runs complete | Same pass criteria as Phase 4 (loss decreases, no NaN, checkpoints saved, macro-F1 > 0.55, covers both classes, valid probs) | Checked across all succeeded cells |
| **T5.3** | English F1 > Bengali F1 | Expected; documents English as higher-resource language | `check_english_vs_bengali_f1()` compares against Bengali `phase4_summary.json` |
| **T5.4** | Shift magnitude documented | MMD computed for English source-to-target pairs | `check_english_shift_magnitude_documented()` verifies English MMD in `outputs/results/shift_magnitudes.csv` |
