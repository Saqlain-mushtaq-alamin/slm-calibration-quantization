# Phase 6 Run Guide: Post-Training Quantization — Track A (GPU / Hugging Face)

This document provides instructions for executing and verifying **Phase 6: Post-Training Quantization (Track A)** in accordance with [`planning/phases/phase_06_post_training_quantization_track_a_gpu_hf.md`](file:///d:/canvas/slm-calibration-quantization/planning/phases/phase_06_post_training_quantization_track_a_gpu_hf.md).

---

## 1. Overview & Architecture

Phase 6 quantizes the 19 fine-tuned base checkpoints produced in Phase 4 (Bengali) and Phase 5 (English control) across four Hugging Face / GPU quantization formats:
1. **`bnb_int8`** (BitsAndBytes 8-bit LLM.int8())
2. **`bnb_nf4`** (BitsAndBytes 4-bit NormalFloat)
3. **`gptq_int4`** (GPTQ 4-bit weight-only PTQ, 128 group size, act-order)
4. **`awq_int4`** (AWQ 4-bit weight-only PTQ; gracefully marked unavailable if `llmcompressor` is absent)

The orchestrator extracts per-example logit distributions across `id_val`, `id_test`, and `ood_test` splits, validates calibration safety gates, and evaluates accuracy and ECE.

### Validation Gates (T6.1 – T6.9)
- **T6.1**: BNB INT8 models load and evaluate on GPU.
- **T6.2**: BNB NF4 models load and evaluate on GPU.
- **T6.3**: GPTQ-4 models quantize and reload without error.
- **T6.4**: AWQ-4 completes or records `status: "unavailable"` without breaking the sweep.
- **T6.5 (Critical Gate)**: INT8 is near-lossless vs. BF16 FP reference:
  - Macro-F1 drop $\le 0.02$
  - Prediction agreement $\ge 0.95$
  - Mean probability shift $|\Delta p| \le 0.05$
- **T6.6**: All logits saved in Parquet and CSV manifests with complete column schemas.
- **T6.7**: Softmax probabilities sum to $1.0 \pm 10^{-4}$ for every row.
- **T6.8 (Zero-Leakage Assertion)**: Calibration samples ($N=512$, $L=256$, seed $42$) are drawn strictly from `train` IDs, completely disjoint from `id_val`, `id_test`, and `ood_test`.
- **T6.9 (Language Matching)**: Calibration language matches model language (Bengali for Phase 4, English for Phase 5).

---

## 2. Prerequisites & Environment

Activate your virtual environment and check GPU availability:
```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Verify CUDA & bitsandbytes / gptqmodel
python -c "import torch, bitsandbytes, gptqmodel; print('CUDA available:', torch.cuda.is_available(), '| GPTQ:', gptqmodel.__version__)"
```

---

## 3. Running Phase 6

### Step 1: Inspect the Quantization Matrix (Dry-Run)
Verify all 72 planned quantization artifacts (Phase 4 Bengali + Phase 5 English):
```bash
python scripts/04_quantize.py --dry-run
```

### Step 2: Check Existing Progress & Status
Display current completion status, elapsed durations, and sanity check results:
```bash
python scripts/04_quantize.py --status
```

### Step 3: Fast Smoke Rehearsal
Before executing the full sweep, perform a rapid end-to-end rehearsal with truncated calibration ($N=32$) and evaluation ($N=32$) sets:
```bash
python scripts/04_quantize.py --smoke
```
*(Outputs from `--smoke` are isolated to avoid overwriting production artifacts).*

### Step 4: Targeted Execution (Single Model or Method)
You can run specific subsets of the matrix:

- **Run only BitsAndBytes INT8 and NF4**:
  ```bash
  python scripts/04_quantize.py --methods bnb_int8 bnb_nf4
  ```

- **Run only Bengali (Phase 4) models**:
  ```bash
  python scripts/04_quantize.py --languages bn
  ```

- **Run a single model family**:
  ```bash
  python scripts/04_quantize.py --models qwen2_5_0_5b_instruct
  ```

### Step 5: Full Experiment Execution
To execute all 72 quantization artifacts:
```bash
python scripts/04_quantize.py
```
> [!NOTE]
> The full sweep takes several hours depending on your GPU. The script is fully resumable: completed artifacts are skipped automatically on subsequent runs. To force recalculation of existing models, pass `--force`.

---

## 4. Output Artifacts

Upon completion, all outputs are organized under standard directories:

| Output Directory / File | Description |
|---|---|
| `outputs/quantized/track_a/{artifact_name}/` | Saved standalone model weights for weight-only PTQ (`gptq_int4`) |
| `outputs/logits/phase6/{artifact_name}_{split}.parquet` | Per-example logit Parquet files (`example_id`, `y_true`, `prob_hate`, `prob_nonhate`, `prediction`) |
| `outputs/results/phase6_metrics.csv` | Unified metrics table across all 72 artifacts (Macro-F1, Accuracy, ECE, Brier score, latency) |
| `outputs/results/phase6_summary.json` | Comprehensive run summary with T6.1–T6.9 validation check outcomes and hardware snapshots |

---

## 5. Verification & Unit Testing

To verify code integrity and confirm zero regressions across the codebase:
```bash
# Run Phase 6 unit tests
pytest tests/test_phase6.py -v

# Run complete test suite (220+ tests)
pytest tests/ -q
```

