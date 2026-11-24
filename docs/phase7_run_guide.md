# Phase 7 Run Guide: Post-Training Quantization — Track B (GGUF / llama.cpp)

This document provides instructions for executing and verifying **Phase 7: Post-Training Quantization (Track B)** in accordance with [`planning/phases/phase_07_post_training_quantization_track_b_gguf_llama_cpp.md`](file:///d:/canvas/slm-calibration-quantization/planning/phases/phase_07_post_training_quantization_track_b_gguf_llama_cpp.md).

> [!IMPORTANT]
> **Global Rule 4**: Track A (GPU / PyTorch / HF) and Track B (GGUF / llama.cpp) are strictly separate experimental tracks. Never pool their results.

---

## 1. Overview & Architecture

Phase 7 converts the 18 fine-tuned base checkpoints produced in Phase 4 (Bengali) and Phase 5 (English control) into GGUF format and quantizes them across four standard quantization formats:
1. **`f16`** (Half-precision unquantized GGUF baseline converted via `convert_hf_to_gguf.py`)
2. **`q8_0`** (8-bit symmetric integer quantization)
3. **`q4_k_m`** (4-bit medium k-quant with mixed block sizes)
4. **`q3_k_m`** (3-bit medium k-quant with mixed block sizes)
*(Optional: `q2_k` 2-bit quantization)*

The pipeline extracts per-example logits across `id_val`, `id_test`, and `ood_test` splits using `llama-cpp-python`, validates the T7.5 baseline fidelity gate, and outputs standard evaluation metrics (Macro-F1, Accuracy, ECE, Brier score).

### Validation Gates (T7.1 – T7.7)
- **T7.1**: GGUF F16 conversion completes; file is valid GGUF and `llama-cpp-python` loads it.
- **T7.2**: Q8_0 quantization completes; file size is strictly smaller than F16 ($< \text{F16}$).
- **T7.3**: Q4_K_M quantization completes; file size is approximately 25–30% of F16 (ratio $0.15 \le r \le 0.45$).
- **T7.4**: Q3_K_M quantization completes; file size is smaller than F16.
- **T7.5 (Critical Gate)**: GGUF F16 approximately matches the HF BF16 reference:
  - Prediction agreement $\ge 0.90$
  - Macro-F1 drop $\le 0.05$
  - Mean probability shift $|\Delta p| \le 0.10$
- **T7.6**: Label-token logits parsed correctly from llama.cpp output; softmax probabilities sum to $1.0 \pm 10^{-4}$ for all rows.
- **T7.7**: All logits saved in Parquet files for every Track B artifact $\times$ eval set.

---

## 2. Prerequisites & Environment

Verify `llama_cpp` is available:
```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Verify llama_cpp version
python -c "import llama_cpp; print('llama_cpp version:', llama_cpp.__version__)"
```

The conversion script is located at `outputs/quantized/llama.cpp/convert_hf_to_gguf.py`. Quantization executes in-process via `llama_cpp.llama_model_quantize` C-API, or via standalone `llama-quantize` if compiled.

---

## 3. Running Phase 7

### Step 1: Inspect the Quantization Matrix (Dry-Run)
Verify all 72 planned quantization artifacts (18 models $\times$ 4 methods):
```bash
python scripts/04_quantize_track_b.py --dry-run
# or alternatively:
python scripts/04_quantize.py --track b --dry-run
```

### Step 2: Check Existing Progress & Status
Display current completion status, file sizes, and agreement results:
```bash
python scripts/04_quantize_track_b.py --status
```

### Step 3: Fast Smoke Rehearsal
Perform a rapid end-to-end rehearsal with truncated evaluation sets ($N=32$ samples):
```bash
python scripts/04_quantize_track_b.py --smoke
```
*(Outputs from `--smoke` are isolated to `outputs/.../phase7_smoke` to protect production artifacts).*

### Step 4: Targeted Execution (Single Model or Method)
You can target specific subsets of the matrix:

- **Run only F16 and Q8_0**:
  ```bash
  python scripts/04_quantize_track_b.py --methods f16 q8_0
  ```

- **Run only Bengali (Phase 4) models**:
  ```bash
  python scripts/04_quantize_track_b.py --languages bn
  ```

- **Run a single model family**:
  ```bash
  python scripts/04_quantize_track_b.py --models qwen2_5_0_5b_instruct
  ```

- **Run with CPU-only (no GPU offload)**:
  ```bash
  python scripts/04_quantize_track_b.py --n-gpu-layers 0
  ```

### Step 5: Full Experiment Execution
To execute all 72 quantization artifacts:
```bash
python scripts/04_quantize_track_b.py
```
> [!NOTE]
> The script is fully resumable: completed artifacts with matching configuration and valid GGUF/Parquet files are automatically skipped on subsequent runs. To force recalculation of existing models, pass `--force`.

---

## 4. Output Artifacts

All Track B outputs are organized under standard directories:

| Output Directory / File | Description |
|---|---|
| `outputs/quantized/track_b/{artifact_name}.gguf` | Quantized standalone GGUF model files |
| `outputs/logits/phase7/{artifact_name}/{method}__{split}.parquet` | Per-example logit Parquet files (`example_id`, `y_true`, `prob_hate`, `prob_nonhate`, `prediction`, etc.) |
| `outputs/runs/phase7/{artifact_name}.json` | Per-run execution manifests with timing, size, and gate outcomes |
| `outputs/results/phase7_metrics.csv` | Unified metrics table across all Track B artifacts |
| `outputs/results/phase7_summary.json` | Comprehensive run summary with T7.1–T7.7 validation check outcomes |

---

## 5. Verification & Unit Testing

To verify code integrity and confirm zero regressions:
```bash
# Run Phase 7 unit tests
pytest tests/test_phase7.py -v

# Run Phase 6 & Phase 7 test suites
pytest tests/test_phase6.py tests/test_phase7.py -v
```

