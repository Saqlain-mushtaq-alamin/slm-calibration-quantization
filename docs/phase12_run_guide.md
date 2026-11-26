# Phase 12: Ablation studies

The runner implements A1–A8 for the two Bengali headline models at seed 42
by default. It saves one Parquet logit file per new evaluation, one JSON
manifest per cell, `outputs/results/ablation_results.csv` and `.parquet`, and
`outputs/results/phase12_run.json`. It checks frozen evaluation IDs and labels,
finite logits, probability normalization, and training-only calibration IDs.

## Prerequisites

- Phase 1 processed data and frozen splits, Phase 4 merged BF16 checkpoints,
  Phase 6 comparator logits, Phase 7 F16 GGUF files for A7, and Phase 11
  recalibration results must be present.
- GPTQModel must provide its quantize/save driver. This installation does.
- AWQ is unavailable in the current environment. Its A1 cells are marked
  `unavailable` and produce no score. There is no AWQ substitution.
- The frozen splits contain no OOD validation data, so A8 has no oracle fit.

## Commands

```powershell
& .venv\Scripts\python.exe -m pytest tests\test_phase12.py -q
& .venv\Scripts\python.exe scripts\08_ablations.py --dry-run
& .venv\Scripts\python.exe scripts\08_ablations.py --smoke --ablations A1 --models qwen2_5_0_5b_instruct --variants bn --methods gptq_int4
& .venv\Scripts\python.exe scripts\08_ablations.py
```

The final command starts the **full A1–A8 sweep**. It was not run during
implementation. `--smoke` uses eight GPTQ calibration prompts, up to 16
examples per test set, and an isolated `outputs/phase12_smoke/` root. Use
`--ablations`, `--models`, `--languages`, `--seeds`, `--variants`, and
`--methods` to run targeted cells. The runner resumes successful cells when
their config hash and saved logit files match; `--force` recomputes them.
Set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` when network access is
unavailable and the original HF model snapshots are cached locally.

## Methods and provenance

| ID | Comparison |
|---|---|
| A1 | GPTQ calibrated on 512 Bengali, 512 English, 256+256 mixed, or 512 deterministic synthetic random strings. Real IDs are checked against validation/test splits. AWQ is unavailable. |
| A2 | Three label-token prompt formats, plus a linear classification head trained on up to 2,048 frozen **training** examples from the fine-tuned causal model's last-token features. The head is a probe, not a separately trained encoder. |
| A3 | QLoRA trained on an NF4 base and evaluated with its adapter, versus Phase 6 LoRA then NF4 PTQ saved logits. Training loss must decrease. |
| A4 | Attention-only and MLP-only GPTQ using explicit layer skip rules, versus the saved full GPTQ comparator. Standard GPTQ already keeps embeddings and `lm_head` in full precision. |
| A5 | GPTQ group size 128 versus 32 with the same training sample and seed. |
| A6 | 10% and 25% LoRA training subsets versus the saved 100% baseline, with full held-out evaluation. |
| A7 | GPTQ INT3 and GGUF Q2_K stress settings. The Q2_K file can contain fallback tensor types if llama.cpp cannot encode every tensor as Q2_K. |
| A8 | Phase 11 raw, own-fit, FP-transfer, and oracle if available, with original fit provenance. |

## Estimated runtime

Phase 6 measured full GPTQ cells at about 8–16 minutes each on this RTX 4060
including three evaluation sets; Phase 12 uses two. For the default two-model,
one-seed matrix, A1 has eight GPTQ cells (~1–2 hours); A4, A5, and the INT3
part of A7 add ten cells (~1.5–3 hours). A2's eight inference/head cells may
take ~0.5–2 hours. The 160-example QLoRA smoke run took about 31 seconds of
training, suggesting ~2–4 hours for two full A3 trainings. A6's 10% and 25%
LoRA runs may take ~1–3 hours. GGUF Q2_K conversion and inference may add
~0.5–1.5 hours. A8 is seconds. **Allow roughly 5–13 GPU hours** overall,
plus space for approximately 18 new quantized checkpoints. This is an
extrapolation, not a Phase 12 full-run measurement; each manifest records its
actual `wall_s`.

The run status remains `partial` while required arms are unavailable. Only
successful arms appear in the results CSV. The AWQ and absent OOD oracle arms
must be resolved before claiming the complete planned Phase 12 matrix.
