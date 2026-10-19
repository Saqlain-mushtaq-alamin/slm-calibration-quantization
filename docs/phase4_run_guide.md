# Phase 4 — Bengali Baseline Fine-Tuning: Run Guide

How to run, verify, and interpret Phase 4 of
*Quantization × Calibration × Selective Prediction for Sub-3B SLMs on Bengali
Hate-Speech Classification under Domain Shift*.

**Status: code complete and tested (168 passing). Nothing has been trained yet.**
The 16-run matrix below is ready to launch and takes roughly 25 GPU-hours.

---

## 1. What Phase 4 produces

Every `(model, seed)` cell of the matrix produces three artifacts:

| Artifact | Path | Why it matters |
|---|---|---|
| LoRA adapter | `outputs/models/phase4/<run_name>/adapter` | Reproducible delta from the base model |
| Merged BF16 checkpoint | `outputs/models/phase4/<run_name>/merged` | **FP reference for Phase 6 quantization** — this is the model that later gets quantized to INT8/NF4/INT4 |
| Per-example logits | `outputs/logits/phase4/<run_name>/bf16__{id_val,id_test,ood_test}.parquet` | Recomputable metrics without re-running inference (global rule 2) |

Plus one manifest per run at `outputs/runs/phase4/<run_name>.json` carrying the git
commit, config hash, seed, split SHA-256, timing, VRAM, library versions, and every
T4.x gate result.

After the sweep completes:

- `outputs/results/phase4_summary.json` — aggregate gate verdict + seed variance
- `outputs/results/phase4_metrics.csv` — one row per (model, seed, eval set)

---

## 2. Prerequisites

Everything is already installed and verified in `.venv`. Confirm before a long run:

```powershell
& .venv\Scripts\python.exe -m pytest tests\ -q
& .venv\Scripts\python.exe scripts\00_verify_environment.py --skip-heavy
```

The two input files Phase 4 depends on must exist (they are gitignored, produced by
Phase 1):

- `data/processed/bd_shs_processed.csv` — 49,941 rows
- `data/splits/bd_shs_loco_splits.json` + `.sha256` — frozen LOCO splits

If either is missing, run Phase 1 first:

```powershell
& .venv\Scripts\python.exe scripts\01_prepare_data.py
```

### One thing to fix first: no git commit exists

The repository currently has **zero commits**, so `git_commit` is `null` in every run
manifest and **T4.7 will fail** for all 16 cells. Commit the code before launching —
that commit is the provenance anchor for the whole sweep.

```powershell
git add -A
git commit -m "Phase 4: baseline fine-tuning orchestrator, split assembly, tests"
```

Never commit `outputs/` or `data/` (both are gitignored).

---

## 3. The run matrix

Config: `configs/baseline/default.yaml`. Language `bn`, dataset `bd_shs`, split
`loco_slander` (matching the Phase 3 pilot).

| Model key | Canonical HF id | Mirror used | Params | Seeds | Cells |
|---|---|---|---|---|---|
| `qwen2_5_0_5b_instruct` | `Qwen/Qwen2.5-0.5B-Instruct` | — | 0.50B | 13, 42, 1234, 2024, 7 | 5 |
| `qwen2_5_1_5b_instruct` | `Qwen/Qwen2.5-1.5B-Instruct` | — | 1.56B | 13, 42, 1234, 2024, 7 | 5 |
| `llama3_2_1b_instruct` | `meta-llama/Llama-3.2-1B-Instruct` | `unsloth/Llama-3.2-1B-Instruct` | 1.25B | 13, 42, 1234 | 3 |
| `gemma2_2b_it` | `google/gemma-2-2b-it` | `unsloth/gemma-2-2b-it` | 2.64B | 13, 42, 1234 | 3 |
| | | | | **Total** | **16** |

`loco_slander` split sizes: **26,446** train / 3,778 `id_val` / 7,557 `id_test` /
12,160 `ood_test`.

### Why mirrors are used

`meta-llama/*` and `google/gemma-*` are gated and no Hugging Face token is available in
this environment, so fetching them fails with:

```
OSError: You are trying to access a gated repo.
huggingface_hub.errors.LocalTokenNotFoundError
```

`resolve_model_id()` in `src/data/assemble.py` tries the canonical id first, then each
mirror, and accepts the first that loads. The mirrors are community re-uploads of the
**same weights** (identical architecture, vocab, and parameter count). Every substitution
is written into the run manifest under `mirror_notes`, alongside both
`canonical_hf_id` and `resolved_hf_id`, so the substitution can never be silently lost.
This is the decision you made; it applies to all 6 Llama/Gemma cells.

If you later obtain an HF token, delete the `mirrors:` lists in the config and re-run
with `--force` to use the official repos instead.

### Gemma-2 requires `eager` attention

Gemma-2 combines sliding-window attention with logit soft-capping, and transformers
warns that the fused `sdpa` kernel does not handle this correctly:

> It is strongly recommended to train Gemma2 models with the `eager` attention
> implementation instead of `sdpa`.

`configs/baseline/default.yaml` pins `attn_implementation: eager` for Gemma-2 only.
Measured cost: **+7%** (0.459 → 0.490 s/example). The other three models keep the
transformers default. `scripts/03_finetune.py` threads this flag into both the training
load and the evaluation load so train-time and eval-time kernels match.

---

## 4. How to run it

### Step 1 — rehearse (always do this first)

A 64-example / 32-eval-example run over all four models. Takes ~25 minutes total and
exercises training, merging, logits extraction, manifest writing, and validation. Every
run is expected to **fail T4.2/T4.3** — 64 examples and 4 optimiser updates cannot
produce a good classifier. What you are checking is that T4.1, T4.4, T4.5, and T4.6 pass
and that no model OOMs.

```powershell
& .venv\Scripts\python.exe scripts\03_finetune.py --smoke
```

Then clean up so the real sweep cannot mistake smoke output for real results:

```powershell
Remove-Item -Recurse -Force outputs\models\phase4, outputs\logits\phase4, outputs\runs\phase4
```

### Step 2 — inspect the matrix without touching the GPU

```powershell
& .venv\Scripts\python.exe scripts\03_finetune.py --dry-run
```

Prints all 16 cells and exits. Verify the count and seed lists match Section 3.

### Step 3 — launch the full sweep

```powershell
& .venv\Scripts\python.exe scripts\03_finetune.py
```

Expect ~25 GPU-hours. It is safe to interrupt and restart: see Section 6.

### Selective and bounded variants

```powershell
# One model, all its seeds
& .venv\Scripts\python.exe scripts\03_finetune.py --models qwen2_5_1_5b_instruct

# One seed across all models (fastest useful check: ~7 h)
& .venv\Scripts\python.exe scripts\03_finetune.py --seeds 42

# Two models, three seeds
& .venv\Scripts\python.exe scripts\03_finetune.py --models qwen2_5_1_5b_instruct gemma2_2b_it --seeds 13 42 1234

# Cap training examples to bound cost (~8k examples per cell, ~9 h)
& .venv\Scripts\python.exe scripts\03_finetune.py --max-train 8000

# Different LOCO fold
& .venv\Scripts\python.exe scripts\03_finetune.py --split loco_gender

# Re-run cells that already succeeded
& .venv\Scripts\python.exe scripts\03_finetune.py --force
```

### Full CLI reference

| Flag | Effect |
|---|---|
| `--smoke` | 64 train / 32 eval rehearsal |
| `--models KEYS...` | restrict to these `model_key` values |
| `--seeds N...` | restrict to these seeds (overrides the tier defaults) |
| `--max-train N` | cap training examples; `0` = whole frozen split |
| `--max-eval N` | cap each eval set; also sets `max_val_samples = N // 2` |
| `--split NAME` | override the LOCO split |
| `--no-keep-merged` | delete the merged BF16 checkpoint after logits |
| `--force` | re-run cells whose manifest already says `success` |
| `--dry-run` | print the matrix and exit |

Caps are recorded in every run manifest, so a capped run can never be mistaken for a
full-data run.

### Disk note

`keep_merged: true` in the config keeps ~44 GB of merged BF16 checkpoints, which is
required because Phase 6 quantizes them. Free space at the time of writing was
103.7 GB, so this fits. If disk becomes tight, `--no-keep-merged` drops to ~4 GB — but
Phase 6 will then need a re-run to materialise the checkpoints.

---

## 5. Training recipe

Identical to the Phase 3 pilot (`configs/pilot/default.yaml`); only the model and seed
matrices widen.

| Setting | Value |
|---|---|
| Precision | BF16 LoRA |
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 |
| Target modules | `q,k,v,o,gate,up,down_proj` |
| Learning rate | 2e-4, cosine schedule, 3% warmup |
| Epochs | 1 |
| Batch size / grad accumulation | 2 / 8 (effective 16) |
| Gradient checkpointing | on |
| Max sequence length | 256 |
| Eval batch size | 8 |
| Precision of base weights | BF16 |
| Verbalizer | `" A"` = hate, `" B"` = non-hate |
| Prompt | `{text}\nবাক্যটি ঘৃণামূলক কি না? উত্তর:` |

The prompt ends in a trailing cue that carries the answer position. Training masks every
prompt token and supervises only the single label token, so the next-token distribution
at the answer position is a proper two-way classifier.

### Verbalizer token IDs (verified single-token)

| Model | `" A"` (hate) | `" B"` (non-hate) |
|---|---|---|
| Qwen2.5-0.5B | 362 | 425 |
| Qwen2.5-1.5B | 362 | 425 |
| Llama-3.2-1B (mirror) | 362 | 426 |
| Gemma-2-2B (mirror) | 586 | 599 |

A verbalizer that maps to more than one token raises immediately — the two-way softmax
read-out requires single-token labels.

---

## 6. Cost model (measured, not estimated)

Per-example training cost measured on this machine with a 7-micro-step probe, excluding
model load / merge / save overhead. Projections use the full 26,446-example train split.

| Model | s/example | Peak VRAM (train) | 1 epoch | Cells | Total |
|---|---|---|---|---|---|
| Llama-3.2-1B | 0.063 | 4.04 GiB | 0.46 h | 3 | 1.4 h |
| Qwen2.5-0.5B | 0.092 | 2.37 GiB | 0.67 h | 5 | 3.4 h |
| Qwen2.5-1.5B | 0.112 | 5.17 GiB | 0.82 h | 5 | 4.1 h |
| Gemma-2-2B (eager) | 0.490 | 6.97 GiB | 3.60 h | 3 | 10.8 h |
| **Training subtotal** | | | | | **19.7 h** |
| Logit extraction (23,495 eval examples × 16 runs) | | | | | ~5 h |
| Per-run load / merge / save overhead | | | | | ~0.6 h |
| **Total** | | | | | **~25 h** |

Llama-3.2-1B is faster than Qwen2.5-0.5B despite being 2.5× larger: logit-projection
cost is dominated by vocabulary size, and Qwen's 151,936-token vocab costs more than
Llama's 128,256.

Hardware: RTX 4060, 8.585 GB VRAM (8.0 GiB usable), CUDA 12.1, torch 2.5.1+cu121.
Gemma-2B is the largest cell at 6.97 GiB peak — it fits, but with under 1 GiB of
headroom, so do not run anything else on the GPU concurrently.

**A caveat worth knowing:** at tiny scale the wall-clock per run is dominated by fixed
overhead, not compute. The Gemma smoke run took 492 s for 64 examples even though only
~31 s was actual training. Do not extrapolate a cost plan from smoke runs — use the
table above.

---

## 7. Resumability and failure handling

- A cell is skipped only when its manifest says `status: "success"` **and** all three
  logits parquet files exist. Anything else is retrained.
- `--force` overrides the skip.
- A cell that raises is recorded with `status: "failed"` plus `error` and a traceback,
  and **the sweep continues to the next cell**. Failures are never silently dropped
  (global rule 6).
- Exit code is `0` for `passed` / `passed_with_failures` / `dry_run`, `1` otherwise.
- Summary status is `passed` if all 16 succeed, `passed_with_failures` if some do, and
  `failed` if none do.

To re-run only the failures, just launch the same command again — successful cells are
skipped automatically.

---

## 8. Acceptance criteria and how each is checked

| ID | Criterion | Implementation |
|---|---|---|
| T4.1 | Train loss decreases, no NaN, checkpoint saved, loss did not explode | `run_one()` after training; aborts the gate if final loss > 5.0 |
| T4.2 | `id_test` macro-F1 > 0.55 | `sklearn.f1_score(average="macro")` on `id_test` |
| T4.3 | Both classes predicted on `id_test` | `np.unique(prediction).size == 2` |
| T4.4 | Logits exist for all three eval sets | three parquet files per run |
| T4.5 | `prob_hate + prob_nonhate == 1` | max deviation across all eval sets |
| T4.6 | No NaN/inf in logits or probabilities | finiteness check |
| T4.7 | Manifest has git commit, config hash, seed | `src/utils/reproducibility.py` |
| T4.8 | Every model saw the same split and training size | identical split SHA-256 and `n_train` across successes |
| T4.9 | Seed macro-F1 std < 0.05 | across seeds of one model, on `id_test`; needs >1 seed to pass |

Thresholds live under `sanity:` in the config, so they are visible and adjustable
without editing code.

### Single-class eval sets

`bd_shs` derives its LOCO `domain` axis from the hate subtype, so **every `ood_test` set
is single-class** (100% hate for the type folds). On those sets macro-F1, AUROC, average
precision, balanced accuracy, MCC, and failure-prediction AUROC are mathematically
undefined. Rather than failing the run, they are recorded as `null` with
`single_class: true`. Calibration metrics (ECE, MCE, Brier, NLL, confidence gap) remain
valid there. T4.2 and T4.3 are enforced on `id_test` only, which is two-class.

This is a property of the dataset, not a bug — Phase 1 notes it explicitly.

---

## 9. Verifying a completed sweep

```powershell
# Gate-by-gate verdict
& .venv\Scripts\python.exe -c "import json;d=json.load(open('outputs/results/phase4_summary.json',encoding='utf-8'));print(json.dumps({'status':d['status'],'n_success':d['n_success'],'n_failed':d['n_failed'],'tests':d['tests']},indent=2))"

# Seed variance per model (T4.9)
& .venv\Scripts\python.exe -c "import json;d=json.load(open('outputs/results/phase4_summary.json',encoding='utf-8'));print(json.dumps(d['seed_variance'],indent=2))"

# Per-run macro-F1 on id_test
& .venv\Scripts\python.exe -c "import pandas as pd;print(pd.read_csv('outputs/results/phase4_metrics.csv').query(\"split=='id_test'\")[['model_key','seed','macro_f1','accuracy','n']].to_string(index=False))"

# Confirm every mirror substitution was recorded
& .venv\Scripts\python.exe -c "import json,glob;[print(p.split(chr(92))[-1], json.load(open(p,encoding='utf-8')).get('canonical_hf_id'),'->',json.load(open(p,encoding='utf-8')).get('resolved_hf_id')) for p in glob.glob('outputs/runs/phase4/*.json')]"

# Confirm all 16 cells exist with all three logits files
(Get-ChildItem outputs\models\phase4 -Directory).Count
(Get-ChildItem outputs\logits\phase4 -Directory | ForEach-Object { (Get-ChildItem $_.FullName -Filter *.parquet).Count }) | Group-Object
```

Also confirm each merged checkpoint recorded a non-null `git_commit`.

---

## 10. Known limitations to record in the write-up

1. **6 of 16 cells used mirrors.** Llama and Gemma ran from `unsloth/` re-uploads
   because the canonical repos are gated without a token. Same weights, but the manifest
   records a substitution and the paper should state it.
2. **Single `ood_test` fold.** Only `loco_slander` is run. The other four LOCO folds are
   available via `--split`.
3. **Single-class OOD sets** make rank metrics undefined there; only calibration and
   selective metrics are meaningful on `ood_test`.
4. **One epoch, one LoRA configuration.** No hyperparameter search; Phase 4 establishes
   a fixed baseline for later phases to measure quantization against.
5. **BF16 only.** Track A quantization (INT8/NF4/GPTQ) is Phase 6. These merged BF16
   checkpoints are its inputs.

---

## 11. What comes after

| Phase | Uses Phase 4 output for |
|---|---|
| 5 | English-control baselines using the identical recipe |
| 6 | Quantizes the merged BF16 checkpoints; needs `keep_merged: true` |
| 7 | GGUF / llama.cpp Track B |
| 8–10 | Full metric, selective-prediction, and efficiency sweeps over the matrix |
| 11–14 | Recalibration, ablations, statistics, Pareto analysis |

After the sweep, the Phase 4 section of `docs/phase_overview.md` should be filled in with
the per-model macro-F1 table, seed variance, gate verdicts, measured wall-clock, and the
mirror provenance note.

---

## 12. File map

| Path | Role |
|---|---|
| `configs/baseline/default.yaml` | Model/seed matrix, mirrors, recipe, thresholds, paths |
| `scripts/03_finetune.py` | Orchestrator, T4.1–T4.9 gates, aggregation, resumability |
| `src/data/assemble.py` | Split materialization, mirror resolution, provenance |
| `src/models/finetune.py` | LoRA training, merge, tokenizer, `attn_implementation` support |
| `src/models/logit_extraction.py` | Padding-safe two-way logits extraction |
| `src/metrics/aggregate.py` | Per-run metric computation |
| `tests/test_phase4.py` | 34 tests: matrix, splits, mirrors, gates, seed variance |
| `outputs/runs/phase4/` | One manifest per cell |
| `outputs/logits/phase4/` | Per-example BF16 logits |
| `outputs/models/phase4/` | Adapters and merged BF16 checkpoints |
| `outputs/results/phase4_summary.json` | Aggregate verdict |
| `outputs/results/phase4_metrics.csv` | Per-cell metrics table |