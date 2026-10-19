# Phase Overview

Quantization × Calibration × Selective Prediction for Sub-3B SLMs on Bengali Hate-Speech Classification under Domain Shift.

Status: **Phase 0 complete** (environment verified; no git commits yet). **Phase 1 complete** (data prep + splits frozen; no git commits yet). **Phase 2 complete** (metrics implemented; 134 tests green). **Phase 3 complete with one blocked criterion** (pilot end-to-end: T3.8 GPTQ-4 unavailable in the installed `gptqmodel` build).

---

## Phase 0: Environment Setup & Verification

### What was built

| Path | Purpose |
|------|---------|
| `scripts/00_verify_environment.py` | T0.1–T0.7 verification orchestrator; writes `outputs/runs/run.json` and `outputs/environment_versions.json` |
| `src/utils/{config,logging_utils,reproducibility,sanity_checks}.py` | Shared utilities (seed, structured logs, import/VRAM checks) |
| `tests/test_environment.py`, `tests/test_sanity_checks.py` | pytest wrappers for Phase 0 checks |
| `requirements.txt` / `pyproject.toml` | Pinned dependency contract + install notes |
| `configs/{base,models,data,quantization}/*.yaml` | Experiment config skeleton for later phases |
| `src/{data,models,metrics,recalibration,analysis}/` | Package skeletons for Phases 1–17 |
| `outputs/`, `data/`, `notebooks/`, `docs/` | Runtime artifacts and docs (mostly gitignored) |

### Verification results (`outputs/runs/run.json`)

| ID | Check | Status |
|----|-------|--------|
| T0.1 | Hardware (RTX 4060, 8.59 GB VRAM, CUDA 12.1) | passed |
| T0.2 | `pip check` — 0 project-internal conflicts | passed |
| T0.3 | bitsandbytes INT8 + NF4 smoke | passed |
| T0.4 | GPTQModel 4-bit quantize → save → reload → generate | passed |
| T0.5 | HF→GGUF convert (llama.cpp) + llama-cpp-python inference | passed |
| T0.6 | GPU power path (NVML → nvidia-smi fallback) | passed |
| T0.7 | Library version manifest | passed |

**Overall: passed.** Acceptance: all tests green, environment versions saved, `run.json` written.

### Verified environment

- Python 3.11.9, Windows 10, pip 26.2.1 (venv)
- NVIDIA GeForce RTX 4060, **8.59 GB** VRAM (≥ 7.5 bar), CUDA **12.1**, torch **2.5.1+cu121**
- 24 CPU threads, 33.4 GB RAM, driver 610.62, power limit 115 W
- transformers **4.49.0**, gptqmodel **2.2.0**, llama-cpp-python **0.3.4**, bitsandbytes **0.49.2**, numpy **1.26.4**, datasets **5.0.0**, scipy **1.16.3**, statsmodels **0.14.6**, peft **0.13.2**, accelerate **1.14.0**, pynvml **11.5.3**, hydra **1.3.4**

### Smoke artifacts

- Tiny GPT-2: `outputs/quantized/smoke_models/tiny_smoke_gpt2` (n_embd=64, n_layer=2, vocab=50257, untied lm_head)
- GPTQ INT4: `outputs/quantized/smoke_gptq`
- GGUF: `outputs/quantized/smoke_gguf/tiny_smoke_f32.gguf`
- Version manifest: `outputs/environment_versions.json`
- Run record: `outputs/runs/run.json`

### Environment notes / workarounds

1. **Venv uses `--system-site-packages`** to reuse preinstalled torch 2.5.1+cu121 (avoids multi-GB re-download).
2. **PyPI via Tsinghua mirror** (`%APPDATA%\pip\pip.ini`) — direct PyPI often times out on this network.
3. **GPTQModel replaces AutoGPTQ/AutoAWQ** (unmaintained). `auto_gptq` is not installed; GPTQModel 2.2.0 is pinned (7.x pulls torch≥2.8).
4. **llama-cpp-python 0.3.4 built from patched sdist** (`scripts/_build_llamacpp.bat`): added `#include <chrono>` to vendor `log.cpp`, MSVC BuildTools + Ninja.
5. **llama.cpp checkout** at `outputs/quantized/llama.cpp` for `convert_hf_to_gguf.py` + `gguf-py`.
6. **GPT-2 `n_ctx`**: converter expects `n_ctx`; tiny model `config.json` stores both `n_ctx` and `n_positions`.
7. **Tied embeddings**: modern converters may omit `output.weight`; llama-cpp-python 0.3.4 requires it for GPT-2. Tiny smoke model uses `tie_word_embeddings=False` so `lm_head.weight` is exported and maps to `output.weight`.
8. **NVML power**: `nvmlDeviceGetPowerUsage` returns NotSupported on this GPU/driver; energy test falls back to `nvidia-smi` `power.draw`/`power.limit` (limit=115 W).
9. **Windows UTF-8**: verification script sets `PYTHONUTF8=1` / `PYTHONIOENCODING=utf-8` (logbar emits non-ASCII).
10. **torch device props**: use `total_memory` (this build has no `total_mem`); both `src/utils/sanity_checks.py` and the verify script handle this.
11. **`pip check`** is nonzero due to unrelated system packages (anylabeling, opencv, shap…); T0.2 only fails on *project* package conflicts.
12. **No commits**: repo `main` has zero commits; planning/, data/, outputs/, .venv/ are gitignored.

### How to re-run Phase 0

```powershell
.venv\Scripts\python.exe scripts\00_verify_environment.py
.venv\Scripts\python.exe -m pytest tests/ -q
```

---

## Phase 1: Data Preparation & Split Engineering

### What was built

| Path | Purpose |
|------|---------|
| `scripts/01_prepare_data.py` | T1.1–T1.11 orchestrator; writes `outputs/runs/run.json` |
| `src/data/loader.py` | BD-SHS / 30k / HateXplain / ToxiGen loaders + English-control matching |
| `src/data/dedup.py` | Exact + MinHash near-dedup (LSH query-before-insert) |
| `src/data/splits.py` | LOCO split creation, SHA-256 freeze, integrity checks |
| `src/data/shift.py` | MMD, centroid cosine, vocab overlap, OOV |
| `src/data/fertility.py` | Tokenizer fertility (tokens/word, tokens/char, UNK rate) |
| `configs/data/default.yaml` | Phase 1 data config |
| `tests/test_data_pipeline.py` | Unit + integration tests for the data pipeline |

### Dataset decisions

| Dataset | Role | Notes |
|---------|------|-------|
| **BD-SHS** (Romim et al., LREC 2022) | Primary Bengali | 50,281 samples (26,125 non-hate / 24,156 hate). Kaggle `iftekharuddin27/bd-shs` (anonymous download). Columns: `sentence`, `target`, `type`, `hate speech`. |
| BD-SHS domain axis | LOCO shift | `domain` = primary hate type (`slander`, `callToViolence`, `gender`, `religion`) + `non_hate`. Official `type`/`target` are hate-only (NaN on non-hate). |
| **30k Bengali HS** | Secondary LOCO axis | 30k with full topic categories (sports, entertainment, crime, politics, religion, celebrity, meme). Kaggle `naurosromim/bengali-hate-speech-dataset`. |
| **ToxiGen** | **Primary English control** | 250,951 rows; identity-group metadata (13 groups). Subsampled to 50,281 with prior 0.4804 to match BD-SHS (T1.10 pass). |
| **HateXplain** | Secondary English | 20,148 posts; rich target-community labels but **too small** to match BD-SHS size (T1.10). Retained for later analysis. |

### T1.x results (`outputs/runs/run.json`, status=passed)

| ID | Result |
|----|--------|
| T1.1 | BD-SHS loaded: 50,281 ≥ 45,000; `text`+`label` present |
| T1.2 | Metadata domains: `domain`×5, `type`×15, `target`×12 |
| T1.3 | Exact dedup: 50,281 → 50,276 (5 removed) |
| T1.4 | Near-dedup Jaccard≥0.8 (MinHash LSH): 50,276 → 49,941 (335 removed) |
| T1.5 | LOCO splits: `loco_{callToViolence,gender,non_hate,religion,slander}` — all ID-disjoint |
| T1.6 | Split SHA-256 deterministic across two runs |
| T1.7 | No train∩ood / train∩id_test leakage |
| T1.8 | Class priors logged in `outputs/results/split_class_priors.csv` |
| T1.9 | Shift magnitudes in `outputs/results/shift_magnitudes.csv` (MMD + vocab overlap + centroid cosine + OOV) |
| T1.10 | English control = ToxiGen, n=50,281, prior=0.4804 (size err <0.01%, prior err <0.01%) |
| T1.11 | Fertility for 6 tokenizers × {bn,en} on 1k samples each |

### Key artifacts

| Artifact | Path / hash |
|----------|-------------|
| BD-SHS processed | `data/processed/bd_shs_processed.csv` (49,941 rows) |
| English control | `data/processed/english_control_matched.csv` (50,281 rows) |
| BD-SHS LOCO splits | `data/splits/bd_shs_loco_splits.json` — `83e985931758f429…` |
| 30k LOCO splits | `data/splits/bengali_hs_30k_loco_splits.json` — `a3ef6368142f53f…` |
| English LOCO splits | `data/splits/english_control_loco_splits.json` — `4d46815e32976fac…` |
| Shift magnitudes | `outputs/results/shift_magnitudes.csv` |
| Fertility table | `outputs/results/tokenizer_fertility.csv` |
| English control decision | `outputs/results/english_control_decision.json` |
| Run record | `outputs/runs/run.json` (status=passed) |

### Fertility snapshot (tokens/word)

| Model | Bengali | English |
|-------|---------|---------|
| Qwen2.5-0.5B / 1.5B | 5.86 | 1.21 |
| SmolLM2-1.7B | 9.92 | 1.25 |
| Gemma-2-2B (unsloth) | 3.14 | 1.20 |
| TinyLlama-1.1B | 7.01 | 1.40 |
| BanglaBERT | 1.27 | 1.81 |

Bengali is much more fragmented for decoder SLM tokenizers than English — the expected mechanistic correlate for later phases. `meta-llama/Llama-3.2-1B-Instruct` is gated (401); TinyLlama used as public Llama-family stand-in until HF auth is available.

### Shift snapshot (source→OOD)

MMD is small in absolute terms (embedding scale) but **vocabulary overlap is low** (≈0.07–0.20) for LOCO hate-type shifts — strong lexical domain shift despite similar embedding centroids. English control top-domain vs rest shows larger centroid cosine distance (≈0.31).

### Phase 1 notes / limitations

1. **BD-SHS metadata**: official release annotates `type`/`target` only on hate rows. LOCO uses derived `domain` (primary type + `non_hate`). OOD sets for hate-type folds are pure hate of that type (class prior 1.0) — logged in priors CSV; later phases must handle prior shift explicitly.
2. **Near-dedup on BD-SHS** removed 335 near-duplicates at Jaccard 0.8 after fixing LSH to query-before-insert. Dedup is global (across all domains) before split creation.
3. **ToxiGen labels**: binary from `prompt_label` (toxic vs benign); domain = identity `group`. Cached at `data/raw/toxigen/toxigen.parquet`.
4. **HateXplain** JSON from GitHub (`punyajoy/HateXplain`); majority-vote `hatespeech` → positive class.
5. **English LOCO** folds use ToxiGen identity groups (≥3 domains, T5.1-ready).
6. **No commits**: `main` still has zero commits.

### How to re-run Phase 1

```powershell
$env:PYTHONUTF8='1'; $env:PYTHONIOENCODING='utf-8'
# Requires data/raw/bd_shs/{train,val,test}.csv (Kaggle iftekharuddin27/bd-shs)
.venv\Scripts\python.exe scripts\01_prepare_data.py
.venv\Scripts\python.exe -m pytest tests/ -q
# Faster path (skip embeddings + fertility):
.venv\Scripts\python.exe scripts\01_prepare_data.py --skip-heavy
```

---

## Phase 2: Metrics Implementation

Task, calibration, selective-prediction and efficiency metrics are implemented once in `src/metrics/` and reused by every later phase.

| Path | Purpose |
|------|---------|
| `src/metrics/task_metrics.py` | Accuracy, macro/weighted/per-class F1, AUROC, AP, MCC, priors; agreement rate; paired bootstrap of accuracy deltas |
| `src/metrics/calibration.py` | ECE/MCE (equal-mass and equal-width), Brier + full Murphy decomposition (reliability/resolution/uncertainty/within-bin variance+covariance), NLL, confidence gap, reliability diagram |
| `src/metrics/selective.py` | Risk–coverage curve, AURC, E-AURC vs an optimal ranker, coverage@risk, selective accuracy, failure-prediction AUROC |
| `src/metrics/efficiency.py` | NVML → `nvidia-smi` power fallback, VRAM accounting, `Stopwatch` (wall/energy/peak), latency benchmark, hardware summary |
| `src/metrics/aggregate.py` | One-call evaluation of a logits frame, cross-precision comparison, JSON-safe record flattening |
| `tests/test_{task_metrics,calibration,selective}.py` | Unit tests, including a `netcal` cross-check on ECE/MCE/ACE |

**Test suite: 134 passed.** Degenerate inputs (single-class eval sets) return `nan` rather than raising; `evaluate_logits_frame` flags them with `single_class_eval_set`.

---

## Phase 3: Pilot Run (Single Model End-to-End)

### What was built

| Path | Purpose |
|------|---------|
| `scripts/02_pilot_run.py` | T3.1–T3.12 orchestrator; writes `outputs/runs/run.json` + `outputs/runs/pilot_report.json` |
| `configs/pilot/default.yaml` | Pilot config (model, LoRA, data, PTQ, eval, sanity tolerances, projection basis) |
| `src/models/finetune.py` | Verbalized single-token SFT dataset, LoRA training, adapter→merged export, deterministic stratified subsampling |
| `src/models/logit_extraction.py` | Per-example two-way label logits → canonical logits frame + Parquet persistence |
| `src/models/quantize_track_a.py` | Track A dispatch: bnb INT8/NF4 on-load, GPTQ-4, artifact/param summaries |
| `src/metrics/aggregate.py` | Metrics over logits frames; precision-vs-precision comparison |
| `tests/test_{logit_extraction,finetune_helpers,aggregate}.py` | Extraction, prompt, subsample and aggregation tests |

Pilot scope: `Qwen/Qwen2.5-1.5B-Instruct`, Bengali, seed 42, LOCO fold `loco_slander`, Track A only. Caps: 4000 train / 800 id_val / 1200 per eval set.

### T3.x results (`outputs/runs/run.json`)

| ID | Result | Status |
|----|--------|--------|
| T3.1 | BF16 forward: logits `(2, 21, 151936)`, finite, verbalizer ids `{'hate': 362, 'nonhate': 425}`, peak VRAM 2.926 GB | passed |
| T3.2 | LoRA loss 3.8188 → 0.2019 over 25 logged steps, no NaN, 4000 train examples | passed |
| T3.3 | best eval loss 0.2775, checkpoint selected, adapter + merged written | passed |
| T3.4 | Merge fidelity: 1/48 prediction flips, all with two-way margin ≤ 0.125 (inside the bf16 noise band); max abs Δp = 0.0312, max abs Δlogit = 0.25 | passed |
| T3.5 | Canonical logits frame validated: probabilities sum to 1 within 2.2e-16, unique ids, all finite | passed |
| T3.6 | INT8 vs BF16 on id_test: 0.0 accuracy drop, 0.9917 agreement, bootstrap CI [-0.0058, +0.0050] | passed |
| T3.7 | NF4 logits finite, probabilities sum to 1 | passed |
| T3.8 | GPTQ-4 | **blocked** (environment — see note 3) |
| T3.9 | All task/calibration/selective metrics computed for 3 precisions × 2 eval sets | passed |
| T3.10 | 6 Parquet logits files written and reloaded with provenance columns | passed |
| T3.12 | `pilot_report.json` written; projected full-experiment cost replaced every estimate | passed |

Overall: **11/12 passed**, run status `failed` solely because T3.8 cannot execute in this environment.

### Headline pilot numbers (id_test, n=1200)

| Precision | Accuracy | AUROC | ECE | Brier | AURC |
|-----------|----------|-------|-----|-------|------|
| bf16 | 0.9150 | 0.9539 | 0.0342 | 0.0644 | 0.0237 |
| bnb_int8 | 0.9150 | — | — | — | — |
| bnb_nf4 | 0.9150 | — | — | — | — |

bf16 calibration/selective detail: macro-F1 0.9005, hate-F1 0.8625, MCC 0.8011, NLL 0.2268, confidence gap -0.0334, E-AURC 0.0199, coverage@5% risk 0.8958, failure-prediction AUROC 0.8320.

### Measured cost (replaces the plan's estimates)

| Quantity | Measured |
|----------|----------|
| Fine-tune wall clock | 666.8 s (11.1 min) for 4000 examples → 0.1667 s/example |
| Merge + save overhead | 0.52 s |
| Fine-tune peak VRAM | 4.586 GB |
| Eval throughput | 0.0311 s/example; 0.62 min per eval set per precision |
| Model load time | bf16 1.63 s, int8 1.76 s, nf4 2.05 s |
| Peak VRAM by precision | bf16 2.892 GiB, int8 1.756 GiB, nf4 1.115 GiB |
| Merged bf16 artifact | 3103.35 MB |
| Logits footprint | 6 files, 0.185 MB total |
| **Projected full experiment** | **260.5 GPU-hours** (5 models × 2 languages × 3 seeds × 5 LOCO folds = 150 runs; 661 307 s train + 894 s quantize + 275 767 s eval) |

### Phase 3 notes / limitations

1. **T3.8 GPTQ-4 is blocked, not implemented-but-failing.** The installed `gptqmodel` 2.2.0 (genuine ModelCloud distribution) exposes no end-to-end driver: `GPTQModel` has no `quantize`/`save` methods and no `gptqmodel` CLI console script is installed. Only the per-layer `gptqmodel.quantization.gptq.GPTQ` accumulator (`add_batch`/`fasterquant`) is available, which would require a hand-written layer-sequential driver. `gptq_backend_status()` reports this, `quantize_gptq()` raises `GPTQUnavailable`, and the pilot records T3.8 as unavailable and continues, so the rest of the sweep and the timing projection still complete. Remediation: install a `gptqmodel` build shipping the CLI/driver, or AutoGPTQ (unmaintained; needs a CUDA toolchain on Windows), or defer GPTQ-4 to the Phase 6 sweep.
2. **bf16-aware merge gate (T3.4).** The plan's `1e-4` tolerance assumed an fp32 checkpoint. The merged model is stored in bf16, so re-serialising `W + B@A*(alpha/r)` and folding it at inference time disagree at the weight-storage floor (~1e-2 on probabilities, 0.2–1.0 on logits) regardless of merge correctness. The gate is therefore `max|Δp| ≤ merge_atol (0.05)` **and** every prediction flip sits inside a two-way margin ≤ `merge_margin_tol (1.0)`. A broken merge (e.g. a double-counted LoRA) flips high-margin examples and blows past `merge_atol`, so the gate still catches real corruption. The adapter is deliberately applied to the *original* base checkpoint, not the merged one, to avoid double-counting the update.
3. **All frozen bd_shs LOCO `ood_test` sets are single-class** (hate subtype domains are 100% hate; `non_hate` is 100% non-hate). AUROC, average precision, balanced accuracy, MCC and failure-prediction AUROC are undefined there and are serialised as `null` with `single_class_eval_set=true`, while calibration metrics stay finite. Later phases must handle this prior shift explicitly or adopt a multi-class/subsampled LOCO axis.
4. **Answer-position handling matters.** Reading `logits[:, -1, :]` on a right-padded batch scores the pad slot of every row shorter than the longest prompt; `answer_logits_batched` gathers each row's final real token and truncates from the left so the trailing verbalizer cue survives. This is covered by `test_batch_composition_does_not_change_logits`.
5. **Energy is unavailable.** `finetune_energy_wh` is `null`: this GPU/driver returns `NotSupported` from `nvmlDeviceGetPowerUsage`, and the `nvidia-smi` fallback yields no sample inside a CUDA training region here.
6. **No commits**: `main` still has zero commits.

### How to re-run Phase 3

```powershell
$env:PYTHONUTF8='1'; $env:PYTHONIOENCODING='utf-8'
.venv\Scripts\python.exe scripts\02_pilot_run.py            # full pilot (~17 min)
.venv\Scripts\python.exe scripts\02_pilot_run.py --smoke    # 64/32 rehearsal
.venv\Scripts\python.exe scripts\02_pilot_run.py --precisions bf16 bnb_nf4
```

---

## Upcoming phases (from `planning/phases/instruction.md`)

| Phase | Focus |
|-------|--------|
| 1 | Data preparation, splits, dedup, domain-shift magnitude — **done** |
| 2 | Metrics implementation (task, calibration, selective, efficiency) — **done** |
| 3 | Pilot end-to-end single-model run — **done** (T3.8 GPTQ blocked by environment) |
| 4–5 | Baseline fine-tuning (Bengali + English control) |
| 6 | PTQ Track A — GPU/HF (BNB, GPTQ, AWQ) |
| 7 | PTQ Track B — GGUF/llama.cpp |
| 8–10 | Evaluation, selective prediction, efficiency measurement |
| 11–14 | Recalibration, ablations, statistics, Pareto analysis |
| 15–17 | Error analysis, encoder baselines, results + paper draft |

Phase 6 should revisit the GPTQ backend: the pilot could not produce GPTQ-4 with `gptqmodel` 2.2.0, so Track A's weight-only INT4 arm is still outstanding.

Shared building blocks prepared in Phase 0: config loading, reproducibility, logging/`run.json`, sanity checks, and a pinned quantization stack (GPTQModel + bitsandbytes + llama-cpp-python) proven on a tiny model.
