# SLM Calibration × Quantization

> Quantization × Calibration × Selective Prediction for Sub-3B SLMs on Bengali Hate-Speech Classification under Domain Shift

## Quick start (Phase 0)

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Unix
source .venv/bin/activate

pip install -U pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

python scripts/00_verify_environment.py
pytest tests/ -q
```

## Project layout

See `planning/phases/instruction.md` for the full plan and directory contract.

| Path | Purpose |
|------|---------|
| `configs/` | YAML experiment configs |
| `src/` | Library code (data, models, metrics, analysis, utils) |
| `scripts/` | Numbered pipeline entrypoints |
| `tests/` | Pytest suites |
| `data/` | Raw / processed / frozen splits (gitignored) |
| `outputs/` | Models, logits, runs, results, figures (gitignored) |
| `docs/` | Phase overviews and notes |
| `planning/` | Original experiment plan (not committed) |

## Global rules

1. Never advance a phase with unmet acceptance criteria.
2. Save per-example logits for every model × eval-set.
3. Write `run.json` for every training/quantization/evaluation run.
4. Track A (GPU/HF) and Track B (GGUF/llama.cpp) stay separate.
5. OOD data is never used for training/tuning/temperature fitting (except oracle arm).
6. Seeds fixed; splits frozen + hashed; library versions pinned.
