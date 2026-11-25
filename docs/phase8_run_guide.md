# Phase 8: Task performance and calibration

Run from the repository root with the project's virtual environment:

```powershell
.venv\Scripts\python.exe scripts\05_evaluate.py
.venv\Scripts\python.exe -m pytest tests\test_phase8.py -q
```

The evaluator reads saved Parquet logits from `outputs/logits/phase4`,
`phase5`, `phase6`, `phase7`, and `pilot`. Smoke-run directories are excluded
because they contain deliberately truncated samples and do not provide
like-for-like comparisons to full BF16 references. No model is loaded.

Results are saved in `outputs/results/all_metrics.csv` and `.parquet`,
`outputs/results/delta_vs_fp.csv` and `.parquet`, with
`outputs/results/phase8_run.json` recording status and coverage. Each row
includes the source logit file and a fingerprint of example IDs and labels.
Reference matching requires the same base run, eval set, model, seed,
language, LOCO fold, row count, and sample fingerprint. The script fails
before writing tables if a reference is missing or misaligned.

`ece_equal_mass_15` is the primary ECE used for quantization deltas.
Equal-width 15-bin and equal-mass 10-bin ECE are also reported. Brier
decomposition uses 15 equal-width bins. Relative ECE change divides by
`max(reference ECE, 1e-6)`; a zero reference F1 makes relative F1 undefined
and fails validation.

For single-class OOD sets, balanced accuracy, AUROC, average precision,
and MCC are mathematically undefined. Their table cells are empty/null and
`single_class_eval_set` is true. All defined metrics must be finite. These
nulls are the only allowed missing metric cells.

The current full-run sweep processes 438 logit files and produces 382
reference deltas. Seventy-five rows are single-class OOD sets. Rerunning
from the same logits produces byte-identical CSV tables.
