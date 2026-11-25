# Phase 9: Selective prediction evaluation

Run from the repository root:

```powershell
.venv\Scripts\python.exe scripts\05_evaluate_selective.py
.venv\Scripts\python.exe -m pytest tests\test_phase9.py tests\test_selective.py -q --basetemp=.pytest_tmp_phase9
```

The evaluator reads the full-run Parquet logits from `outputs/logits/phase4`,
`phase5`, `phase6`, `phase7`, and `pilot`. It excludes deliberately truncated
smoke runs. The output is `outputs/results/selective_metrics.csv` and
`.parquet`, one risk–coverage Parquet file per input under
`outputs/results/rc_curves/`, and `outputs/results/phase9_run.json`.

Confidence is the top-class probability; ties retain input order. AURC is
the mean of prefix risks (the discrete Riemann definition), and E-AURC
subtracts the optimal ranking for the same number of errors. Selective
accuracy uses the nearest integer accepted count, with a minimum of one
example for small samples. Coverage at target risk is the largest prefix
whose observed risk is at or below the target. Failure AUROC is null only
when every prediction is correct or every prediction is wrong; the
`failure_auroc_defined` flag identifies that case.

The plan's T9.3 states that selective accuracy must decrease as coverage
grows. This is not a mathematical property of prefix accuracy: adding
correct examples can increase it. The evaluator reports
`selective_accuracy_monotonic_70_90` as a diagnostic and checks every
reported selective accuracy against its saved risk–coverage curve instead.

The current sweep contains 438 metric rows and 438 curves. All AURC values
are in [0, 1], all E-AURC values are nonnegative, and all coverage-at-risk
values are in [0, 1]. Sixty-seven rows fail the plan's monotonicity
assertion; their values and curves are retained. No failure AUROC is
undefined in the current full-run data.
