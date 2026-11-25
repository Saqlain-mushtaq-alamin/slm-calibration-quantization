# Phase 11: Post-hoc recalibration

Run from the repository root:

```powershell
.venv\Scripts\python.exe scripts\07_recalibrate.py
.venv\Scripts\python.exe -m pytest tests\test_phase11.py -q
```

The sweep reads full-run saved logits from Phases 4–7. It fits one positive
temperature by minimizing ID-validation negative log likelihood for each
artifact (`T_own`), and transfers the corresponding BF16 ID-validation fit
(`T_fp`). It evaluates the raw, own-fit, and FP-transfer arms on both ID-test
and OOD-test. It never fits on either test set. The script checks that saved
probabilities and decisions agree with the raw logits and that each artifact's
test examples match its BF16 reference by ID and label.

The outputs are `outputs/results/recalibration_results.{csv,parquet}`,
`recalibration_temperatures.{csv,parquet}`, and `phase11_run.json`. Results
include the temperature, fitting file and split, raw and scaled ECE (equal
mass and equal width), Brier score, NLL, and their differences. Temperature
fits at the search boundary are retained and visible in the table.

An oracle fit is emitted only when a matching `ood_val` file exists for that
artifact. Its result is limited to OOD-test and labeled
`oracle -- not deployable`. The current full-run data has no OOD-validation
files, so the oracle arm is unavailable. Pilot logits also lack ID validation
and are listed as skipped in the run manifest. The current run has 144
complete full-run artifacts and 864 results: 144 artifacts × 3 arms × 2 test
sets. Own-fit ID-test equal-mass ECE improved or stayed equal in 130/144
artifacts. This diagnostic does not guarantee improvement on every held-out
sample; the fitted objective is validation NLL, not test ECE.
