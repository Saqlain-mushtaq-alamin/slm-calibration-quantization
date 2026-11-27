# Phase 13: statistical analysis and hypothesis testing

Run from the repository root:

```powershell
.venv\Scripts\python.exe scripts\09_statistical_analysis.py
```

The run reads the existing Phase 8–12 Parquet tables and writes only these Phase 13 files under `outputs/results`: `hypothesis_tests.csv`, `hypothesis_tests.parquet`, `phase13_interactions.csv`, and `phase13_run.json`. It takes a few seconds on a CPU with the default 10,000 bootstrap resamples. It does not launch quantization, evaluation, or the ablation sweep.

Each precision result is paired with BF16 on model, seed, language, LOCO fold, evaluation split, sample count, and sample fingerprint. Seeds are averaged within each model family before inference; there are three independent family values for most contrasts and two for A1. The CSV reports the mean paired difference, percentile bootstrap 95% confidence interval, exact paired sign-flip p-value for the small family count, Holm-adjusted p-value across the 138 contrasts, paired Cohen's d, and descriptive Cliff's delta. The interval is descriptive at this sample size; a nonzero interval alone is not evidence of significance. The smallest possible two-sided exact p-value with three families is 0.25. TOST equivalence uses the registered ECE margin 0.02 and F1 margin 0.01, with a separate Holm adjustment over equivalence tests. A nonsignificant difference does not establish equivalence.

H2 and H3 use paired difference-in-differences as the primary contrasts. `phase13_interactions.csv` contains supplementary GPTQ-versus-BF16 mixed models with model and nested seed random effects; these p-values are exploratory given three model families. OOD sets may be single-class and have different class priors, so H2 describes these observed shifts rather than a general OOD population.

The run is **PARTIAL**. Phase 12 remains **PARTIAL**, with 176 saved result rows. AWQ calibration variants are unavailable, leaving H7 based on GPTQ alone. A8 still lacks an OOD-validation oracle temperature arm: validation would require a frozen, disjoint OOD validation set, provenance-checked temperature fitting on that set, and paired evaluation against own-fit and FP-transfer temperatures on held-out OOD test examples. H5b cannot resolve its oracle comparison until then. H6 cannot claim Pareto preference until the Phase 14 frontier is computed; the currently measured INT8 peak-VRAM ratios are also below the proposed 2× threshold. These missing requirements are recorded in the Phase 13 manifest and are not treated as completed hypotheses.

To rerun with a different bootstrap count:

```powershell
.venv\Scripts\python.exe scripts\09_statistical_analysis.py --n-boot 20000 --seed 42
```

Only Phase 13 output files are refreshed. To check the implementation:

```powershell
.venv\Scripts\python.exe -m pytest tests\test_statistical_tests.py tests\test_phase13.py
```
