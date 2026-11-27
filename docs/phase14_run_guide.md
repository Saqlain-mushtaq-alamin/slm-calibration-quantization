# Phase 14 trade-off and Pareto analysis

Phase 14 runs entirely on CPU from the existing Phase 8 task metrics, Phase 9 selective metrics, and Phase 10 efficiency measurements. It does not load models or use the GPU.

## Run

From PowerShell:

```powershell
Set-Location 'D:\canvas\slm-calibration-quantization'
& .\.venv\Scripts\python.exe scripts\10_pareto_analysis.py
```

The command returns a nonzero exit code while required objective pairs are unavailable. Read the generated status and report:

```powershell
Get-Content outputs\results\phase14_run.json | ConvertFrom-Json | Select-Object status,n_frontiers_computed,n_frontiers_expected,energy_limitation
Get-Content outputs\results\phase14_report.md
```

The script saves `phase14_frontier_points.csv` and `.parquet`, `phase14_frontier_summary.csv`, `phase14_id_ood_comparison.csv`, `phase14_derived_ratios.csv` and `.parquet`, `phase14_report.md`, and `phase14_run.json` under `outputs/results/`. Source Parquet SHA-256 values are recorded in the manifest. Existing source result tables are read-only.

## Current measured result

The executed analysis is **PARTIAL**: 12 of 20 required language × evaluation × objective frontiers are computable. All eight energy frontiers are unavailable because Phase 10 recorded no usable GPU power telemetry (`energy_per_1k_joules` is null for every efficiency row). Memory and stable-latency frontiers use measured rows only. The report lists all computed Pareto points and paired descriptive ratios; it leaves joules ratios undefined.

Six pilot task rows have no matching Phase 10 efficiency record and are excluded with an explicit count. The 432 joined research rows match on artifact, method, model, seed, and language. Task and selective rows also match frozen evaluation fingerprints. Bengali OOD ECE is from a single-class target and cannot be compared directly with conventional two-class OOD ECE.

Phase 14 must remain **PARTIAL** until energy is measured with a stable, comparable protocol and all missing frontiers are rerun. The current GPU reports power draw as `N/A` through `nvidia-smi`; no energy values were inferred or entered by hand. Do not run a new GPU efficiency sweep while the BanglaBERT encoder is training.

## Tests

```powershell
& .\.venv\Scripts\python.exe -m ruff check src\analysis\pareto.py scripts\10_pareto_analysis.py tests\test_pareto.py
& .\.venv\Scripts\python.exe -m pytest tests\test_pareto.py tests\test_phase8.py tests\test_phase9.py tests\test_phase10.py -q
```

## BanglaBERT GPU load

The encoder training command is active independently of Phase 14. The observed GPU load during this analysis was about 5.8 GB of 8.2 GB VRAM and below 72°C, with checkpoints continuing to advance. Phase 14 makes no GPU allocations. Monitor training without launching a second model:

```powershell
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits
Get-ChildItem outputs\encoders\banglabert\checkpoints -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 3 Name,LastWriteTime
```

If you need to stop training, create the STOP file and wait for the next optimizer step to save a checkpoint:

```powershell
New-Item -ItemType File -Path outputs\encoders\banglabert\STOP -Force
```

Remove that file before resuming. Do not edit the existing run's batch size between checkpoints; the resume guard rejects a changed training configuration. The default batch size 8 has not shown an out-of-memory error in the observed run.
