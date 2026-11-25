# Phase 10: Efficiency measurement

Run from the repository root on the target GPU after closing workload-heavy
applications and allowing the device to reach a stable idle temperature:

```powershell
.venv\Scripts\python.exe scripts\06_measure_efficiency.py
.venv\Scripts\python.exe -m pytest tests\test_phase10.py -q --basetemp=.pytest_tmp_phase10
```

The default protocol uses ten warmups and two blocks of 100 synchronized
single-example inferences per artifact. Model loading and tokenization
are outside the timed inference region. The same representative `id_test`
prompt is used within a model run; token counts summarize the first 100
`id_test` texts. Runs are sequential. `--limit 1` rehearses one artifact;
`--force` repeats existing measurements. Individual JSON records under
`outputs/results/phase10_runs/` make the sweep resumable. Each artifact runs
in a fresh process so GPTQ and other backend CUDA allocations cannot affect
the next artifact's memory baseline.

`outputs/results/efficiency_metrics.csv` and `.parquet` hold one row per
artifact. The `phase10_run.json` gate report records whether acceptance
conditions were met. A nonzero exit status with `status: incomplete` means
the measured values were saved but at least one gate is unmet.

Disk size is the size of a stored BF16, GPTQ, or GGUF artifact. BNB INT8
and NF4 quantize when loaded; they have no separate stored quantized file,
so `disk_size_mb` is null and `source_checkpoint_mb` gives the shared BF16
source size. Peak VRAM is sampled device usage above the pre-load baseline;
this can include other GPU activity. Host RAM is process RSS above its
pre-load baseline. Latency is the time to run the model on one prepared
prompt through the answer-logit position. Throughput is repeated scored
examples divided by active wall time. BF16, Track A and Track B retain
their backend identities in the table.

Energy requires actual GPU board-power samples at 10 Hz. When NVML and
`nvidia-smi` return no power draw, idle power and energy per 1,000 remain
null, and T10.3/T10.4 fail. Values are never estimated from TDP. The
current RTX 4060 driver reports `N/A` for power draw, so full Phase 10
energy acceptance requires a power-capable driver or external meter.
Temperature stability requires a difference below 5 °C from after warmup
to the end of the timed blocks;
latency stability requires CV below 0.15. Memory ordering compares matched
precisions of the same base run within each backend track.

## Current run

The full sweep measured 144/144 artifacts (18 each of BF16, BNB INT8,
BNB NF4, GPTQ INT4, GGUF F16, Q8_0, Q4_K_M, and Q3_K_M). The 90 required
matched memory comparisons passed. One additional Q3_K_M versus Q4_K_M
memory comparison failed and remains a diagnostic because that ordering
is outside T10.2's INT8/INT4 requirement.

Latency CV passed for 82/144 artifacts; thermal stability passed for
66/144. All 144 idle-power and energy fields are unavailable because the
GPU driver returns no board-power readings. The run manifest therefore
records `status: incomplete`. These measurements support analysis of disk
size, memory, latency, throughput, and tokens per example, but the Phase 10
acceptance criteria are **not met** on this hardware/session. Reaching them
requires a power-readable GPU or external meter and a controlled thermal
environment, followed by `--force` to replace the current records.
