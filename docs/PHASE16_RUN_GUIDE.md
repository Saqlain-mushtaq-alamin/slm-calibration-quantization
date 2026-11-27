# Phase 16 — encoder baselines run guide

Status: **PARTIAL** until both FP32 training runs and both dynamic INT8 CPU evaluations finish and the final gate passes. All four real-model smoke arms passed after the model files and Bangla normalizer were installed. Smoke outputs are rehearsal artifacts, never research results.

Run the commands below from `D:\canvas\slm-calibration-quantization` in PowerShell. Run one model at a time on the GPU. Training commands automatically resume the latest valid checkpoint; `--resume` additionally refuses to start if none exists. Remove a STOP file before resuming. The training script selects the checkpoint using validation Macro-F1, evaluates frozen ID and OOD test sets only after training, and writes `run.json`. INT8 uses a new FP32 model loaded from the saved checkpoint, converts supported Linear modules, and evaluates both FP32 and INT8 on CPU.

## Prerequisites and rehearsal

```powershell
Set-Location 'D:\canvas\slm-calibration-quantization'
& .\.venv\Scripts\python.exe -m pip install git+https://github.com/csebuetnlp/normalizer.git
& .\.venv\Scripts\python.exe scripts\12_smoke_encoders.py --allow-download
& .\.venv\Scripts\python.exe -m pytest tests\test_encoder.py -q
```

The normalizer command is needed for BanglaBERT. The smoke script downloads model files if required and performs separate FP32 forward/backward, checkpoint reload/continuation, and INT8 inference rehearsals for each model. It writes only under `outputs\encoders\smoke\`. Check `availability.json` in each model's smoke directory before full training. If a download fails, restore network access and rerun the rehearsal.

## BanglaBERT FP32 full training

```powershell
Set-Location 'D:\canvas\slm-calibration-quantization'
& .\.venv\Scripts\python.exe scripts\10_train_encoders.py --config configs\encoder_banglabert.yaml
```

## BanglaBERT resume after interruption

```powershell
Set-Location 'D:\canvas\slm-calibration-quantization'
Remove-Item -LiteralPath 'outputs\encoders\banglabert\STOP' -ErrorAction SilentlyContinue
& .\.venv\Scripts\python.exe scripts\10_train_encoders.py --config configs\encoder_banglabert.yaml --resume
```

## BanglaBERT dynamic INT8 CPU evaluation

```powershell
Set-Location 'D:\canvas\slm-calibration-quantization'
& .\.venv\Scripts\python.exe scripts\11_evaluate_encoder_int8.py --config configs\encoder_banglabert.yaml
```

## XLM-R FP32 full training

```powershell
Set-Location 'D:\canvas\slm-calibration-quantization'
& .\.venv\Scripts\python.exe scripts\10_train_encoders.py --config configs\encoder_xlmr.yaml
```

## XLM-R resume after interruption

```powershell
Set-Location 'D:\canvas\slm-calibration-quantization'
Remove-Item -LiteralPath 'outputs\encoders\xlmr\STOP' -ErrorAction SilentlyContinue
& .\.venv\Scripts\python.exe scripts\10_train_encoders.py --config configs\encoder_xlmr.yaml --resume
```

## XLM-R dynamic INT8 CPU evaluation

```powershell
Set-Location 'D:\canvas\slm-calibration-quantization'
& .\.venv\Scripts\python.exe scripts\11_evaluate_encoder_int8.py --config configs\encoder_xlmr.yaml
```

## Monitoring background training

To start BanglaBERT in a hidden background process, run this command instead of its foreground command. Replace `encoder_banglabert.yaml` and `banglabert` with `encoder_xlmr.yaml` and `xlmr` for XLM-R. Do not start both simultaneously.

```powershell
New-Item -ItemType Directory -Path 'outputs\encoders\banglabert\logs' -Force | Out-Null
$p = Start-Process -FilePath (Resolve-Path '.\.venv\Scripts\python.exe') -ArgumentList 'scripts\10_train_encoders.py','--config','configs\encoder_banglabert.yaml' -WorkingDirectory (Get-Location).Path -RedirectStandardOutput 'outputs\encoders\banglabert\logs\console.out' -RedirectStandardError 'outputs\encoders\banglabert\logs\console.err' -WindowStyle Hidden -PassThru
$p.Id
Get-Content 'outputs\encoders\banglabert\logs\console.err' -Wait
```

To inspect progress without a live stream:

```powershell
Get-Content 'outputs\encoders\banglabert\run.json' | ConvertFrom-Json | Select-Object status,global_step,epoch,checkpoint,runtime_seconds
Get-ChildItem 'outputs\encoders\banglabert\checkpoints' -Directory | Sort-Object Name
```

For XLM-R, replace `banglabert` with `xlmr` in the two inspection commands.

## Graceful stop

Create the model's STOP file. The callback saves a checkpoint at the next optimizer step and marks the run `interrupted`. Remove the file before resuming.

```powershell
New-Item -ItemType File -Path 'outputs\encoders\banglabert\STOP' -Force
```

For XLM-R:

```powershell
New-Item -ItemType File -Path 'outputs\encoders\xlmr\STOP' -Force
```

## Verify completed outputs

```powershell
Get-Content 'outputs\encoders\banglabert\run.json' | ConvertFrom-Json | Select-Object status,global_step,epoch,checkpoint
Get-Content 'outputs\encoders\banglabert\int8_run.json' | ConvertFrom-Json | Select-Object status,quantized_modules
Get-ChildItem 'outputs\encoders\banglabert\predictions','outputs\encoders\banglabert\metrics'
Get-Content 'outputs\encoders\xlmr\run.json' | ConvertFrom-Json | Select-Object status,global_step,epoch,checkpoint
Get-Content 'outputs\encoders\xlmr\int8_run.json' | ConvertFrom-Json | Select-Object status,quantized_modules
Get-ChildItem 'outputs\encoders\xlmr\predictions','outputs\encoders\xlmr\metrics'
```

## Final Phase 16 validation gate

```powershell
& .\.venv\Scripts\python.exe scripts\13_validate_encoders.py
Get-Content 'outputs\encoders\phase16_status.json' | ConvertFrom-Json | Format-List
```

The gate returns exit code 1 and keeps Phase 16 `PARTIAL` if any arm is missing or invalid. It checks saved ID/OOD example alignment, labels, logits, metrics, and frozen fingerprints. Bengali OOD is single-class; its ECE is reported with an explicit comparability note. Dynamic INT8 latency and memory are CPU measurements. No GPU INT8 speed or energy claim is made.

## Output and configuration notes

Each model directory contains `checkpoints/`, `best_checkpoint/`, `predictions/`, `metrics/`, `logs/`, and `run.json`; `int8_run.json` records the separate INT8 arm. The default microbatch is 8 with accumulation 4 and max length 256. If 8 GB VRAM is insufficient, edit `batch_size` to 4 in the relevant YAML **before its first run** and record that configuration; do not change it midway through a resumable run. Existing frozen splits and registered thresholds are never rewritten.
