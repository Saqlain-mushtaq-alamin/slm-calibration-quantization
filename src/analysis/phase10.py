"""Phase 10 live inference efficiency measurements for the full artifact grid."""

from __future__ import annotations

import gc
import json
import statistics
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.metrics.efficiency import (
    gpu_power_limit_w,
    gpu_temperature_c,
    hardware_summary,
    measure_inference_blocks,
    read_gpu_power_w,
)
from src.utils.atomic import atomic_write_json

PHASES = ("phase4", "phase5", "phase6", "phase7")
METHODS = {"bf16", "bnb_int8", "bnb_nf4", "gptq_int4", "f16", "q8_0", "q4_k_m", "q3_k_m"}


def discover_artifacts(root: Path) -> list[dict[str, Any]]:
    manifests = []
    for phase in PHASES:
        for path in sorted((root / "outputs" / "runs" / phase).glob("*.json")):
            manifest = json.loads(path.read_text(encoding="utf-8"))
            name = manifest.get("artifact_name") or manifest.get("run_name")
            method = manifest.get("method") or "bf16"
            if not name or method not in METHODS:
                raise ValueError(f"{path}: invalid artifact identity")
            manifests.append({"phase": phase, "artifact": name, "method": method,
                              "manifest_path": path, "manifest": manifest})
    if not manifests:
        raise ValueError("No Phase 4–7 artifact manifests found")
    if len({(m["phase"], m["artifact"]) for m in manifests}) != len(manifests):
        raise ValueError("Duplicate artifact manifests")
    return manifests


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def artifact_storage(root: Path, item: dict[str, Any]) -> dict[str, Any]:
    manifest = item["manifest"]
    phase = item["phase"]
    method = item["method"]
    if phase in {"phase4", "phase5"}:
        path = _resolve(root, manifest["artifacts"]["merged_path"])
        kind = "stored_bf16"
    elif phase == "phase6" and method in {"bnb_int8", "bnb_nf4"}:
        path = _resolve(root, manifest["model_path"])
        kind = "shared_bf16_source_on_load_quantization"
    elif phase == "phase6":
        path = _resolve(root, manifest["artifact_dir"])
        kind = "stored_quantized"
    else:
        path = root / "outputs" / "quantized" / "track_b" / f"{item['artifact']}.gguf"
        kind = "stored_gguf"
    if not path.exists():
        raise FileNotFoundError(path)
    files = [path] if path.is_file() else list(path.rglob("*"))
    size = sum(p.stat().st_size for p in files if p.is_file())
    if size <= 0:
        raise ValueError(f"Empty model artifact: {path}")
    return {"model_path": str(path), "storage_kind": kind,
            "disk_size_mb": size / 1e6 if kind != "shared_bf16_source_on_load_quantization" else None,
            "source_checkpoint_mb": size / 1e6 if kind == "shared_bf16_source_on_load_quantization" else None}


def _gpu_used_mb() -> float | None:
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            return pynvml.nvmlDeviceGetMemoryInfo(handle).used / 1e6
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        pass
    try:
        proc = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, timeout=20, check=False)
        return float(proc.stdout.strip().splitlines()[0]) if proc.returncode == 0 else None
    except Exception:
        return None


class _ResourceMonitor:
    """Sample system GPU memory and process RSS through load and inference."""

    def __init__(self, interval_s: float = 0.1) -> None:
        self.interval_s = interval_s
        self.gpu_mb: list[float] = []
        self.rss_mb: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self):
        import psutil

        process = psutil.Process()

        def sample() -> None:
            while not self._stop.is_set():
                gpu = _gpu_used_mb()
                if gpu is not None:
                    self.gpu_mb.append(gpu)
                self.rss_mb.append(process.memory_info().rss / 1e6)
                self._stop.wait(self.interval_s)

        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_args):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)


def _idle_power(seconds: float, sampling_hz: float = 10.0) -> float | None:
    first = read_gpu_power_w()
    if first is None:
        return None
    readings = [first]
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        time.sleep(min(1 / sampling_hz, max(0, deadline - time.perf_counter())))
        value = read_gpu_power_w()
        if value is not None and np.isfinite(value) and value > 0:
            readings.append(value)
    return statistics.fmean(readings) if len(readings) >= 2 else None


def _sample_texts(root: Path, item: dict[str, Any], limit: int = 100) -> list[str]:
    base = item["manifest"].get("base_run_name") or item["artifact"]
    language = item["manifest"].get("language")
    source_phase = "phase4" if language == "bn" else "phase5"
    logits = root / "outputs" / "logits" / source_phase / base / "bf16__id_test.parquet"
    ids = pd.read_parquet(logits, columns=["example_id"])["example_id"].astype(str).head(limit)
    data_file = root / "data" / "processed" / (
        "bd_shs_processed.csv" if language == "bn" else "english_control_matched.csv"
    )
    texts = pd.read_csv(data_file, usecols=["example_id", "text"], dtype={"example_id": str})
    selected = texts.set_index("example_id")["text"].reindex(ids)
    if selected.isna().any() or selected.empty:
        raise ValueError(f"Missing benchmark text for {item['artifact']}")
    template = (
        "{text}\nবাক্যটি ঘৃণামূলক কি না? উত্তর:"
        if language == "bn" else
        "Classify the following text as hate speech or not.\nText: {text}\nAnswer:"
    )
    from src.models.finetune import build_prompt

    return [build_prompt(text, template) for text in selected.tolist()]


def _load_inference(root: Path, item: dict[str, Any], path: Path):
    import torch
    from src.models import finetune as ft

    method = item["method"]
    if item["phase"] == "phase7":
        from src.models.quantize_track_b import load_gguf_model

        offload = int(item["manifest"].get("n_gpu_layers", -1))
        model = load_gguf_model(path, n_ctx=512, n_gpu_layers=offload, verbose=False, logits_all=True)
        return model, None, "gguf"
    if method == "gptq_int4":
        from src.models.quantize_track_a import load_gptq

        model, tokenizer = load_gptq(str(path))
    elif method in {"bnb_int8", "bnb_nf4"}:
        from src.models.quantize_track_a import load_quantized_hf

        model = load_quantized_hf(str(path), method)
        tokenizer = ft.load_tokenizer(str(path))
    else:
        attn = "eager" if "gemma" in item["manifest"].get("model_key", "") else None
        model = ft.load_base_model(str(path), attn_implementation=attn)
        tokenizer = ft.load_tokenizer(str(path))
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return model, tokenizer, "hf"


def _benchmark_fn(model, tokenizer, backend: str, prompts: list[str]):
    import torch

    prompt = prompts[len(prompts) // 2]
    if backend == "hf":
        from src.models.logit_extraction import answer_logits_batched, encode_label_token

        hate = encode_label_token(tokenizer, " A")
        nonhate = encode_label_token(tokenizer, " B")
        lengths = [min(len(tokenizer(p, add_special_tokens=False)["input_ids"]) + 1, 256) for p in prompts]

        def fn():
            with torch.inference_mode():
                answer_logits_batched(model, tokenizer, [prompt], hate, nonhate, max_len=256)

        sync = torch.cuda.synchronize if torch.cuda.is_available() else lambda: None
        return fn, sync, lengths
    tokens = model.tokenize(prompt.encode("utf-8"), add_bos=True)
    if len(tokens) > 256:
        bos = model.token_bos()
        tokens = [tokens[0]] + tokens[-255:] if tokens[0] == bos else tokens[-256:]
    lengths = [min(len(model.tokenize(p.encode("utf-8"), add_bos=True)), 256) for p in prompts]
    token_id = model.tokenize(b" A", add_bos=False)[-1]

    def fn():
        model.reset()
        model.eval(tokens)
        float(model.scores[model.n_tokens - 1, token_id])

    return fn, lambda: None, lengths


def measure_artifact(root: Path, item: dict[str, Any], *, warmup: int = 10,
                     repeats: int = 100, idle_seconds: float = 60.0) -> dict[str, Any]:
    import psutil
    import torch

    manifest = item["manifest"]
    storage = artifact_storage(root, item)
    row: dict[str, Any] = {
        "phase": item["phase"], "artifact": item["artifact"], "method": item["method"],
        "base_run": manifest.get("base_run_name") or item["artifact"],
        "model": manifest.get("model_key"), "seed": manifest.get("seed"),
        "language": manifest.get("language"), "source_status": manifest.get("status"),
        "backend": "llama.cpp" if item["phase"] == "phase7" else "HF",
        **storage, "warmup": warmup, "repeats_per_block": repeats, "blocks": 2,
    }
    if manifest.get("status") != "success":
        row.update({"status": "source_failed", "error": manifest.get("error")})
        return row
    model = tokenizer = None
    before_gpu = _gpu_used_mb()
    before_rss = psutil.Process().memory_info().rss / 1e6
    row["gpu_temperature_before_c"] = gpu_temperature_c()
    try:
        prompts = _sample_texts(root, item)
        row["idle_power_watts"] = _idle_power(idle_seconds)
        with _ResourceMonitor() as resources:
            load_start = time.perf_counter()
            model, tokenizer, backend = _load_inference(root, item, Path(storage["model_path"]))
            row["load_s"] = time.perf_counter() - load_start
            fn, sync, lengths = _benchmark_fn(model, tokenizer, backend, prompts)
            row["tokens_per_example_mean"] = float(np.mean(lengths))
            row["tokens_per_example_p95"] = float(np.percentile(lengths, 95))
            row["benchmark_prompt_tokens"] = lengths[len(lengths) // 2]
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            benchmark = measure_inference_blocks(fn, synchronize=sync, warmup=warmup, repeats=repeats)
        row.update({k: v for k, v in benchmark.items() if k != "blocks"})
        row["block_details"] = benchmark["blocks"]
        torch_peak = torch.cuda.max_memory_allocated() / 1e6 if backend == "hf" and torch.cuda.is_available() else None
        row["peak_vram_mb"] = (max(max(resources.gpu_mb) - before_gpu, 0.0) if before_gpu is not None and resources.gpu_mb else torch_peak)
        row["vram_measurement"] = "NVML_sampled_device_delta" if before_gpu is not None and resources.gpu_mb else "torch_allocated" if torch_peak is not None else None
        row["peak_host_ram_mb"] = max(max(resources.rss_mb) - before_rss, 0.0) if resources.rss_mb else None
        row["gpu_temperature_after_c"] = gpu_temperature_c()
        limit = gpu_power_limit_w()
        idle = row["idle_power_watts"]
        if idle is not None and limit is not None and 0 < idle < limit:
            energies = [
                max(0.0, b["active_energy_j"] - idle * b["active_duration_s"]) * 1000 / repeats
                for b in benchmark["blocks"] if b["active_energy_j"] is not None
            ]
            row["energy_per_1k_joules"] = statistics.fmean(energies) if len(energies) == 2 else None
            row["energy_repeat_relative_diff"] = (
                abs(energies[0] - energies[1]) / max(statistics.fmean(energies), 1e-9)
                if len(energies) == 2 else None
            )
        else:
            row["energy_per_1k_joules"] = None
            row["energy_repeat_relative_diff"] = None
        temp_a, temp_b = row["gpu_temperature_before_c"], row["gpu_temperature_after_c"]
        row["latency_stable"] = bool(row["latency_cv_max"] < 0.15)
        row["thermal_stable"] = bool(temp_a is not None and temp_b is not None and abs(temp_b - temp_a) < 5)
        row["energy_stable"] = bool(row["energy_repeat_relative_diff"] is not None and row["energy_repeat_relative_diff"] < 0.15)
        row["status"] = "complete" if all((row["latency_stable"], row["thermal_stable"], row["energy_stable"], row["peak_vram_mb"] is not None)) else "incomplete"
        return row
    except Exception as exc:
        row.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        return row
    finally:
        if item["phase"] == "phase7" and model is not None:
            from src.models.quantize_track_b import free_gguf

            free_gguf(model)
        model = tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_sweep(root: Path, results_dir: Path, *, warmup: int = 10, repeats: int = 100,
              idle_seconds: float = 60.0, limit: int | None = None, force: bool = False) -> pd.DataFrame:
    items = discover_artifacts(root)
    selected = items[:limit] if limit is not None else items
    run_dir = results_dir / "phase10_runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, item in enumerate(selected, 1):
        target = run_dir / item["phase"] / f"{item['artifact']}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_file() and not force:
            row = json.loads(target.read_text(encoding="utf-8"))
            if row.get("warmup") != warmup or row.get("repeats_per_block") != repeats:
                raise ValueError(f"Benchmark protocol changed for {target}; use --force")
        else:
            row = measure_artifact(root, item, warmup=warmup, repeats=repeats, idle_seconds=idle_seconds)
            atomic_write_json(target, row)
        rows.append(row)
        print(f"[{i}/{len(selected)}] {item['artifact']}: {row['status']}", flush=True)
    frame = pd.DataFrame(rows)
    return frame
