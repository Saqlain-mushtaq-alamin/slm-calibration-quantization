"""Efficiency measurement: latency, memory, and GPU energy.

The Phase 3 pilot only needs wall-clock/VRAM/power bookkeeping; Phase 10 reuses
the same helpers for the full efficiency sweep.
"""

from __future__ import annotations

import platform
import statistics
import subprocess
import threading
import time
from contextlib import contextmanager
from typing import Any, Iterator

_GIB = 1024.0**3


# ---------------------------------------------------------------------------
# GPU power
# ---------------------------------------------------------------------------

def read_gpu_power_w() -> float | None:
    """Current GPU board power in watts, or None if unavailable.

    NVML ``nvmlDeviceGetPowerUsage`` is preferred; this consumer GPU/driver
    combination returns ``NotSupported`` (see Phase 0 note 8), so we fall back
    to ``nvidia-smi --query-gpu=power.draw``.
    """
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mw = int(pynvml.nvmlDeviceGetPowerUsage(handle))
            if mw > 0:
                return mw / 1000.0
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        pass

    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=False,
            timeout=20,
        )
        token = (res.stdout or "").strip().splitlines()
        if token:
            val = float(token[0].strip())
            return val if val > 0 else None
    except Exception:
        pass
    return None


def gpu_temperature_c() -> float | None:
    """GPU core temperature in Celsius, or None."""
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=False,
            timeout=20,
        )
        line = (res.stdout or "").strip().splitlines()
        if line:
            return float(line[0].strip())
    except Exception:
        pass
    return None


def gpu_power_limit_w() -> float | None:
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.limit", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=False,
            timeout=20,
        )
        line = (res.stdout or "").strip().splitlines()
        if line:
            return float(line[0].strip())
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

def current_vram_gb() -> float | None:
    """Currently allocated + reserved CUDA memory in GiB, or None on CPU."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return (torch.cuda.memory_allocated() + torch.cuda.memory_reserved()) / _GIB
    except Exception:
        return None


def peak_vram_gb() -> float | None:
    """Peak allocated CUDA memory since the last ``reset_peak_vram``."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return torch.cuda.max_memory_allocated() / _GIB
    except Exception:
        return None


def reset_peak_vram() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def total_vram_gb() -> float | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        props = torch.cuda.get_device_properties(0)
        total = getattr(props, "total_memory", None) or props.total_mem
        return total / _GIB
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Timed execution
# ---------------------------------------------------------------------------

class Stopwatch:
    """Wall-clock + power/energy accounting for a named code region."""

    def __init__(self, name: str, sample_interval_s: float = 0.0) -> None:
        self.name = name
        self.sample_interval_s = sample_interval_s
        self.wall_s = 0.0
        self.energy_wh: float | None = None
        self.mean_power_w: float | None = None
        self.peak_power_w: float | None = None
        self.peak_vram_gb: float | None = None
        self.gpu_temp_c: float | None = None
        self._powers: list[float] = []

    def __enter__(self) -> "Stopwatch":
        reset_peak_vram()
        self._powers = []
        self._start_wall = time.perf_counter()
        p = read_gpu_power_w()
        if p is not None:
            self._powers.append(p)
        return self

    def sample(self) -> None:
        p = read_gpu_power_w()
        if p is not None:
            self._powers.append(p)

    def __exit__(self, *exc: object) -> None:
        self.wall_s = time.perf_counter() - self._start_wall
        self.peak_vram_gb = peak_vram_gb()
        p = read_gpu_power_w()
        if p is not None:
            self._powers.append(p)
        self.gpu_temp_c = gpu_temperature_c()
        if self._powers:
            self.mean_power_w = float(statistics.fmean(self._powers))
            self.peak_power_w = float(max(self._powers))
            self.energy_wh = self.mean_power_w * (self.wall_s / 3600.0)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "wall_s": round(self.wall_s, 4),
            "wall_min": round(self.wall_s / 60.0, 4),
            "peak_vram_gb": None if self.peak_vram_gb is None else round(self.peak_vram_gb, 3),
            "mean_power_w": None if self.mean_power_w is None else round(self.mean_power_w, 2),
            "peak_power_w": None if self.peak_power_w is None else round(self.peak_power_w, 2),
            "energy_wh": None if self.energy_wh is None else round(self.energy_wh, 6),
            "gpu_temp_c": self.gpu_temp_c,
        }


@contextmanager
def timed(name: str, record: dict[str, Any] | None = None) -> Iterator[Stopwatch]:
    """Context manager yielding a :class:`Stopwatch`; merges result into ``record``."""
    sw = Stopwatch(name)
    with sw:
        yield sw
    if record is not None:
        record[name] = sw.as_dict()


# ---------------------------------------------------------------------------
# Latency benchmarking
# ---------------------------------------------------------------------------

def benchmark_callable(
    fn,
    *,
    n_warmup: int = 3,
    n_runs: int = 20,
) -> dict[str, float]:
    """Latency distribution (ms) for ``fn()`` with warmup."""
    for _ in range(max(0, n_warmup)):
        fn()
    latencies: list[float] = []
    for _ in range(max(1, n_runs)):
        t0 = time.perf_counter()
        fn()
        latencies.append((time.perf_counter() - t0) * 1000.0)
    arr = sorted(latencies)
    return {
        "n_runs": int(n_runs),
        "mean_ms": float(statistics.fmean(arr)),
        "median_ms": float(statistics.median(arr)),
        "p95_ms": float(arr[min(len(arr) - 1, int(0.95 * len(arr)))]),
        "min_ms": float(arr[0]),
        "max_ms": float(arr[-1]),
        "std_ms": float(statistics.pstdev(arr)) if len(arr) > 1 else 0.0,
    }


def measure_inference_blocks(
    fn,
    *,
    synchronize=lambda: None,
    warmup: int = 10,
    repeats: int = 100,
    blocks: int = 2,
    power_reader=read_gpu_power_w,
    temperature_reader=None,
    sampling_hz: float = 10.0,
) -> dict[str, Any]:
    """Time repeated synchronized inferences and integrate sampled board power.

    Power is sampled concurrently with actual timestamps. A missing power trace
    remains unavailable; it is never inferred from elapsed time or GPU TDP.
    """
    import numpy as np

    if repeats < 2 or blocks < 1 or warmup < 0 or sampling_hz <= 0:
        raise ValueError("invalid benchmark repetitions or sampling rate")
    for _ in range(warmup):
        fn()
        synchronize()
    temperature_start = temperature_reader() if temperature_reader is not None else None
    initial_power = power_reader()
    out = []
    for _ in range(blocks):
        samples: list[tuple[float, float]] = []
        stop = threading.Event()

        def sample(stop=stop, samples=samples) -> None:
            while not stop.is_set():
                watts = power_reader()
                if watts is not None and np.isfinite(watts) and watts > 0:
                    samples.append((time.perf_counter(), float(watts)))
                stop.wait(1.0 / sampling_hz)

        thread = None
        if initial_power is not None:
            thread = threading.Thread(target=sample, daemon=True)
            thread.start()
        latencies = []
        start = time.perf_counter()
        try:
            for _ in range(repeats):
                synchronize()
                t0 = time.perf_counter()
                fn()
                synchronize()
                latencies.append((time.perf_counter() - t0) * 1000.0)
        finally:
            stop.set()
            if thread is not None:
                thread.join(timeout=2)
        elapsed = time.perf_counter() - start
        arr = np.asarray(latencies, dtype=np.float64)
        energy_j = None
        if len(samples) >= 2:
            ts = np.asarray([x[0] for x in samples])
            pw = np.asarray([x[1] for x in samples])
            integrate = getattr(np, "trapezoid", None) or np.trapz
            inner = ts[(ts > start) & (ts < start + elapsed)]
            grid = np.concatenate(([start], inner, [start + elapsed]))
            watts = np.interp(grid, ts, pw, left=pw[0], right=pw[-1])
            energy_j = float(integrate(watts, grid))
        out.append({
            "latency_p50_ms": float(np.percentile(arr, 50)),
            "latency_p95_ms": float(np.percentile(arr, 95)),
            "latency_mean_ms": float(arr.mean()),
            "latency_cv": float(arr.std() / arr.mean()),
            "throughput_examples_per_sec": float(repeats / elapsed),
            "active_duration_s": elapsed,
            "active_energy_j": energy_j,
            "n_power_samples": len(samples),
        })
    temperature_end = temperature_reader() if temperature_reader is not None else None
    return {
        "blocks": out,
        "latency_p50_ms": float(np.median([b["latency_p50_ms"] for b in out])),
        "latency_p95_ms": float(np.median([b["latency_p95_ms"] for b in out])),
        "latency_cv_max": max(b["latency_cv"] for b in out),
        "throughput_examples_per_sec": float(np.median([b["throughput_examples_per_sec"] for b in out])),
        "power_available": initial_power is not None,
        "measurement_temperature_start_c": temperature_start,
        "measurement_temperature_end_c": temperature_end,
    }


def hardware_summary() -> dict[str, Any]:
    """Static hardware description attached to every efficiency record."""
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": _cpu_count(),
        "total_vram_gib": None,
        "gpu_name": None,
        "power_limit_w": None,
    }
    try:
        import torch

        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["torch_cuda"] = torch.version.cuda
            vram = total_vram_gb()
            info["total_vram_gib"] = None if vram is None else round(vram, 2)
    except Exception:
        pass
    info["power_limit_w"] = gpu_power_limit_w()
    return info


def _cpu_count() -> int:
    try:
        import os

        return int(os.cpu_count() or 0)
    except Exception:
        return 0


def model_size_bytes(model) -> int:
    """Total on-disk-equivalent parameter bytes of a (possibly quantized) model."""
    total = 0
    try:
        for p in model.parameters():
            total += p.numel() * p.element_size()
    except Exception:
        return 0
    return int(total)


def peak_vram_measurement() -> dict[str, Any]:
    """One-shot VRAM snapshot used in run manifests."""
    return {
        "peak_vram_gib": None if peak_vram_gb() is None else round(peak_vram_gb() or 0.0, 3),
        "current_vram_gib": None if current_vram_gb() is None else round(current_vram_gb() or 0.0, 3),
        "total_vram_gib": None if total_vram_gb() is None else round(total_vram_gb() or 0.0, 2),
    }
