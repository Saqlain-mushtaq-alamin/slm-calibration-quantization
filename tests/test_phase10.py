"""Phase 10 artifact coverage, size semantics and benchmark validation."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.analysis.phase10 import artifact_storage, discover_artifacts, memory_order_checks
from src.metrics.efficiency import measure_inference_blocks


def _manifest(root, phase, name, method=None):
    path = root / "outputs" / "runs" / phase / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"status": "success", "artifact_name": name, "method": method,
            "model_path": "model/bf16", "artifact_dir": "model/gptq",
            "artifacts": {"merged_path": "model/bf16"}}
    path.write_text(json.dumps(data), encoding="utf-8")
    return data


def test_phase10_inventory_and_disk_size_are_honest(tmp_path):
    bf = tmp_path / "model" / "bf16"
    bf.mkdir(parents=True)
    (bf / "weights.safetensors").write_bytes(b"x" * 100)
    _manifest(tmp_path, "phase4", "base", "bf16")
    _manifest(tmp_path, "phase6", "base__bnb_int8", "bnb_int8")
    items = discover_artifacts(tmp_path)
    assert len(items) == 2
    stored = artifact_storage(tmp_path, items[0])
    on_load = artifact_storage(tmp_path, items[1])
    assert stored["disk_size_mb"] == pytest.approx(100 / 1e6)
    assert on_load["disk_size_mb"] is None
    assert on_load["source_checkpoint_mb"] == pytest.approx(100 / 1e6)


def test_phase10_benchmark_warmup_and_missing_power():
    calls = []

    def infer():
        calls.append(1)

    result = measure_inference_blocks(infer, warmup=3, repeats=5, blocks=2,
                                      power_reader=lambda: None)
    assert len(calls) == 13
    assert result["power_available"] is False
    assert result["latency_p50_ms"] > 0
    assert result["throughput_examples_per_sec"] > 0
    assert all(block["active_energy_j"] is None for block in result["blocks"])


def test_phase10_memory_order_compares_only_matched_models():
    frame = pd.DataFrame([
        {"base_run": "base", "method": "bf16", "peak_vram_mb": 3000},
        {"base_run": "base", "method": "bnb_int8", "peak_vram_mb": 1800},
        {"base_run": "base", "method": "bnb_nf4", "peak_vram_mb": 1000},
        {"base_run": "base", "method": "gptq_int4", "peak_vram_mb": 1200},
        {"base_run": "base", "method": "f16", "peak_vram_mb": 3100},
        {"base_run": "base", "method": "q8_0", "peak_vram_mb": 1700},
        {"base_run": "base", "method": "q4_k_m", "peak_vram_mb": 1100},
        {"base_run": "base", "method": "q3_k_m", "peak_vram_mb": 900},
    ])
    checks = memory_order_checks(frame)
    assert checks["n_comparisons"] == 6
    assert checks["passed"]
    frame.loc[frame["method"] == "gptq_int4", "peak_vram_mb"] = 2000
    assert not memory_order_checks(frame)["passed"]
