"""Phase 6 Unit Tests: Post-Training Quantization Track A (GPU / Hugging Face).

Validates:
- Configuration loading (configs/quantization/default.yaml)
- Matrix expansion logic (Phase 4 and Phase 5 base models, 4 Track A methods)
- Zero-leakage calibration data sampling (T6.8, T6.9)
- T6.5 INT8 near-lossless sanity check (agreement, macro-F1 drop, probability drift)
- Driver status detection and graceful fallback (GPTQ, AWQ unavailable)
- CLI dry-run and status matrix display
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import load_config
from src.utils.sanity_checks import sanity_check_int8
from src.models.quantize_track_a import (
    gptq_backend_status,
    quantize_awq,
    AWQUnavailable,
    GPTQUnavailable,
)

def _load_quantize_script():
    spec = importlib.util.spec_from_file_location("quantize_script", ROOT / "scripts" / "04_quantize.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["quantize_script"] = mod
    spec.loader.exec_module(mod)
    return mod

_quant_module = _load_quantize_script()
build_quant_matrix = _quant_module.build_quant_matrix
prepare_calibration_data = _quant_module.prepare_calibration_data
parse_run_name = _quant_module.parse_run_name


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def quant_config():
    """Load default quantization config."""
    cfg_path = ROOT / "configs" / "quantization" / "default.yaml"
    assert cfg_path.exists(), f"Missing config: {cfg_path}"
    return load_config(cfg_path)


# ---------------------------------------------------------------------------
# Test: Configuration Schema & Integrity
# ---------------------------------------------------------------------------
def test_quant_config_structure(quant_config):
    """Ensure all required Track A sections, methods, and evaluation splits exist."""
    assert "track_a" in quant_config
    assert "methods" in quant_config["track_a"]
    assert set(quant_config["track_a"]["methods"]) == {
        "bnb_int8",
        "bnb_nf4",
        "gptq_int4",
        "awq_int4",
    }

    assert "calibration" in quant_config["track_a"]
    calib = quant_config["track_a"]["calibration"]
    assert calib["n_samples"] == 512
    assert calib["seed"] == 42
    assert calib["max_len"] == 256

    assert "eval" in quant_config["track_a"]
    assert "splits" in quant_config["track_a"]["eval"]
    assert set(quant_config["track_a"]["eval"]["splits"]) == {
        "id_val",
        "id_test",
        "ood_test",
    }

    assert "int8_tolerance" in quant_config["track_a"]
    sc = quant_config["track_a"]["int8_tolerance"]
    assert sc["max_f1_drop"] <= 0.02
    assert sc["min_agreement"] >= 0.95
    assert sc["max_mean_delta_p"] <= 0.05

    assert "sources" in quant_config
    assert "phase4" in quant_config["sources"]
    assert "phase5" in quant_config["sources"]


# ---------------------------------------------------------------------------
# Test: Run Name Parser & Matrix Expansion
# ---------------------------------------------------------------------------
def test_parse_run_name():
    """Verify regex extraction of language, model, and seed from run directory name."""
    parsed_bn = parse_run_name("bn_qwen2_5_0_5b_instruct_lora_bf16_s13")
    assert parsed_bn["language"] == "bn"
    assert parsed_bn["model_key"] == "qwen2_5_0_5b_instruct"
    assert parsed_bn["seed"] == 13

    parsed_en = parse_run_name("en_llama3_2_1b_instruct_lora_bf16_s42")
    assert parsed_en["language"] == "en"
    assert parsed_en["model_key"] == "llama3_2_1b_instruct"
    assert parsed_en["seed"] == 42


def test_build_experiment_matrix(quant_config):
    """Verify matrix accurately pairs Phase 4 and Phase 5 models with Track A methods."""
    matrix = build_quant_matrix(quant_config)
    assert len(matrix) > 0

    # Ensure all items in matrix have required keys
    for item in matrix:
        assert item["language"] in {"bn", "en"}
        assert item["method"] in {
            "bnb_int8",
            "bnb_nf4",
            "gptq_int4",
            "awq_int4",
        }
        assert isinstance(item["seed"], int)
        assert Path(item["run_dir"]).exists()
        assert "target_model_dir" in item
        assert "merged_dir" in item

    # Verify method counts match methods x models
    methods = set(quant_config["track_a"]["methods"])
    matrix_methods = {item["method"] for item in matrix}
    assert matrix_methods == methods


# ---------------------------------------------------------------------------
# Test: Calibration Sampling & Leakage Verification (T6.8 & T6.9)
# ---------------------------------------------------------------------------
def test_calibration_zero_leakage_and_matching(quant_config):
    """Test calibration sampling respects zero leakage into evaluation sets and matches language."""
    for source_key in ["phase4", "phase5"]:
        source_cfg = quant_config["sources"][source_key]
        calib_texts, calib_ids, prov = prepare_calibration_data(
            source_cfg,
            n_samples=100,  # small slice for test speed
            seed=42,
        )
        assert len(calib_texts) == 100
        assert len(calib_ids) == 100
        assert all(isinstance(t, str) and len(t.strip()) > 0 for t in calib_texts)

        # Check provenance guarantees
        assert prov["verified_train_only"] is True
        assert prov["leaked_into_eval"] is False
        assert prov["language"] == source_cfg["language"]


# ---------------------------------------------------------------------------
# Test: T6.5 INT8 Near-Lossless Sanity Check
# ---------------------------------------------------------------------------
def test_sanity_check_int8_passes():
    """Verify sanity_check_int8 succeeds when logits and probabilities are nearly identical."""
    np.random.seed(42)
    n = 200
    y_true = np.random.randint(0, 2, size=n)
    prob_hate_fp = np.clip(y_true * 0.8 + np.random.normal(0.1, 0.05, size=n), 0.01, 0.99)
    pred_fp = (prob_hate_fp >= 0.5).astype(int)

    fp_df = pd.DataFrame({"y_true": y_true, "prediction": pred_fp, "prob_hate": prob_hate_fp})

    # INT8 with tiny drift
    prob_hate_int8 = np.clip(prob_hate_fp + np.random.normal(0, 0.005, size=n), 0.0, 1.0)
    pred_int8 = (prob_hate_int8 >= 0.5).astype(int)
    int8_df = pd.DataFrame({"y_true": y_true, "prediction": pred_int8, "prob_hate": prob_hate_int8})

    res = sanity_check_int8(fp_df, int8_df)
    assert res["passed"] is True
    assert res["agreement"] >= 0.95
    assert res["f1_drop"] <= 0.02
    assert res["mean_delta_p"] <= 0.05


def test_sanity_check_int8_fails_on_high_prob_drift():
    """Verify sanity_check_int8 fails when probability drift exceeds tolerance."""
    np.random.seed(42)
    n = 200
    y_true = np.random.randint(0, 2, size=n)
    prob_hate_fp = np.random.uniform(0.1, 0.9, size=n)
    pred_fp = (prob_hate_fp >= 0.5).astype(int)

    fp_df = pd.DataFrame({"y_true": y_true, "prediction": pred_fp, "prob_hate": prob_hate_fp})

    # INT8 with massive drift
    prob_hate_int8 = np.clip(prob_hate_fp + 0.3, 0.0, 1.0)
    pred_int8 = (prob_hate_int8 >= 0.5).astype(int)
    int8_df = pd.DataFrame({"y_true": y_true, "prediction": pred_int8, "prob_hate": prob_hate_int8})

    res = sanity_check_int8(fp_df, int8_df)
    assert res["passed"] is False
    assert res["checks"]["delta_p_ok"] is False


def test_sanity_check_int8_fails_on_macro_f1_drop():
    """Verify sanity_check_int8 flags drop in Macro-F1."""
    y_true = np.array([0] * 50 + [1] * 50)
    pred_fp = y_true.copy()
    prob_fp = np.where(y_true == 1, 0.9, 0.1)
    fp_df = pd.DataFrame({"y_true": y_true, "prediction": pred_fp, "prob_hate": prob_fp})

    # INT8 that mispredicts all class 1 as class 0
    pred_int8 = np.zeros(100, dtype=int)
    prob_int8 = np.full(100, 0.1)
    int8_df = pd.DataFrame({"y_true": y_true, "prediction": pred_int8, "prob_hate": prob_int8})

    res = sanity_check_int8(fp_df, int8_df)
    assert res["passed"] is False
    assert res["checks"]["f1_drop_ok"] is False
    assert res["f1_drop"] > 0.02


def test_sanity_check_int8_single_class_edge_case():
    """Ensure sanity_check_int8 handles single-class datasets (e.g. OOD split) gracefully."""
    y_true = np.ones(50, dtype=int)
    pred_fp = np.ones(50, dtype=int)
    prob_fp = np.full(50, 0.8)
    fp_df = pd.DataFrame({"y_true": y_true, "prediction": pred_fp, "prob_hate": prob_fp})

    pred_int8 = np.ones(50, dtype=int)
    prob_int8 = np.full(50, 0.79)
    int8_df = pd.DataFrame({"y_true": y_true, "prediction": pred_int8, "prob_hate": prob_int8})

    res = sanity_check_int8(fp_df, int8_df)
    assert res["single_class"] is True
    assert res["f1_drop"] is None
    assert res["passed"] is True


# ---------------------------------------------------------------------------
# Test: Driver Status & Graceful Fallbacks
# ---------------------------------------------------------------------------
def test_gptq_backend_status():
    """Check GPTQ backend status detects installed gptqmodel build."""
    status = gptq_backend_status()
    assert isinstance(status, dict)
    assert status["installed"] is True
    assert status["has_model_quantize"] is True
    assert status["has_model_save"] is True
    assert "GPTQModel" in status["driver_api"]


def test_awq_unavailable_exception():
    """Ensure quantize_awq cleanly raises AWQUnavailable when backend is missing."""
    with pytest.raises(AWQUnavailable) as exc_info:
        quantize_awq(
            model_path="dummy_path",
            calib_texts=["test text"],
            output_path="dummy_out",
        )
    assert "AWQ requires llmcompressor" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test: CLI Dry-Run & Status Execution
# ---------------------------------------------------------------------------
def test_cli_dry_run():
    """Ensure scripts/04_quantize.py --dry-run completes cleanly with code 0."""
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "04_quantize.py"), "--dry-run"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Dry-run failed: {result.stderr}"
    assert "Track A Quantization Matrix" in result.stdout


def test_cli_status():
    """Ensure scripts/04_quantize.py --status completes cleanly with code 0."""
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "04_quantize.py"), "--status"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Status command failed: {result.stderr}"
    assert "Phase 6 Post-Training Quantization (Track A) Status" in result.stdout

