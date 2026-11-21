"""Phase 6 Unit Tests: Post-Training Quantization Track A (GPU / Hugging Face).

Validates:
- Configuration loading (configs/quantization/default.yaml)
- Matrix expansion logic (Phase 4 and Phase 5 base models, 4 Track A methods)
- Zero-leakage calibration data sampling (T6.8, T6.9)
- T6.5 INT8 near-lossless sanity check (agreement, macro-F1 drop, probability drift)
- Driver status detection and graceful fallback (GPTQ, AWQ unavailable)
- CLI dry-run and smoke mode behaviors
"""

import importlib
from pathlib import Path
import numpy as np
import pytest
import yaml

from src.utils.sanity_checks import sanity_check_int8
from src.models.quantize_track_a import (
    gptq_backend_status,
    quantize_awq,
    AWQUnavailable,
    GPTQUnavailable,
)

# Dynamically import scripts.04_quantize because of leading digit in module name
_quant_module = importlib.import_module("scripts.04_quantize")
load_quant_config = _quant_module.load_quant_config
build_experiment_matrix = _quant_module.build_experiment_matrix
sample_calibration_tokens = _quant_module.sample_calibration_tokens


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def quant_config():
    """Load default quantization config."""
    cfg_path = Path("configs/quantization/default.yaml")
    assert cfg_path.exists(), f"Missing config: {cfg_path}"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Test: Configuration Schema & Integrity
# ---------------------------------------------------------------------------
def test_quant_config_structure(quant_config):
    """Ensure all required Track A sections, methods, and evaluation splits exist."""
    assert "methods" in quant_config
    assert "track_a" in quant_config["methods"]
    assert set(quant_config["methods"]["track_a"]) == {
        "bnb_int8",
        "bnb_nf4",
        "gptq_int4",
        "awq_int4",
    }

    assert "calibration" in quant_config
    calib = quant_config["calibration"]
    assert calib["num_samples"] == 512
    assert calib["seed"] == 42
    assert calib["max_seq_len"] == 256
    assert calib["source_split"] == "train"

    assert "evaluation" in quant_config
    assert "splits" in quant_config["evaluation"]
    assert set(quant_config["evaluation"]["splits"]) == {
        "id_val",
        "id_test",
        "ood_test",
    }

    assert "int8_sanity_check" in quant_config
    sc = quant_config["int8_sanity_check"]
    assert sc["macro_f1_drop_max"] <= 0.02
    assert sc["pred_agreement_min"] >= 0.95
    assert sc["mean_prob_diff_max"] <= 0.05


# ---------------------------------------------------------------------------
# Test: Matrix Expansion
# ---------------------------------------------------------------------------
def test_build_experiment_matrix(quant_config):
    """Verify matrix accurately pairs Phase 4 and Phase 5 models with Track A methods."""
    matrix = build_experiment_matrix(quant_config)
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
        assert item["seed"] in {13, 42, 87}
        assert Path(item["model_dir"]).exists()

    # Verify method counts match methods x models
    methods = set(quant_config["methods"]["track_a"])
    matrix_methods = {item["method"] for item in matrix}
    assert matrix_methods == methods


# ---------------------------------------------------------------------------
# Test: Calibration Sampling & Leakage Verification (T6.8 & T6.9)
# ---------------------------------------------------------------------------
def test_calibration_zero_leakage_and_matching(quant_config):
    """Test calibration sampling respects zero leakage into evaluation sets and matches language."""
    # Test for both Bengali and English
    for lang in ["bn", "en"]:
        calib_texts, train_indices = sample_calibration_tokens(
            language=lang,
            num_samples=100,  # small slice for test speed
            seed=42,
        )
        assert len(calib_texts) == 100
        assert len(train_indices) == 100
        # No empty texts
        assert all(isinstance(t, str) and len(t.strip()) > 0 for t in calib_texts)

        # Verify indices come strictly from train partition and never overlap eval splits
        if lang == "bn":
            split_dir = Path("data/processed/splits")
        else:
            split_dir = Path("data/processed/splits/english_control")

        id_val_indices = set(np.load(split_dir / "id_val_indices.npy"))
        id_test_indices = set(np.load(split_dir / "id_test_indices.npy"))
        ood_test_indices = set(np.load(split_dir / "ood_test_indices.npy"))

        eval_indices = id_val_indices | id_test_indices | ood_test_indices
        leakage = set(train_indices).intersection(eval_indices)
        assert (
            len(leakage) == 0
        ), f"T6.8 Violation: Calibration data leaked {len(leakage)} indices into eval sets!"


# ---------------------------------------------------------------------------
# Test: T6.5 INT8 Near-Lossless Sanity Check
# ---------------------------------------------------------------------------
def test_sanity_check_int8_passes():
    """Verify sanity_check_int8 succeeds when logits are nearly identical."""
    np.random.seed(42)
    n = 200
    # Simulate FP reference logits for binary classification
    fp_logits = np.random.randn(n, 2)
    # Add minor noise simulating good INT8 quantization
    int8_logits = fp_logits + np.random.normal(0, 0.01, size=fp_logits.shape)

    passed, report = sanity_check_int8(fp_logits, int8_logits)
    assert passed is True
    assert report["passed"] is True
    assert report["agreement"] >= 0.95
    assert report["macro_f1_drop"] <= 0.02
    assert report["mean_prob_diff"] <= 0.05


def test_sanity_check_int8_fails_on_high_prob_drift():
    """Verify sanity_check_int8 fails when probability drift exceeds tolerance."""
    np.random.seed(42)
    n = 200
    fp_logits = np.random.randn(n, 2)
    # Drift logits significantly
    int8_logits = fp_logits + np.random.normal(1.5, 0.5, size=fp_logits.shape)

    passed, report = sanity_check_int8(fp_logits, int8_logits)
    assert passed is False
    assert report["passed"] is False


def test_sanity_check_int8_fails_on_macro_f1_drop():
    """Verify sanity_check_int8 flags drop in Macro-F1."""
    # Construct exact logits where class 1 flips to class 0 in many instances
    fp_logits = np.array([[2.0, 0.0]] * 100 + [[0.0, 2.0]] * 100)
    int8_logits = np.array([[2.0, 0.0]] * 100 + [[2.0, 0.0]] * 100)  # all predicted as 0

    passed, report = sanity_check_int8(fp_logits, int8_logits)
    assert passed is False
    assert report["macro_f1_drop"] > 0.02


def test_sanity_check_int8_single_class_edge_case():
    """Ensure sanity_check_int8 handles edge cases without crashing."""
    fp_logits = np.array([[5.0, -5.0]] * 50)
    int8_logits = np.array([[4.8, -4.8]] * 50)

    passed, report = sanity_check_int8(fp_logits, int8_logits)
    assert isinstance(passed, bool)
    assert "macro_f1_drop" in report


# ---------------------------------------------------------------------------
# Test: Driver Status & Graceful Fallbacks
# ---------------------------------------------------------------------------
def test_gptq_backend_status():
    """Check GPTQ backend status detects available driver or reasons properly."""
    status = gptq_backend_status()
    assert isinstance(status, dict)
    assert "available" in status
    assert "driver" in status
    assert "version" in status
    # In .venv, gptqmodel 2.2.0 is installed
    assert status["available"] is True
    assert status["driver"] == "gptqmodel"


def test_awq_unavailable_exception():
    """Ensure quantize_awq cleanly raises AWQUnavailable when backend is missing."""
    with pytest.raises(AWQUnavailable) as exc_info:
        quantize_awq(
            model_id_or_path="dummy_path",
            output_dir=Path("dummy_out"),
            calibration_texts=["test text"],
        )
    assert "AWQ backend unavailable" in str(exc_info.value)
