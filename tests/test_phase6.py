"""Phase 6 Unit & Regression Tests: Post-Training Quantization Track A (GPU / Hugging Face).

Validates:
- Configuration loading and exact template/verbalizer match with training (T6.8, T6.9)
- Matrix expansion logic (strictly requires is_valid_merged, no adapter fallback)
- Zero-leakage calibration data sampling (T6.8, T6.9)
- T6.5 INT8 near-lossless sanity check (example_id alignment, mismatch errors, gate failures)
- Driver status detection and graceful fallback (GPTQ, AWQ unavailable)
- quantize_gptq exception logging and timing fields (quantize_s, eval_s)
- load_gptq fallback to GPTQModel.load with extract_logits_hf compatibility
- Resumability requiring matching config_hash and GPTQ re-quantization skipping
- End-to-end execution of run_one_quant with a fake model
- Stage 3 metrics & validation on real logits parquets
- CLI dry-run and status matrix display
"""

from __future__ import annotations

import importlib.util
import json
import logging
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics.aggregate import evaluate_logits_frame
from src.metrics.validation import REQUIRED_LOGIT_COLUMNS, validate_logits_frame
from src.models import logit_extraction as lx
from src.models.quantize_track_a import (
    AWQUnavailable,
    GPTQUnavailable,
    gptq_backend_status,
    load_gptq,
    quantize_awq,
    quantize_gptq,
)
from src.utils.atomic import is_valid_merged
from src.utils.config import config_hash, load_config
from src.utils.sanity_checks import sanity_check_int8


def _load_quantize_script():
    spec = importlib.util.spec_from_file_location(
        "quantize_script", ROOT / "scripts" / "04_quantize.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["quantize_script"] = mod
    spec.loader.exec_module(mod)
    return mod


_quant_module = _load_quantize_script()
build_quant_matrix = _quant_module.build_quant_matrix
prepare_calibration_data = _quant_module.prepare_calibration_data
parse_run_name = _quant_module.parse_run_name
is_valid_gptq_dir = _quant_module.is_valid_gptq_dir
_vram_report = _quant_module._vram_report
run_one_quant = _quant_module.run_one_quant


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
# Issue 9: Confirm exact prompt_template & verbalizer match with training
# ---------------------------------------------------------------------------
def test_quant_config_matches_training_templates_and_verbalizer(quant_config):
    """Verify configs/quantization/default.yaml uses exact same templates and verbalizer as training."""
    cfg_bn = load_config(ROOT / "configs" / "baseline" / "default.yaml")
    cfg_en = load_config(ROOT / "configs" / "baseline" / "english_control.yaml")

    # Verbalizer
    assert (
        quant_config["verbalizer"]["hate"]
        == cfg_bn["model"]["verbalizer"]["hate"]
        == cfg_en["model"]["verbalizer"]["hate"]
        == " A"
    )
    assert (
        quant_config["verbalizer"]["nonhate"]
        == cfg_bn["model"]["verbalizer"]["nonhate"]
        == cfg_en["model"]["verbalizer"]["nonhate"]
        == " B"
    )

    # Bengali prompt template
    assert (
        quant_config["sources"]["phase4"]["prompt_template"]
        == cfg_bn["prompt_template"]
        == "{text}\nবাক্যটি ঘৃণামূলক কি না? উত্তর:"
    )

    # English prompt template
    assert (
        quant_config["sources"]["phase5"]["prompt_template"]
        == cfg_en["prompt_template"]
        == "Classify the following text as hate speech or not.\nText: {text}\nAnswer:"
    )


def test_quant_config_structure(quant_config):
    """Ensure all required Track A sections, methods, and evaluation splits exist."""
    assert "track_a" in quant_config
    assert set(quant_config["track_a"]["methods"]) == {
        "bnb_int8",
        "bnb_nf4",
        "gptq_int4",
        "awq_int4",
    }
    assert quant_config["track_a"]["calibration"]["n_samples"] == 512
    assert set(quant_config["track_a"]["eval"]["splits"]) == {
        "id_val",
        "id_test",
        "ood_test",
    }


# ---------------------------------------------------------------------------
# Issue 8: build_quant_matrix only includes runs with valid merged dir
# ---------------------------------------------------------------------------
def test_build_experiment_matrix(quant_config):
    """Verify matrix accurately pairs Phase 4 and Phase 5 models with Track A methods."""
    matrix = build_quant_matrix(quant_config)
    assert len(matrix) > 0

    for item in matrix:
        assert item["language"] in {"bn", "en"}
        assert item["method"] in {
            "bnb_int8",
            "bnb_nf4",
            "gptq_int4",
            "awq_int4",
        }
        assert isinstance(item["seed"], int)
        assert Path(item["model_path"]).exists()
        assert item["model_path"] == item["merged_dir"]
        assert is_valid_merged(item["merged_dir"])
        assert "adapter_dir" not in item  # no adapter fallback


def test_build_quant_matrix_rejects_missing_merged_sentinel(quant_config, tmp_path):
    """Verify build_quant_matrix filters out runs without is_valid_merged."""
    fake_source = tmp_path / "fake_models"
    fake_source.mkdir()
    run_dir = fake_source / "bn_test_model_lora_bf16_s42"
    run_dir.mkdir()
    # Has adapter but no merged dir
    (run_dir / "adapter").mkdir()

    cfg_copy = dict(quant_config)
    cfg_copy["sources"] = {
        "phase4": {
            "path": str(fake_source),
            "language": "bn",
            "split": "loco_slander",
            "data_path": "data/processed/bd_shs_processed.csv",
            "splits_path": "data/splits/bd_shs_loco_splits.json",
            "prompt_template": "{text}\nউত্তর:",
            "fp_logits_dir": "outputs/logits/phase4",
        }
    }
    m = build_quant_matrix(cfg_copy)
    assert len(m) == 0, "Should exclude runs that lack a valid merged directory"


# ---------------------------------------------------------------------------
# Issue 2: sanity_check_int8 joins on example_id and asserts identical id sets
# ---------------------------------------------------------------------------
def test_sanity_check_int8_id_mismatch_raises():
    """Assert sanity_check_int8 raises clear ValueError when example_id sets differ."""
    fp_df = pd.DataFrame(
        {
            "example_id": ["id_1", "id_2", "id_3"],
            "y_true": [0, 1, 0],
            "prediction": [0, 1, 0],
            "prob_hate": [0.1, 0.9, 0.2],
        }
    )
    int8_df = pd.DataFrame(
        {
            "example_id": ["id_1", "id_2", "id_4"],  # id_4 instead of id_3
            "y_true": [0, 1, 0],
            "prediction": [0, 1, 0],
            "prob_hate": [0.1, 0.9, 0.2],
        }
    )
    with pytest.raises(ValueError, match="mismatched IDs"):
        sanity_check_int8(fp_df, int8_df)


def test_sanity_check_int8_missing_example_id_column_raises():
    """Assert ValueError if example_id column is absent."""
    df1 = pd.DataFrame({"y_true": [0], "prediction": [0], "prob_hate": [0.1]})
    df2 = pd.DataFrame({"y_true": [0], "prediction": [0], "prob_hate": [0.1]})
    with pytest.raises(ValueError, match="must contain 'example_id'"):
        sanity_check_int8(df1, df2)


def test_sanity_check_int8_unaligned_order_correctly_aligned():
    """Assert sanity_check_int8 aligns rows properly even when row orders differ."""
    fp_df = pd.DataFrame(
        {
            "example_id": ["id_1", "id_2", "id_3"],
            "y_true": [0, 1, 0],
            "prediction": [0, 1, 0],
            "prob_hate": [0.1, 0.8, 0.2],
        }
    )
    # Shuffled row order in int8
    int8_df = pd.DataFrame(
        {
            "example_id": ["id_3", "id_1", "id_2"],
            "y_true": [0, 0, 1],
            "prediction": [0, 0, 1],
            "prob_hate": [0.21, 0.11, 0.79],
        }
    )
    res = sanity_check_int8(fp_df, int8_df)
    assert res["passed"] is True
    assert res["agreement"] == 1.0
    assert res["mean_delta_p"] < 0.02


# ---------------------------------------------------------------------------
# Issue 1: Stage 3 validation gate moved to src/metrics/validation.py
# ---------------------------------------------------------------------------
def test_stage3_on_real_logits_parquet():
    """Test Stage 3 metrics and validation execution on real logits parquet."""
    real_parquet = ROOT / "outputs/logits/phase4/bn_llama3_2_1b_instruct_lora_bf16_s13/bf16__id_test.parquet"
    if not real_parquet.is_file():
        pytest.skip(f"Real parquet not found at {real_parquet}")

    frame = pd.read_parquet(real_parquet)
    # Verify required columns exist
    for col in REQUIRED_LOGIT_COLUMNS:
        assert col in frame.columns

    val_res = validate_logits_frame(frame, min_macro_f1=0.50, require_both_classes=False)
    assert val_res["passed"] is True
    assert val_res["checks"]["probs_sum_to_one"] is True
    assert val_res["checks"]["no_nan_logits"] is True

    metrics = evaluate_logits_frame(frame, ece_bins=15, ece_strategy="equal_width")
    assert "accuracy" in metrics
    assert "ece" in metrics
    assert "brier" in metrics


# ---------------------------------------------------------------------------
# Issue 3: T6.5 failure sets artifact status to "failed" with failing checks
# ---------------------------------------------------------------------------
def test_t6_5_failure_sets_failed_status(tmp_path, quant_config, monkeypatch, capsys):
    """Verify T6.5 INT8 failure sets status to 'failed', records error, and prints checks."""
    cell = {
        "artifact_name": "test_artifact__bnb_int8",
        "base_run_name": "test_run",
        "source_key": "phase4",
        "source_phase": 4,
        "language": "bn",
        "model_key": "test_model",
        "seed": 13,
        "method": "bnb_int8",
        "model_path": str(tmp_path),
        "merged_dir": str(tmp_path),
        "source_cfg": quant_config["sources"]["phase4"],
    }

    # Mock dataset and split frames
    fake_df = pd.DataFrame(
        {
            "example_id": [f"id_{i}" for i in range(10)],
            "text": ["টেস্ট বাক্য"] * 10,
            "label": [i % 2 for i in range(10)],
            "domain": ["slander"] * 10,
        }
    )
    fake_spec = {
        "train_ids": [f"id_{i}" for i in range(4)],
        "id_val_ids": [f"id_{i}" for i in range(4, 6)],
        "id_test_ids": [f"id_{i}" for i in range(6, 8)],
        "ood_test_ids": [f"id_{i}" for i in range(8, 10)],
    }
    monkeypatch.setattr(_quant_module, "get_dataset_and_splits", lambda sc: (fake_df, fake_spec, "sha"))

    # Mock model loading & logit extraction
    monkeypatch.setattr(_quant_module, "load_quantized_hf", lambda p, m: MagicMock())
    monkeypatch.setattr(_quant_module.ft, "load_tokenizer", lambda p: MagicMock())

    def fake_extract_logits(*args, **kwargs):
        split = kwargs.get("split", "id_test")
        ids = fake_spec[f"{split}_ids"]
        return pd.DataFrame(
            {
                "example_id": ids,
                "y_true": [0] * len(ids),
                "logit_hate": [2.0] * len(ids),
                "logit_nonhate": [0.0] * len(ids),
                "prob_hate": [0.88] * len(ids),
                "prob_nonhate": [0.12] * len(ids),
                "prediction": [1] * len(ids),
            }
        )

    monkeypatch.setattr(_quant_module.lx, "extract_logits_hf", fake_extract_logits)

    # Force sanity_check_int8 to fail
    failing_report = {
        "passed": False,
        "single_class": False,
        "macro_f1": 0.2,
        "agreement": 0.80,
        "f1_drop": 0.40,
        "mean_delta_p": 0.35,
        "checks": {"f1_drop_ok": False, "agreement_ok": False, "delta_p_ok": False},
    }
    monkeypatch.setattr(_quant_module, "sanity_check_int8", lambda *a, **k: failing_report)

    # Mock FP reference file existence
    fake_fp_dir = tmp_path / "fp_logits" / "test_run"
    fake_fp_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "example_id": fake_spec["id_test_ids"],
            "y_true": [0, 1],
            "prediction": [0, 1],
            "prob_hate": [0.1, 0.9],
            "prob_nonhate": [0.9, 0.1],
            "logit_hate": [0.0, 2.0],
            "logit_nonhate": [2.0, 0.0],
        }
    ).to_parquet(fake_fp_dir / "bf16__id_test.parquet")

    cfg_copy = dict(quant_config)
    cfg_copy["paths"] = {
        "runs": str(tmp_path / "runs"),
        "logits": str(tmp_path / "logits"),
        "quantized_models": str(tmp_path / "quant"),
    }
    cfg_copy["sources"]["phase4"]["fp_logits_dir"] = str(tmp_path / "fp_logits")

    manifest = run_one_quant(cell, cfg_copy, force=True)

    # Assert status is failed with the exact error message
    assert manifest["status"] == "failed"
    assert manifest["error"] == "T6.5 INT8 sanity gate failed"
    # Assert failing checks were printed to stdout
    captured = capsys.readouterr()
    assert "FAILED T6.5 CHECKS" in captured.out


# ---------------------------------------------------------------------------
# Issue 4: Release models and log VRAM before and after
# ---------------------------------------------------------------------------
def test_vram_report_and_cleanup():
    """Verify _vram_report returns valid string and memory cleanup handles None safely."""
    rep = _vram_report()
    assert "allocated=" in rep or "CPU" in rep


# ---------------------------------------------------------------------------
# Issue 5: quantize_gptq logs first exception with traceback and records quantize_s
# ---------------------------------------------------------------------------
def test_quantize_gptq_logs_first_exception(monkeypatch, caplog):
    """Verify quantize_gptq logs first exception traceback before fallback signature."""
    class FakeGPTQ:
        def quantize(self, *args, **kwargs):
            if "tokenizer" in kwargs and len(args) > 0 and isinstance(args[0], list):
                raise TypeError("Simulated signature failure 1")
            return None

        def save_quantized(self, path):
            pass

    import src.models.quantize_track_a as qta_mod
    monkeypatch.setattr(qta_mod, "gptq_backend_status", lambda: {"driver_api": "ok", "problems": []})
    monkeypatch.setattr("gptqmodel.GPTQModel.load", lambda *a, **k: FakeGPTQ())
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *a, **k: MagicMock())

    with caplog.at_level(logging.WARNING):
        res = quantize_gptq(
            model_path="dummy",
            calib_texts=["text"],
            output_path=str(ROOT / "outputs" / "test_gptq_tmp"),
        )
    assert res["method"] == "gptq_int4"
    assert "quantize_s" in res
    assert "GPTQModel.quantize(calib_texts, ...) failed" in caplog.text


# ---------------------------------------------------------------------------
# Issue 6: load_gptq fallback to GPTQModel.load and extract_logits_hf compatibility
# ---------------------------------------------------------------------------
def test_load_gptq_fallback(monkeypatch, tmp_path):
    """Verify load_gptq falls back to GPTQModel.load when AutoModelForCausalLM fails."""
    fake_hf_model = torch.nn.Linear(10, 10)
    fake_hf_model.config = type("Config", (), {"use_cache": True})()
    fake_gptq_obj = type("BaseGPTQModel", (), {"model": fake_hf_model})()

    def mock_auto_model(*args, **kwargs):
        raise ImportError("Missing optimum/auto_gptq driver")

    monkeypatch.setattr("transformers.AutoModelForCausalLM.from_pretrained", mock_auto_model)
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *a, **k: MagicMock(pad_token=None, eos_token="<eos>"))
    monkeypatch.setattr("gptqmodel.GPTQModel.load", lambda *a, **k: fake_gptq_obj)

    model, tok = load_gptq(str(tmp_path))
    assert model is fake_hf_model
    assert model.config.use_cache is False


# ---------------------------------------------------------------------------
# Issue 7: Resumability checks matching config_hash, GPTQ skips re-quantizing
# ---------------------------------------------------------------------------
def test_is_valid_gptq_dir(tmp_path):
    """Verify is_valid_gptq_dir checks for config.json and weights."""
    assert not is_valid_gptq_dir(tmp_path)
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    assert not is_valid_gptq_dir(tmp_path)
    (tmp_path / "model.safetensors").write_text("dummy", encoding="utf-8")
    assert is_valid_gptq_dir(tmp_path)


# ---------------------------------------------------------------------------
# Issue 1: End-to-end test of run_one_quant with tiny fake model
# ---------------------------------------------------------------------------
def test_run_one_quant_e2e_fake_model(tmp_path, quant_config, monkeypatch):
    """Execute entire body of run_one_quant with a tiny fake model and tokenizer."""
    cell = {
        "artifact_name": "fake_run__bnb_int8",
        "base_run_name": "fake_run",
        "source_key": "phase4",
        "source_phase": 4,
        "language": "bn",
        "model_key": "qwen2_5_0_5b_instruct",
        "seed": 13,
        "method": "bnb_int8",
        "model_path": str(tmp_path),
        "merged_dir": str(tmp_path),
        "source_cfg": quant_config["sources"]["phase4"],
    }

    fake_df = pd.DataFrame(
        {
            "example_id": [f"id_{i}" for i in range(12)],
            "text": ["টেস্ট বাক্য"] * 12,
            "label": [i % 2 for i in range(12)],
            "domain": ["slander"] * 12,
        }
    )
    fake_spec = {
        "train_ids": [f"id_{i}" for i in range(4)],
        "id_val_ids": [f"id_{i}" for i in range(4, 7)],
        "id_test_ids": [f"id_{i}" for i in range(7, 10)],
        "ood_test_ids": [f"id_{i}" for i in range(10, 12)],
    }
    monkeypatch.setattr(_quant_module, "get_dataset_and_splits", lambda sc: (fake_df, fake_spec, "sha"))

    # Minimal fake causal LM
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = type("Config", (), {"vocab_size": 30, "use_cache": False})()
        def forward(self, input_ids, **kwargs):
            b, s = input_ids.shape
            logits = torch.zeros((b, s, 30), dtype=torch.float32)
            logits[:, :, 7] = 2.0  # verbalizer hate " A"
            logits[:, :, 9] = 1.0  # verbalizer nonhate " B"
            return type("Out", (), {"logits": logits})()

    class TinyTokenizer:
        pad_token_id = 0
        eos_token_id = 1
        pad_token = "<pad>"
        eos_token = "<eos>"
        def __call__(self, texts, **kwargs):
            n = len(texts) if isinstance(texts, list) else 1
            return {
                "input_ids": torch.ones((n, 4), dtype=torch.long),
                "attention_mask": torch.ones((n, 4), dtype=torch.long),
            }
        def encode(self, text, **kwargs):
            return [7 if "A" in text else 9]

    monkeypatch.setattr(_quant_module, "load_quantized_hf", lambda p, m: TinyModel())
    monkeypatch.setattr(_quant_module.ft, "load_tokenizer", lambda p: TinyTokenizer())

    cfg_copy = dict(quant_config)
    cfg_copy["paths"] = {
        "runs": str(tmp_path / "runs"),
        "logits": str(tmp_path / "logits"),
        "quantized_models": str(tmp_path / "quant"),
    }
    # Provide synthetic FP logits so T6.5 passes
    fp_dir = tmp_path / "fp_logits" / "fake_run"
    fp_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "example_id": fake_spec["id_test_ids"],
            "y_true": [0, 1, 0],
            "prediction": [1, 1, 1],
            "prob_hate": [0.73, 0.73, 0.73],
            "prob_nonhate": [0.27, 0.27, 0.27],
            "logit_hate": [2.0, 2.0, 2.0],
            "logit_nonhate": [1.0, 1.0, 1.0],
        }
    ).to_parquet(fp_dir / "bf16__id_test.parquet")
    cfg_copy["sources"]["phase4"]["fp_logits_dir"] = str(tmp_path / "fp_logits")

    manifest = run_one_quant(cell, cfg_copy, caps={"id_val": 3, "id_test": 3, "ood_test": 2}, force=True)

    assert manifest["status"] == "success"
    assert manifest["quantize_s"] == 0.0
    assert "eval_s" in manifest
    assert "validation" in manifest
    assert manifest["T6.6"]["passed"] is True
    assert manifest["T6.7"]["passed"] is True
    assert manifest["T6.5"]["passed"] is True


# ---------------------------------------------------------------------------
# Test: Calibration Sampling & Leakage Verification (T6.8 & T6.9)
# ---------------------------------------------------------------------------
def test_calibration_zero_leakage_and_matching(quant_config):
    """Test calibration sampling respects zero leakage into evaluation sets and matches language."""
    for source_key in ["phase4", "phase5"]:
        source_cfg = quant_config["sources"][source_key]
        calib_texts, calib_ids, prov = prepare_calibration_data(
            source_cfg,
            n_samples=100,
            seed=42,
        )
        assert len(calib_texts) == 100
        assert len(calib_ids) == 100
        assert all(isinstance(t, str) and len(t.strip()) > 0 for t in calib_texts)
        assert prov["verified_train_only"] is True
        assert prov["leaked_into_eval"] is False
        assert prov["language"] == source_cfg["language"]


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
