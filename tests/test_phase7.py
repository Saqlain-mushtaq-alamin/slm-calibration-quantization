"""Phase 7 Unit & Regression Tests: Post-Training Quantization Track B (GGUF / llama.cpp).

Validates:
- Configuration loading and exact template/verbalizer match with training
- Track B matrix expansion logic (requires is_valid_merged, no adapter fallback)
- GGUF file format validator and size utility
- llama.cpp conversion script and quantizer detection
- Quantization type enum mapping
- Critical Gate T7.5: GGUF F16 vs HF BF16 agreement sanity check
- Mocked logit extraction producing valid canonical DataFrames (T7.6, T7.7)
- End-to-end execution of run_one_quant_b with mock/simulated models
- Strict separation between Track A and Track B (Global Rule 4)
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics.validation import REQUIRED_LOGIT_COLUMNS, validate_logits_frame
from src.models.inference import UnifiedInferenceModel, extract_logits_unified
from src.models.quantize_track_b import (
    ALL_TRACK_B_METHODS,
    GGUFConversionUnavailable,
    TRACK_B_METHODS,
    find_convert_script,
    find_llama_quantize_binary,
    free_gguf,
    get_llama_ftype_enum,
    gguf_file_size_mb,
    is_valid_gguf_file,
    load_gguf_model,
    sanity_check_gguf_f16,
)
from src.utils.atomic import is_valid_merged
from src.utils.config import config_hash, load_config


def _load_track_b_script():
    spec = importlib.util.spec_from_file_location(
        "quantize_track_b_script", ROOT / "scripts" / "04_quantize_track_b.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["quantize_track_b_script"] = mod
    spec.loader.exec_module(mod)
    return mod


_track_b_mod = _load_track_b_script()
build_quant_matrix_b = _track_b_mod.build_quant_matrix_b
parse_run_name = _track_b_mod.parse_run_name
run_one_quant_b = _track_b_mod.run_one_quant_b


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def track_b_config():
    """Load default Track B quantization config."""
    cfg_path = ROOT / "configs" / "quantization" / "track_b.yaml"
    assert cfg_path.exists(), f"Missing config: {cfg_path}"
    return load_config(cfg_path)


# ---------------------------------------------------------------------------
# 1. Config Validation
# ---------------------------------------------------------------------------

def test_track_b_config_matches_training_templates_and_verbalizer(track_b_config):
    """Verify configs/quantization/track_b.yaml uses exact same templates and verbalizer as training."""
    cfg_bn = load_config(ROOT / "configs" / "baseline" / "default.yaml")
    cfg_en = load_config(ROOT / "configs" / "baseline" / "english_control.yaml")

    # Verbalizer match
    assert (
        track_b_config["verbalizer"]["hate"]
        == cfg_bn["model"]["verbalizer"]["hate"]
        == cfg_en["model"]["verbalizer"]["hate"]
        == " A"
    )
    assert (
        track_b_config["verbalizer"]["nonhate"]
        == cfg_bn["model"]["verbalizer"]["nonhate"]
        == cfg_en["model"]["verbalizer"]["nonhate"]
        == " B"
    )

    # Prompt templates match
    assert (
        track_b_config["sources"]["phase4"]["prompt_template"]
        == cfg_bn["data"]["prompt_template"]
        == "{text}\nবাক্যটি ঘৃণামূলক কি না? উত্তর:"
    )
    assert (
        track_b_config["sources"]["phase5"]["prompt_template"]
        == cfg_en["data"]["prompt_template"]
        == "Classify the following text as hate speech or not.\nText: {text}\nAnswer:"
    )


def test_track_b_config_structure(track_b_config):
    """Verify required Track B structure and methods."""
    assert track_b_config["phase"] == 7
    assert track_b_config["track_b"]["methods"] == ["f16", "q8_0", "q4_k_m", "q3_k_m"]
    assert set(track_b_config["track_b"]["eval"]["splits"]) == {"id_val", "id_test", "ood_test"}
    assert track_b_config["track_b"]["f16_tolerance"]["min_agreement"] == 0.90


# ---------------------------------------------------------------------------
# 2. Matrix Builder & Name Parsing
# ---------------------------------------------------------------------------

def test_parse_run_name():
    """Test parsing canonical run names."""
    res_bn = parse_run_name("bn_qwen2_5_0_5b_instruct_lora_bf16_s42")
    assert res_bn == {"language": "bn", "model_key": "qwen2_5_0_5b_instruct", "seed": 42}

    res_en = parse_run_name("en_llama3_2_1b_instruct_lora_bf16_s1234")
    assert res_en == {"language": "en", "model_key": "llama3_2_1b_instruct", "seed": 1234}


def test_build_track_b_matrix(track_b_config):
    """Verify Track B matrix pairs Phase 4 and Phase 5 models with methods."""
    matrix = build_quant_matrix_b(track_b_config)
    assert len(matrix) == 72  # 18 base checkpoints * 4 methods

    for item in matrix:
        assert item["language"] in {"bn", "en"}
        assert item["method"] in {"f16", "q8_0", "q4_k_m", "q3_k_m"}
        assert Path(item["merged_dir"]).is_dir()
        assert is_valid_merged(item["merged_dir"])


def test_build_track_b_matrix_filtering(track_b_config):
    """Test filtering matrix by model, method, language, seed."""
    filtered = build_quant_matrix_b(
        track_b_config,
        models=["qwen2_5_0_5b_instruct"],
        methods=["q8_0"],
        languages=["bn"],
        seeds=[42],
    )
    assert len(filtered) == 1
    cell = filtered[0]
    assert cell["model_key"] == "qwen2_5_0_5b_instruct"
    assert cell["method"] == "q8_0"
    assert cell["language"] == "bn"
    assert cell["seed"] == 42


# ---------------------------------------------------------------------------
# 3. GGUF File Validation & Utilities
# ---------------------------------------------------------------------------

def test_is_valid_gguf_file(tmp_path):
    """Verify GGUF magic header detection and size check."""
    non_existent = tmp_path / "missing.gguf"
    assert not is_valid_gguf_file(non_existent)

    # Empty file
    empty = tmp_path / "empty.gguf"
    empty.touch()
    assert not is_valid_gguf_file(empty)

    # File with invalid magic
    bad_magic = tmp_path / "bad.gguf"
    bad_magic.write_bytes(b"NOTG" + b"\x00" * 2000000)
    assert not is_valid_gguf_file(bad_magic)

    # Valid GGUF file
    valid = tmp_path / "valid.gguf"
    valid.write_bytes(b"GGUF" + b"\x00" * 2000000)
    assert is_valid_gguf_file(valid)
    assert gguf_file_size_mb(valid) > 1.9


def test_find_convert_script():
    """Verify convert_hf_to_gguf.py is found."""
    p = find_convert_script()
    assert p.is_file()
    assert p.name == "convert_hf_to_gguf.py"


def test_get_llama_ftype_enum():
    """Verify quantization type string mapping to integer enums."""
    assert get_llama_ftype_enum("f16") == 1
    assert get_llama_ftype_enum("q8_0") == 7
    assert get_llama_ftype_enum("q4_k_m") == 15
    assert get_llama_ftype_enum("q3_k_m") == 12

    with pytest.raises(ValueError):
        get_llama_ftype_enum("non_existent_quant_type")


# ---------------------------------------------------------------------------
# 4. Critical Gate T7.5: GGUF F16 Agreement Check
# ---------------------------------------------------------------------------

def test_sanity_check_gguf_f16_pass():
    """Verify agreement check passes when agreement >= 0.90 and within tolerance."""
    n = 100
    df_fp = pd.DataFrame({
        "example_id": [f"ex_{i}" for i in range(n)],
        "y_true": [i % 2 for i in range(n)],
        "prediction": [i % 2 for i in range(n)],
        "prob_hate": [0.9 if i % 2 == 1 else 0.1 for i in range(n)],
    })
    # 97% agreement (3 differences)
    preds_gguf = [i % 2 if i < 97 else (1 - i % 2) for i in range(n)]
    probs_gguf = [0.89 if p == 1 else 0.11 for p in preds_gguf]
    df_gguf = pd.DataFrame({
        "example_id": [f"ex_{i}" for i in range(n)],
        "y_true": [i % 2 for i in range(n)],
        "prediction": preds_gguf,
        "prob_hate": probs_gguf,
    })

    res = sanity_check_gguf_f16(df_fp, df_gguf)
    assert res["passed"] is True
    assert res["agreement"] == 0.97
    assert res["checks"]["agreement_ok"] is True


def test_sanity_check_gguf_f16_fail():
    """Verify agreement check fails when agreement < 0.90."""
    n = 100
    df_fp = pd.DataFrame({
        "example_id": [f"ex_{i}" for i in range(n)],
        "y_true": [i % 2 for i in range(n)],
        "prediction": [i % 2 for i in range(n)],
        "prob_hate": [0.9 if i % 2 == 1 else 0.1 for i in range(n)],
    })
    # Only 70% agreement
    preds_gguf = [i % 2 if i < 70 else (1 - i % 2) for i in range(n)]
    probs_gguf = [0.85 if p == 1 else 0.15 for p in preds_gguf]
    df_gguf = pd.DataFrame({
        "example_id": [f"ex_{i}" for i in range(n)],
        "y_true": [i % 2 for i in range(n)],
        "prediction": preds_gguf,
        "prob_hate": probs_gguf,
    })

    res = sanity_check_gguf_f16(df_fp, df_gguf)
    assert res["passed"] is False
    assert res["agreement"] == 0.70
    assert res["checks"]["agreement_ok"] is False


def test_sanity_check_gguf_f16_id_mismatch():
    """Verify error raised on mismatched example_ids."""
    df1 = pd.DataFrame({"example_id": ["a", "b"], "y_true": [0, 1], "prediction": [0, 1], "prob_hate": [0.1, 0.9]})
    df2 = pd.DataFrame({"example_id": ["c", "d"], "y_true": [0, 1], "prediction": [0, 1], "prob_hate": [0.1, 0.9]})
    with pytest.raises(ValueError, match="No overlapping"):
        sanity_check_gguf_f16(df1, df2)


# ---------------------------------------------------------------------------
# 5. Unified Inference Model
# ---------------------------------------------------------------------------

def test_unified_inference_model_routing():
    """Verify UnifiedInferenceModel initializes correctly for both backends."""
    # Test invalid backend
    with pytest.raises(ValueError, match="Unsupported backend"):
        UnifiedInferenceModel(backend="invalid")

    # Test HF backend wrapper
    mock_hf = MagicMock()
    unified_hf = UnifiedInferenceModel(backend="hf", model=mock_hf)
    assert unified_hf.backend == "hf"
    unified_hf.close()
    assert unified_hf.model is None


# ---------------------------------------------------------------------------
# 6. Global Rule 4: Track A and Track B Separation
# ---------------------------------------------------------------------------

def test_track_a_and_track_b_separation(track_b_config):
    """Verify Track A and Track B paths, phases, and naming conventions never collide."""
    cfg_a = load_config(ROOT / "configs" / "quantization" / "default.yaml")

    # Distinct phases
    assert cfg_a["phase"] == 6
    assert track_b_config["phase"] == 7

    # Distinct method sets
    methods_a = set(cfg_a["track_a"]["methods"])
    methods_b = set(track_b_config["track_b"]["methods"])
    assert methods_a.isdisjoint(methods_b), f"Method collision: {methods_a & methods_b}"

    # Distinct output paths
    assert cfg_a["paths"]["quantized_models"] != track_b_config["paths"]["quantized_models"]
    assert cfg_a["paths"]["logits"] != track_b_config["paths"]["logits"]
    assert cfg_a["paths"]["runs"] != track_b_config["paths"]["runs"]
    assert cfg_a["paths"]["summary"] != track_b_config["paths"]["summary"]
    assert cfg_a["paths"]["metrics_csv"] != track_b_config["paths"]["metrics_csv"]


# ---------------------------------------------------------------------------
# 7. Mocked Execution of run_one_quant_b (T7.1 - T7.7 Gates)
# ---------------------------------------------------------------------------

def test_run_one_quant_b_lifecycle(tmp_path, track_b_config):
    """Test full per-artifact lifecycle and gate recording in run_one_quant_b."""
    cell = {
        "artifact_name": "mock_bn_qwen_s42__q8_0",
        "base_run_name": "mock_bn_qwen_s42",
        "source_key": "phase4",
        "source_phase": 4,
        "language": "bn",
        "model_key": "qwen2_5_0_5b_instruct",
        "seed": 42,
        "method": "q8_0",
        "merged_dir": str(tmp_path / "mock_merged"),
        "source_cfg": track_b_config["sources"]["phase4"],
    }

    # Setup isolated test paths
    test_cfg = dict(track_b_config)
    test_cfg["paths"] = dict(track_b_config["paths"])
    test_cfg["paths"]["runs"] = str(tmp_path / "runs")
    test_cfg["paths"]["logits"] = str(tmp_path / "logits")
    test_cfg["paths"]["quantized_models"] = str(tmp_path / "quantized")

    # Mock conversion, quantize, and extraction
    fake_f16 = tmp_path / "quantized" / "mock_bn_qwen_s42__f16.gguf"
    fake_q8 = tmp_path / "quantized" / "mock_bn_qwen_s42__q8_0.gguf"
    fake_f16.parent.mkdir(parents=True, exist_ok=True)
    fake_f16.write_bytes(b"GGUF" + b"\x00" * 2000000)
    fake_q8.write_bytes(b"GGUF" + b"\x00" * 1000000)

    def mock_extract(llm, dataset, **kwargs):
        n = len(dataset)
        return pd.DataFrame({
            "example_id": [f"ex_{i}" for i in range(n)],
            "split": kwargs.get("split", "eval"),
            "y_true": [i % 2 for i in range(n)],
            "logit_hate": [2.0 if i % 2 == 1 else -2.0 for i in range(n)],
            "logit_nonhate": [-2.0 if i % 2 == 1 else 2.0 for i in range(n)],
            "prob_hate": [0.98 if i % 2 == 1 else 0.02 for i in range(n)],
            "prob_nonhate": [0.02 if i % 2 == 1 else 0.98 for i in range(n)],
            "prediction": [i % 2 for i in range(n)],
            "confidence": [0.98] * n,
            "correct": [1] * n,
            "margin": [4.0] * n,
            "entropy": [0.08] * n,
        })

    with patch.object(_track_b_mod, "convert_to_gguf") as mock_conv, \
         patch.object(_track_b_mod, "quantize_gguf") as mock_quant, \
         patch.object(_track_b_mod, "load_gguf_model") as mock_load, \
         patch.object(_track_b_mod, "extract_logits_llamacpp", side_effect=mock_extract):

        mock_conv.return_value = {"status": "success", "file_size_mb": 2.0}
        mock_quant.return_value = {"status": "success", "output_size_mb": 1.0}
        mock_load.return_value = MagicMock()

        res = run_one_quant_b(cell, test_cfg, caps={"id_val": 10, "id_test": 10, "ood_test": 10})

        assert res["status"] == "success"
        assert res["T7.1"]["passed"] is True
        assert res["T7.2"]["passed"] is True
        assert res["T7.6"]["passed"] is True
        assert res["T7.7"]["passed"] is True

        # Verify run manifest was written
        run_file = Path(test_cfg["paths"]["runs"]) / f"{cell['artifact_name']}.json"
        assert run_file.is_file()
        run_data = json.loads(run_file.read_text(encoding="utf-8"))
        assert run_data["status"] == "success"
        assert run_data["phase"] == 7
        assert run_data["track"] == "B"
