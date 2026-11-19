"""Phase 5 unit tests: English control fine-tuning, run matrix, matching verification, and validation gates.

Covers T5.1-T5.4 from planning/phases/phase_05_english_control_fine_tuning.md.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.assemble import build_split_frames, resolve_model_id, split_frame_meta, split_sizes
from src.data.loader import load_dataset_stats, verify_english_control_matching
from src.data.splits import load_splits
from src.models import logit_extraction as lx
from src.utils.config import load_config


def _load_script():
    spec = importlib.util.spec_from_file_location("finetune_script", ROOT / "scripts" / "03_finetune.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["finetune_script"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ph5():
    return _load_script()


@pytest.fixture(scope="module")
def cfg():
    return load_config(ROOT / "configs" / "baseline" / "english_control.yaml")


# ---------------------------------------------------------------------------
# 5.1 Run Matrix & Configuration
# ---------------------------------------------------------------------------

def test_english_config_exists_and_matches_phase5(cfg):
    assert cfg["phase"] == 5
    assert cfg["language"] == "en"
    assert cfg["dataset"] == "toxigen"
    assert cfg["split"] == "loco_lgbtq"
    assert cfg["paths"]["runs"] == "outputs/runs/phase5"
    assert cfg["paths"]["models"] == "outputs/models/phase5"
    assert cfg["paths"]["logits"] == "outputs/logits/phase5"


def test_english_run_matrix_has_sixteen_cells(ph5, cfg):
    matrix = ph5.build_run_matrix(cfg, None, None)
    assert len(matrix) == 16


def test_english_run_matrix_matches_plan_seeds(ph5, cfg):
    matrix = ph5.build_run_matrix(cfg, None, None)
    seeds_by_model: dict[str, list[int]] = {}
    for cell in matrix:
        seeds_by_model.setdefault(cell["model_key"], []).append(cell["seed"])
    assert seeds_by_model == {
        "qwen2_5_0_5b_instruct": [13, 42, 1234, 2024, 7],
        "qwen2_5_1_5b_instruct": [13, 42, 1234, 2024, 7],
        "llama3_2_1b_instruct": [13, 42, 1234],
        "gemma2_2b_it": [13, 42, 1234],
    }


def test_english_run_name_prefix_and_formatting(ph5, cfg):
    matrix = ph5.build_run_matrix(cfg, None, None)
    for cell in matrix:
        name = cell["run_name"]
        assert name.startswith("en_"), f"Run name {name} should start with 'en_'"
        assert f"_s{cell['seed']}" in name
        assert set(name) <= set("abcdefghijklmnopqrstuvwxyz0123456789_")


def test_run_name_function_backward_compatible(ph5):
    # Default language is 'bn' (Phase 4 compatibility)
    assert ph5.run_name("qwen2_5_1_5b_instruct", 42) == "bn_qwen2_5_1_5b_instruct_lora_bf16_s42"
    # Explicit 'en' (Phase 5)
    assert ph5.run_name("qwen2_5_1_5b_instruct", 42, language="en") == "en_qwen2_5_1_5b_instruct_lora_bf16_s42"


def test_gemma_is_forced_to_eager_attention(ph5, cfg):
    matrix = ph5.build_run_matrix(cfg, None, None)
    impl = {c["model_key"]: c["attn_implementation"] for c in matrix}
    assert impl["gemma2_2b_it"] == "eager"
    assert impl["qwen2_5_0_5b_instruct"] is None
    assert impl["llama3_2_1b_instruct"] is None


def test_hyperparameters_identical_to_bengali(cfg):
    bn_cfg = load_config(ROOT / "configs" / "baseline" / "default.yaml")
    # LoRA and training hyperparameters must be identical
    assert cfg["train"]["rank"] == bn_cfg["train"]["rank"]
    assert cfg["train"]["alpha"] == bn_cfg["train"]["alpha"]
    assert cfg["train"]["dropout"] == bn_cfg["train"]["dropout"]
    assert cfg["train"]["target_modules"] == bn_cfg["train"]["target_modules"]
    assert cfg["train"]["lr"] == bn_cfg["train"]["lr"]
    assert cfg["train"]["epochs"] == bn_cfg["train"]["epochs"]
    assert cfg["train"]["batch_size"] == bn_cfg["train"]["batch_size"]
    assert cfg["train"]["grad_accum"] == bn_cfg["train"]["grad_accum"]
    assert cfg["train"]["bf16"] == bn_cfg["train"]["bf16"]
    assert cfg["sanity"] == bn_cfg["sanity"]


# ---------------------------------------------------------------------------
# 5.2 Matching Verification (T5.1 & T5.4)
# ---------------------------------------------------------------------------

def test_load_dataset_stats_bengali():
    stats = load_dataset_stats("bn")
    assert stats["lang"] == "bn"
    assert stats["dataset"] == "bd_shs"
    assert stats["n"] == 49941
    assert pytest.approx(stats["hate_rate"], abs=0.01) == 0.48
    assert stats["n_domains"] == 5
    assert stats["mean_mmd"] is not None and stats["mean_mmd"] > 0


def test_load_dataset_stats_english():
    stats = load_dataset_stats("en")
    assert stats["lang"] == "en"
    assert stats["dataset"] == "toxigen"
    assert stats["n"] == 50281
    assert pytest.approx(stats["hate_rate"], abs=0.01) == 0.48
    assert stats["n_domains"] == 13
    assert stats["mean_mmd"] is not None and stats["mean_mmd"] > 0


def test_verify_english_control_matching_passes_t5_1():
    result = verify_english_control_matching()
    assert result["t5_1_passed"] is True
    assert result["passed"] is True

    checks = result["checks"]
    # Pass condition: Size within 10%, class prior within 2%, >= 3 shift domains
    assert checks["size_within_10pct"] is True
    assert checks["prior_within_2pct"] is True
    assert checks["has_shift_axis"] is True

    # Check exact values
    assert result["size_diff_ratio"] < 0.10
    assert result["prior_diff"] < 0.02
    assert result["en_stats"]["n_domains"] >= 3


def test_verify_matching_rejects_size_mismatch():
    bn = {"lang": "bn", "n": 50000, "hate_rate": 0.48, "n_domains": 5, "mean_mmd": 0.01}
    en = {"lang": "en", "n": 35000, "hate_rate": 0.48, "n_domains": 10, "mean_mmd": 0.01}  # 30% smaller
    res = verify_english_control_matching(bn_stats=bn, en_stats=en)
    assert res["checks"]["size_within_10pct"] is False
    assert res["t5_1_passed"] is False


def test_verify_matching_rejects_prior_mismatch():
    bn = {"lang": "bn", "n": 50000, "hate_rate": 0.48, "n_domains": 5, "mean_mmd": 0.01}
    en = {"lang": "en", "n": 50100, "hate_rate": 0.35, "n_domains": 10, "mean_mmd": 0.01}  # prior diff 0.13 > 0.02
    res = verify_english_control_matching(bn_stats=bn, en_stats=en)
    assert res["checks"]["prior_within_2pct"] is False
    assert res["t5_1_passed"] is False


def test_verify_matching_rejects_insufficient_domains():
    bn = {"lang": "bn", "n": 50000, "hate_rate": 0.48, "n_domains": 5, "mean_mmd": 0.01}
    en = {"lang": "en", "n": 50100, "hate_rate": 0.48, "n_domains": 2, "mean_mmd": 0.01}  # < 3 domains
    res = verify_english_control_matching(bn_stats=bn, en_stats=en)
    assert res["checks"]["has_shift_axis"] is False
    assert res["t5_1_passed"] is False


def test_t5_4_shift_magnitude_documented(ph5):
    res = ph5.check_english_shift_magnitude_documented()
    assert res["passed"] is True
    assert res["n_english_pairs"] >= 1
    assert any(p["target_domain"] == "lgbtq" for p in res["pairs"])


# ---------------------------------------------------------------------------
# English Split Assembly & Balanced Eval Sets
# ---------------------------------------------------------------------------

def test_english_dataset_clean_and_loadable(ph5, cfg):
    df, dataset = ph5._load_dataset(cfg)
    assert len(df) == 50281
    assert not df["text"].isna().any()
    assert not (df["text"].str.strip() == "").any()
    assert dataset["name"] == "loco_lgbtq"
    assert "lgbtq" == dataset["spec"]["target_domain"]


def test_english_loco_splits_contain_both_classes_in_all_partitions(ph5, cfg):
    df, dataset = ph5._load_dataset(cfg)
    spec = dataset["spec"]
    frames = build_split_frames(df, spec, seed=42)

    # In ToxiGen LOCO, unlike BD-SHS, every partition including ood_test has both classes!
    for part_name, frame in frames.items():
        assert len(frame) > 0, f"Partition {part_name} is empty"
        assert frame["label"].nunique() == 2, f"Partition {part_name} must contain both classes"


def test_english_split_caps_work_properly(ph5, cfg):
    df, dataset = ph5._load_dataset(cfg)
    spec = dataset["spec"]
    caps = {"train": 64, "id_val": 32, "id_test": 32, "ood_test": 32}
    frames = build_split_frames(df, spec, caps=caps, seed=13)
    assert len(frames["train"]) == 64
    assert len(frames["id_val"]) == 32
    assert len(frames["id_test"]) == 32
    assert len(frames["ood_test"]) == 32
    for name, f in frames.items():
        assert set(f["label"].unique()) == {0, 1}


# ---------------------------------------------------------------------------
# T5.3 English F1 > Bengali F1 Comparison Logic
# ---------------------------------------------------------------------------

def test_t5_3_passes_when_english_f1_higher(ph5, tmp_path):
    bn_summary = tmp_path / "bn_summary.json"
    bn_summary.write_text(
        json.dumps(
            {
                "per_run": [
                    {"run_name": "bn_m1_s42", "model_key": "m1", "status": "success", "macro_f1_id_test": 0.72},
                    {"run_name": "bn_m2_s42", "model_key": "m2", "status": "success", "macro_f1_id_test": 0.68},
                ]
            }
        ),
        encoding="utf-8",
    )
    en_records = [
        {"model_key": "m1", "status": "success", "T4.2": {"macro_f1": 0.85}},
        {"model_key": "m2", "status": "success", "T4.2": {"macro_f1": 0.79}},
    ]
    res = ph5.check_english_vs_bengali_f1(en_records, bn_summary)
    assert res["passed"] is True
    assert res["comparison"]["m1"]["en_greater_than_bn"] is True
    assert res["comparison"]["m2"]["en_greater_than_bn"] is True


def test_t5_3_flags_when_english_f1_lower(ph5, tmp_path):
    bn_summary = tmp_path / "bn_summary.json"
    bn_summary.write_text(
        json.dumps(
            {
                "per_run": [
                    {"run_name": "bn_m1_s42", "model_key": "m1", "status": "success", "macro_f1_id_test": 0.88},
                ]
            }
        ),
        encoding="utf-8",
    )
    en_records = [
        {"model_key": "m1", "status": "success", "T4.2": {"macro_f1": 0.75}},  # Lower!
    ]
    res = ph5.check_english_vs_bengali_f1(en_records, bn_summary)
    assert res["passed"] is False
    assert res["comparison"]["m1"]["en_greater_than_bn"] is False


def test_t5_3_graceful_when_bengali_results_not_yet_present(ph5, tmp_path):
    en_records = [
        {"model_key": "m1", "status": "success", "T4.2": {"macro_f1": 0.85}},
    ]
    res = ph5.check_english_vs_bengali_f1(en_records, tmp_path / "does_not_exist.json")
    assert res["passed"] is None
    assert "not yet available" in res["reason"]


# ---------------------------------------------------------------------------
# CLI & Orchestrator Execution
# ---------------------------------------------------------------------------

def test_cli_dry_run_with_english_flag(ph5):
    import argparse
    args = argparse.Namespace(
        config="",
        english=True,
        language="en",
        smoke=False,
        models=["qwen2_5_0_5b_instruct"],
        seeds=[13],
        max_train=0,
        max_eval=0,
        split="",
        no_keep_merged=False,
        force=False,
        status=False,
        dry_run=True,
    )
    res = ph5.run(args)
    assert res["status"] == "dry_run"
    assert res["matrix"] == ["en_qwen2_5_0_5b_instruct_lora_bf16_s13"]


def test_cli_smoke_routing_isolates_phase5_smoke_dir(ph5):
    import argparse
    args = argparse.Namespace(
        config="",
        english=True,
        language="en",
        smoke=True,
        models=["qwen2_5_0_5b_instruct"],
        seeds=[13],
        max_train=64,
        max_eval=32,
        split="",
        no_keep_merged=False,
        force=False,
        status=False,
        dry_run=True,
    )
    res = ph5.run(args)
    assert res["status"] == "dry_run"

