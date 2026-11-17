"""Tests for Phase 4 resumability, atomic writes, and crash recovery (Part B, B1-B8).

Simulates interruptions across each stage of execution:
- Mid-training checkpoint fallback on corruption
- Interruption during merge (rebuilds merged model from valid adapter without retraining)
- Interruption during eval (keeps existing valid split parquets, computes only missing splits)
- Corrupted parquet / manifest detection and regeneration
- Config hash change refusing to skip an outdated successful run
- Git commit change warning while still skipping identical configs
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models import finetune as ft
from src.utils.atomic import (
    atomic_write_csv,
    atomic_write_json,
    atomic_write_parquet,
    atomic_write_text,
    find_last_valid_checkpoint,
    is_valid_adapter,
    is_valid_checkpoint,
    is_valid_merged,
    is_valid_parquet,
)
from src.utils.config import load_config


def _load_ph4():
    spec = importlib.util.spec_from_file_location("phase4_script", ROOT / "scripts" / "03_finetune.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["phase4_script"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ph4():
    return _load_ph4()


@pytest.fixture
def dummy_dataset():
    df = pd.DataFrame(
        {
            "example_id": [str(i) for i in range(40)],
            "text": [f"Text sample {i}" for i in range(40)],
            "label": [i % 2 for i in range(40)],
            "domain": ["sports" if i < 30 else "religion" for i in range(40)],
        }
    )
    spec = {
        "name": "dummy_loco",
        "train_ids": [str(i) for i in range(20)],
        "id_val_ids": [str(i) for i in range(20, 26)],
        "id_test_ids": [str(i) for i in range(26, 32)],
        "ood_test_ids": [str(i) for i in range(32, 40)],
        "sizes": {"train": 20, "id_val": 6, "id_test": 6, "ood_test": 8},
    }
    return {
        "df": df,
        "spec": spec,
        "sha": "dummy_sha_12345",
        "path": "data/splits/dummy.json",
        "name": "dummy_loco",
    }


def _create_mock_adapter(adapter_dir: Path) -> None:
    adapter_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(adapter_dir / "adapter_config.json", {"r": 16, "lora_alpha": 32})
    atomic_write_text(adapter_dir / "adapter_model.safetensors", "MOCK_WEIGHTS_BINARY_DATA")
    atomic_write_text(adapter_dir / ".completed", "ok\n")


def _create_mock_checkpoint(ckpt_dir: Path, step: int, valid: bool = True) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if valid:
        atomic_write_json(ckpt_dir / "trainer_state.json", {"global_step": step, "loss": 1.5})
        atomic_write_json(ckpt_dir / "adapter_config.json", {"r": 16})
        atomic_write_text(ckpt_dir / "adapter_model.safetensors", "MOCK_WEIGHTS")
        atomic_write_text(ckpt_dir / "optimizer.pt", "MOCK_OPTIMIZER")
    else:
        # Corrupted checkpoint (truncated/empty files)
        (ckpt_dir / "trainer_state.json").write_text("{corrupt json", encoding="utf-8")
        (ckpt_dir / "adapter_model.safetensors").touch()


def _create_mock_parquet(parquet_path: Path, n_rows: int = 6) -> None:
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "example_id": [str(i) for i in range(n_rows)],
            "y_true": [i % 2 for i in range(n_rows)],
            "logit_hate": [1.2] * n_rows,
            "logit_nonhate": [-0.5] * n_rows,
            "prob_hate": [0.845] * n_rows,
            "prob_nonhate": [0.155] * n_rows,
            "prediction": [1] * n_rows,
        }
    )
    atomic_write_parquet(parquet_path, df)


# ---------------------------------------------------------------------------
# Test Cases (B8)
# ---------------------------------------------------------------------------

def test_find_last_valid_checkpoint_skips_corrupted(tmp_path):
    """B1 & B8: Trainer checkpoint fallback when the latest checkpoint is incomplete/corrupted."""
    ckpts_dir = tmp_path / "checkpoints"
    _create_mock_checkpoint(ckpts_dir / "checkpoint-100", step=100, valid=True)
    _create_mock_checkpoint(ckpts_dir / "checkpoint-200", step=200, valid=True)
    _create_mock_checkpoint(ckpts_dir / "checkpoint-300", step=300, valid=False)

    latest = find_last_valid_checkpoint(ckpts_dir)
    assert latest is not None
    assert latest.name == "checkpoint-200"


def test_resume_interrupted_during_merge(ph4, dummy_dataset, tmp_path, monkeypatch):
    """B3 & B8: Interruption during merge: finds valid adapter, skips train, merges standalone model."""
    output_dir = tmp_path / "outputs" / "models" / "phase4" / "bn_test_model_lora_bf16_s42"
    adapter_dir = output_dir / "adapter"
    merged_dir = output_dir / "merged"
    _create_mock_adapter(adapter_dir)

    # Save mock train metrics
    atomic_write_json(
        output_dir / "train_metrics.json",
        {
            "n_train": 20,
            "n_id_val": 6,
            "trainable_params": 1000,
            "total_params": 1000000,
            "log_history": [{"loss": 2.5}, {"loss": 1.2}],
            "label_token_ids": {"nonhate": 10, "hate": 20},
            "wall_s": 12.5,
            "peak_vram_gib": 2.5,
        },
    )

    finetune_called = False
    merge_called = False

    def mock_finetune(*args, **kwargs):
        nonlocal finetune_called
        finetune_called = True
        raise AssertionError("finetune_lora should not be called when valid adapter exists!")

    def mock_merge(hf_id, adir, mdir, **kwargs):
        nonlocal merge_called
        merge_called = True
        mdir = Path(mdir)
        mdir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(mdir / "config.json", {"vocab_size": 1000})
        atomic_write_text(mdir / "model.safetensors", "MERGED_WEIGHTS")
        atomic_write_text(mdir / ".completed", "ok\n")
        return mdir

    monkeypatch.setattr(ph4.ft, "finetune_lora", mock_finetune)
    monkeypatch.setattr(ph4.ft, "merge_adapter_to_base", mock_merge)

    # Pre-populate valid parquets so eval step also completes cleanly
    logits_dir = tmp_path / "outputs" / "logits" / "phase4" / "bn_test_model_lora_bf16_s42"
    _create_mock_parquet(logits_dir / "bf16__id_val.parquet", n_rows=6)
    _create_mock_parquet(logits_dir / "bf16__id_test.parquet", n_rows=6)
    _create_mock_parquet(logits_dir / "bf16__ood_test.parquet", n_rows=8)

    cfg = load_config(ROOT / "configs" / "baseline" / "default.yaml")
    cfg["paths"]["runs"] = str(tmp_path / "outputs" / "runs" / "phase4")
    cfg["paths"]["models"] = str(tmp_path / "outputs" / "models" / "phase4")
    cfg["paths"]["logits"] = str(tmp_path / "outputs" / "logits" / "phase4")

    cell = {
        "model_key": "qwen2_5_0_5b_instruct",
        "hf_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "resolved_hf_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "tier": "headline",
        "seed": 42,
        "run_name": "bn_test_model_lora_bf16_s42",
        "attn_implementation": None,
    }

    manifest = ph4.run_one(cell, cfg, dummy_dataset, force=False)
    assert not finetune_called, "Training was retrained despite valid adapter!"
    assert merge_called, "Merge was not invoked when merged model was missing!"
    assert is_valid_merged(merged_dir)


def test_resume_interrupted_during_eval(ph4, dummy_dataset, tmp_path, monkeypatch):
    """B3 & B8: Interruption during eval: preserves completed split 1 parquet, computes splits 2 and 3."""
    output_dir = tmp_path / "outputs" / "models" / "phase4" / "bn_test_eval_resume_s42"
    adapter_dir = output_dir / "adapter"
    merged_dir = output_dir / "merged"
    _create_mock_adapter(adapter_dir)
    merged_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(merged_dir / "config.json", {"vocab_size": 1000})
    atomic_write_text(merged_dir / "model.safetensors", "MERGED_WEIGHTS")
    atomic_write_text(merged_dir / ".completed", "ok\n")

    atomic_write_json(
        output_dir / "train_metrics.json",
        {
            "n_train": 20,
            "n_id_val": 6,
            "trainable_params": 1000,
            "total_params": 1000000,
            "log_history": [{"loss": 2.0}, {"loss": 1.0}],
            "label_token_ids": {"nonhate": 10, "hate": 20},
            "wall_s": 10.0,
        },
    )

    logits_dir = tmp_path / "outputs" / "logits" / "phase4" / "bn_test_eval_resume_s42"
    # Pre-create valid id_val parquet
    _create_mock_parquet(logits_dir / "bf16__id_val.parquet", n_rows=6)
    mtime_id_val = (logits_dir / "bf16__id_val.parquet").stat().st_mtime

    computed_splits: list[str] = []

    def mock_extract(model, tok, frame, split, **kwargs):
        computed_splits.append(split)
        return pd.DataFrame(
            {
                "example_id": [str(i) for i in range(len(frame))],
                "y_true": frame["label"].tolist(),
                "logit_hate": [1.0] * len(frame),
                "logit_nonhate": [-1.0] * len(frame),
                "prob_hate": [0.88] * len(frame),
                "prob_nonhate": [0.12] * len(frame),
                "prediction": [1] * len(frame),
            }
        )

    monkeypatch.setattr(ph4.lx, "extract_logits_hf", mock_extract)
    monkeypatch.setattr(ph4.ft, "load_base_model", lambda *args, **kwargs: MagicMock())
    monkeypatch.setattr(ph4.ft, "load_tokenizer", lambda *args, **kwargs: MagicMock())

    cfg = load_config(ROOT / "configs" / "baseline" / "default.yaml")
    cfg["paths"]["runs"] = str(tmp_path / "outputs" / "runs" / "phase4")
    cfg["paths"]["models"] = str(tmp_path / "outputs" / "models" / "phase4")
    cfg["paths"]["logits"] = str(tmp_path / "outputs" / "logits" / "phase4")

    cell = {
        "model_key": "qwen2_5_0_5b_instruct",
        "hf_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "resolved_hf_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "tier": "headline",
        "seed": 42,
        "run_name": "bn_test_eval_resume_s42",
        "attn_implementation": None,
    }

    ph4.run_one(cell, cfg, dummy_dataset, force=False)

    # id_val should have been skipped! Only id_test and ood_test should be computed.
    assert "id_val" not in computed_splits, "id_val was recomputed despite valid existing parquet!"
    assert "id_test" in computed_splits
    assert "ood_test" in computed_splits
    assert (logits_dir / "bf16__id_val.parquet").stat().st_mtime == mtime_id_val


def test_corrupted_parquet_detection_triggers_recomputation(ph4, dummy_dataset, tmp_path, monkeypatch):
    """B2 & B8: Corrupted or truncated parquet is detected and recomputed."""
    output_dir = tmp_path / "outputs" / "models" / "phase4" / "bn_test_corrupt_parquet_s42"
    adapter_dir = output_dir / "adapter"
    merged_dir = output_dir / "merged"
    _create_mock_adapter(adapter_dir)
    merged_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(merged_dir / "config.json", {"vocab_size": 1000})
    atomic_write_text(merged_dir / "model.safetensors", "MERGED_WEIGHTS")
    atomic_write_text(merged_dir / ".completed", "ok\n")

    atomic_write_json(
        output_dir / "train_metrics.json",
        {
            "n_train": 20,
            "n_id_val": 6,
            "trainable_params": 1000,
            "total_params": 1000000,
            "log_history": [{"loss": 1.5}, {"loss": 0.8}],
            "label_token_ids": {"nonhate": 10, "hate": 20},
            "wall_s": 10.0,
        },
    )

    logits_dir = tmp_path / "outputs" / "logits" / "phase4" / "bn_test_corrupt_parquet_s42"
    # id_val is valid
    _create_mock_parquet(logits_dir / "bf16__id_val.parquet", n_rows=6)
    # id_test is corrupt (0 bytes)
    (logits_dir / "bf16__id_test.parquet").parent.mkdir(parents=True, exist_ok=True)
    (logits_dir / "bf16__id_test.parquet").touch()
    assert not is_valid_parquet(logits_dir / "bf16__id_test.parquet")
    # ood_test is valid
    _create_mock_parquet(logits_dir / "bf16__ood_test.parquet", n_rows=8)

    recomputed: list[str] = []

    def mock_extract(model, tok, frame, split, **kwargs):
        recomputed.append(split)
        return pd.DataFrame(
            {
                "example_id": [str(i) for i in range(len(frame))],
                "y_true": frame["label"].tolist(),
                "logit_hate": [1.0] * len(frame),
                "logit_nonhate": [-1.0] * len(frame),
                "prob_hate": [0.88] * len(frame),
                "prob_nonhate": [0.12] * len(frame),
                "prediction": [1] * len(frame),
            }
        )

    monkeypatch.setattr(ph4.lx, "extract_logits_hf", mock_extract)
    monkeypatch.setattr(ph4.ft, "load_base_model", lambda *args, **kwargs: MagicMock())
    monkeypatch.setattr(ph4.ft, "load_tokenizer", lambda *args, **kwargs: MagicMock())

    cfg = load_config(ROOT / "configs" / "baseline" / "default.yaml")
    cfg["paths"]["runs"] = str(tmp_path / "outputs" / "runs" / "phase4")
    cfg["paths"]["models"] = str(tmp_path / "outputs" / "models" / "phase4")
    cfg["paths"]["logits"] = str(tmp_path / "outputs" / "logits" / "phase4")

    cell = {
        "model_key": "qwen2_5_0_5b_instruct",
        "hf_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "resolved_hf_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "tier": "headline",
        "seed": 42,
        "run_name": "bn_test_corrupt_parquet_s42",
        "attn_implementation": None,
    }

    ph4.run_one(cell, cfg, dummy_dataset, force=False)
    assert recomputed == ["id_test"], f"Expected only id_test to be recomputed, got {recomputed}"
    assert is_valid_parquet(logits_dir / "bf16__id_test.parquet", expected_rows=6)


def test_config_hash_mismatch_refuses_to_skip(ph4, dummy_dataset, tmp_path, monkeypatch):
    """A7 & B8: Config change causes run_one to reject stale success manifest and re-execute."""
    runs_dir = tmp_path / "outputs" / "runs" / "phase4"
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_json = runs_dir / "bn_test_cfg_mismatch_s42.json"

    # Previous run succeeded under old config hash
    atomic_write_json(
        run_json,
        {
            "status": "success",
            "config_hash": "OLD_STALE_CONFIG_HASH",
            "split_sha256": dummy_dataset["sha"],
            "resolved_hf_id": "Qwen/Qwen2.5-0.5B-Instruct",
            "git_commit": "abc1234",
        },
    )

    logits_dir = tmp_path / "outputs" / "logits" / "phase4" / "bn_test_cfg_mismatch_s42"
    _create_mock_parquet(logits_dir / "bf16__id_val.parquet", n_rows=6)
    _create_mock_parquet(logits_dir / "bf16__id_test.parquet", n_rows=6)
    _create_mock_parquet(logits_dir / "bf16__ood_test.parquet", n_rows=8)

    cfg = load_config(ROOT / "configs" / "baseline" / "default.yaml")
    cfg["paths"]["runs"] = str(tmp_path / "outputs" / "runs" / "phase4")
    cfg["paths"]["models"] = str(tmp_path / "outputs" / "models" / "phase4")
    cfg["paths"]["logits"] = str(tmp_path / "outputs" / "logits" / "phase4")

    executed = False

    def mock_finetune(*args, **kwargs):
        nonlocal executed
        executed = True
        output_dir = tmp_path / "outputs" / "models" / "phase4" / "bn_test_cfg_mismatch_s42"
        _create_mock_adapter(output_dir / "adapter")
        (output_dir / "merged").mkdir(parents=True, exist_ok=True)
        atomic_write_text(output_dir / "merged" / ".completed", "ok\n")
        atomic_write_json(output_dir / "merged" / "config.json", {})
        atomic_write_text(output_dir / "merged" / "model.safetensors", "MOCK")
        return ft.FinetuneResult(
            adapter_path=str(output_dir / "adapter"),
            merged_path=str(output_dir / "merged"),
            history=[{"loss": 2.0}, {"loss": 1.0}],
            best_eval_loss=1.0,
            best_checkpoint="best",
            n_train=20,
            n_id_val=6,
            loss_decreased=True,
            final_train_loss=1.0,
            peak_vram_gb=1.0,
            wall_s=5.0,
            trainable_params=100,
            total_params=1000,
            label_token_ids={"nonhate": 10, "hate": 20},
            max_seq_len=64,
        )

    monkeypatch.setattr(ph4.ft, "finetune_lora", mock_finetune)
    monkeypatch.setattr(ph4.ft, "load_base_model", lambda *args, **kwargs: MagicMock())
    monkeypatch.setattr(ph4.ft, "load_tokenizer", lambda *args, **kwargs: MagicMock())

    cell = {
        "model_key": "qwen2_5_0_5b_instruct",
        "hf_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "resolved_hf_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "tier": "headline",
        "seed": 42,
        "run_name": "bn_test_cfg_mismatch_s42",
        "attn_implementation": None,
    }

    manifest = ph4.run_one(cell, cfg, dummy_dataset, force=False)
    assert not manifest.get("skipped", False), "Run was skipped despite config hash mismatch!"
    assert executed, "Fine-tuning was not executed when config hash changed!"

