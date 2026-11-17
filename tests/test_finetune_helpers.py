"""Tests for data subsampling and prompt construction helpers (Phase 3 pilot)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models import finetune as ft
from src.models import logit_extraction as lx


def _frame(n: int = 1000, pos_frac: float = 0.3) -> pd.DataFrame:
    n_pos = int(round(n * pos_frac))
    labels = np.array([1] * n_pos + [0] * (n - n_pos))
    rng = np.random.default_rng(0)
    rng.shuffle(labels)
    return pd.DataFrame(
        {
            "example_id": [f"ex{i:05d}" for i in range(n)],
            "text": [f"text {i}" for i in range(n)],
            "label": labels,
        }
    )


@pytest.mark.parametrize("n", [1, 2, 17, 250, 999, 1000])
def test_subsample_returns_exactly_n_rows(n):
    out = ft.subsample(_frame(1000), n, seed=42)
    assert len(out) == n
    assert out["example_id"].nunique() == n
    assert "label" in out.columns, "grouping column must survive subsampling"


def test_subsample_caps_at_frame_size():
    df = _frame(50)
    assert len(ft.subsample(df, 5000, seed=1)) == 50


def test_subsample_is_deterministic_for_a_seed():
    df = _frame(1000)
    a = ft.subsample(df, 200, seed=7)
    b = ft.subsample(df, 200, seed=7)
    c = ft.subsample(df, 200, seed=8)
    assert list(a["example_id"]) == list(b["example_id"])
    assert list(a["example_id"]) != list(c["example_id"])


def test_subsample_preserves_class_ratio_and_keeps_both_classes():
    df = _frame(1000, pos_frac=0.3)
    out = ft.subsample(df, 300, seed=42)
    assert set(out["label"].unique()) == {0, 1}
    assert out["label"].mean() == pytest.approx(0.3, abs=0.02)


def test_subsample_noop_when_n_ge_len_or_none():
    df = _frame(50)
    assert len(ft.subsample(df, None, seed=1)) == 50
    assert len(ft.subsample(df, 50, seed=1)) == 50
    assert len(ft.subsample(df, 999, seed=1)) == 50


def test_subsample_requires_label_column():
    with pytest.raises(ValueError, match="label"):
        ft.subsample(pd.DataFrame({"example_id": ["a", "b"], "text": ["t", "u"]}), 1)


def test_frame_hash_detects_content_change():
    df = _frame(20)
    h1 = ft.frame_hash(df)
    assert h1 == ft.frame_hash(df.copy())
    changed = df.copy()
    changed.loc[0, "label"] = 1 - changed.loc[0, "label"]
    assert ft.frame_hash(changed) != h1


def test_build_prompt_uses_default_template_cue():
    prompt = ft.build_prompt("আমি   তোমাকে ঘৃণা করি")
    assert prompt.endswith("Answer:")
    body = prompt.split("Text: ")[1].split("\n")[0]
    assert body == "আমি তোমাকে ঘৃণা করি"


def test_build_prompt_accepts_custom_template():
    template = "{text}\nবাক্যটি ঘৃণামূলক কি না? উত্তর:"
    assert ft.build_prompt("test", template) == "test\nবাক্যটি ঘৃণামূলক কি না? উত্তর:"


def test_frame_to_examples_and_eval_examples_agree():
    df = _frame(6)
    df["domain"] = "sarcasm"
    sft = ft.frame_to_examples(df)
    ev = lx.build_eval_examples(df)
    assert len(sft) == len(ev) == 6
    assert [e.example_id for e in sft] == [e["example_id"] for e in ev]
    assert sft[0].prompt == ev[0]["prompt"]
    assert [e.label for e in sft] == [e["y_true"] for e in ev]


class _DummyTokenizer:
    def __init__(self, bos_token_id: int | None = 128000):
        self.bos_token_id = bos_token_id
        self.pad_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if " X" in text or text == " X":
            return [10]
        if " Y" in text or text == " Y":
            return [20]
        # Return deterministic tokens based on character ordinals
        return [100 + (ord(c) % 50) for c in text]


def test_verbalized_sft_dataset_prepends_bos_and_respects_custom_verbalizer():
    tok = _DummyTokenizer(bos_token_id=128000)
    examples = [
        ft.TrainingExample(example_id="1", prompt="Hello world", label=0),
        ft.TrainingExample(example_id="2", prompt="Test prompt", label=1),
    ]
    ds = ft.VerbalizedSFTDataset(
        examples,
        tok,
        max_seq_len=64,
        verbalizer={0: " X", 1: " Y"},
    )
    # Check BOS is prepended
    assert ds[0]["input_ids"][0] == 128000
    assert ds[1]["input_ids"][0] == 128000
    # Check label tokens match custom verbalizer
    assert ds[0]["input_ids"][-1] == 10
    assert ds[1]["input_ids"][-1] == 20
    # Check target position is masked except last
    assert ds[0]["labels"][-1] == 10
    assert ds[0]["labels"][-2] == -100

    # Test truncation keeps BOS at index 0
    ds_trunc = ft.VerbalizedSFTDataset(
        examples,
        tok,
        max_seq_len=6,
        verbalizer={0: " X", 1: " Y"},
    )
    assert len(ds_trunc[0]["input_ids"]) == 6
    assert ds_trunc[0]["input_ids"][0] == 128000
    assert ds_trunc[0]["input_ids"][-1] == 10


def test_sft_collator_hygiene():
    tok = _DummyTokenizer(bos_token_id=2)
    collator = ft.SFTCollator(tok)
    batch_features = [
        {"input_ids": [2, 101, 10], "labels": [-100, -100, 10], "example_id": "ex1", "extra": "drop_me"},
        {"input_ids": [2, 102, 103, 20], "labels": [-100, -100, -100, 20], "example_id": "ex2", "extra": "drop_me"},
    ]
    batch = collator(batch_features)
    assert set(batch.keys()) == {"input_ids", "attention_mask", "labels"}
    assert "example_id" not in batch
    assert "extra" not in batch
    assert batch["input_ids"].shape == (2, 4)
    assert batch["labels"].shape == (2, 4)


def test_load_base_model_cpu_offload_detection(monkeypatch):
    class MockModel:
        def __init__(self):
            self.hf_device_map = {"model.embed_tokens": "cuda:0", "model.layers.0": "cpu"}

    monkeypatch.setattr(
        "transformers.AutoModelForCausalLM.from_pretrained",
        lambda *args, **kwargs: MockModel(),
    )
    with pytest.raises(RuntimeError, match="offloaded to CPU or disk"):
        ft.load_base_model("dummy_path", "bfloat16")


def test_divergence_check_callback():
    from transformers import TrainerControl, TrainerState, TrainingArguments

    cb = ft.DivergenceCheckCallback(max_final_train_loss=5.0, consecutive_diverged_steps=2)
    args = TrainingArguments(output_dir="tmp_dummy")
    state = TrainerState()
    control = TrainerControl()

    # Step 1: normal loss
    cb.on_log(args, state, control, logs={"loss": 2.0})
    assert control.should_training_stop is False

    # Step 2: loss > max_loss (1st time)
    cb.on_log(args, state, control, logs={"loss": 6.0})
    assert control.should_training_stop is False

    # Step 3: loss > max_loss (2nd consecutive time) -> should stop!
    cb.on_log(args, state, control, logs={"loss": 6.5})
    assert control.should_training_stop is True
