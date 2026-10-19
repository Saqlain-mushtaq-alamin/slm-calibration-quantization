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