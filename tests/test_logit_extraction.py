"""Tests for per-example logit extraction (Phase 3, T3.5 / T3.10).

The central regression test is :func:`test_batch_composition_does_not_change_logits`.
Reading ``logits[:, -1, :]`` on a right-padded batch silently scores the pad slot
of every row shorter than the longest prompt, so a short example gets a different
answer depending on what it was batched with.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from src.models import logit_extraction as lx

HATE_ID = 7
NONHATE_ID = 9
VERBALIZER_IDS = {" A": HATE_ID, " B": NONHATE_ID}
VERBALIZER = {"hate": " A", "nonhate": " B"}


class FakeTokenizer:
    """Minimal right-padding tokenizer that honours ``truncation_side``."""

    pad_token_id = 0

    def __init__(self, truncation_side: str = "right") -> None:
        self.truncation_side = truncation_side
        self.seen_truncation_side: str | None = None
        self.n_calls = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if text in VERBALIZER_IDS:
            return [VERBALIZER_IDS[text]]
        return [ord(c) % 20 + 1 for c in text]

    def __call__(
        self,
        texts,
        add_special_tokens: bool = False,
        padding: bool = True,
        truncation: bool = False,
        max_length: int | None = None,
        return_tensors: str | None = None,
    ) -> dict[str, torch.Tensor]:
        self.n_calls += 1
        self.seen_truncation_side = self.truncation_side
        if isinstance(texts, str):
            texts = [texts]
        seqs = [self.encode(t) for t in texts]
        if truncation and max_length is not None:
            if self.truncation_side == "left":
                seqs = [s[-max_length:] for s in seqs]
            else:
                seqs = [s[:max_length] for s in seqs]
        width = max(len(s) for s in seqs)
        ids = torch.zeros(len(seqs), width, dtype=torch.long)
        mask = torch.zeros(len(seqs), width, dtype=torch.long)
        for i, seq in enumerate(seqs):
            ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
            mask[i, : len(seq)] = 1
        return {"input_ids": ids, "attention_mask": mask}


class PositionEchoModel(torch.nn.Module):
    """``logits[b, t, v] = t + 0.001 * v``.

    Reading the correct answer position is therefore observable directly: the
    returned value must equal ``(last_real_index + 0.001 * token_id)``.
    """

    def __init__(self, vocab_size: int = 32) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids=None, attention_mask=None, use_cache=None, **kwargs):
        b, t = input_ids.shape
        pos = torch.arange(t, dtype=torch.float32).view(1, t, 1)
        vid = torch.arange(self.vocab_size, dtype=torch.float32).view(1, 1, -1)
        return SimpleNamespace(logits=(pos + 0.001 * vid).expand(b, t, self.vocab_size).clone())


def _frame(n: int = 6) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "example_id": [f"ex{i}" for i in range(n)],
            "text": [f"বাক্য {i}" for i in range(n)],
            "label": [i % 2 for i in range(n)],
            "domain": ["sarcasm"] * n,
        }
    )


# ---------------------------------------------------------------------------
# Answer-position selection
# ---------------------------------------------------------------------------

def test_answer_logits_uses_last_real_token_per_row():
    model, tok = PositionEchoModel(), FakeTokenizer()
    prompts = ["a", "abcde", "abcd"]  # lengths 1, 5, 4
    lh, ln = lx.answer_logits_batched(model, tok, prompts, HATE_ID, NONHATE_ID)

    expected_pos = np.array([0, 4, 3], dtype=np.float64)
    np.testing.assert_allclose(lh, expected_pos + 0.001 * HATE_ID, rtol=0, atol=1e-4)
    np.testing.assert_allclose(ln, expected_pos + 0.001 * NONHATE_ID, rtol=0, atol=1e-4)
    # The batch-max index would be 4 for every row; that is the bug.
    assert not np.allclose(lh, 4 + 0.001 * HATE_ID)


def test_batch_composition_does_not_change_logits():
    model, tok = PositionEchoModel(), FakeTokenizer()
    short = "abc"
    long = "abcdefghij"
    alone, _ = lx.answer_logits_batched(model, tok, [short], HATE_ID, NONHATE_ID)
    batched, _ = lx.answer_logits_batched(model, tok, [long, short], HATE_ID, NONHATE_ID)
    assert batched[1] == pytest.approx(alone[0], abs=1e-5)
    # Sanity: the long row does read its own final position.
    assert batched[0] == pytest.approx(9 + 0.001 * HATE_ID, abs=1e-4)


def test_left_truncation_keeps_answer_position_and_restores_setting():
    tok = FakeTokenizer(truncation_side="right")
    lh, _ = lx.answer_logits_batched(PositionEchoModel(), tok, ["abcdefghij"], HATE_ID, NONHATE_ID, max_len=4)

    assert tok.seen_truncation_side == "left", "eval must truncate from the left"
    assert tok.truncation_side == "right", "caller tokenizer must be left unchanged"
    assert lh[0] == pytest.approx(3 + 0.001 * HATE_ID, abs=1e-4)


# ---------------------------------------------------------------------------
# Probability helpers
# ---------------------------------------------------------------------------

def test_softmax_pair_normalises_and_orders():
    probs = lx.softmax_pair(np.array([0.0, 10.0]), np.array([1.0, 0.0]))
    assert probs.shape == (2, 2)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)
    # columns are [nonhate, hate]
    assert probs[0, 0] > probs[0, 1]  # logit_nonhate == 1.0 beats logit_hate == 0.0
    assert probs[1, 1] > probs[1, 0]  # logit_hate == 10.0 beats logit_nonhate == 0.0
    assert probs[1, 1] == pytest.approx(1 / (1 + np.exp(-10)), abs=1e-12)
    assert probs[0, 1] == pytest.approx(1 / (1 + np.exp(1)), abs=1e-12)


def test_softmax_pair_is_numerically_stable_for_large_logits():
    probs = lx.softmax_pair(np.array([1e4]), np.array([-1e4]))
    assert np.all(np.isfinite(probs))
    assert probs.sum() == pytest.approx(1.0)


def test_probs_to_frame_derived_columns():
    frame = lx.probs_to_frame(
        ["a", "b"], "id_test", [1, 0],
        np.array([2.0, 0.0]), np.array([0.0, 2.0]),
        extra={"domain": ["sarcasm", "slur"], "model": "m"},
    )
    assert list(frame["prediction"]) == [1, 0]
    assert list(frame["correct"]) == [1, 1]
    np.testing.assert_allclose(frame["confidence"].to_numpy(), [0.880797, 0.880797], atol=1e-5)
    np.testing.assert_allclose(frame["margin"].to_numpy(), [2.0, 2.0])
    assert frame.loc[0, "entropy"] > 0
    assert list(frame["model"]) == ["m", "m"]
    lx.validate_logits_frame(frame)


def test_probs_to_frame_handles_extreme_margin_entropy():
    frame = lx.probs_to_frame(["a"], "s", [1], np.array([50.0]), np.array([-50.0]))
    assert frame.loc[0, "entropy"] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_validate_logits_frame_rejects_missing_columns():
    frame = lx.probs_to_frame(["a"], "s", [1], np.array([1.0]), np.array([0.0]))
    with pytest.raises(ValueError, match="Missing required logits columns"):
        lx.validate_logits_frame(frame.drop(columns=["prob_nonhate"]))


def test_validate_logits_frame_rejects_unnormalised_probabilities():
    frame = lx.probs_to_frame(["a"], "s", [1], np.array([1.0]), np.array([0.0]))
    frame.loc[0, "prob_nonhate"] = 0.5
    with pytest.raises(ValueError, match="deviates from 1"):
        lx.validate_logits_frame(frame, prob_tol=1e-9)


def test_validate_logits_frame_rejects_duplicates_and_nonfinite():
    frame = lx.probs_to_frame(["a", "a"], "s", [1, 0], np.array([1.0, 0.0]), np.array([0.0, 1.0]))
    with pytest.raises(ValueError, match="Duplicate example_id"):
        lx.validate_logits_frame(frame)

    frame = lx.probs_to_frame(["a"], "s", [1], np.array([1.0]), np.array([0.0]))
    frame.loc[0, "logit_hate"] = np.nan
    with pytest.raises(ValueError, match="Non-finite"):
        lx.validate_logits_frame(frame)


def test_validate_logits_frame_rejects_empty():
    with pytest.raises(ValueError, match="Empty logits frame"):
        lx.validate_logits_frame(pd.DataFrame(columns=lx.REQUIRED_COLUMNS))


# ---------------------------------------------------------------------------
# End-to-end extraction + persistence
# ---------------------------------------------------------------------------

def test_extract_logits_hf_frame_and_state_restoration():
    model = PositionEchoModel()
    model.train()
    df = _frame(5)
    out = lx.extract_logits_hf(
        model, FakeTokenizer(), df,
        verbalizer=VERBALIZER, batch_size=2, split="id_test", progress_every=0,
    )
    assert len(out) == len(df)
    assert list(out["example_id"]) == list(df["example_id"])
    assert list(out["y_true"]) == list(df["label"])
    assert set(lx.REQUIRED_COLUMNS).issubset(out.columns)
    lx.validate_logits_frame(out)
    assert model.training, "extraction must not leave the model in eval mode"


def test_extract_logits_hf_batching_is_order_and_size_invariant():
    df = _frame(7)
    kwargs = dict(verbalizer=VERBALIZER, split="id_test", progress_every=0)
    model = PositionEchoModel()
    one = lx.extract_logits_hf(model, FakeTokenizer(), df, batch_size=1, max_len=64, **kwargs)
    many = lx.extract_logits_hf(model, FakeTokenizer(), df, batch_size=7, max_len=64, **kwargs)
    np.testing.assert_allclose(one["prob_hate"].to_numpy(), many["prob_hate"].to_numpy(), atol=1e-6)
    np.testing.assert_allclose(one["logit_hate"].to_numpy(), many["logit_hate"].to_numpy(), atol=1e-4)


def test_extract_logits_hf_rejects_colliding_verbalizer():
    same = {"hate": " A", "nonhate": " A"}
    with pytest.raises(ValueError, match="collide"):
        lx.extract_logits_hf(
            PositionEchoModel(), FakeTokenizer(), _frame(2), verbalizer=same, progress_every=0
        )


def test_save_and_load_logits_roundtrip(tmp_path):
    frame = lx.probs_to_frame(["a", "b"], "id_test", [1, 0], np.array([1.0, 0.0]), np.array([0.0, 1.0]))
    path = lx.save_logits(frame, tmp_path / "sub" / "bf16__id_test.parquet", precision="bf16", seed=42)
    assert path.exists()
    back = lx.load_logits(path)
    assert back["precision"].eq("bf16").all()
    assert back["seed"].eq(42).all()
    np.testing.assert_allclose(back["prob_hate"].to_numpy(), frame["prob_hate"].to_numpy())


def test_agreement_between_frames():
    a = lx.probs_to_frame(["a", "b"], "s", [1, 0], np.array([2.0, 2.0]), np.array([0.0, 0.0]))
    b = lx.probs_to_frame(["a", "b"], "s", [1, 0], np.array([2.0, -2.0]), np.array([0.0, 0.0]))
    rep = lx.agreement_between_frames(a, b)
    assert rep["n"] == 2
    assert rep["agreement_rate"] == pytest.approx(0.5)
    assert rep["max_abs_prob_delta"] > 0.5


def test_agreement_between_frames_requires_full_overlap():
    a = lx.probs_to_frame(["a", "b"], "s", [1, 0], np.array([1.0, 1.0]), np.array([0.0, 0.0]))
    b = lx.probs_to_frame(["a"], "s", [1], np.array([1.0]), np.array([0.0]))
    with pytest.raises(ValueError, match="missing from the second logits frame"):
        lx.agreement_between_frames(a, b)