"""Tests for one-call metric aggregation over a logits frame (Phase 3, T3.9)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.metrics.aggregate import (
    compare_precision_frames,
    evaluate_logits_frame,
    metrics_frame,
    metrics_to_record,
)


def _frame(y_true, y_prob, ids=None, split: str = "id_test") -> pd.DataFrame:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    ids = ids if ids is not None else [str(i) for i in range(len(y_true))]
    return pd.DataFrame(
        {
            "example_id": ids,
            "split": split,
            "y_true": y_true,
            "prob_hate": y_prob,
            "prediction": (y_prob >= 0.5).astype(np.int64),
        }
    )


def _balanced(n: int = 300, seed: int = 0):
    rng = np.random.default_rng(seed)
    y = (rng.uniform(size=n) < 0.31).astype(int)
    p = np.clip(0.75 * y + rng.uniform(0.05, 0.35, n), 0.01, 0.99)
    return y, p


def test_evaluate_logits_frame_on_balanced_set():
    y, p = _balanced()
    m = evaluate_logits_frame(_frame(y, p))
    assert m["single_class_eval_set"] is False
    for key in ("accuracy", "macro_f1", "hate_f1", "auroc", "mcc", "ece", "brier", "nll", "aurc"):
        assert isinstance(m[key], float) and np.isfinite(m[key]), key
    assert m["n"] == len(y)


def test_evaluate_logits_frame_on_single_class_set_flags_undefined():
    """bd_shs LOCO target domains are single-class; rank metrics are undefined there."""
    y = np.ones(80, dtype=int)
    p = np.linspace(0.55, 0.99, 80)
    m = evaluate_logits_frame(_frame(y, p, split="ood_test"))
    assert m["single_class_eval_set"] is True
    assert not np.isfinite(m["auroc"])
    assert not np.isfinite(m["failure_prediction_auroc"])
    # Calibration metrics stay finite and meaningful.
    for key in ("ece", "mce", "brier", "nll", "aurc"):
        assert np.isfinite(m[key]), key


def test_metrics_to_record_nulls_non_finite_values():
    m = evaluate_logits_frame(_frame(np.ones(50, dtype=int), np.linspace(0.6, 0.99, 50)))
    record = metrics_to_record(m, precision="bf16", split="ood_test")
    assert record["precision"] == "bf16"
    assert record["auroc"] is None
    assert np.isfinite(record["brier"])


def test_metrics_to_record_preserves_scalars_and_sequences():
    rec = metrics_to_record({"a": 1, "b": "x", "c": True, "d": [1, 2], "e": 0.5})
    assert rec == {"a": 1, "b": "x", "c": True, "d": [1, 2], "e": 0.5}


def test_metrics_frame_orders_leading_columns():
    table = metrics_frame(
        [
            {"model": "m", "precision": "bf16", "split": "id_test", "accuracy": 0.9, "ece": 0.1},
            {"model": "m", "precision": "bnb_nf4", "split": "id_test", "accuracy": 0.8, "ece": 0.2},
        ]
    )
    assert list(table.columns[:3]) == ["model", "precision", "split"]
    assert list(table["precision"]) == ["bf16", "bnb_nf4"]


def test_metrics_frame_handles_empty_input():
    assert metrics_frame([]).empty


def test_compare_precision_frames_reports_drop_and_agreement():
    y, p = _balanced()
    ref = _frame(y, p)
    # Variant flips exactly 10% of predictions.
    var_prob = np.where(np.arange(len(p)) % 10 == 0, 1.0 - p, p)
    var = _frame(y, var_prob)
    cmp = compare_precision_frames(ref, var, )
    assert cmp["n_aligned"] == len(y)
    assert cmp["label_mismatch"] == 0
    assert cmp["accuracy_drop"] >= 0.0
    assert cmp["agreement_rate"] == pytest.approx(0.9, abs=0.01)
    assert "ci_low" in cmp["bootstrap"] and "ci_high" in cmp["bootstrap"]
    # Every variant example is strictly more confident than the reference.
    assert cmp["mean_abs_prob_hate_delta"] > 0


def test_compare_precision_frames_detects_label_mismatch():
    y, p = _balanced(60)
    ref = _frame(y, p)
    var = _frame(1 - y, p)
    assert compare_precision_frames(ref, var)["label_mismatch"] == 60


def test_compare_precision_frames_requires_overlap():
    y, p = _balanced(20)
    ref = _frame(y, p)
    var = _frame(y, p, ids=[f"z{i}" for i in range(20)])
    with pytest.raises(ValueError, match="No overlapping example_id"):
        compare_precision_frames(ref, var)


def test_evaluate_logits_frame_requires_probability_column():
    bad = _frame([0, 1], [0.2, 0.8]).drop(columns=["prob_hate"])
    with pytest.raises(ValueError, match="No positive-class probability"):
        evaluate_logits_frame(bad)