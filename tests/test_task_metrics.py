"""Task-performance metric unit tests (Phase 2, 2.3)."""

from __future__ import annotations

import numpy as np
import pytest

from src.metrics.task_metrics import (
    agreement_rate,
    bootstrap_accuracy_delta,
    compute_all_task_metrics,
)


class TestComputeAllTaskMetrics:
    def test_perfect_classifier(self):
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = y_true.copy()
        y_prob = np.array([0.01, 0.02, 0.98, 0.99, 0.05, 0.95])
        m = compute_all_task_metrics(y_true, y_pred, y_prob)
        assert m["accuracy"] == pytest.approx(1.0)
        assert m["macro_f1"] == pytest.approx(1.0)
        assert m["hate_f1"] == pytest.approx(1.0)
        assert m["mcc"] == pytest.approx(1.0)
        assert m["auroc"] == pytest.approx(1.0)

    def test_all_wrong_classifier(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = 1 - y_true
        y_prob = np.array([0.9, 0.8, 0.2, 0.1])
        m = compute_all_task_metrics(y_true, y_pred, y_prob)
        assert m["accuracy"] == pytest.approx(0.0)
        assert m["auroc"] == pytest.approx(0.0)
        assert m["mcc"] == pytest.approx(-1.0)

    def test_matches_sklearn_per_class_f1(self):
        from sklearn.metrics import f1_score

        rng = np.random.default_rng(0)
        n = 500
        y_true = rng.integers(0, 2, n)
        y_prob = rng.uniform(0, 1, n)
        y_pred = (y_prob >= 0.5).astype(int)
        m = compute_all_task_metrics(y_true, y_pred, y_prob)
        assert m["macro_f1"] == pytest.approx(f1_score(y_true, y_pred, average="macro"))
        assert m["hate_f1"] == pytest.approx(f1_score(y_true, y_pred, pos_label=1))
        assert m["nonhate_f1"] == pytest.approx(f1_score(y_true, y_pred, pos_label=0))

    def test_single_class_input_does_not_crash(self):
        """OOD folds can be single-class; AUROC must degrade gracefully."""
        y_true = np.zeros(20, dtype=int)
        y_pred = np.zeros(20, dtype=int)
        y_prob = np.full(20, 0.2)
        m = compute_all_task_metrics(y_true, y_pred, y_prob)
        assert np.isnan(m["auroc"])
        assert m["accuracy"] == pytest.approx(1.0)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="equal length"):
            compute_all_task_metrics(np.array([0, 1]), np.array([0]), np.array([0.5, 0.5]))

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty input"):
            compute_all_task_metrics(np.array([]), np.array([]), np.array([]))

    def test_all_values_finite(self):
        rng = np.random.default_rng(1)
        n = 1000
        y_prob = rng.uniform(0, 1, n)
        y_true = (rng.uniform(0, 1, n) < y_prob).astype(int)
        m = compute_all_task_metrics(y_true, (y_prob >= 0.5).astype(int), y_prob)
        for k, v in m.items():
            assert isinstance(v, (float, int)), f"{k}={v!r}"


class TestAgreementRate:
    def test_identical_is_one(self):
        assert agreement_rate(np.array([0, 1, 1]), np.array([0, 1, 1])) == pytest.approx(1.0)

    def test_known_value(self):
        assert agreement_rate(np.array([0, 1, 1, 0]), np.array([0, 1, 0, 0])) == pytest.approx(0.75)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="length mismatch"):
            agreement_rate(np.array([0, 1]), np.array([0]))


class TestBootstrapAccuracyDelta:
    def test_identical_models_have_zero_delta(self):
        rng = np.random.default_rng(2)
        y_true = rng.integers(0, 2, 400)
        y_pred = (rng.uniform(0, 1, 400) >= 0.5).astype(int)
        out = bootstrap_accuracy_delta(y_true, y_pred, y_pred, n_boot=100, seed=42)
        assert out["delta"] == pytest.approx(0.0)
        assert out["ci_low"] <= 0.0 <= out["ci_high"]

    def test_detects_real_degradation(self):
        rng = np.random.default_rng(3)
        n = 2000
        y_true = rng.integers(0, 2, n)
        sign = 2 * y_true - 1
        good = (rng.uniform(0, 1, n) >= 0.5 - 0.35 * sign).astype(int)  # ~85% acc
        bad = (rng.uniform(0, 1, n) >= 0.5 - 0.10 * sign).astype(int)  # ~60% acc
        out = bootstrap_accuracy_delta(y_true, good, bad, n_boot=200, seed=7)
        assert out["delta"] == pytest.approx(0.25, abs=0.05)
        assert out["ci_low"] > 0
        assert out["p_delta_le_0"] == pytest.approx(0.0)

    def test_deterministic_given_seed(self):
        rng = np.random.default_rng(4)
        y_true = rng.integers(0, 2, 200)
        a = (rng.uniform(0, 1, 200) >= 0.5).astype(int)
        b = (rng.uniform(0, 1, 200) >= 0.5).astype(int)
        o1 = bootstrap_accuracy_delta(y_true, a, b, n_boot=50, seed=1)
        o2 = bootstrap_accuracy_delta(y_true, a, b, n_boot=50, seed=1)
        assert o1 == o2

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty input"):
            bootstrap_accuracy_delta(np.array([]), np.array([]), np.array([]))