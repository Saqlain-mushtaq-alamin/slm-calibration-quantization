"""T2.8: selective-prediction metric unit tests."""

from __future__ import annotations

import numpy as np
import pytest

from src.metrics.selective import (
    aurc,
    compute_all_selective_metrics,
    coverage_at_risk,
    e_aurc,
    failure_prediction_auroc,
    optimal_aurc,
    risk_coverage_curve,
    selective_accuracy_at_coverage,
)


def _perfect_ranker(n_errors: int = 1, n: int = 100):
    """Errors placed at the lowest confidence -> optimal risk-coverage ranking.

    Every example is predicted positive (``y_prob >= 0.5``); the ``n_errors``
    negatives get the smallest confidence so they are accepted last.
    """
    n_errors = min(n_errors, n)
    y_true = np.ones(n, dtype=int)
    y_prob = np.full(n, 0.9)
    y_true[:n_errors] = 0
    y_prob[:n_errors] = 0.51
    return y_true, y_prob


def _worst_ranker(n_errors: int = 1, n: int = 100):
    """Same error count, but errors carry the *highest* confidence."""
    n_errors = min(n_errors, n)
    y_true = np.ones(n, dtype=int)
    y_prob = np.full(n, 0.6)
    y_true[:n_errors] = 0
    y_prob[:n_errors] = 0.99
    return y_true, y_prob


class TestRiskCoverageCurve:
    def test_shapes_and_monotonic_coverage(self):
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 2, 200)
        y_prob = rng.uniform(0, 1, 200)
        cov, risk = risk_coverage_curve(y_true, y_prob)
        assert cov.shape == risk.shape == (200,)
        assert cov[0] == pytest.approx(1 / 200)
        assert cov[-1] == pytest.approx(1.0)
        assert np.all(np.diff(cov) > 0)
        assert np.all((risk >= 0) & (risk <= 1))

    def test_final_risk_equals_error_rate(self):
        rng = np.random.default_rng(1)
        y_true = rng.integers(0, 2, 500)
        y_prob = rng.uniform(0, 1, 500)
        _, risk = risk_coverage_curve(y_true, y_prob)
        expected = ((y_prob >= 0.5).astype(int) != y_true).mean()
        assert risk[-1] == pytest.approx(expected)

    def test_perfect_predictions_risk_is_zero(self):
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.01, 0.99, 0.02, 0.98])
        _, risk = risk_coverage_curve(y_true, y_prob)
        assert np.allclose(risk, 0.0)


class TestAURC:
    def test_perfect_ranker(self):
        """Perfect ranker (errors at lowest confidence) gives minimal AURC."""
        y_true, y_prob = _perfect_ranker(1, 5)
        val = aurc(y_true, y_prob)
        assert val >= 0
        assert val == pytest.approx(optimal_aurc(5, 1))

    def test_aurc_bounds(self):
        """T2.8: AURC must be in [0, 1]."""
        rng = np.random.default_rng(2)
        y_true = rng.integers(0, 2, 200)
        y_prob = rng.uniform(0, 1, 200)
        val = aurc(y_true, y_prob)
        assert 0 <= val <= 1

    def test_aurc_is_zero_for_perfect_classifier(self):
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_prob = np.array([0.01, 0.99, 0.02, 0.98, 0.05, 0.95])
        assert aurc(y_true, y_prob) == pytest.approx(0.0)

    def test_aurc_increases_when_ranking_is_destroyed(self):
        rng = np.random.default_rng(3)
        y_true = rng.integers(0, 2, 300)
        good = np.where(y_true == 1, 0.9, 0.1) + rng.uniform(0, 0.05, 300)
        bad = rng.uniform(0.4, 0.6, 300)
        assert aurc(y_true, good) < aurc(y_true, bad)


class TestOptimalAURC:
    def test_no_errors_is_zero(self):
        assert optimal_aurc(50, 0) == pytest.approx(0.0)

    def test_all_errors_equals_full_error_rate(self):
        """If every example is wrong, risk == 1 at every coverage."""
        assert optimal_aurc(10, 10) == pytest.approx(1.0)
        assert optimal_aurc(500, 500) == pytest.approx(1.0)

    def test_matches_explicit_simulation(self):
        """Cross-check the closed form against an explicit per-rank sum."""
        for n, e in ((50, 3), (200, 20), (1000, 137)):
            ks = np.arange(1, n + 1)
            risks = np.maximum(0, ks - (n - e)) / ks
            assert optimal_aurc(n, e) == pytest.approx(risks.mean())

    def test_monotone_in_error_count(self):
        vals = [optimal_aurc(100, k) for k in (0, 10, 25, 50, 100)]
        assert all(a <= b for a, b in zip(vals, vals[1:]))

    def test_error_count_clipped_to_n(self):
        assert optimal_aurc(10, 99) == optimal_aurc(10, 10)

    def test_trapezoid_variant_available(self):
        assert 0 < optimal_aurc(1000, 100, "trapezoid") <= 1.0


class TestEAURC:
    def test_zero_for_perfect_ranker(self):
        y_true, y_prob = _perfect_ranker(2, 20)
        assert e_aurc(y_true, y_prob) == pytest.approx(0.0, abs=1e-12)

    def test_positive_for_misranked_errors(self):
        """Errors ranked above correct predictions -> E-AURC > 0."""
        y_true, y_prob = _worst_ranker(5, 40)
        assert e_aurc(y_true, y_prob) > 0

    def test_non_negative_when_error_count_matches_optimal(self):
        y_true, y_prob = _perfect_ranker(3, 30)
        assert e_aurc(y_true, y_prob) >= -1e-12

    def test_perfect_ranker_minimises_e_aurc(self):
        good_t, good_p = _perfect_ranker(5, 50)
        bad_t, bad_p = _worst_ranker(5, 50)
        assert e_aurc(good_t, good_p) < e_aurc(bad_t, bad_p)
        assert aurc(good_t, good_p) < aurc(bad_t, bad_p)


class TestSelectiveAccuracy:
    def test_full_coverage_equals_accuracy(self):
        rng = np.random.default_rng(4)
        y_true = rng.integers(0, 2, 200)
        y_prob = rng.uniform(0, 1, 200)
        acc = (y_prob >= 0.5).astype(int).__eq__(y_true).mean()
        assert selective_accuracy_at_coverage(y_true, y_prob, 1.0) == pytest.approx(acc)

    def test_low_coverage_at_least_full_coverage(self):
        """Confident-first selection should not be worse than using everything."""
        rng = np.random.default_rng(5)
        y_true = rng.integers(0, 2, 1000)
        y_prob = rng.uniform(0, 1, 1000)
        assert selective_accuracy_at_coverage(y_true, y_prob, 0.5) >= (
            selective_accuracy_at_coverage(y_true, y_prob, 1.0) - 0.05
        )

    def test_invalid_coverage_raises(self):
        with pytest.raises(ValueError, match="coverage"):
            selective_accuracy_at_coverage(np.array([0, 1]), np.array([0.2, 0.8]), 1.5)


class TestCoverageAtRisk:
    def test_perfect_model_gives_full_coverage(self):
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.01, 0.99, 0.02, 0.98])
        assert coverage_at_risk(y_true, y_prob, max_risk=0.05) == pytest.approx(1.0)

    def test_random_model_gives_low_coverage(self):
        rng = np.random.default_rng(6)
        y_true = rng.integers(0, 2, 2000)
        y_prob = rng.uniform(0.4, 0.6, 2000)
        assert coverage_at_risk(y_true, y_prob, max_risk=0.05) < 0.5

    def test_zero_when_risk_unreachable(self):
        y_true = np.array([0, 0, 0, 0])
        y_prob = np.array([0.99, 0.99, 0.99, 0.99])
        assert coverage_at_risk(y_true, y_prob, max_risk=0.01) == 0.0


class TestFailurePredictionAUROC:
    def test_perfect_confidence_ordering_gives_high_auroc(self):
        y_true, y_prob = _perfect_ranker(10, 200)
        val = failure_prediction_auroc(y_true, y_prob)
        assert 0.9 <= val <= 1.0

    def test_inverted_ordering_gives_low_auroc(self):
        y_true, y_prob = _worst_ranker(10, 200)
        val = failure_prediction_auroc(y_true, y_prob)
        assert 0.0 <= val <= 0.1

    def test_nan_when_no_errors(self):
        y_true = np.zeros(10, dtype=int)
        y_prob = np.full(10, 0.9)
        assert np.isnan(failure_prediction_auroc(y_true, y_prob))


class TestComputeAllSelective:
    def test_keys_present_and_finite(self):
        rng = np.random.default_rng(9)
        n = 3000
        y_prob = rng.uniform(0.05, 0.95, n)
        y_true = (rng.uniform(0, 1, n) < y_prob).astype(int)
        m = compute_all_selective_metrics(y_true, y_prob)
        assert {"aurc", "e_aurc", "failure_prediction_auroc"} <= set(m)
        assert 0 <= m["aurc"] <= 1
        for k, v in m.items():
            assert np.isfinite(v), f"{k}={v}"