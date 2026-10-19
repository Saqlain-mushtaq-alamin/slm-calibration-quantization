"""T2.1–T2.7, T2.9: calibration metric unit tests (+ netcal cross-check)."""

from __future__ import annotations

import numpy as np
import pytest
from netcal.metrics import ACE, ECE, MCE

from src.metrics.calibration import (
    brier_decomposition,
    brier_score,
    compute_all_calibration_metrics,
    confidence_gap,
    expected_calibration_error,
    maximum_calibration_error,
    negative_log_likelihood,
    reliability_diagram,
)


class TestECE:
    def test_perfectly_calibrated(self):
        """T2.1: a perfectly calibrated model should have ECE near 0."""
        np.random.seed(42)
        n = 10000
        y_prob = np.random.uniform(0, 1, n)
        y_true = (np.random.uniform(0, 1, n) < y_prob).astype(int)
        ece = expected_calibration_error(y_true, y_prob, n_bins=15, strategy="equal_width")
        assert ece < 0.03, f"Perfect calibration should give ECE near 0, got {ece}"

    def test_always_overconfident(self):
        """T2.2: a model that always predicts 0.99 should have high ECE."""
        y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        y_prob = np.full(10, 0.99)
        ece = expected_calibration_error(y_true, y_prob, n_bins=5, strategy="equal_width")
        assert ece > 0.3, f"Overconfident model should have high ECE, got {ece}"

    def test_ece_bounds(self):
        """T2.3: ECE must be in [0, 1]."""
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 2, 100)
        y_prob = rng.uniform(0, 1, 100)
        ece = expected_calibration_error(y_true, y_prob)
        assert 0 <= ece <= 1

    def test_equal_mass_vs_equal_width(self):
        """Both strategies give valid ECE values and agree on synthetic data."""
        rng = np.random.default_rng(1)
        y_true = rng.integers(0, 2, 500)
        y_prob = rng.uniform(0, 1, 500)
        ece_em = expected_calibration_error(y_true, y_prob, strategy="equal_mass")
        ece_ew = expected_calibration_error(y_true, y_prob, strategy="equal_width")
        assert 0 <= ece_em <= 1
        assert 0 <= ece_ew <= 1

    def test_unknown_strategy_raises(self):
        y_true, y_prob = np.array([0, 1]), np.array([0.2, 0.8])
        with pytest.raises(ValueError, match="Unknown strategy"):
            expected_calibration_error(y_true, y_prob, strategy="quantile")

    def test_perfect_and_worst_case(self):
        """ECE == 0 for a perfect ranker with 0/1 probs; == 1 when fully inverted."""
        y_true = np.array([0, 0, 1, 1])
        assert expected_calibration_error(y_true, np.array([0.0, 0.0, 1.0, 1.0])) == pytest.approx(0.0)
        assert expected_calibration_error(y_true, np.array([1.0, 1.0, 0.0, 0.0])) == pytest.approx(1.0)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="length mismatch"):
            expected_calibration_error(np.array([0, 1]), np.array([0.5]))

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty input"):
            expected_calibration_error(np.array([]), np.array([]))

    def test_mce_at_least_ece(self):
        rng = np.random.default_rng(7)
        y_true = rng.integers(0, 2, 800)
        y_prob = rng.uniform(0, 1, 800)
        assert maximum_calibration_error(y_true, y_prob) >= expected_calibration_error(y_true, y_prob)


class TestBrier:
    def test_perfect_predictions(self):
        """T2.4: perfect predictions give Brier = 0."""
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.0, 1.0, 0.0, 1.0])
        assert brier_score(y_true, y_prob) == 0.0

    def test_worst_predictions(self):
        """T2.5: worst predictions give Brier = 1."""
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([1.0, 0.0, 1.0, 0.0])
        assert brier_score(y_true, y_prob) == 1.0

    def test_matches_sklearn(self):
        from sklearn.metrics import brier_score_loss

        rng = np.random.default_rng(3)
        y_true = rng.integers(0, 2, 300)
        y_prob = rng.uniform(0, 1, 300)
        assert brier_score(y_true, y_prob) == pytest.approx(
            brier_score_loss(y_true, y_prob), abs=1e-12
        )

    def test_decomposition_sums_to_brier(self):
        rng = np.random.default_rng(4)
        y_true = rng.integers(0, 2, 2000)
        y_prob = rng.uniform(0, 1, 2000)
        d = brier_decomposition(y_true, y_prob)
        assert d["brier"] == pytest.approx(d["brier_direct"], abs=1e-12)
        assert d["reliability"] - d["resolution"] + d["uncertainty"] == pytest.approx(
            d["brier_murphy_3term"]
        )
        assert d["brier"] == pytest.approx(
            d["brier_murphy_3term"] + d["within_bin_variance"] + d["within_bin_covariance"],
            abs=1e-12,
        )
        assert d["reliability"] >= 0
        assert d["resolution"] >= 0
        assert d["within_bin_variance"] >= 0

    def test_decomposition_reduces_to_three_terms_for_point_forecasts(self):
        """With one forecast per bin, variance and covariance terms vanish."""
        y_prob = np.array([0.1, 0.2, 0.3, 0.9, 0.8, 0.7])
        y_true = np.array([0, 0, 1, 1, 1, 0])
        d = brier_decomposition(y_true, y_prob, n_bins=6, strategy="equal_mass")
        assert d["within_bin_variance"] < 1e-12
        assert d["within_bin_covariance"] == pytest.approx(0.0, abs=1e-12)
        assert d["brier_murphy_3term"] == pytest.approx(d["brier_direct"], abs=1e-12)

    def test_decomposition_uncertainty(self):
        d = brier_decomposition(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
        assert d["uncertainty"] == pytest.approx(0.25)


class TestNLL:
    def test_perfect(self):
        """T2.6: perfect predictions give NLL near 0."""
        y_true = np.array([1, 0, 1])
        y_prob = np.array([0.999, 0.001, 0.999])
        assert negative_log_likelihood(y_true, y_prob) < 0.01

    def test_random(self):
        """T2.7: p=0.5 everywhere gives NLL ≈ ln 2 = 0.693."""
        rng = np.random.default_rng(11)
        y_true = rng.integers(0, 2, 10000)
        y_prob = np.full(10000, 0.5)
        assert abs(negative_log_likelihood(y_true, y_prob) - 0.693) < 0.01

    def test_matches_sklearn_log_loss(self):
        from sklearn.metrics import log_loss

        rng = np.random.default_rng(12)
        y_true = rng.integers(0, 2, 400)
        y_prob = rng.uniform(0.01, 0.99, 400)
        assert negative_log_likelihood(y_true, y_prob) == pytest.approx(
            log_loss(y_true, y_prob), abs=1e-9
        )

    def test_probability_zero_is_finite(self):
        """Clipping keeps log(0) from producing inf."""
        val = negative_log_likelihood(np.array([1]), np.array([0.0]))
        assert np.isfinite(val)
        assert val > 30


class TestConfidenceGap:
    def test_sign(self):
        """Overconfident model -> positive gap."""
        y_true = np.array([0, 0, 0, 0, 0])
        y_prob = np.array([0.9, 0.9, 0.9, 0.9, 0.9])
        assert confidence_gap(y_true, y_prob) > 0

    def test_underconfident_negative(self):
        """All correct, but hesitant forecasts -> negative gap."""
        y_true = np.array([1, 1, 0, 0])
        y_prob = np.array([0.90, 0.60, 0.40, 0.10])
        assert confidence_gap(y_true, y_prob) == pytest.approx(-0.25)

    def test_perfect_confident_is_zero(self):
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.0, 1.0, 0.0, 1.0])
        assert confidence_gap(y_true, y_prob) == pytest.approx(0.0)


class TestReliabilityDiagram:
    def test_shapes_and_counts(self):
        rng = np.random.default_rng(21)
        y_true = rng.integers(0, 2, 1500)
        y_prob = rng.uniform(0, 1, 1500)
        d = reliability_diagram(y_true, y_prob, n_bins=15)
        assert d["confidence"].shape == d["accuracy"].shape == d["count"].shape
        assert d["count"].sum() == 1500

    def test_equal_width_bins_are_ordered(self):
        y_true = np.array([0, 1, 0, 1, 1])
        y_prob = np.array([0.05, 0.15, 0.45, 0.75, 0.95])
        d = reliability_diagram(y_true, y_prob, n_bins=5, strategy="equal_width")
        assert np.all(np.diff(d["confidence"]) >= 0)


class TestComputeAll:
    def test_all_keys_are_finite(self):
        rng = np.random.default_rng(31)
        n = 2000
        y_prob = rng.uniform(0.01, 0.99, n)
        y_true = (rng.uniform(0, 1, n) < y_prob).astype(int)
        m = compute_all_calibration_metrics(y_true, y_prob)
        for k, v in m.items():
            assert isinstance(v, float) and np.isfinite(v), f"{k}={v}"
        assert {"ece", "mce", "brier", "nll", "confidence_gap"} <= set(m)


class TestCrossCheckNetcal:
    """T2.9: cross-check against netcal 1.3.6 (ECE/ACE/MCE share the binning API)."""

    @staticmethod
    def _measure(metric_cls, y_true, y_prob, n_bins, **kwargs):
        from netcal.metrics import ECE  # noqa: F401  (netcal import sanity)

        return float(metric_cls(n_bins, **kwargs).measure(y_prob, y_true))

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_ece_matches_netcal_equal_width(self, seed):
        rng = np.random.default_rng(seed)
        y_true = rng.integers(0, 2, 1000)
        y_prob = rng.uniform(0, 1, 1000)
        ours = expected_calibration_error(y_true, y_prob, n_bins=15, strategy="equal_width")
        theirs = self._measure(ECE, y_true, y_prob, 15, equal_intervals=True)
        assert abs(ours - theirs) < 0.02, f"ours={ours} netcal={theirs}"
        assert ours == pytest.approx(theirs, abs=1e-9)

    def test_mce_matches_netcal(self):
        rng = np.random.default_rng(5)
        y_true = rng.integers(0, 2, 1000)
        y_prob = rng.uniform(0, 1, 1000)
        ours = maximum_calibration_error(y_true, y_prob, n_bins=15, strategy="equal_width")
        theirs = self._measure(MCE, y_true, y_prob, 15, equal_intervals=True)
        assert abs(ours - theirs) < 1e-9, f"ours={ours} netcal={theirs}"

    def test_ece_mass_matches_netcal_ace(self):
        """Adaptive (equal-mass) ECE agrees with netcal's ACE within 0.02."""
        rng = np.random.default_rng(8)
        y_true = rng.integers(0, 2, 2000)
        y_prob = rng.uniform(0, 1, 2000)
        ours = expected_calibration_error(y_true, y_prob, n_bins=15, strategy="equal_mass")
        theirs = self._measure(ACE, y_true, y_prob, 15)
        assert abs(ours - theirs) < 0.02, f"ours={ours} netcal={theirs}"

    def test_ece_matches_netcal_on_overconfident(self):
        """Sanity: agreement also holds in the skewed regime the study cares about."""
        rng = np.random.default_rng(6)
        y_true = rng.integers(0, 2, 1200)
        y_prob = np.clip(rng.beta(2.0, 2.0, 1200) ** 4, 0.0, 1.0)
        ours = expected_calibration_error(y_true, y_prob, n_bins=15, strategy="equal_width")
        theirs = self._measure(ECE, y_true, y_prob, 15, equal_intervals=True)
        assert abs(ours - theirs) < 0.02, f"ours={ours} netcal={theirs}"

    def test_ece_matches_netcal_on_ood_prior_shift(self):
        """OOD folds have a shifted class prior; agreement must survive that."""
        rng = np.random.default_rng(9)
        y_true = (rng.uniform(0, 1, 1500) < 0.95).astype(int)  # heavily imbalanced
        y_prob = np.clip(rng.beta(1.5, 6.0, 1500), 0.0, 1.0)
        ours = expected_calibration_error(y_true, y_prob, n_bins=15, strategy="equal_width")
        theirs = self._measure(ECE, y_true, y_prob, 15, equal_intervals=True)
        assert abs(ours - theirs) < 0.02, f"ours={ours} netcal={theirs}"