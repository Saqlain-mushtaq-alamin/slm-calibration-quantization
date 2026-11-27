import numpy as np
import pytest

from src.analysis.statistical_tests import (
    apply_correction, cliffs_delta, cohens_d, paired_bootstrap_delta, tost_equivalence,
)


def test_paired_bootstrap_and_exact_small_sample_p():
    a = np.zeros(3)
    b = np.array([0.1, 0.2, 0.3])
    result = paired_bootstrap_delta(a, b, n_resamples=500, seed=7)
    assert result["ci_lower"] <= 0.2 <= result["ci_upper"]
    assert result["p_value"] == 0.25
    assert paired_bootstrap_delta(a, a, n_resamples=100)["p_value"] == 1
    with pytest.raises(ValueError):
        paired_bootstrap_delta(a, b[:2])


def test_tost_equivalence_and_non_equivalence():
    a = np.zeros(4)
    assert tost_equivalence(a, a, 0.02)["equivalent_at_05"] is True
    assert tost_equivalence(a, np.full(4, 0.03), 0.02)["equivalent_at_05"] is False
    assert tost_equivalence(a, np.array([0.0, 0.01, 0.0, 0.01]), 0.02)["p_tost"] < 0.05


def test_holm_and_effect_sizes():
    raw = [0.01, 0.03, 0.8]
    corrected = apply_correction(raw)
    assert all(p >= original for p, original in zip(corrected, raw))
    assert cohens_d([1, 2, 3], [1, 2, 3]) == 0
    assert cliffs_delta([0, 0], [1, 1]) == 1
