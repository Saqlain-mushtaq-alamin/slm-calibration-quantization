"""Paired inference utilities for Phase 13.

The analysis layer aggregates seeds within model family before inference;
per-example pairing remains available here for focused bootstrap checks.
"""

from __future__ import annotations

from itertools import product

import numpy as np
from scipy.stats import t as student_t
from statsmodels.stats.multitest import multipletests


def _paired(a, b):
    left = np.asarray(a, dtype=np.float64).reshape(-1)
    right = np.asarray(b, dtype=np.float64).reshape(-1)
    if len(left) == 0 or left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("paired arrays must have equal nonzero length and finite values")
    return left, right


def paired_bootstrap_delta(metric_a, metric_b, n_resamples: int = 10000, seed: int = 42):
    """Percentile CI and two-sided bootstrap sign test for B−A.

    Resampling draws *paired* indices. Callers must pass independent units
    (e.g. model-family means) when making a population-level claim.
    """
    a, b = _paired(metric_a, metric_b)
    if n_resamples < 100:
        raise ValueError("n_resamples must be at least 100")
    diff = b - a
    observed = float(diff.mean())
    rng = np.random.default_rng(seed)
    draws = np.empty(n_resamples, dtype=np.float64)
    for start in range(0, n_resamples, 256):
        end = min(start + 256, n_resamples)
        indices = rng.integers(0, len(diff), size=(end - start, len(diff)))
        draws[start:end] = diff[indices].mean(axis=1)
    low, high = np.percentile(draws, [2.5, 97.5])
    # With only a few model families, bootstrap tail frequencies are badly
    # anti-conservative. Use the exact paired sign-flip null for the p-value.
    if len(diff) <= 20:
        null_means = np.fromiter((np.mean(diff * signs) for signs in product((-1, 1), repeat=len(diff))),
                                 dtype=float, count=2 ** len(diff))
        p_value = float(np.mean(np.abs(null_means) >= abs(observed) - 1e-12))
    else:
        p_value = min(1.0, 2 * min((np.count_nonzero(draws <= 0) + 1) / (n_resamples + 1),
                                    (np.count_nonzero(draws >= 0) + 1) / (n_resamples + 1)))
    return {"observed_delta": observed, "ci_lower": float(low), "ci_upper": float(high),
            "p_value": float(p_value), "significant_at_05": bool(low > 0 or high < 0)}


def tost_equivalence(metric_a, metric_b, margin: float):
    """Correct paired two one-sided t tests for |mean(B−A)| < margin."""
    a, b = _paired(metric_a, metric_b)
    if not np.isfinite(margin) or margin <= 0:
        raise ValueError("equivalence margin must be positive and finite")
    diff = b - a
    mean = float(diff.mean())
    if len(diff) < 2:
        return {"mean_diff": mean, "margin": float(margin), "p_tost": None,
                "equivalent_at_05": None, "reason": "fewer than two paired units"}
    se = float(diff.std(ddof=1) / np.sqrt(len(diff)))
    if se == 0:
        p = 0.0 if abs(mean) < margin else 1.0
    else:
        df = len(diff) - 1
        p_lower = float(student_t.sf((mean + margin) / se, df))
        p_upper = float(student_t.cdf((mean - margin) / se, df))
        p = max(p_lower, p_upper)
    return {"mean_diff": mean, "margin": float(margin), "p_tost": float(p),
            "equivalent_at_05": bool(p < .05), "reason": None}


def apply_correction(p_values, method: str = "holm"):
    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("p-values must be finite numbers in [0, 1]")
    if len(values) == 0:
        return []
    return multipletests(values, method=method)[1].tolist()


def cohens_d(group_a, group_b):
    """Paired Cohen's dz; undefined for a constant nonzero difference."""
    a, b = _paired(group_a, group_b)
    diff = b - a
    if np.all(diff == 0):
        return 0.0
    if len(diff) < 2 or diff.std(ddof=1) == 0:
        return None
    return float(diff.mean() / diff.std(ddof=1))


def cliffs_delta(group_a, group_b):
    """Unpaired descriptive rank effect, P(B>A)−P(B<A)."""
    a = np.asarray(group_a, dtype=np.float64).reshape(-1)
    b = np.asarray(group_b, dtype=np.float64).reshape(-1)
    if not len(a) or not len(b) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("groups must be finite and nonempty")
    comparisons = b[:, None] - a[None, :]
    return float((np.count_nonzero(comparisons > 0) - np.count_nonzero(comparisons < 0)) / comparisons.size)
