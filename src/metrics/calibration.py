"""Calibration metrics: ECE, Brier (+ Murphy decomposition), NLL, confidence gap.

All functions take binary labels ``y_true`` (0/1) and ``y_prob`` = P(class 1).
Conventions follow Guo et al. (2017) and Murphy (1973); netcal is used as an
independent cross-check in tests.
"""

from __future__ import annotations

from typing import Any

import numpy as np

_EPS = 1e-15


def _as_float_array(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(-1)


def _validate(y_true: Any, y_prob: Any) -> tuple[np.ndarray, np.ndarray]:
    yt = _as_float_array(y_true)
    yp = _as_float_array(y_prob)
    if yt.shape[0] != yp.shape[0]:
        raise ValueError(f"length mismatch: y_true={yt.shape[0]} y_prob={yp.shape[0]}")
    if yt.shape[0] == 0:
        raise ValueError("empty input")
    return yt, yp


def bin_indices(y_prob: np.ndarray, n_bins: int = 15, strategy: str = "equal_mass") -> list[np.ndarray]:
    """Return a list of index arrays, one per bin.

    ``equal_mass``  -> quantile bins over the sorted probabilities (equal count).
    ``equal_width`` -> uniform bins on [0, 1].
    """
    yp = _as_float_array(y_prob)
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    if strategy == "equal_mass":
        order = np.argsort(yp, kind="stable")
        return [np.asarray(chunk, dtype=int) for chunk in np.array_split(order, n_bins)]
    if strategy == "equal_width":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        yp_clip = np.clip(yp, 0.0, 1.0)
        out = []
        for i in range(n_bins):
            lo, hi = edges[i], edges[i + 1]
            mask = (yp_clip >= lo) & (yp_clip < hi) if i < n_bins - 1 else (yp_clip >= lo) & (yp_clip <= hi)
            out.append(np.where(mask)[0])
        return [np.asarray(c, dtype=int) for c in out]
    raise ValueError(f"Unknown strategy: {strategy}")


def expected_calibration_error(
    y_true: Any,
    y_prob: Any,
    n_bins: int = 15,
    strategy: str = "equal_mass",
) -> float:
    """Expected Calibration Error (lower is better).

    ECE = sum_b (|B_b| / N) * |acc(B_b) - conf(B_b)|
    """
    yt, yp = _validate(y_true, y_prob)
    ece = 0.0
    for idx in bin_indices(yp, n_bins=n_bins, strategy=strategy):
        if idx.size == 0:
            continue
        ece += idx.size / yt.shape[0] * abs(yt[idx].mean() - yp[idx].mean())
    return float(ece)


def maximum_calibration_error(
    y_true: Any, y_prob: Any, n_bins: int = 15, strategy: str = "equal_mass"
) -> float:
    """MCE: the largest single-bin |acc - conf| gap."""
    yt, yp = _validate(y_true, y_prob)
    worst = 0.0
    for idx in bin_indices(yp, n_bins=n_bins, strategy=strategy):
        if idx.size == 0:
            continue
        worst = max(worst, abs(yt[idx].mean() - yp[idx].mean()))
    return float(worst)


def brier_score(y_true: Any, y_prob: Any) -> float:
    """Brier score = mean squared error of the positive-class probability."""
    yt, yp = _validate(y_true, y_prob)
    return float(np.mean((np.clip(yp, 0.0, 1.0) - yt) ** 2))


def brier_decomposition(
    y_true: Any, y_prob: Any, n_bins: int = 15, strategy: str = "equal_width"
) -> dict[str, float]:
    """Exact binned Brier decomposition.

    For bins ``b`` with mass ``m_b``, mean forecast ``f_b`` and observed
    frequency ``o_b`` (Murphy 1973; Kumar et al. 2019):

        Brier = reliability - resolution + uncertainty
                + within_bin_variance + within_bin_covariance

    * ``reliability          = sum_b m_b (o_b - f_b)^2``
    * ``resolution            = sum_b m_b (o_b - o_bar)^2``
    * ``uncertainty           = o_bar (1 - o_bar)``
    * ``within_bin_variance   = sum_b m_b Var(p | b)``
    * ``within_bin_covariance = -2 sum_b m_b Cov(y, p | b)``

    The classic three-term Murphy form drops the last two terms, so it equals the
    Brier score only when each bin holds a single forecast value.  All terms are
    reported here so ``brier == brier_score`` exactly; ``brier_murphy_3term``
    keeps the textbook value for comparability with published results.
    """
    yt, yp = _validate(y_true, y_prob)
    yp = np.clip(yp, 0.0, 1.0)
    n = yt.shape[0]
    base_rate = float(yt.mean())
    uncertainty = float(base_rate * (1.0 - base_rate))

    reliability = 0.0
    resolution = 0.0
    within_bin_variance = 0.0
    within_bin_covariance = 0.0
    for idx in bin_indices(yp, n_bins=n_bins, strategy=strategy):
        if idx.size == 0:
            continue
        m_b = idx.size / n
        conf_b = float(yp[idx].mean())
        acc_b = float(yt[idx].mean())
        reliability += m_b * (acc_b - conf_b) ** 2
        resolution += m_b * (acc_b - base_rate) ** 2
        within_bin_variance += m_b * float(yp[idx].var())
        centred = (yt[idx] - acc_b) * (yp[idx] - conf_b)
        within_bin_covariance += -2.0 * m_b * float(centred.mean())

    three_term = reliability - resolution + uncertainty
    return {
        "reliability": float(reliability),
        "resolution": float(resolution),
        "uncertainty": uncertainty,
        "within_bin_variance": float(within_bin_variance),
        "within_bin_covariance": float(within_bin_covariance),
        "brier_murphy_3term": float(three_term),
        "brier": float(three_term + within_bin_variance + within_bin_covariance),
        "brier_direct": brier_score(yt, yp),
    }


def negative_log_likelihood(y_true: Any, y_prob: Any, eps: float = _EPS) -> float:
    """Mean binary NLL / log loss."""
    yt, yp = _validate(y_true, y_prob)
    yp = np.clip(yp, eps, 1.0 - eps)
    return float(-np.mean(yt * np.log(yp) + (1.0 - yt) * np.log(1.0 - yp)))


def confidence_gap(y_true: Any, y_prob: Any) -> float:
    """Signed top-class confidence minus accuracy.

    Positive => overconfident; negative => underconfident.
    """
    yt, yp = _validate(y_true, y_prob)
    confidence = np.where(yt == 1, yp, 1.0 - yp)
    accuracy = (np.round(yp) == yt).mean()
    return float(confidence.mean() - accuracy)


def reliability_diagram(
    y_true: Any, y_prob: Any, n_bins: int = 15, strategy: str = "equal_mass"
) -> dict[str, np.ndarray]:
    """Per-bin (confidence, accuracy, count) for reliability plots."""
    yt, yp = _validate(y_true, y_prob)
    confs, accs, counts = [], [], []
    for idx in bin_indices(yp, n_bins=n_bins, strategy=strategy):
        if idx.size == 0:
            continue
        confs.append(float(yp[idx].mean()))
        accs.append(float(yt[idx].mean()))
        counts.append(int(idx.size))
    return {
        "confidence": np.asarray(confs, dtype=np.float64),
        "accuracy": np.asarray(accs, dtype=np.float64),
        "count": np.asarray(counts, dtype=np.int64),
    }


def compute_all_calibration_metrics(
    y_true: Any, y_prob: Any, n_bins: int = 15, strategy: str = "equal_width"
) -> dict[str, float]:
    """All calibration metrics in one call (used by evaluation phases).

    Defaults to ``equal_width`` so the reported ECE/MCE use the same bins as the
    Brier decomposition and stay internally consistent.
    """
    decomp = brier_decomposition(y_true, y_prob, n_bins=n_bins, strategy=strategy)
    return {
        "ece": expected_calibration_error(y_true, y_prob, n_bins=n_bins, strategy=strategy),
        "mce": maximum_calibration_error(y_true, y_prob, n_bins=n_bins, strategy=strategy),
        "brier": decomp["brier"],
        "brier_reliability": decomp["reliability"],
        "brier_resolution": decomp["resolution"],
        "brier_uncertainty": decomp["uncertainty"],
        "brier_within_bin_variance": decomp["within_bin_variance"],
        "brier_within_bin_covariance": decomp["within_bin_covariance"],
        "brier_murphy_3term": decomp["brier_murphy_3term"],
        "nll": negative_log_likelihood(y_true, y_prob),
        "confidence_gap": confidence_gap(y_true, y_prob),
    }