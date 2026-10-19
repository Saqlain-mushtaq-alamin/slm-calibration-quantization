"""Selective-prediction metrics: risk-coverage, AURC, E-AURC, coverage@risk.

Confidence is the top-class probability ``max(p, 1-p)``; examples are ranked by
descending confidence and the risk-coverage curve reports the error rate over the
accepted prefix.
"""

from __future__ import annotations

from typing import Any

import numpy as np

_EPS = 1e-15


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    """Trapezoidal integration (numpy renamed trapz -> trapezoid in 1.26/2.x)."""
    fn = getattr(np, "trapezoid", None) or np.trapz
    return float(fn(y, x))


def _prepare(y_true: Any, y_prob: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yp = np.asarray(y_prob, dtype=np.float64).reshape(-1)
    if yt.shape[0] != yp.shape[0]:
        raise ValueError(f"length mismatch: y_true={yt.shape[0]} y_prob={yp.shape[0]}")
    if yt.shape[0] == 0:
        raise ValueError("empty input")
    yp = np.clip(yp, 0.0, 1.0)
    confidence = np.maximum(yp, 1.0 - yp)
    prediction = (yp >= 0.5).astype(np.int64)
    errors = (prediction != yt).astype(np.float64)
    return yt, yp, confidence, errors


def _integrate(risks: np.ndarray, coverages: np.ndarray, method: str) -> float:
    """Area under the discrete risk-coverage curve.

    ``riemann`` (default) is the standard definition used in the selective
    prediction literature: the mean of the per-coverage risks.  ``trapezoid``
    is the variant sketched in the project plan.
    """
    if method == "riemann":
        return float(risks.mean())
    if method == "trapezoid":
        return _trapz(risks, coverages)
    raise ValueError(f"Unknown method: {method}")


def risk_coverage_curve(y_true: Any, y_prob: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(coverages, risks)`` with one point per accepted example."""
    _, _, confidence, errors = _prepare(y_true, y_prob)
    n = confidence.shape[0]
    order = np.argsort(-confidence, kind="stable")
    sorted_errors = errors[order]
    coverages = np.arange(1, n + 1) / n
    risks = np.cumsum(sorted_errors) / np.arange(1, n + 1)
    return coverages, risks


def aurc(y_true: Any, y_prob: Any, method: str = "riemann") -> float:
    """Area Under the Risk-Coverage curve (lower is better)."""
    coverages, risks = risk_coverage_curve(y_true, y_prob)
    return _integrate(risks, coverages, method)


def optimal_aurc(n: int, n_errors: int, method: str = "riemann") -> float:
    """AURC of a perfect ranker: every error sits at the *lowest* confidence.

    With all ``n_errors`` errors deferred to the final ranks, the cumulative
    error count at rank ``k`` is ``max(0, k - (n - n_errors))``, so
    ``risk(k) = max(0, k - (n - n_errors)) / k``.
    """
    n = int(n)
    n_errors = int(n_errors)
    if n <= 0:
        raise ValueError("n must be positive")
    n_errors = int(np.clip(n_errors, 0, n))
    ks = np.arange(1, n + 1)
    risks = np.maximum(0.0, ks - (n - n_errors)) / ks
    coverages = ks / n
    return _integrate(risks, coverages, method)


def e_aurc(y_true: Any, y_prob: Any, method: str = "riemann") -> float:
    """Excess AURC = AURC(model) - AURC(optimal ranker).

    Isolates *ranking* quality from raw accuracy (Geifman & El-Yaniv, 2017).
    """
    _, _, _, errors = _prepare(y_true, y_prob)
    n = errors.shape[0]
    return float(aurc(y_true, y_prob, method) - optimal_aurc(n, int(errors.sum()), method))


def selective_accuracy_at_coverage(y_true: Any, y_prob: Any, coverage: float) -> float:
    """Accuracy on the most-confident ``coverage`` fraction of examples."""
    if not 0.0 < coverage <= 1.0:
        raise ValueError(f"coverage must be in (0, 1], got {coverage}")
    yt, yp, confidence, _ = _prepare(y_true, y_prob)
    n = yt.shape[0]
    n_select = int(round(n * coverage))
    if n_select == 0:
        return float("nan")
    top = np.argsort(-confidence, kind="stable")[:n_select]
    return float(((yp[top] >= 0.5).astype(np.int64) == yt[top]).mean())


def coverage_at_risk(y_true: Any, y_prob: Any, max_risk: float = 0.05) -> float:
    """Largest coverage whose running risk stays <= ``max_risk``."""
    coverages, risks = risk_coverage_curve(y_true, y_prob)
    ok = risks <= max_risk
    if not ok.any():
        return 0.0
    return float(np.max(coverages[ok]))


def failure_prediction_auroc(y_true: Any, y_prob: Any) -> float:
    """AUROC of using ``1 - confidence`` to predict an error (higher is better)."""
    from sklearn.metrics import roc_auc_score

    yt, _, confidence, errors = _prepare(y_true, y_prob)
    n = yt.shape[0]
    if errors.sum() == 0 or errors.sum() == n:
        return float("nan")
    return float(roc_auc_score(errors.astype(int), 1.0 - confidence))


def compute_all_selective_metrics(y_true: Any, y_prob: Any) -> dict[str, float]:
    """All selective-prediction metrics in one call."""
    out: dict[str, float] = {
        "aurc": aurc(y_true, y_prob),
        "e_aurc": e_aurc(y_true, y_prob),
        "coverage_at_10pct_risk": coverage_at_risk(y_true, y_prob, 0.10),
        "coverage_at_5pct_risk": coverage_at_risk(y_true, y_prob, 0.05),
        "coverage_at_2pct_risk": coverage_at_risk(y_true, y_prob, 0.02),
        "failure_prediction_auroc": failure_prediction_auroc(y_true, y_prob),
    }
    for cov in (0.1, 0.3, 0.5, 0.7, 0.9, 1.0):
        out[f"selective_accuracy_at_{int(cov * 100)}pct"] = selective_accuracy_at_coverage(
            y_true, y_prob, cov
        )
    return out