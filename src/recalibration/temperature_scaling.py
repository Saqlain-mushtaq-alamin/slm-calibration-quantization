"""Stable binary temperature scaling on the saved two-label logits."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, logsumexp


def _inputs(logits: np.ndarray, y_true: np.ndarray | None = None):
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("logits must be a nonempty, finite (N, 2) array")
    if y_true is None:
        return values
    labels = np.asarray(y_true)
    if labels.shape != (len(values),) or not np.isin(labels, [0, 1]).all():
        raise ValueError("labels must be a binary (N,) array matching logits")
    return values, labels.astype(np.int64)


def fit_temperature(logits: np.ndarray, y_true: np.ndarray, *, bounds=(0.01, 10.0)) -> float:
    """Minimize held-out NLL over one positive scalar temperature.

    The search uses log-temperature and an unclipped log-sum-exp loss. This
    remains finite for very large raw logits and avoids saturated gradients.
    """
    values, labels = _inputs(logits, y_true)
    low, high = map(float, bounds)
    if not np.isfinite([low, high]).all() or low <= 0 or high <= low:
        raise ValueError("temperature bounds must satisfy 0 < low < high")
    if np.all(values[:, 1] == values[:, 0]):
        return 1.0

    def loss(log_t: float) -> float:
        scaled = values / np.exp(log_t)
        return float(np.mean(logsumexp(scaled, axis=1) - scaled[np.arange(len(labels)), labels]))

    lo, hi = np.log(low), np.log(high)
    result = minimize_scalar(loss, bounds=(lo, hi), method="bounded", options={"xatol": 1e-10})
    if not result.success:
        raise RuntimeError(f"temperature optimization failed: {result.message}")
    candidates = [lo, float(result.x), hi]
    if lo <= 0 <= hi:
        candidates.append(0.0)
    return float(np.exp(min(candidates, key=loss)))


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Return [P(non-hate), P(hate)] without changing the decision boundary."""
    values = _inputs(logits)
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    p_hate = expit((values[:, 1] - values[:, 0]) / temperature)
    return np.column_stack((1.0 - p_hate, p_hate))
