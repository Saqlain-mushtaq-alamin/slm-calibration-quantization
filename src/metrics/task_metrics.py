"""Task-performance metrics for binary hate-speech classification."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

_NEG_INF = float("nan")


def _safe(fn, *args, **kwargs) -> float:
    """Run a sklearn metric, mapping degenerate inputs (single class) to NaN."""
    try:
        return float(fn(*args, **kwargs))
    except ValueError:
        return _NEG_INF


def compute_all_task_metrics(
    y_true: Any, y_pred: Any, y_prob: Any
) -> dict[str, float]:
    """Compute all task-performance metrics for a binary task."""
    yt = np.asarray(y_true).reshape(-1)
    yp = np.asarray(y_pred).reshape(-1)
    yprob = np.asarray(y_prob, dtype=np.float64).reshape(-1)
    if not (yt.shape[0] == yp.shape[0] == yprob.shape[0]):
        raise ValueError("y_true, y_pred and y_prob must have equal length")
    if yt.shape[0] == 0:
        raise ValueError("empty input")

    single_class = np.unique(yt).size < 2
    return {
        "n": int(yt.shape[0]),
        "accuracy": _safe(accuracy_score, yt, yp),
        "balanced_accuracy": _safe(
            lambda a, b: float(
                0.5 * ((np.asarray(b) == 1)[np.asarray(a) == 0].mean())
                + ((np.asarray(b) == 0)[np.asarray(a) == 1].mean())
            )
            if np.unique(yt).size == 2
            else float("nan"),
            yt,
            yp,
        ),
        "macro_f1": _safe(f1_score, yt, yp, average="macro", zero_division=0),
        "weighted_f1": _safe(f1_score, yt, yp, average="weighted", zero_division=0),
        "hate_f1": _safe(f1_score, yt, yp, pos_label=1, zero_division=0),
        "nonhate_f1": _safe(f1_score, yt, yp, pos_label=0, zero_division=0),
        "macro_precision": _safe(precision_score, yt, yp, average="macro", zero_division=0),
        "macro_recall": _safe(recall_score, yt, yp, average="macro", zero_division=0),
        "hate_precision": _safe(precision_score, yt, yp, pos_label=1, zero_division=0),
        "hate_recall": _safe(recall_score, yt, yp, pos_label=1, zero_division=0),
        "auroc": _NEG_INF if single_class else _safe(roc_auc_score, yt, yprob),
        "average_precision": _NEG_INF
        if single_class
        else _safe(average_precision_score, yt, yprob),
        "mcc": _safe(matthews_corrcoef, yt, yp),
        "positive_rate": float(yt.mean()),
        "pred_positive_rate": float(np.mean(np.asarray(yp) == 1)),
    }


def agreement_rate(pred_a: Any, pred_b: Any) -> float:
    """Fraction of examples where two prediction vectors agree."""
    a = np.asarray(pred_a).reshape(-1)
    b = np.asarray(pred_b).reshape(-1)
    if a.shape[0] != b.shape[0]:
        raise ValueError("length mismatch")
    return float((a == b).mean())


def bootstrap_accuracy_delta(
    y_true: Any, pred_a: Any, pred_b: Any, n_boot: int = 1000, seed: int = 42
) -> dict[str, float]:
    """Paired bootstrap of ``acc(a) - acc(b)`` (used for quantization deltas)."""
    yt = np.asarray(y_true).reshape(-1)
    a = (np.asarray(pred_a).reshape(-1) == yt).astype(np.float64)
    b = (np.asarray(pred_b).reshape(-1) == yt).astype(np.float64)
    n = yt.shape[0]
    if n == 0:
        raise ValueError("empty input")
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        deltas[i] = a[idx].mean() - b[idx].mean()
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {
        "delta": float(a.mean() - b.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_delta_le_0": float((deltas <= 0).mean()),
        "n_boot": int(n_boot),
    }