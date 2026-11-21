"""Validation of model output logits frames (Phase 4 / Phase 6 quality gates)."""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

REQUIRED_LOGIT_COLUMNS: tuple[str, ...] = (
    "example_id",
    "y_true",
    "logit_hate",
    "logit_nonhate",
    "prob_hate",
    "prob_nonhate",
    "prediction",
)


def validate_logits_frame(
    frame: pd.DataFrame,
    *,
    min_macro_f1: float = 0.50,
    require_both_classes: bool = False,
) -> dict[str, Any]:
    """Phase 4 plan 4.3 / T4.2, T4.3, T4.5, T4.6, T6.6, T6.7 quality gate.

    ``single_class`` marks a set whose ground truth has one class only.
    The bd_shs LOCO axis makes every ``ood_test`` single-class, so macro-F1 and the
    both-classes-predicted gate are undefined there and are reported as ``null``
    instead of failing the run. Structural checks always apply.
    """
    missing = [c for c in REQUIRED_LOGIT_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"logits frame is missing columns {missing}")
    if frame.empty:
        raise ValueError("logits frame is empty")

    y_true = frame["y_true"].to_numpy(dtype=np.int64)
    y_pred = frame["prediction"].to_numpy(dtype=np.int64)
    prob_sum = frame["prob_hate"].to_numpy(dtype=np.float64) + frame[
        "prob_nonhate"
    ].to_numpy(dtype=np.float64)
    single_class = bool(np.unique(y_true).size < 2)

    macro_f1 = (
        float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        if not single_class
        else None
    )
    n_pred_classes = int(np.unique(y_pred).size)

    checks: dict[str, bool | None] = {
        "has_all_columns": True,
        "probs_sum_to_one": bool(np.allclose(prob_sum, 1.0, atol=1e-5)),
        "no_nan_logits": not bool(
            frame[["logit_hate", "logit_nonhate"]].isna().any().any()
            or not np.isfinite(
                frame[["logit_hate", "logit_nonhate", "prob_hate", "prob_nonhate"]]
                .to_numpy(dtype=np.float64)
            ).all()
        ),
        "f1_above_random": None if single_class else bool(macro_f1 > min_macro_f1),
        "covers_both_classes": None
        if single_class or not require_both_classes
        else bool(n_pred_classes == 2),
    }

    return {
        "n_rows": int(len(frame)),
        "n_true_classes": int(np.unique(y_true).size),
        "n_pred_classes": n_pred_classes,
        "single_class": single_class,
        "macro_f1": macro_f1,
        "accuracy": float((y_pred == y_true).mean()),
        "max_prob_sum_deviation": float(np.max(np.abs(prob_sum - 1.0))),
        "checks": checks,
        "passed": all(v for v in checks.values() if v is not None),
        "note": (
            "ground truth is single-class; macro-F1 and the both-classes-predicted gate "
            "are undefined and are reported as null"
            if single_class
            else ""
        ),
    }
