"""One-call aggregation of every metric from a canonical logits frame.

Phase 3 uses this to prove T3.9 ("all metrics computable on pilot logits");
Phase 8 reuses it for the full evaluation sweep, so the metric definition is
written exactly once.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.metrics.calibration import compute_all_calibration_metrics
from src.metrics.selective import compute_all_selective_metrics
from src.metrics.task_metrics import agreement_rate, compute_all_task_metrics


def _probability_column(df: pd.DataFrame) -> np.ndarray:
    """P(hate) from whichever probability column the frame carries."""
    for col in ("prob_hate", "y_prob", "prob_positive", "probability"):
        if col in df.columns:
            return df[col].to_numpy(dtype=np.float64)
    raise ValueError(f"No positive-class probability column in frame: {list(df.columns)}")


def evaluate_logits_frame(
    df: pd.DataFrame,
    *,
    ece_bins: int = 15,
    ece_strategy: str = "equal_width",
    aurc_method: str = "riemann",
) -> dict[str, Any]:
    """Compute task + calibration + selective metrics from a logits frame.

    ``prediction`` is used when present (so quantization variants are scored on
    their own decisions); otherwise it is thresholded at ``p = 0.5``.
    """
    y_true = df["y_true"].to_numpy(dtype=np.int64)
    y_prob = _probability_column(df)
    if "prediction" in df.columns:
        y_pred = df["prediction"].to_numpy(dtype=np.int64)
    else:
        y_pred = (y_prob >= 0.5).astype(np.int64)

    out: dict[str, Any] = {}
    out.update(compute_all_task_metrics(y_true, y_pred, y_prob))
    out.update(compute_all_calibration_metrics(y_true, y_prob, n_bins=ece_bins, strategy=ece_strategy))
    sel = compute_all_selective_metrics(y_true, y_prob)
    if aurc_method != "riemann":
        from src.metrics.selective import aurc as _aurc
        from src.metrics.selective import e_aurc as _e_aurc

        sel["aurc"] = _aurc(y_true, y_prob, aurc_method)
        sel["e_aurc"] = _e_aurc(y_true, y_prob, aurc_method)
    out.update(sel)
    # bd_shs derives `domain` from the hate subtype, so every LOCO target domain is
    # single-class (slander/callToViolence/gender/religion are all hate; non_hate is
    # all non-hate).  Rank metrics such as AUROC are undefined there; flag it rather
    # than letting -inf leak into the results table.
    out["single_class_eval_set"] = bool(np.unique(y_true).size < 2)
    if "split" in df.columns:
        out["splits"] = sorted(set(df["split"].astype(str)))
    return out


def compare_precision_frames(
    reference: pd.DataFrame, variant: pd.DataFrame
) -> dict[str, Any]:
    """Accuracy / agreement deltas of a quantized variant against a reference."""
    from src.metrics.task_metrics import bootstrap_accuracy_delta

    merged = reference.merge(variant, on="example_id", suffixes=("_ref", "_var"), how="inner")
    if len(merged) == 0:
        raise ValueError("No overlapping example_id between reference and variant logits")
    if "y_true_ref" in merged and "y_true_var" in merged:
        mismatch = int((merged["y_true_ref"] != merged["y_true_var"]).sum())
    else:
        mismatch = 0

    y_true = merged["y_true_ref"].to_numpy(dtype=np.int64)
    out = {
        "n_aligned": int(len(merged)),
        "label_mismatch": mismatch,
        "acc_reference": float((merged["prediction_ref"] == y_true).mean()),
        "acc_variant": float((merged["prediction_var"] == y_true).mean()),
        "agreement_rate": agreement_rate(merged["prediction_ref"], merged["prediction_var"]),
        "mean_abs_prob_hate_delta": float(
            np.mean(np.abs(merged["prob_hate_ref"] - merged["prob_hate_var"]))
        ),
        "max_abs_prob_hate_delta": float(
            np.max(np.abs(merged["prob_hate_ref"] - merged["prob_hate_var"]))
        ),
    }
    out["accuracy_drop"] = out["acc_reference"] - out["acc_variant"]
    out["bootstrap"] = bootstrap_accuracy_delta(
        y_true, merged["prediction_ref"], merged["prediction_var"], n_boot=1000, seed=42
    )
    return out


def metrics_to_record(metrics: dict[str, Any], **context: Any) -> dict[str, Any]:
    """Flatten a metrics dict (plus context) into a JSON-safe record."""
    record: dict[str, Any] = {**context}
    for key, value in metrics.items():
        if isinstance(value, (str, bool, int)):
            record[key] = value
        elif isinstance(value, (list, tuple)):
            record[key] = list(value)
        elif isinstance(value, float) and not np.isfinite(value):
            record[key] = None
        else:
            record[key] = value
    return record


def metrics_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Assemble per-(model, precision, split) metric rows into one table."""
    flat = [metrics_to_record(r) for r in records]
    if not flat:
        return pd.DataFrame()
    frame = pd.DataFrame(flat)
    leading = [c for c in ("model", "precision", "split", "language", "loco", "seed") if c in frame.columns]
    rest = [c for c in frame.columns if c not in leading]
    return frame[leading + sorted(rest)]