"""Phase 9 selective-prediction sweep over full-run saved logits."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.phase8 import discover_logits, evaluate_file
from src.metrics.selective import (
    compute_all_selective_metrics,
    risk_coverage_curve,
    selective_accuracy_at_coverage,
)


def evaluate_selective_file(path: Path, logits_root: Path) -> tuple[dict, pd.DataFrame]:
    """Validate one logits file, compute metrics, and retain its full curve."""
    context = evaluate_file(path, logits_root)
    frame = pd.read_parquet(path, columns=["y_true", "prob_hate"])
    y_true = frame["y_true"].to_numpy(dtype=np.int64)
    y_prob = frame["prob_hate"].to_numpy(dtype=np.float64)
    coverage, risk = risk_coverage_curve(y_true, y_prob)
    metrics = compute_all_selective_metrics(y_true, y_prob)
    acc = {
        f"selective_acc_{int(cov * 100)}": selective_accuracy_at_coverage(y_true, y_prob, cov)
        for cov in (0.70, 0.80, 0.90)
    }
    metrics.update(acc)
    metrics["failure_auroc"] = metrics["failure_prediction_auroc"]
    metrics["selective_accuracy_monotonic_70_90"] = bool(
        acc["selective_acc_70"] >= acc["selective_acc_80"] >= acc["selective_acc_90"]
    )
    n_errors = int(round(float(risk[-1]) * len(risk)))
    metrics["failure_auroc_defined"] = bool(0 < n_errors < len(risk))
    if not metrics["failure_auroc_defined"]:
        metrics["failure_auroc"] = None
        metrics["failure_prediction_auroc"] = None
    for name, value in metrics.items():
        if value is None or isinstance(value, bool):
            continue
        if not np.isfinite(value):
            raise ValueError(f"{path}: nonfinite selective metric {name}")
    if not 0 <= metrics["aurc"] <= 1 or metrics["e_aurc"] < -1e-12:
        raise ValueError(f"{path}: invalid AURC/E-AURC")
    if not 0 <= metrics["coverage_at_5pct_risk"] <= 1:
        raise ValueError(f"{path}: invalid coverage at 5% risk")
    if not np.isclose(metrics["aurc"], risk.mean(), atol=1e-12):
        raise ValueError(f"{path}: AURC disagrees with saved curve")
    if not np.isclose(acc["selective_acc_70"], 1 - risk[round(.70 * len(risk)) - 1], atol=1e-12):
        raise ValueError(f"{path}: selective accuracy disagrees with curve")
    curve = pd.DataFrame({"coverage": coverage, "risk": risk})
    curve_file = path.relative_to(logits_root).with_suffix("").as_posix() + "_rc.parquet"
    row = {
        key: context[key] for key in (
            "artifact", "base_run", "phase", "method", "model", "seed", "lang",
            "loco", "eval_set", "n_samples", "sample_fingerprint", "logit_file",
            "single_class_eval_set",
        )
    }
    row["curve_file"] = curve_file
    row.update(metrics)
    return row, curve


def evaluate_selective_prediction(logits_root: Path, curves_root: Path) -> pd.DataFrame:
    files = discover_logits(logits_root)
    if not files:
        raise ValueError(f"No full-run logits found under {logits_root}")
    rows = []
    for path in files:
        row, curve = evaluate_selective_file(path, logits_root)
        curve_path = curves_root / row["curve_file"]
        curve_path.parent.mkdir(parents=True, exist_ok=True)
        curve.to_parquet(curve_path, index=False)
        rows.append(row)
    result = pd.DataFrame(rows)
    if result.duplicated(["phase", "artifact", "eval_set"]).any():
        raise ValueError("Duplicate artifact/eval-set rows")
    if not all((curves_root / name).is_file() for name in result["curve_file"]):
        raise ValueError("Missing risk-coverage curve")
    return result
