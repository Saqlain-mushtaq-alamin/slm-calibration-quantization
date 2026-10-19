"""Task, calibration, selective-prediction, and efficiency metrics."""

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
from src.metrics.selective import (
    aurc,
    compute_all_selective_metrics,
    coverage_at_risk,
    e_aurc,
    failure_prediction_auroc,
    optimal_aurc,
    risk_coverage_curve,
    selective_accuracy_at_coverage,
)
from src.metrics.task_metrics import (
    agreement_rate,
    bootstrap_accuracy_delta,
    compute_all_task_metrics,
)

__all__ = [
    "agreement_rate",
    "aurc",
    "bootstrap_accuracy_delta",
    "brier_decomposition",
    "brier_score",
    "compute_all_calibration_metrics",
    "compute_all_selective_metrics",
    "compute_all_task_metrics",
    "confidence_gap",
    "coverage_at_risk",
    "e_aurc",
    "expected_calibration_error",
    "failure_prediction_auroc",
    "maximum_calibration_error",
    "negative_log_likelihood",
    "optimal_aurc",
    "reliability_diagram",
    "risk_coverage_curve",
    "selective_accuracy_at_coverage",
]