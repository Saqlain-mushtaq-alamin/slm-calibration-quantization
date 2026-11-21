"""Automated sanity checks used across phases."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""
    extras: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "detail": self.detail,
            **self.extras,
        }


def check_gpu_min_vram(min_vram_gb: float = 7.5) -> CheckResult:
    """Verify CUDA GPU is present with at least min_vram_gb VRAM."""
    try:
        import torch
    except ImportError:
        return CheckResult("gpu_vram", False, "torch not installed")
    if not torch.cuda.is_available():
        return CheckResult("gpu_vram", False, "CUDA not available")
    props = torch.cuda.get_device_properties(0)
    total_bytes = getattr(props, "total_memory", None) or props.total_mem
    vram_gb = round(total_bytes / 1e9, 2)
    name = torch.cuda.get_device_name(0)
    passed = vram_gb >= min_vram_gb
    return CheckResult(
        "gpu_vram",
        passed,
        f"{name}: {vram_gb} GB (need >= {min_vram_gb} GB)",
        {"gpu_name": name, "vram_gb": vram_gb},
    )


def check_required_imports(modules: list[str]) -> CheckResult:
    """Verify a list of import names are importable."""
    missing = []
    versions: dict[str, str] = {}
    import importlib

    for mod in modules:
        try:
            m = importlib.import_module(mod)
            versions[mod] = getattr(m, "__version__", "unknown")
        except ImportError:
            missing.append(mod)
    passed = not missing
    detail = "all present" if passed else f"missing: {missing}"
    return CheckResult("required_imports", passed, detail, {"versions": versions})


def sanity_check_int8(
    fp_logits: pd.DataFrame,
    int8_logits: pd.DataFrame,
    tolerance: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Phase 6 critical gate (T6.5): INT8 MUST be near-lossless vs BF16 FP reference.

    Joins fp and int8 logits on example_id, asserts the example_id sets are
    identical, and computes all metrics on the aligned rows.

    Pass condition:
    - F1 drop <= max_f1_drop (0.02)
    - Agreement >= min_agreement (0.95)
    - Mean abs(delta_p) <= max_mean_delta_p (0.05)
    """
    if tolerance is None:
        tolerance = {"max_f1_drop": 0.02, "min_agreement": 0.95, "max_mean_delta_p": 0.05}

    if "example_id" not in fp_logits.columns or "example_id" not in int8_logits.columns:
        raise ValueError("Both fp_logits and int8_logits must contain 'example_id' column.")

    fp_ids = set(fp_logits["example_id"].astype(str))
    int8_ids = set(int8_logits["example_id"].astype(str))
    if fp_ids != int8_ids:
        diff = fp_ids.symmetric_difference(int8_ids)
        raise ValueError(
            f"example_id sets are not identical: {len(diff)} mismatched IDs (e.g. {list(diff)[:5]})"
        )

    # Align on example_id
    aligned = pd.merge(
        fp_logits,
        int8_logits,
        on="example_id",
        suffixes=("_fp", "_int8"),
    )

    y_true_col = "y_true_fp" if "y_true_fp" in aligned.columns else "y_true"
    y_true = aligned[y_true_col].to_numpy(dtype=np.int64)

    fp_pred = aligned["prediction_fp"].to_numpy(dtype=np.int64)
    int8_pred = aligned["prediction_int8"].to_numpy(dtype=np.int64)

    prob_fp = aligned["prob_hate_fp"].to_numpy(dtype=np.float64)
    prob_int8 = aligned["prob_hate_int8"].to_numpy(dtype=np.float64)

    # Check for single-class eval set (e.g. bd_shs ood_test)
    single_class = bool(np.unique(y_true).size < 2)

    fp_f1 = float(f1_score(y_true, fp_pred, average="macro", zero_division=0)) if not single_class else None
    int8_f1 = float(f1_score(y_true, int8_pred, average="macro", zero_division=0)) if not single_class else None

    agreement = float((fp_pred == int8_pred).mean())
    mean_delta_p = float(np.abs(prob_fp - prob_int8).mean())
    max_delta_p = float(np.abs(prob_fp - prob_int8).max())

    f1_drop = (fp_f1 - int8_f1) if (fp_f1 is not None and int8_f1 is not None) else None

    checks = {
        "f1_drop_ok": None if single_class else bool(f1_drop <= tolerance["max_f1_drop"]),
        "agreement_ok": bool(agreement >= tolerance["min_agreement"]),
        "delta_p_ok": bool(mean_delta_p <= tolerance["max_mean_delta_p"]),
    }

    passed = all(v for v in checks.values() if v is not None)
    return {
        "passed": passed,
        "single_class": single_class,
        "fp_macro_f1": fp_f1,
        "int8_macro_f1": int8_f1,
        "f1_drop": f1_drop,
        "agreement": agreement,
        "mean_delta_p": mean_delta_p,
        "max_delta_p": max_delta_p,
        "tolerance": tolerance,
        "checks": checks,
    }
