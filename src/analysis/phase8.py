"""Phase 8 evaluation from saved, canonical per-example logits."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from src.metrics.calibration import (
    brier_decomposition,
    confidence_gap,
    expected_calibration_error,
    negative_log_likelihood,
)
from src.metrics.task_metrics import compute_all_task_metrics

EVAL_SETS = {"id_val", "id_test", "ood_test"}
PHASES = {"phase4", "phase5", "phase6", "phase7", "pilot"}
UNDEFINED_ON_SINGLE_CLASS = {"balanced_accuracy", "auroc", "average_precision", "mcc"}


def discover_logits(root: Path, *, include_smoke: bool = False) -> list[Path]:
    phases = PHASES | ({"phase5_smoke", "phase7_smoke"} if include_smoke else set())
    return sorted(p for p in root.rglob("*.parquet") if p.relative_to(root).parts[0] in phases)


def _constant(frame: pd.DataFrame, column: str, path: Path):
    if column not in frame or frame[column].nunique(dropna=False) != 1:
        raise ValueError(f"{path}: missing or nonconstant {column}")
    return frame[column].iloc[0]


def _sample_fingerprint(frame: pd.DataFrame) -> str:
    pairs = frame[["example_id", "y_true"]].copy()
    pairs["example_id"] = pairs["example_id"].astype(str)
    pairs = pairs.sort_values("example_id", kind="stable")
    digest = hashlib.sha256()
    for example_id, label in pairs.itertuples(index=False, name=None):
        encoded = example_id.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(bytes([int(label)]))
    return digest.hexdigest()


def evaluate_file(path: Path, root: Path) -> dict:
    frame = pd.read_parquet(path)
    required = {"example_id", "split", "y_true", "prob_hate", "prob_nonhate", "prediction", "precision", "seed", "language"}
    missing = required - set(frame.columns)
    if missing or frame.empty:
        raise ValueError(f"{path}: missing {sorted(missing)} or empty logits")
    phase = path.relative_to(root).parts[0]
    method, separator, eval_set = path.stem.partition("__")
    if not separator or eval_set not in EVAL_SETS or method != _constant(frame, "precision", path):
        raise ValueError(f"{path}: filename does not match precision/eval set")
    if _constant(frame, "split", path) != eval_set:
        raise ValueError(f"{path}: split column does not match filename")
    artifact = path.parent.name if phase != "pilot" else f"pilot__{method}"
    base_run = artifact.rsplit("__", 1)[0] if phase in {"phase6", "phase7", "phase7_smoke"} else (
        "pilot" if phase == "pilot" else artifact
    )
    for column in ("seed", "language", "model_key" if "model_key" in frame else "model"):
        _constant(frame, column, path)
    if frame["example_id"].isna().any() or frame["example_id"].duplicated().any():
        raise ValueError(f"{path}: null or duplicate example_id")
    y = frame["y_true"].to_numpy(dtype=np.int64)
    pred = frame["prediction"].to_numpy(dtype=np.int64)
    p = frame["prob_hate"].to_numpy(dtype=np.float64)
    other = frame["prob_nonhate"].to_numpy(dtype=np.float64)
    if not np.isin(y, [0, 1]).all() or not np.isin(pred, [0, 1]).all():
        raise ValueError(f"{path}: labels/predictions must be binary")
    if not np.isfinite(p).all() or not np.isfinite(other).all() or (p < 0).any() or (p > 1).any() or not np.allclose(p + other, 1, atol=1e-5):
        raise ValueError(f"{path}: invalid probabilities")
    if ("logit_hate" in frame and "logit_nonhate" in frame
            and not np.isfinite(frame[["logit_hate", "logit_nonhate"]].to_numpy(dtype=np.float64)).all()):
        raise ValueError(f"{path}: nonfinite logits")
    task = compute_all_task_metrics(y, pred, p)
    single_class = bool(np.unique(y).size == 1)
    if single_class:
        task = {key: (None if key in UNDEFINED_ON_SINGLE_CLASS else value) for key, value in task.items()}
    decomp = brier_decomposition(y, p, n_bins=15, strategy="equal_width")
    metrics = {
        **task,
        "ece_equal_mass_15": expected_calibration_error(y, p, 15, "equal_mass"),
        "ece_equal_width_15": expected_calibration_error(y, p, 15, "equal_width"),
        "ece_equal_mass_10": expected_calibration_error(y, p, 10, "equal_mass"),
        "nll": negative_log_likelihood(y, p),
        "confidence_gap": confidence_gap(y, p),
        **decomp,
    }
    for key, value in metrics.items():
        if value is not None and isinstance(value, (float, np.floating)) and not np.isfinite(value):
            raise ValueError(f"{path}: nonfinite metric {key}")
    return {
        "artifact": artifact, "base_run": base_run, "phase": phase,
        "method": method, "model": str(frame["model_key"].iloc[0]) if "model_key" in frame else str(_constant(frame, "model", path)),
        "seed": int(frame["seed"].iloc[0]), "lang": str(frame["language"].iloc[0]),
        "loco": str(_constant(frame, "split_name", path)) if "split_name" in frame else str(_constant(frame, "loco", path)),
        "eval_set": eval_set, "n_samples": len(frame),
        "sample_fingerprint": _sample_fingerprint(frame),
        "logit_file": path.relative_to(root).as_posix(),
        "single_class_eval_set": single_class,
        **metrics,
    }


def evaluate_all_artifacts(root: Path) -> pd.DataFrame:
    files = discover_logits(root)
    if not files:
        raise ValueError(f"No full-run logits found under {root}")
    result = pd.DataFrame([evaluate_file(path, root) for path in files])
    if result.duplicated(["artifact", "eval_set"]).any():
        raise ValueError("Duplicate artifact/eval-set rows")
    return result


def compute_deltas_vs_fp(results: pd.DataFrame) -> pd.DataFrame:
    keys = ["base_run", "eval_set"]
    references = results[results["method"] == "bf16"]
    if references.duplicated(keys).any():
        raise ValueError("Ambiguous BF16 references")
    variants = results[results["method"] != "bf16"]
    joined = variants.merge(references, on=keys, how="left", suffixes=("", "_fp"), validate="many_to_one", indicator=True)
    if (joined["_merge"] != "both").any():
        raise ValueError(f"Missing BF16 reference: {joined.loc[joined['_merge'] != 'both', ['artifact', 'eval_set']].to_dict('records')[:5]}")
    for key in ("model", "seed", "lang", "loco", "n_samples", "sample_fingerprint"):
        if (joined[key] != joined[f"{key}_fp"]).any():
            bad = joined.loc[joined[key] != joined[f"{key}_fp"], "artifact"].iloc[0]
            raise ValueError(f"BF16 reference mismatch for {bad}: {key}")
    out = joined[["artifact", "base_run", "phase", "method", "model", "seed", "lang", "loco", "eval_set", "n_samples", "artifact_fp"]].rename(columns={"artifact_fp": "fp_reference"}).copy()
    for metric, delta_name in (("macro_f1", "delta_f1"), ("ece_equal_mass_15", "delta_ece"), ("brier", "delta_brier"), ("nll", "delta_nll")):
        out[delta_name] = joined[metric] - joined[f"{metric}_fp"]
    out["relative_f1_change"] = out["delta_f1"] / joined["macro_f1_fp"].replace(0, np.nan)
    out["relative_ece_change"] = out["delta_ece"] / joined["ece_equal_mass_15_fp"].clip(lower=1e-6)
    if out.select_dtypes(include="number").isna().any().any():
        raise ValueError("Undefined relative delta: zero BF16 F1")
    return out
