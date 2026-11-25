"""Phase 11 recalibration sweep over saved full-run logits."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.phase8 import discover_logits, evaluate_file
from src.metrics.calibration import brier_score, expected_calibration_error, negative_log_likelihood
from src.recalibration.temperature_scaling import apply_temperature, fit_temperature


def _read(path: Path, root: Path) -> tuple[dict, pd.DataFrame, np.ndarray]:
    context = evaluate_file(path, root, extra_eval_sets={"ood_val"})
    frame = pd.read_parquet(path)
    required = {"logit_nonhate", "logit_hate"}
    if not required <= set(frame):
        raise ValueError(f"{path}: missing raw logits {sorted(required - set(frame))}")
    logits = frame[["logit_nonhate", "logit_hate"]].to_numpy(dtype=np.float64)
    raw = apply_temperature(logits, 1.0)[:, 1]
    if not np.allclose(raw, frame["prob_hate"].to_numpy(dtype=np.float64), atol=1e-5, rtol=0):
        raise ValueError(f"{path}: saved probabilities disagree with raw logits")
    if not np.array_equal((raw >= .5).astype(int), frame["prediction"].to_numpy(dtype=int)):
        raise ValueError(f"{path}: saved predictions disagree with logits")
    return context, frame, logits


def _metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    return {
        "ece_equal_mass_15": expected_calibration_error(labels, probabilities, 15, "equal_mass"),
        "ece_equal_width_15": expected_calibration_error(labels, probabilities, 15, "equal_width"),
        "brier": brier_score(labels, probabilities),
        "nll": negative_log_likelihood(labels, probabilities),
    }


def run_recalibration_experiment(logits_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Fit only on validation sets; evaluate only on held-out test sets.

    An oracle arm is emitted only if an explicit OOD validation file exists.
    Pilot artifacts without ID validation are recorded as skipped.
    """
    paths = discover_logits(logits_root)
    if not paths:
        raise ValueError(f"No full-run logits found under {logits_root}")
    by_key: dict[tuple[str, str], tuple[dict, pd.DataFrame, np.ndarray]] = {}
    for path in paths:
        context, frame, logits = _read(path, logits_root)
        key = context["artifact"], context["eval_set"]
        if key in by_key:
            raise ValueError(f"Duplicate artifact/eval set: {key}")
        by_key[key] = context, frame, logits

    artifacts = sorted({key[0] for key in by_key})
    rows: list[dict] = []
    temperatures: list[dict] = []
    skipped: list[dict] = []
    for artifact in artifacts:
        own = by_key.get((artifact, "id_val"))
        tests = [by_key[(artifact, split)] for split in ("id_test", "ood_test") if (artifact, split) in by_key]
        if own is None or len(tests) != 2:
            skipped.append({"artifact": artifact, "reason": "missing ID validation or test logits"})
            continue
        context, frame, logits = own
        fp = by_key.get((context["base_run"], "id_val"))
        if fp is None or fp[0]["method"] != "bf16":
            raise ValueError(f"{artifact}: missing BF16 ID validation reference")
        fp_context, fp_frame, fp_logits = fp
        for field in ("base_run", "model", "seed", "lang", "loco", "n_samples", "sample_fingerprint"):
            if context[field] != fp_context[field]:
                raise ValueError(f"{artifact}: BF16 ID validation mismatch in {field}")
        own_t = fit_temperature(logits, frame["y_true"].to_numpy())
        fp_t = fit_temperature(fp_logits, fp_frame["y_true"].to_numpy())
        arms = [("none", 1.0, None, None, True),
                ("T_own", own_t, "id_val", context["logit_file"], True),
                ("T_fp", fp_t, "id_val", fp_context["logit_file"], True)]
        ood_path = logits_root / context["phase"] / artifact / f"{context['method']}__ood_val.parquet"
        if ood_path.is_file():
            oracle_context, oracle_frame, oracle_logits = _read(ood_path, logits_root)
            for field in ("artifact", "model", "seed", "lang", "loco"):
                if context[field] != oracle_context[field]:
                    raise ValueError(f"{artifact}: oracle validation mismatch in {field}")
            arms.append(("T_oracle", fit_temperature(oracle_logits, oracle_frame["y_true"].to_numpy()),
                         "ood_val", oracle_context["logit_file"], False))
        for arm, temperature, fit_split, fit_file, deployable in arms:
            temperatures.append({**{k: context[k] for k in ("artifact", "base_run", "phase", "method", "model", "seed", "lang", "loco")},
                                 "arm": arm, "temperature": temperature, "fit_split": fit_split,
                                 "fit_file": fit_file, "deployable": deployable,
                                 "label": "oracle -- not deployable" if arm == "T_oracle" else arm})
            for test_context, test_frame, test_logits in tests:
                if arm == "T_oracle" and test_context["eval_set"] != "ood_test":
                    continue
                fp_test = by_key.get((context["base_run"], test_context["eval_set"]))
                if fp_test is None:
                    raise ValueError(f"{artifact}: missing BF16 {test_context['eval_set']} reference")
                for field in ("model", "seed", "lang", "loco", "n_samples", "sample_fingerprint"):
                    if test_context[field] != fp_test[0][field]:
                        raise ValueError(f"{artifact}: BF16 test mismatch in {field}")
                labels = test_frame["y_true"].to_numpy(dtype=np.int64)
                raw_metrics = _metrics(labels, apply_temperature(test_logits, 1.0)[:, 1])
                scaled = apply_temperature(test_logits, temperature)
                if not np.allclose(scaled.sum(axis=1), 1.0, atol=1e-12):
                    raise ValueError(f"{artifact}: scaled probabilities do not sum to one")
                metrics = _metrics(labels, scaled[:, 1])
                row = {k: test_context[k] for k in ("artifact", "base_run", "phase", "method", "model", "seed", "lang", "loco", "eval_set", "n_samples", "sample_fingerprint", "logit_file", "single_class_eval_set")}
                row.update({"arm": arm, "temperature": temperature, "fit_split": fit_split,
                            "fit_file": fit_file, "deployable": deployable,
                            "label": "oracle -- not deployable" if arm == "T_oracle" else arm})
                for name, value in metrics.items():
                    row[f"raw_{name}"] = raw_metrics[name]
                    row[name] = value
                    row[f"delta_{name}"] = value - raw_metrics[name]
                rows.append(row)
    if not rows:
        raise ValueError("No complete artifacts with ID validation and both test sets")
    result = pd.DataFrame(rows)
    fits = pd.DataFrame(temperatures)
    summary = {"n_artifacts": len(fits[fits.arm == "none"]), "n_rows": len(result),
               "n_oracle_artifacts": int((fits.arm == "T_oracle").sum()), "skipped": skipped,
               "id_ece_improved_fraction": float((result.query("arm == 'T_own' and eval_set == 'id_test'")["delta_ece_equal_mass_15"] <= 0).mean()),
               "temperature_in_0_1_to_5_fraction": float(fits.query("arm != 'none'")["temperature"].between(.1, 5, inclusive="neither").mean())}
    return result, fits, summary
