"""Phase 11 fitting, provenance, and held-out evaluation gates."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from src.analysis.phase11 import run_recalibration_experiment
from src.recalibration.temperature_scaling import apply_temperature, fit_temperature


def _write(root, phase, artifact, method, split, margins, labels=None, ids=None):
    margins = np.asarray(margins, dtype=float)
    labels = np.asarray([0, 1, 0, 1] if labels is None else labels)
    ids = [f"{split}_{i}" for i in range(len(labels))] if ids is None else ids
    p = expit(margins)
    path = root / phase / artifact / f"{method}__{split}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"example_id": ids, "split": split, "y_true": labels,
                  "logit_nonhate": np.zeros(len(labels)), "logit_hate": margins,
                  "prob_hate": p, "prob_nonhate": 1 - p,
                  "prediction": (p >= .5).astype(int), "precision": method,
                  "seed": 42, "language": "bn", "model_key": "model_a",
                  "split_name": "loco_x"}).to_parquet(path, index=False)
    return path


def _grid(root, *, oracle=False):
    base = "bn_model_a_lora_bf16_s42"
    variant = base + "__bnb_int8"
    for phase, artifact, method, scale in [("phase4", base, "bf16", 1),
                                            ("phase6", variant, "bnb_int8", 2)]:
        for split in ("id_val", "id_test", "ood_test"):
            _write(root, phase, artifact, method, split, np.array([-2, 2, -1, 1]) * scale)
        if oracle:
            _write(root, phase, artifact, method, "ood_val", np.array([-3, 3, -2, 2]) * scale)
    return base, variant


def test_temperature_is_stable_and_rejects_invalid_input():
    logits = np.array([[1000, -1000], [-1000, 1000], [1000, -1000]], float)
    t = fit_temperature(logits, np.array([0, 1, 1]))
    assert .01 <= t <= 10 and np.isfinite(t)
    probs = apply_temperature(logits, t)
    np.testing.assert_allclose(probs.sum(axis=1), 1)
    with pytest.raises(ValueError):
        apply_temperature(logits, 0)
    with pytest.raises(ValueError):
        fit_temperature(logits, [0, 2, 1])


def test_recalibration_uses_only_validation_and_transfers_fp(tmp_path):
    base, variant = _grid(tmp_path)
    result, fits, summary = run_recalibration_experiment(tmp_path)
    assert len(result) == 12
    assert summary["n_artifacts"] == 2 and summary["n_oracle_artifacts"] == 0
    assert set(result.arm) == {"none", "T_own", "T_fp"}
    assert set(fits.query("arm == 'T_own'").fit_split) == {"id_val"}
    assert set(fits.query("arm == 'T_fp'").fit_split) == {"id_val"}
    assert fits.query("artifact == @variant and arm == 'T_fp'").iloc[0].fit_file.startswith("phase4/")
    assert fits.query("artifact == @variant and arm == 'T_fp'").iloc[0].temperature == pytest.approx(
        fits.query("artifact == @base and arm == 'T_own'").iloc[0].temperature)
    assert (result.query("arm == 'none'").filter(like="delta_").abs() < 1e-12).all().all()


def test_oracle_is_labeled_and_only_evaluated_on_ood(tmp_path):
    _grid(tmp_path, oracle=True)
    result, fits, summary = run_recalibration_experiment(tmp_path)
    oracle = result.query("arm == 'T_oracle'")
    assert len(oracle) == 2 and set(oracle.eval_set) == {"ood_test"}
    assert set(oracle.label) == {"oracle -- not deployable"}
    assert not oracle.deployable.any()
    assert set(fits.query("arm == 'T_oracle'").fit_split) == {"ood_val"}


def test_missing_reference_and_misalignment_fail(tmp_path):
    base, variant = _grid(tmp_path)
    (tmp_path / "phase4" / base / "bf16__id_val.parquet").unlink()
    with pytest.raises(ValueError, match="missing BF16 ID validation"):
        run_recalibration_experiment(tmp_path)
    _grid(tmp_path)
    _write(tmp_path, "phase6", variant, "bnb_int8", "id_test", [-2, 2, -1, 1],
           ids=["wrong", "id_test_1", "id_test_2", "id_test_3"])
    with pytest.raises(ValueError, match="BF16 test mismatch"):
        run_recalibration_experiment(tmp_path)
