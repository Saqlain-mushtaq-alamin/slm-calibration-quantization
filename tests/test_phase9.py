"""Phase 9 selective tables and risk-coverage curve integration tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.phase9 import evaluate_selective_file, evaluate_selective_prediction


def _write(root, phase, artifact, method, probabilities, labels, *, split="id_test"):
    path = root / phase / artifact / f"{method}__{split}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    p = np.asarray(probabilities)
    pd.DataFrame({
        "example_id": [str(i) for i in range(len(p))],
        "split": split,
        "y_true": labels,
        "prob_hate": p,
        "prob_nonhate": 1 - p,
        "prediction": (p >= 0.5).astype(int),
        "precision": method,
        "seed": 42,
        "language": "bn",
        "model_key": "model_a",
        "split_name": "loco_x",
    }).to_parquet(path, index=False)
    return path


def test_phase9_writes_one_curve_per_full_artifact(tmp_path):
    logits = tmp_path / "logits"
    curves = tmp_path / "curves"
    base = "bn_model_a_lora_bf16_s42"
    _write(logits, "phase4", base, "bf16", [.1, .9, .2, .8], [0, 1, 1, 1])
    _write(logits, "phase6", base + "__bnb_int8", "bnb_int8", [.2, .8, .3, .7], [0, 1, 1, 1])
    _write(logits, "phase7_smoke", base + "__f16", "f16", [.2, .8, .3, .7], [0, 1, 1, 1])
    table = evaluate_selective_prediction(logits, curves)
    assert len(table) == 2
    assert table["curve_file"].nunique() == 2
    for row in table.itertuples():
        curve = pd.read_parquet(curves / row.curve_file)
        assert len(curve) == row.n_samples
        assert curve["coverage"].iloc[-1] == 1
        assert curve["risk"].between(0, 1).all()
        assert row.aurc == pytest.approx(curve["risk"].mean())
        assert row.selective_acc_70 == pytest.approx(1 - curve["risk"].iloc[round(.7 * len(curve)) - 1])
        assert 0 <= row.coverage_at_5pct_risk <= 1
        assert row.e_aurc >= -1e-12
    pd.testing.assert_frame_equal(table, evaluate_selective_prediction(logits, curves))


def test_phase9_failure_auroc_undefined_with_zero_errors(tmp_path):
    logits = tmp_path / "logits"
    path = _write(logits, "phase4", "base", "bf16", [.1, .9, .2, .8], [0, 1, 0, 1])
    row, _ = evaluate_selective_file(path, logits)
    assert row["failure_auroc_defined"] is False
    assert row["failure_auroc"] is None
    assert row["aurc"] == 0


def test_phase9_accuracy_monotonicity_is_diagnostic(tmp_path):
    logits = tmp_path / "logits"
    # An error among the top 70, followed by correct examples, raises accuracy.
    n = 10
    probabilities = np.linspace(.99, .51, n)
    labels = np.ones(n, dtype=int)
    labels[0] = 0
    path = _write(logits, "phase4", "base", "bf16", probabilities, labels)
    row, _ = evaluate_selective_file(path, logits)
    assert row["selective_acc_70"] < row["selective_acc_80"]
    assert row["selective_accuracy_monotonic_70_90"] is False


def test_phase9_rejects_invalid_logits_before_curve_write(tmp_path):
    logits = tmp_path / "logits"
    curves = tmp_path / "curves"
    _write(logits, "phase4", "base", "bf16", [.1, 1.2, .2, .8], [0, 1, 0, 1])
    with pytest.raises(ValueError, match="invalid probabilities"):
        evaluate_selective_prediction(logits, curves)
    assert not list(curves.rglob("*.parquet"))
