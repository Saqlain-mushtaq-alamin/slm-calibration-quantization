"""Phase 8 artifact discovery, calibration, reference matching and integrity gates."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.phase8 import compute_deltas_vs_fp, discover_logits, evaluate_all_artifacts


def _write(root, phase, artifact, method, split, probabilities, *, ids=None, labels=None):
    p = np.asarray(probabilities)
    y = np.asarray(labels if labels is not None else [0, 1, 0, 1])
    path = root / phase / artifact / f"{method}__{split}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "example_id": ids if ids is not None else ["a", "b", "c", "d"],
        "split": split, "y_true": y, "prob_hate": p,
        "prob_nonhate": 1 - p, "prediction": (p >= .5).astype(int),
        "precision": method, "seed": 42, "language": "bn",
        "model_key": "model_a", "split_name": "loco_x",
    }).to_parquet(path, index=False)
    return path


def test_phase8_evaluates_all_full_logits_and_matches_reference(tmp_path):
    base = "bn_model_a_lora_bf16_s42"
    _write(tmp_path, "phase4", base, "bf16", "id_test", [.1, .9, .2, .8])
    _write(tmp_path, "phase6", base + "__bnb_int8", "bnb_int8", "id_test", [.2, .8, .3, .7])
    _write(tmp_path, "phase7", base + "__f16", "f16", "id_test", [.15, .85, .25, .75])
    _write(tmp_path, "phase7_smoke", base + "__f16", "f16", "id_test", [.15, .85, .25, .75])
    assert len(discover_logits(tmp_path)) == 3
    rows = evaluate_all_artifacts(tmp_path)
    assert len(rows) == 3
    assert rows["logit_file"].nunique() == 3
    assert rows["ece_equal_mass_15"].between(0, 1).all()
    assert rows["brier"].between(0, 1).all()
    assert rows["macro_f1"].between(0, 1).all()
    assert not rows.select_dtypes(include="number").isna().any().any()
    deltas = compute_deltas_vs_fp(rows)
    assert len(deltas) == 2
    assert set(deltas["fp_reference"]) == {base}
    pd.testing.assert_frame_equal(rows, evaluate_all_artifacts(tmp_path))


def test_phase8_single_class_has_explicit_undefined_metrics(tmp_path):
    _write(tmp_path, "phase4", "base", "bf16", "ood_test", [.8, .9, .7, .6], labels=[1, 1, 1, 1])
    row = evaluate_all_artifacts(tmp_path).iloc[0]
    assert row["single_class_eval_set"]
    assert pd.isna(row["auroc"])
    assert pd.isna(row["balanced_accuracy"])
    assert np.isfinite(row["ece_equal_mass_15"])
    assert np.isfinite(row["brier"])


def test_phase8_rejects_unmatched_or_misaligned_reference(tmp_path):
    base = "bn_model_a_lora_bf16_s42"
    variant = base + "__bnb_int8"
    _write(tmp_path, "phase6", variant, "bnb_int8", "id_test", [.2, .8, .3, .7])
    with pytest.raises(ValueError, match="Missing BF16 reference"):
        compute_deltas_vs_fp(evaluate_all_artifacts(tmp_path))
    _write(tmp_path, "phase4", base, "bf16", "id_test", [.1, .9, .2, .8], ids=["a", "b", "c", "other"])
    with pytest.raises(ValueError, match="sample_fingerprint"):
        compute_deltas_vs_fp(evaluate_all_artifacts(tmp_path))


def test_phase8_rejects_corrupt_probabilities(tmp_path):
    _write(tmp_path, "phase4", "base", "bf16", "id_test", [.1, 1.2, .2, .8])
    with pytest.raises(ValueError, match="invalid probabilities"):
        evaluate_all_artifacts(tmp_path)
