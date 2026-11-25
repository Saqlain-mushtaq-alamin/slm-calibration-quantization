"""Phase 12 frozen-data, calibration, logit and A8 provenance checks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis.phase12 import (
    CALIB_VARIANTS, calibration_sets, frozen_frames, metric_row,
    temperature_source_rows, validate_ablation_logits,
)


def _frozen(tmp_path):
    ids = [f"x{i}" for i in range(8)]
    data = pd.DataFrame({"example_id": ids, "text": [f"example {i}" for i in range(8)],
                         "label": [0, 1] * 4})
    data.to_csv(tmp_path / "data.csv", index=False)
    spec = {"loco_x": {"train_ids": ids[:4], "id_val_ids": ids[4:5],
                        "id_test_ids": ids[5:7], "ood_test_ids": ids[7:]}}
    (tmp_path / "splits.json").write_text(json.dumps(spec), encoding="utf-8")
    return tmp_path / "data.csv", tmp_path / "splits.json"


def test_frozen_frames_and_calibration_sets_are_train_only(tmp_path):
    data, splits = _frozen(tmp_path)
    frames, sha = frozen_frames(data, splits, "loco_x")
    assert len(sha) == 64
    forbidden = set().union(*(set(frames[k].example_id) for k in ("id_val", "id_test", "ood_test")))
    sets = calibration_sets(frames["train"], frames["train"], n=4, seed=42,
                            template="{text} Answer:", forbidden_bn=forbidden, forbidden_en=forbidden)
    assert set(sets) == set(CALIB_VARIANTS)
    assert all(x["n"] == 4 for x in sets.values())
    assert all(not (set(x["ids"]) & forbidden) for x in sets.values())
    assert sets["random"]["ids"] == []
    assert sets == calibration_sets(frames["train"], frames["train"], n=4, seed=42,
                                    template="{text} Answer:", forbidden_bn=forbidden, forbidden_en=forbidden)


def test_frozen_frames_reject_overlap(tmp_path):
    data, splits = _frozen(tmp_path)
    spec = json.loads(splits.read_text())
    spec["loco_x"]["id_test_ids"][0] = spec["loco_x"]["train_ids"][0]
    splits.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="overlapping"):
        frozen_frames(data, splits, "loco_x")


def test_logit_gate_and_metrics(tmp_path):
    expected = pd.DataFrame({"example_id": ["a", "b"], "label": [0, 1]})
    frame = pd.DataFrame({"example_id": ["a", "b"], "y_true": [0, 1],
                          "logit_nonhate": [2., -2.], "logit_hate": [-2., 2.],
                          "prob_nonhate": [.98, .02], "prob_hate": [.02, .98],
                          "prediction": [0, 1]})
    validate_ablation_logits(frame, expected)
    assert metric_row(frame, ablation_id="A2")["ece_equal_mass_15"] >= 0
    with pytest.raises(ValueError, match="frozen evaluation"):
        validate_ablation_logits(frame.iloc[::-1].reset_index(drop=True), expected)
    bad = frame.copy()
    bad.loc[0, "prob_hate"] = np.nan
    with pytest.raises(ValueError, match="Nonfinite"):
        validate_ablation_logits(bad, expected)


def test_a8_requires_fit_provenance_and_keeps_oracle_tag(tmp_path):
    path = tmp_path / "phase11.parquet"
    pd.DataFrame({"artifact": ["bn_m_lora_bf16_s42", "bn_m_lora_bf16_s42"],
                  "model": ["m", "m"], "lang": ["bn", "bn"], "seed": [42, 42],
                  "eval_set": ["ood_test", "ood_test"], "arm": ["T_own", "T_oracle"],
                  "temperature": [1.1, 1.2], "fit_split": ["id_val", "ood_val"],
                  "fit_file": ["id_val.parquet", "ood_val.parquet"],
                  "raw_ece_equal_mass_15": [.2, .2], "ece_equal_mass_15": [.1, .08],
                  "raw_brier": [.3, .3], "brier": [.2, .15],
                  "raw_nll": [.4, .4], "nll": [.3, .25],
                  "deployable": [True, False]}).to_parquet(path, index=False)
    out = temperature_source_rows(path, models={"m"}, languages={"bn"}, seeds={42})
    assert set(out.ablation_id) == {"A8"}
    assert not out.loc[out.variant == "T_oracle", "deployable"].iloc[0]
    bad = pd.read_parquet(path)
    bad.loc[0, "fit_split"] = "id_test"
    bad.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="ID validation"):
        temperature_source_rows(path, models={"m"}, languages={"bn"}, seeds={42})
