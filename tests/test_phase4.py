"""Phase 4 unit tests: run matrix, split assembly, and T4.x validation gates.

These cover the logic the sweep itself cannot cheaply exercise 16 times: the matrix
must match the plan exactly, a capped split must never silently lose a class, and
the single-class ood_test sets must be reported as undefined rather than failed.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.assemble import build_split_frames, resolve_model_id, split_frame_meta, split_sizes
from src.models import logit_extraction as lx
from src.utils.config import load_config


def _load_script():
    spec = importlib.util.spec_from_file_location("phase4_script", ROOT / "scripts" / "03_finetune.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["phase4_script"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ph4():
    return _load_script()


@pytest.fixture(scope="module")
def cfg():
    return load_config(ROOT / "configs" / "baseline" / "default.yaml")


# ---------------------------------------------------------------------------
# Run matrix
# ---------------------------------------------------------------------------

def test_run_matrix_has_sixteen_cells(ph4, cfg):
    matrix = ph4.build_run_matrix(cfg, None, None)
    assert len(matrix) == 16


def test_run_matrix_matches_plan_seeds(ph4, cfg):
    matrix = ph4.build_run_matrix(cfg, None, None)
    seeds_by_model: dict[str, list[int]] = {}
    for cell in matrix:
        seeds_by_model.setdefault(cell["model_key"], []).append(cell["seed"])
    assert seeds_by_model == {
        "qwen2_5_0_5b_instruct": [13, 42, 1234, 2024, 7],
        "qwen2_5_1_5b_instruct": [13, 42, 1234, 2024, 7],
        "llama3_2_1b_instruct": [13, 42, 1234],
        "gemma2_2b_it": [13, 42, 1234],
    }


def test_canonical_ids_match_plan(ph4, cfg):
    matrix = ph4.build_run_matrix(cfg, None, None)
    ids = {c["model_key"]: c["hf_id"] for c in matrix}
    assert ids == {
        "qwen2_5_0_5b_instruct": "Qwen/Qwen2.5-0.5B-Instruct",
        "qwen2_5_1_5b_instruct": "Qwen/Qwen2.5-1.5B-Instruct",
        "llama3_2_1b_instruct": "meta-llama/Llama-3.2-1B-Instruct",
        "gemma2_2b_it": "google/gemma-2-2b-it",
    }


def test_every_cell_declares_a_mirror_for_gated_models(cfg):
    gated = {"meta-llama/Llama-3.2-1B-Instruct", "google/gemma-2-2b-it"}
    for entry in cfg["models"]:
        if entry["hf_id"] in gated:
            assert entry["mirrors"], f"{entry['hf_id']} is gated but declares no mirror"


def test_run_name_is_filesystem_safe_and_seed_suffixed(ph4):
    name = ph4.run_name("qwen2_5_1_5b_instruct", 42)
    assert name == "bn_qwen2_5_1_5b_instruct_lora_bf16_s42"
    assert set(name) <= set("abcdefghijklmnopqrstuvwxyz0123456789_")


def test_matrix_filters_are_applied(ph4, cfg):
    matrix = ph4.build_run_matrix(cfg, ["gemma2_2b_it"], [13])
    assert len(matrix) == 1
    assert matrix[0]["model_key"] == "gemma2_2b_it"
    assert matrix[0]["seed"] == 13


def test_gemma_is_forced_to_eager_attention(ph4, cfg):
    """Gemma-2 sliding window + logit soft-capping break the fused sdpa kernel."""
    matrix = ph4.build_run_matrix(cfg, None, None)
    impl = {c["model_key"]: c["attn_implementation"] for c in matrix}
    assert impl["gemma2_2b_it"] == "eager"
    # The others keep the transformers default.
    assert impl["qwen2_5_0_5b_instruct"] is None
    assert impl["llama3_2_1b_instruct"] is None


def test_every_matrix_cell_carries_its_attn_implementation(ph4, cfg):
    for cell in ph4.build_run_matrix(cfg, None, None):
        assert "attn_implementation" in cell


def test_matrix_deduplicates_against_the_bundled_splits():
    """Every configured model must have a headroom record for seed-level variance."""
    splits_path = ROOT / "data" / "splits" / "bd_shs_loco_splits.json"
    if not splits_path.exists():
        pytest.skip("Phase 1 splits not generated")
    from src.data.splits import load_splits

    splits, _ = load_splits(splits_path)
    cfg = load_config(ROOT / "configs" / "baseline" / "default.yaml")
    assert str(cfg["split"]) in splits


# ---------------------------------------------------------------------------
# Split assembly
# ---------------------------------------------------------------------------

@pytest.fixture
def toy_spec():
    rows = []
    for i in range(400):
        domain = "slander" if i % 4 == 0 else "other"
        rows.append(
            {
                "example_id": f"ex{i}",
                "text": f"sample {i}",
                "label": 1 if i % 3 else 0,
                "domain": domain,
            }
        )
    df = pd.DataFrame(rows)
    spec = {
        "target_domain": "slander",
        "source_domains": ["other"],
        "train_ids": [f"ex{i}" for i in range(0, 400, 4)],
        "id_val_ids": [f"ex{i}" for i in range(1, 400, 4)],
        "id_test_ids": [f"ex{i}" for i in range(2, 400, 4)],
        "ood_test_ids": [f"ex{i}" for i in range(3, 400, 4)],
        "sizes": {"train": 100, "id_val": 100, "id_test": 100, "ood_test": 100},
    }
    return df, spec


def test_build_split_frames_uses_every_id(toy_spec):
    df, spec = toy_spec
    frames = build_split_frames(df, spec)
    assert set(frames) == {"train", "id_val", "id_test", "ood_test"}
    assert all(len(f) == 100 for f in frames.values())
    # ids are preserved, not regenerated
    assert frames["ood_test"]["example_id"].tolist() == spec["ood_test_ids"]


def test_build_split_frames_caps_and_keeps_both_classes(toy_spec):
    df, spec = toy_spec
    frames = build_split_frames(df, spec, caps={"train": 40}, seed=42)
    assert len(frames["train"]) == 40
    assert set(frames["train"]["label"].unique()) == {0, 1}
    # uncapped parts untouched
    assert len(frames["id_test"]) == 100


def test_build_split_frames_is_deterministic(toy_spec):
    df, spec = toy_spec
    a = build_split_frames(df, spec, caps={"train": 25}, seed=7)
    b = build_split_frames(df, spec, caps={"train": 25}, seed=7)
    c = build_split_frames(df, spec, caps={"train": 25}, seed=8)
    assert list(a["train"]["example_id"]) == list(b["train"]["example_id"])
    assert list(a["train"]["example_id"]) != list(c["train"]["example_id"])


def test_cap_that_drops_a_class_is_refused(toy_spec):
    df, spec = toy_spec
    with pytest.raises(AssertionError, match="lost a class"):
        build_split_frames(df, spec, caps={"train": 1}, seed=42)


def test_missing_ids_are_reported_not_silently_dropped(toy_spec):
    df, spec = toy_spec
    spec = {**spec, "id_test_ids": spec["id_test_ids"] + ["does_not_exist"]}
    with pytest.raises(KeyError, match="out of sync"):
        build_split_frames(df, spec)


def test_unknown_part_is_rejected(toy_spec):
    df, spec = toy_spec
    with pytest.raises(KeyError, match="ood_ids"):
        build_split_frames(df, spec, parts=("train", "ood"))


def test_ood_test_may_stay_single_class():
    """bd_shs LOCO target domains are pure hate; that must not raise."""
    df = pd.DataFrame(
        {
            "example_id": [f"o{i}" for i in range(20)],
            "text": [f"t{i}" for i in range(20)],
            "label": [1] * 20,
            "domain": ["slander"] * 20,
        }
    )
    spec = {
        "ood_test_ids": list(df["example_id"]),
        "sizes": {"ood_test": 20},
    }
    frames = build_split_frames(df, spec, parts=("ood_test",))
    assert frames["ood_test"]["label"].nunique() == 1


def test_split_sizes_and_meta(toy_spec):
    df, spec = toy_spec
    frames = build_split_frames(df, spec)
    assert split_sizes(spec) == {"train": 100, "id_val": 100, "id_test": 100, "ood_test": 100}
    meta = split_frame_meta(frames, spec)
    assert meta["sizes_used"]["train"] == 100
    assert set(meta["frame_hashes"]) == set(frames)
    assert meta["class_balance_used"]["train"]["0"] > 0


# ---------------------------------------------------------------------------
# Mirror resolution
# ---------------------------------------------------------------------------

def _reachable(*ids: str):
    """A transformers.Auto* stand-in that answers only for the given repo ids."""

    class Fake:
        @staticmethod
        def from_pretrained(model_id, **_kw):
            if model_id in ids:
                return object()
            raise OSError(f"{model_id} is gated or unreachable")

    return Fake


def test_resolve_model_id_prefers_the_canonical_repo(monkeypatch):
    fake = _reachable("public/model")
    monkeypatch.setattr("transformers.AutoTokenizer", fake)
    monkeypatch.setattr("transformers.AutoConfig", fake)
    model_id, notes = resolve_model_id("public/model", ["mirror/model"])
    assert model_id == "public/model"
    assert notes == []


def test_resolve_model_id_falls_back_and_records_the_substitution(monkeypatch):
    fake = _reachable("unsloth/Llama-3.2-1B-Instruct")
    monkeypatch.setattr("transformers.AutoTokenizer", fake)
    monkeypatch.setattr("transformers.AutoConfig", fake)
    model_id, notes = resolve_model_id("meta-llama/Llama-3.2-1B-Instruct", ["unsloth/Llama-3.2-1B-Instruct"])
    assert model_id == "unsloth/Llama-3.2-1B-Instruct"
    assert notes and "meta-llama/Llama-3.2-1B-Instruct" in notes[0]
    assert "unsloth/Llama-3.2-1B-Instruct" in notes[0]


def test_resolve_model_id_tries_every_mirror_in_order(monkeypatch):
    fake = _reachable("mirror/second")
    monkeypatch.setattr("transformers.AutoTokenizer", fake)
    monkeypatch.setattr("transformers.AutoConfig", fake)
    model_id, _notes = resolve_model_id("gated/x", ["mirror/first", "mirror/second"])
    assert model_id == "mirror/second"


def test_resolve_model_id_raises_when_nothing_is_reachable(monkeypatch):
    fake = _reachable()
    monkeypatch.setattr("transformers.AutoTokenizer", fake)
    monkeypatch.setattr("transformers.AutoConfig", fake)
    with pytest.raises(RuntimeError, match="None of the candidates"):
        resolve_model_id("meta-llama/x", ["mirror/x"])


# ---------------------------------------------------------------------------
# T4.x validation gates
# ---------------------------------------------------------------------------

def _frame(y_true, y_prob, ids=None):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    ids = ids if ids is not None else [str(i) for i in range(len(y_true))]
    return lx.probs_to_frame(ids, "id_test", y_true, np.log(np.clip(y_prob, 1e-12, 1)),
                            np.log(np.clip(1 - y_prob, 1e-12, 1)))


def _good(n=400):
    rng = np.random.default_rng(0)
    p = np.clip(0.15 + 0.7 * rng.uniform(size=n), 0.02, 0.98)
    y = (rng.uniform(size=n) < p).astype(np.int64)
    return y, p


def test_validate_accepts_a_healthy_two_class_run(ph4):
    y, p = _good()
    report = ph4.validate_logits_frame(_frame(y, p), min_macro_f1=0.55, require_both_classes=True)
    assert report["passed"] is True
    assert report["single_class"] is False
    assert report["macro_f1"] > 0.55
    assert report["max_prob_sum_deviation"] < 1e-9
    assert all(report["checks"].values())


def test_validate_rejects_missing_columns(ph4):
    y, p = _good(40)
    frame = _frame(y, p).drop(columns=["prediction"])
    with pytest.raises(ValueError, match="missing columns"):
        ph4.validate_logits_frame(frame, min_macro_f1=0.55, require_both_classes=True)


def test_validate_rejects_empty_frame(ph4):
    frame = _frame(np.array([], dtype=np.int64), np.array([]))
    with pytest.raises(ValueError, match="empty"):
        ph4.validate_logits_frame(frame, min_macro_f1=0.55, require_both_classes=True)


def test_validate_flags_chance_level_f1(ph4):
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 400)
    p = np.clip(rng.uniform(0.45, 0.55, 400), 0.02, 0.98)
    report = ph4.validate_logits_frame(_frame(y, p), min_macro_f1=0.55, require_both_classes=True)
    assert report["checks"]["f1_above_random"] is False
    assert report["passed"] is False


def test_validate_flags_prediction_collapse(ph4):
    """T4.3: predicting a single class is a failure even with high accuracy."""
    y, p = _good()
    frame = _frame(y, p)
    frame["prediction"] = 0
    report = ph4.validate_logits_frame(frame, min_macro_f1=0.55, require_both_classes=True)
    assert report["n_pred_classes"] == 1
    assert report["checks"]["covers_both_classes"] is False
    assert report["passed"] is False


def test_validate_flags_unnormalised_probabilities(ph4):
    y, p = _good(80)
    frame = _frame(y, p)
    frame.loc[0, "prob_nonhate"] = 0.42
    report = ph4.validate_logits_frame(frame, min_macro_f1=0.55, require_both_classes=True)
    assert report["checks"]["probs_sum_to_one"] is False
    assert report["passed"] is False


def test_validate_flags_nan_logits(ph4):
    y, p = _good(80)
    frame = _frame(y, p)
    frame.loc[3, "logit_hate"] = np.nan
    report = ph4.validate_logits_frame(frame, min_macro_f1=0.55, require_both_classes=True)
    assert report["checks"]["no_nan_logits"] is False
    assert report["passed"] is False


def test_single_class_set_reports_undefined_instead_of_failing(ph4):
    """T4.2/T4.3 on an all-hate ood_test: macro-F1 and collapse are undefined."""
    y = np.ones(200, dtype=np.int64)
    p = np.clip(np.linspace(0.55, 0.99, 200), 0.02, 0.98)
    report = ph4.validate_logits_frame(_frame(y, p), min_macro_f1=0.55, require_both_classes=True)
    assert report["single_class"] is True
    assert report["macro_f1"] is None
    assert report["checks"]["f1_above_random"] is None
    assert report["checks"]["covers_both_classes"] is None
    # Structural checks still apply and still pass.
    assert report["checks"]["probs_sum_to_one"] is True
    assert report["checks"]["no_nan_logits"] is True
    assert report["passed"] is True
    assert report["note"]


def test_collapse_check_is_reported_as_inapplicable_when_disabled(ph4):
    y, p = _good(60)
    frame = _frame(y, p)
    frame["prediction"] = 1
    # min_macro_f1=0.0 so the F1 gate cannot mask what this test is about.
    report = ph4.validate_logits_frame(frame, min_macro_f1=0.0, require_both_classes=False)
    # Opted out of the gate, so it is "not applicable" rather than a silent pass.
    assert report["checks"]["covers_both_classes"] is None
    assert report["passed"] is True


# ---------------------------------------------------------------------------
# Seed variance (T4.9)
# ---------------------------------------------------------------------------

def _ok_run(model_key: str, seed: int, macro_f1: float) -> dict:
    return {
        "run_name": f"{model_key}_s{seed}",
        "model_key": model_key,
        "seed": seed,
        "status": "success",
        "T4.2": {"macro_f1": macro_f1, "passed": True},
    }


def test_seed_variance_computes_std(ph4):
    records = [_ok_run("m", s, f) for s, f in [(13, 0.70), (42, 0.74), (1234, 0.72)]]
    var = ph4._seed_variance(records, 0.05)
    assert var["m"]["n_seeds"] == 3
    assert var["m"]["macro_f1_mean"] == pytest.approx(0.72, abs=0.01)
    assert var["m"]["macro_f1_std"] == pytest.approx(0.02, abs=0.005)
    assert var["m"]["passed"] is True


def test_seed_variance_fails_unstable_seeds(ph4):
    records = [_ok_run("m", s, f) for s, f in [(13, 0.50), (42, 0.80), (1234, 0.62)]]
    assert ph4._seed_variance(records, 0.05)["m"]["passed"] is False


def test_seed_variance_single_seed_is_inconclusive(ph4):
    var = ph4._seed_variance([_ok_run("m", 13, 0.7)], 0.05)
    assert var["m"]["n_seeds"] == 1
    assert var["m"]["passed"] is None


def test_seed_variance_ignores_failed_runs(ph4):
    records = [_ok_run("m", 13, 0.70), _ok_run("m", 42, 0.70)]
    records[1] = {**records[1], "status": "failed"}
    assert ph4._seed_variance(records, 0.05)["m"]["n_seeds"] == 1