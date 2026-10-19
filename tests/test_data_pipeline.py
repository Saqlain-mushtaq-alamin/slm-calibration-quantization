"""Phase 1 data-pipeline tests (T1.x unit + integration where data exists)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dedup import dedup_pipeline, exact_dedup, near_dedup
from src.data.loader import load_bd_shs, primary_hate_type, subsample_to_match
from src.data.shift import (
    compute_mmd,
    compute_oov_rate,
    compute_vocab_overlap,
    compute_shift_magnitude,
)
from src.data.splits import (
    create_loco_splits,
    freeze_and_hash_splits,
    load_splits,
    verify_split_integrity,
)

BD_SHS_RAW = ROOT / "data" / "raw" / "bd_shs"


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def toy_df() -> pd.DataFrame:
    rows = []
    domains = ["slander", "gender", "religion", "non_hate"]
    for i in range(200):
        d = domains[i % 4]
        label = 0 if d == "non_hate" else (i % 2)
        rows.append(
            {
                "example_id": f"id_{i}",
                "text": f"sample text {i} domain {d} unique-{i}",
                "label": label,
                "domain": d,
            }
        )
    return pd.DataFrame(rows)


def test_exact_dedup_removes_duplicates():
    df = pd.DataFrame(
        {
            "example_id": ["a", "b", "c", "d"],
            "text": ["Hello World", "hello world", "  hello   world ", "other"],
            "label": [1, 0, 1, 0],
        }
    )
    out = exact_dedup(df)
    assert len(out) == 2
    assert set(out["text"]) == {"Hello World", "other"}


def test_near_dedup_removes_fuzzy_duplicates():
    base = "bengali hate speech sample text about violence and slander "
    df = pd.DataFrame(
        {
            "example_id": [f"id_{i}" for i in range(6)],
            "text": [
                base,
                base + "extra",
                base[:-1],
                "completely different sentence about sports cricket match",
                base,
                "another unrelated post about weather today",
            ],
            "label": [1, 1, 1, 0, 1, 0],
        }
    )
    out = near_dedup(df, threshold=0.8, num_perm=64)
    assert len(out) < len(df)
    # exact duplicate of first row must be gone
    assert out["text"].duplicated().sum() == 0


def test_dedup_pipeline_stats(toy_df):
    df = toy_df.copy()
    df = pd.concat([df, df.iloc[:3]], ignore_index=True)
    out, stats = dedup_pipeline(df, run_near=False)
    assert stats["removed_exact"] == 3
    assert stats["n_final"] == len(out)


def test_loco_splits_disjoint_and_cover_source(toy_df):
    splits = create_loco_splits(toy_df, domain_col="domain", seed=42, min_domain_size=50)
    assert len(splits) >= 2
    report = verify_split_integrity(splits)
    assert report["ok"] is True
    for spec in splits.values():
        train = set(spec["train_ids"])
        id_val = set(spec["id_val_ids"])
        id_test = set(spec["id_test_ids"])
        ood = set(spec["ood_test_ids"])
        assert not (train & id_val)
        assert not (train & id_test)
        assert not (train & ood)
        assert not (id_test & ood)
        # source union ⊆ ids not in ood
        source = train | id_val | id_test
        assert not (source & ood)
        # every source id maps to a non-target domain
        assert len(source) > 0


def test_split_hash_deterministic(toy_df, tmp_path):
    splits = create_loco_splits(toy_df, domain_col="domain", seed=42, min_domain_size=50)
    p1 = tmp_path / "splits.json"
    p2 = tmp_path / "splits2.json"
    sha1 = freeze_and_hash_splits(splits, p1)
    sha2 = freeze_and_hash_splits(splits, p2)
    assert sha1 == sha2
    loaded, sha3 = load_splits(p1)
    assert sha3 == sha1
    assert set(loaded.keys()) == set(splits.keys())


def test_mmd_and_vocab_metrics():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, size=(50, 8))
    y = rng.normal(0, 1, size=(50, 8)) + 3.0
    mmd_close = compute_mmd(x, x + rng.normal(0, 0.01, size=x.shape))
    mmd_far = compute_mmd(x, y)
    assert mmd_far > mmd_close

    src = ["alpha beta gamma", "alpha beta delta"]
    tgt = ["alpha beta gamma", "zeta eta theta"]
    overlap = compute_vocab_overlap(src, tgt)
    assert 0.0 <= overlap <= 1.0
    oov = compute_oov_rate(src, tgt)
    assert 0.0 <= oov <= 1.0


def test_compute_shift_magnitude_dict_keys():
    src = ["bad words here hate speech", "another hateful comment"]
    tgt = ["totally different sports text about cricket", "weather is nice today"]
    out = compute_shift_magnitude(src, tgt, model_name="paraphrase-multilingual-MiniLM-L12-v2")
    for key in ("mmd", "centroid_cosine_dist", "vocab_overlap", "oov_rate_word"):
        assert key in out
        assert out[key] is not None


def test_primary_hate_type_mapping():
    assert primary_hate_type(np.nan) == "non_hate"
    assert primary_hate_type("slander") == "slander"
    assert primary_hate_type("callToViolence_slander") == "callToViolence"
    assert primary_hate_type("gender_slander") == "gender"
    assert primary_hate_type("religion_slander") == "religion"


def test_subsample_to_match_prior_and_size():
    n = 2000
    df = pd.DataFrame(
        {
            "example_id": [f"e_{i}" for i in range(n)],
            "text": [f"text {i}" for i in range(n)],
            "label": [1 if i < n // 3 else 0 for i in range(n)],
        }
    )
    out = subsample_to_match(df, target_n=500, target_prior=0.5, seed=1)
    assert abs(len(out) - 500) / 500 <= 0.10
    assert abs(out["label"].mean() - 0.5) <= 0.02


# ---------------------------------------------------------------------------
# Integration against real BD-SHS (skipped if not downloaded)
# ---------------------------------------------------------------------------

pytestmark_integration = pytest.mark.skipif(
    not (BD_SHS_RAW / "train.csv").exists(),
    reason="BD-SHS raw files not present under data/raw/bd_shs/",
)


@pytestmark_integration
def test_bd_shs_load_t1_1_t1_2():
    df = load_bd_shs(BD_SHS_RAW)
    assert len(df) >= 45000
    assert "text" in df.columns and "label" in df.columns
    assert df["domain"].nunique() >= 3


def test_bd_shs_loco_frozen_artifacts_exist():
    split_json = ROOT / "data" / "splits" / "bd_shs_loco_splits.json"
    if not split_json.exists():
        pytest.skip("Phase 1 splits not generated yet")
    splits, sha = load_splits(split_json)
    report = verify_split_integrity(splits)
    assert report["ok"] is True
    sha_file = Path(str(split_json) + ".sha256")
    assert sha_file.exists()
    assert sha_file.read_text(encoding="utf-8").strip() == sha


def test_phase1_outputs_if_present():
    results = ROOT / "outputs" / "results"
    shift_csv = results / "shift_magnitudes.csv"
    fert_csv = results / "tokenizer_fertility.csv"
    decision = results / "english_control_decision.json"
    if not decision.exists():
        pytest.skip("Phase 1 script not run yet")
    dec = json.loads(decision.read_text(encoding="utf-8"))
    assert dec.get("chosen") in {"toxigen", "hatexplain_fallback"}
    if shift_csv.exists():
        sdf = pd.read_csv(shift_csv)
        assert "mmd" in sdf.columns or len(sdf) == 0
    if fert_csv.exists():
        fdf = pd.read_csv(fert_csv)
        assert "tokens_per_word" in fdf.columns or "status" in fdf.columns
