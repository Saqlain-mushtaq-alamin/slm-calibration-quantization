from pathlib import Path

import pandas as pd
import pytest

from src.analysis.phase13 import prepare_pairs, run_phase13


RESULTS = Path(__file__).resolve().parents[1] / "outputs/results"


def test_pairing_rejects_example_mismatch():
    frame = pd.read_parquet(RESULTS / "all_metrics.parquet")
    pairs = prepare_pairs(frame)
    assert len(pairs) > 0
    changed = frame.copy()
    idx = changed.index[(changed.method == "gptq_int4") & (changed.eval_set == "id_test")][0]
    changed.loc[idx, "sample_fingerprint"] = "wrong"
    with pytest.raises(ValueError, match="pairing mismatch"):
        prepare_pairs(changed)


def test_phase13_real_results_smoke():
    table, interactions, summary = run_phase13(RESULTS, n_boot=100, seed=42)
    assert set(summary["hypotheses"]) == {"H1", "H2", "H3", "H4", "H5a", "H5b", "H6", "H7"}
    assert len(table) == 138
    assert set(interactions.factor) == {"domain", "language"}
    assert interactions.converged.all()
    assert (table.p_holm >= table.p_value).all()
    assert (table.loc[table.hypothesis == "H7", "status"] == "partial").all()
    assert (table.loc[table.hypothesis == "H7", "n_models"] == 2).all()
