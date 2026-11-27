"""Phase 14 synthetic dominance, provenance, and real artifact checks."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis.pareto import (
    assemble,
    derived_ratios,
    frontiers,
    pareto_frontier,
    run_phase14,
)


def test_pareto_known_points_and_ties():
    points = np.array([[1, 4], [2, 2], [3, 3], [4, 1], [2, 2]])
    assert pareto_frontier(points).tolist() == [True, True, False, True, True]
    assert pareto_frontier(np.array([[1, 1], [2, 2]]), [False, False]).tolist() == [False, True]
    with pytest.raises(ValueError):
        pareto_frontier(np.array([[np.nan, 1]]))


def test_frontiers_all_groups_and_unavailable_energy():
    rows = []
    for lang in ("bn", "en"):
        for split in ("id_test", "ood_test"):
            for method, memory, ece in (("bf16", 2.0, .2), ("int8", 1.0, .1)):
                rows.append({"lang": lang, "eval_set": split, "method": method,
                             "artifact": f"{lang}-{method}", "base_run": lang,
                             "model": "tiny", "seed": 42, "backend": "HF", "loco": "fold",
                             "sample_fingerprint": "same", "peak_vram_mb": memory,
                             "ece_equal_mass_15": ece, "aurc": ece,
                             "latency_p50_ms": memory, "latency_stable": True,
                             "vram_measurement": "measured", "power_available": False,
                             "energy_stable": False, "energy_per_1k_joules": np.nan,
                             "one_minus_f1": ece})
    points, summary, comparison = frontiers(pd.DataFrame(rows))
    assert len(summary) == 20
    assert summary.status.eq("computed").sum() == 12
    assert summary.loc[summary.status.eq("computed"), "n_frontier"].ge(1).all()
    assert summary.loc[summary.pair.eq("peak_vram_mb__ece_equal_mass_15"),
                       "paired_quantized_dominates_bf16"].eq(True).all()
    assert comparison.loc[comparison.status.eq("unavailable"), "changed"].isna().all()
    assert points.loc[points.pair.eq("peak_vram_mb__ece_equal_mass_15") & points.on_frontier,
                      "method"].eq("int8").all()

    full = pd.DataFrame(rows)
    full["energy_per_1k_joules"] = full["peak_vram_mb"] * 10
    full["power_available"] = True
    full["energy_stable"] = True
    _, all_summary, _ = frontiers(full)
    assert all_summary.status.eq("computed").all()


def test_derived_ratios_same_frozen_examples():
    base = {"base_run": "r", "model": "m", "seed": 42, "lang": "en", "loco": "fold",
            "eval_set": "id_test", "n_samples": 2, "sample_fingerprint": "abc",
            "aurc": .1, "ece_equal_mass_15": .1, "energy_per_1k_joules": np.nan,
            "power_available": False, "energy_stable": False}
    frame = pd.DataFrame([{**base, "method": "bf16", "peak_vram_mb": 2048},
                          {**base, "method": "int8", "artifact": "a", "peak_vram_mb": 1024,
                           "aurc": .12, "ece_equal_mass_15": .15}])
    result = derived_ratios(frame)
    assert len(result) == 1
    assert result.memory_saved_gb.iloc[0] == pytest.approx(1)
    assert result.delta_ece_per_gb_saved.iloc[0] == pytest.approx(.05)
    assert pd.isna(result.joules_saved_per_delta_ece.iloc[0])


def test_real_sources_are_joined_without_fabrication():
    directory = Path("outputs/results")
    if not (directory / "all_metrics.parquet").exists():
        pytest.skip("research artifacts absent")
    joined, sources = assemble(directory)
    assert len(joined) == 432
    assert sources["excluded_pilot_rows_without_efficiency"] == 6
    points, summary, comparisons, ratios, manifest = run_phase14(directory)
    assert manifest["status"] == "partial"
    assert len(summary) == 20
    assert len(points) > 0 and len(ratios) > 0 and len(comparisons) == 10
    assert summary.loc[summary.status.eq("computed"), "n_frontier"].ge(1).all()
    assert summary.loc[summary.pair.str.startswith("energy"), "status"].eq("unavailable").all()


def test_provenance_mismatch_is_rejected(tmp_path):
    source = Path("outputs/results")
    if not (source / "all_metrics.parquet").exists():
        pytest.skip("research artifacts absent")
    for name in ("all_metrics", "selective_metrics", "efficiency_metrics"):
        frame = pd.read_parquet(source / f"{name}.parquet")
        if name == "selective_metrics":
            frame.loc[0, "sample_fingerprint"] = "tampered"
        frame.to_parquet(tmp_path / f"{name}.parquet", index=False)
    with pytest.raises(ValueError, match="provenance"):
        assemble(tmp_path)
