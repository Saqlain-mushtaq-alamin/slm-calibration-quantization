"""Phase 14: provenance-checked efficiency versus trust frontiers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

PAIRS = (
    ("energy_per_1k_joules", "ece_equal_mass_15"),
    ("energy_per_1k_joules", "one_minus_f1"),
    ("peak_vram_mb", "ece_equal_mass_15"),
    ("peak_vram_mb", "aurc"),
    ("latency_p50_ms", "ece_equal_mass_15"),
)
EVAL_SETS = ("id_test", "ood_test")
LANGUAGES = ("bn", "en")
JOIN_KEYS = ["artifact", "method", "base_run", "model", "seed"]
EVAL_KEYS = JOIN_KEYS + ["lang", "loco", "eval_set", "n_samples", "sample_fingerprint"]


def pareto_frontier(points: np.ndarray, minimize: list[bool] | None = None) -> np.ndarray:
    """Non-dominated rows; exact ties remain on the frontier."""
    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[1] == 0 or not np.isfinite(values).all():
        raise ValueError("points must be a finite N x D array")
    if minimize is None:
        minimize = [True] * values.shape[1]
    if len(minimize) != values.shape[1]:
        raise ValueError("minimize length must match objective count")
    adjusted = values * np.where(minimize, 1, -1)
    return np.array([not np.any(np.all(adjusted <= row, axis=1) &
                                np.any(adjusted < row, axis=1)) for row in adjusted], dtype=bool)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assemble(results_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Join only exact artifact/method/evaluation matches; fail on ambiguity."""
    files = {name: results_dir / f"{name}.parquet" for name in
             ("all_metrics", "selective_metrics", "efficiency_metrics")}
    tables = {name: pd.read_parquet(path) for name, path in files.items()}
    task, selective, efficiency = (tables[name] for name in files)
    for name, table, keys in (("task", task, EVAL_KEYS), ("selective", selective, EVAL_KEYS),
                              ("efficiency", efficiency, JOIN_KEYS)):
        if table.duplicated(keys).any():
            raise ValueError(f"Duplicate {name} join keys")
    sel = selective[EVAL_KEYS + ["aurc"]]
    joined = task.merge(sel, on=EVAL_KEYS, how="left", validate="one_to_one", indicator="_selective_join")
    if not joined["_selective_join"].eq("both").all() or joined["aurc"].isna().any():
        raise ValueError("Task/selective provenance mismatch")
    joined = joined.drop(columns="_selective_join")
    eff_columns = JOIN_KEYS + ["language", "backend", "status", "source_status",
                               "latency_stable", "thermal_stable", "power_available",
                               "energy_stable", "peak_vram_mb", "vram_measurement",
                               "latency_p50_ms", "energy_per_1k_joules"]
    joined = joined.merge(efficiency[eff_columns], on=JOIN_KEYS, how="left",
                          validate="many_to_one", indicator="_efficiency_join")
    unmatched = joined.loc[~joined["_efficiency_join"].eq("both")]
    if not unmatched.empty and not unmatched.phase.eq("pilot").all():
        raise ValueError("Non-pilot task/efficiency artifact mismatch")
    joined = joined.loc[joined["_efficiency_join"].eq("both")].copy()
    if not joined["lang"].eq(joined["language"]).all():
        raise ValueError("Language mismatch between task and efficiency")
    joined = joined.drop(columns="_efficiency_join")
    joined["one_minus_f1"] = 1 - joined["macro_f1"]
    if not set(joined.eval_set) >= set(EVAL_SETS):
        raise ValueError("ID/OOD evaluation sets missing")
    sources = {name: {"path": str(path), "sha256": file_sha256(path),
                      "rows": len(tables[name])} for name, path in files.items()}
    sources["excluded_pilot_rows_without_efficiency"] = len(unmatched)
    return joined, sources


def eligible(frame: pd.DataFrame, efficiency_metric: str, quality_metric: str) -> tuple[pd.DataFrame, str | None]:
    base = frame.copy()
    for metric in (efficiency_metric, quality_metric):
        base = base.loc[np.isfinite(pd.to_numeric(base[metric], errors="coerce"))]
    base = base.loc[base[efficiency_metric] > 0]
    if efficiency_metric == "energy_per_1k_joules":
        base = base.loc[base.power_available.eq(True) & base.energy_stable.eq(True) &
                        base.thermal_stable.eq(True)]
        reason = "No valid measured energy telemetry" if base.empty else None
    elif efficiency_metric == "latency_p50_ms":
        base = base.loc[base.latency_stable.eq(True)]
        reason = "No stable measured latency" if base.empty else None
    else:
        base = base.loc[base.vram_measurement.notna()]
        reason = "No measured GPU memory" if base.empty else None
    return base, reason


def frontiers(joined: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    point_rows, summary_rows = [], []
    for lang in LANGUAGES:
        for eval_set in EVAL_SETS:
            group = joined.loc[(joined.lang == lang) & (joined.eval_set == eval_set)]
            for x, y in PAIRS:
                usable, reason = eligible(group, x, y)
                pair_id = f"{x}__{y}"
                if usable.empty:
                    summary_rows.append({"lang": lang, "eval_set": eval_set, "pair": pair_id,
                                         "status": "unavailable", "reason": reason,
                                         "n_candidates": len(group), "n_eligible": 0, "n_frontier": 0,
                                         "bf16_dominated": None})
                    continue
                mask = pareto_frontier(usable[[x, y]].to_numpy(), [True, True])
                for (_, row), is_frontier in zip(usable.iterrows(), mask):
                    point_rows.append({"lang": lang, "eval_set": eval_set, "pair": pair_id,
                                       "artifact": row.artifact, "base_run": row.base_run,
                                       "model": row.model, "seed": int(row.seed), "method": row.method,
                                       "backend": row.backend, "loco": row.loco,
                                       "sample_fingerprint": row.sample_fingerprint,
                                       "x_metric": x, "x_value": float(row[x]),
                                       "y_metric": y, "y_value": float(row[y]),
                                       "on_frontier": bool(is_frontier)})
                bf16 = usable.method.eq("bf16").to_numpy()
                paired_quantized_dominates = False
                for _, reference in usable.loc[usable.method == "bf16"].iterrows():
                    candidates = usable.loc[(usable.base_run == reference.base_run) &
                                            (usable.method != "bf16")]
                    if any((candidate[x] <= reference[x] and candidate[y] <= reference[y] and
                            (candidate[x] < reference[x] or candidate[y] < reference[y]))
                           for _, candidate in candidates.iterrows()):
                        paired_quantized_dominates = True
                        break
                summary_rows.append({"lang": lang, "eval_set": eval_set, "pair": pair_id,
                                     "status": "computed", "reason": None,
                                     "n_candidates": len(group), "n_eligible": len(usable),
                                     "n_frontier": int(mask.sum()),
                                     "bf16_dominated": bool(np.any(bf16 & ~mask)) if bf16.any() else None,
                                     "paired_quantized_dominates_bf16": paired_quantized_dominates})
    points = pd.DataFrame(point_rows)
    summary = pd.DataFrame(summary_rows)
    comparisons = []
    for lang in LANGUAGES:
        for x, y in PAIRS:
            pair = f"{x}__{y}"
            a = points.loc[(points.lang == lang) & (points.eval_set == "id_test") & (points.pair == pair) & points.on_frontier]
            b = points.loc[(points.lang == lang) & (points.eval_set == "ood_test") & (points.pair == pair) & points.on_frontier]
            a = set(zip(a.artifact, a.method)) if not a.empty else set()
            b = set(zip(b.artifact, b.method)) if not b.empty else set()
            comparisons.append({"lang": lang, "pair": pair,
                                "status": "computed" if a and b else "unavailable",
                                "changed": bool(a != b) if a and b else None,
                                "id_only": sorted(f"{p[0]}::{p[1]}" for p in a - b) if a and b else None,
                                "ood_only": sorted(f"{p[0]}::{p[1]}" for p in b - a) if a and b else None})
    return points, summary, pd.DataFrame(comparisons)


def derived_ratios(joined: pd.DataFrame) -> pd.DataFrame:
    """Paired descriptive differences versus BF16; undefined ratios are null."""
    keys = ["base_run", "model", "seed", "lang", "loco", "eval_set", "n_samples", "sample_fingerprint"]
    bf = joined.loc[joined.method == "bf16", keys +
                    ["ece_equal_mass_15", "aurc", "peak_vram_mb", "energy_per_1k_joules",
                     "power_available", "energy_stable", "thermal_stable"]]
    variant = joined.loc[joined.method != "bf16"]
    paired = variant.merge(bf, on=keys, suffixes=("", "_bf16"), validate="many_to_one")
    rows = []
    for _, row in paired.iterrows():
        memory_saved_gb = (row.peak_vram_mb_bf16 - row.peak_vram_mb) / 1024
        compression_pct = 100 * (1 - row.peak_vram_mb / row.peak_vram_mb_bf16)
        delta_ece = row.ece_equal_mass_15 - row.ece_equal_mass_15_bf16
        delta_aurc = row.aurc - row.aurc_bf16
        energy_ok = (pd.notna(row.energy_per_1k_joules) and pd.notna(row.energy_per_1k_joules_bf16)
                     and row.energy_per_1k_joules > 0 and row.energy_per_1k_joules_bf16 > 0
                     and bool(row.power_available) and bool(row.power_available_bf16)
                     and bool(row.energy_stable) and bool(row.energy_stable_bf16)
                     and bool(row.thermal_stable) and bool(row.thermal_stable_bf16))
        rows.append({"artifact": row.artifact, "base_run": row.base_run, "model": row.model,
                     "seed": int(row.seed), "lang": row.lang, "loco": row.loco,
                     "eval_set": row.eval_set, "method": row.method,
                     "sample_fingerprint": row.sample_fingerprint,
                     "memory_saved_gb": float(memory_saved_gb),
                     "compression_pct": float(compression_pct),
                     "delta_ece": float(delta_ece), "delta_aurc": float(delta_aurc),
                     "delta_ece_per_gb_saved": float(delta_ece / memory_saved_gb) if memory_saved_gb > 0 else np.nan,
                     "delta_aurc_per_pct_compression": float(delta_aurc / compression_pct) if compression_pct > 0 else np.nan,
                     "joules_saved_per_delta_ece": float((row.energy_per_1k_joules_bf16 - row.energy_per_1k_joules) / delta_ece)
                     if energy_ok and delta_ece != 0 else np.nan})
    return pd.DataFrame(rows)


def run_phase14(results_dir: Path):
    joined, sources = assemble(results_dir)
    points, summary, comparisons = frontiers(joined)
    ratios = derived_ratios(joined.loc[joined.eval_set.isin(EVAL_SETS)])
    phase_statuses = {}
    for phase in (8, 9, 10):
        path = results_dir / f"phase{phase}_run.json"
        phase_statuses[f"phase{phase}"] = json.loads(path.read_text())["status"] if path.exists() else "missing"
    complete = bool(summary.status.eq("computed").all() and not ratios.empty and
                    ratios.joules_saved_per_delta_ece.notna().any() and
                    phase_statuses == {"phase8": "success", "phase9": "success", "phase10": "success"})
    manifest = {"phase": 14, "status": "complete" if complete else "partial",
                "sources": sources, "n_joined": len(joined), "n_frontiers_expected": len(PAIRS) * len(EVAL_SETS) * len(LANGUAGES),
                "n_frontiers_computed": int(summary.status.eq("computed").sum()),
                "n_ratio_rows": len(ratios),
                "source_phase_statuses": phase_statuses,
                "energy_limitation": "All energy frontiers unavailable: Phase 10 has no measured GPU power telemetry" if not joined.energy_per_1k_joules.notna().any() else None,
                "measurement_note": "Memory uses measured peak VRAM; latency frontiers require stable Phase 10 latency. Phase 10 overall status is incomplete.",
                "single_class_ood_note": "Bengali OOD ECE reflects a single-class target and is not directly comparable to conventional two-class ECE."}
    return points, summary, comparisons, ratios, manifest
