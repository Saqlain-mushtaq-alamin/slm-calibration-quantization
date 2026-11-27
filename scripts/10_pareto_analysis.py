"""Run CPU-only Phase 14 Pareto analysis from saved Phase 8–10 artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.pareto import run_phase14
from src.utils.atomic import atomic_write_csv, atomic_write_json, atomic_write_parquet


def write_report(path: Path, summary, comparisons, ratios, manifest):
    lines = ["# Phase 14 Pareto analysis", "",
             f"Status: **{manifest['status'].upper()}**. Computed {manifest['n_frontiers_computed']} of {manifest['n_frontiers_expected']} required frontiers.",
             "", "All values below were computed from the saved source tables recorded by SHA-256 in `phase14_run.json`.",
             "This run used no GPU inference or hand-entered result values.", "",
             "## Frontier coverage", "",
             "| Language | Evaluation | Objectives | Status | Eligible | Frontier | Paired quantized dominates BF16 |",
             "|---|---|---|---|---:|---:|---|"]
    for _, row in summary.iterrows():
        paired = row.get("paired_quantized_dominates_bf16")
        paired_text = "unknown" if paired is None or str(paired) == "nan" else str(bool(paired)).lower()
        lines.append(f"| {row.lang} | {row.eval_set} | {row.pair} | {row.status} | {row.n_eligible} | {row.n_frontier} | {paired_text} |")
    lines += ["", "## ID versus OOD frontier membership", "",
              "| Language | Objectives | Status | Changed |", "|---|---|---|---|"]
    for _, row in comparisons.iterrows():
        changed = "undefined" if row.changed is None else str(bool(row.changed)).lower()
        lines.append(f"| {row.lang} | {row.pair} | {row.status} | {changed} |")
    lines += ["", "## Derived ratios", "",
              (f"Paired BF16 comparisons: {len(ratios)}. Defined ΔECE per GB saved: {ratios.delta_ece_per_gb_saved.notna().sum()}. "
               f"Defined ΔAURC per percentage point of compression: {ratios.delta_aurc_per_pct_compression.notna().sum()}. "
               f"Defined joules saved per ΔECE: {ratios.joules_saved_per_delta_ece.notna().sum()}."), "",
              ("`memory_saved_gb = (BF16 peak VRAM MB − variant peak VRAM MB) / 1024`; "
               "`compression_pct = 100 × (1 − variant VRAM / BF16 VRAM)`. "
               "ΔECE and ΔAURC are variant minus paired BF16. Nonpositive memory savings leave the first two ratios undefined."), "",
              "## Limits", "",
              ((manifest["energy_limitation"] or "Energy pairs use measured power, energy-stable, and thermally stable rows.").rstrip(".") + "." +
               f" Phase 10 status: {manifest['source_phase_statuses']['phase10']}. "
               "Latency frontiers use only records marked stable; memory frontiers use measured VRAM even where other efficiency gates failed."),
              "Bengali OOD is single-class. Its ECE is reported as an observed error against that target prior and is not directly comparable to ordinary two-class OOD ECE.",
              "Global frontier membership compares model configurations and seeds; paired dominance checks compare precision methods within the same base run.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "outputs/results")
    args = parser.parse_args()
    points, summary, comparisons, ratios, manifest = run_phase14(args.results_dir)
    directory = args.results_dir
    atomic_write_csv(directory / "phase14_frontier_points.csv", points)
    atomic_write_parquet(directory / "phase14_frontier_points.parquet", points)
    atomic_write_csv(directory / "phase14_frontier_summary.csv", summary)
    atomic_write_csv(directory / "phase14_id_ood_comparison.csv", comparisons)
    atomic_write_csv(directory / "phase14_derived_ratios.csv", ratios)
    atomic_write_parquet(directory / "phase14_derived_ratios.parquet", ratios)
    atomic_write_json(directory / "phase14_run.json", manifest)
    write_report(directory / "phase14_report.md", summary, comparisons, ratios, manifest)
    print(f"Phase 14 {manifest['status']}: {manifest['n_frontiers_computed']}/{manifest['n_frontiers_expected']} frontiers; "
          f"{len(ratios)} paired ratio rows")
    return 0 if manifest["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
