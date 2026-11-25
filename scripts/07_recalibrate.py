"""Fit and evaluate Phase 11 temperature scaling from saved logits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.phase11 import run_recalibration_experiment
from src.utils.logging_utils import write_run_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logits-dir", type=Path, default=ROOT / "outputs/logits")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "outputs/results")
    args = parser.parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    try:
        results, temperatures, summary = run_recalibration_experiment(args.logits_dir)
        results.to_csv(args.results_dir / "recalibration_results.csv", index=False)
        results.to_parquet(args.results_dir / "recalibration_results.parquet", index=False)
        temperatures.to_csv(args.results_dir / "recalibration_temperatures.csv", index=False)
        temperatures.to_parquet(args.results_dir / "recalibration_temperatures.parquet", index=False)
        write_run_json(args.results_dir / "phase11_run.json", phase=11,
                       phase_name="post_hoc_recalibration", status="success",
                       payload={"logits_dir": str(args.logits_dir), **summary})
        print(f"Phase 11 complete: {summary['n_artifacts']} artifacts, {len(results)} results, "
              f"{summary['n_oracle_artifacts']} oracle fits -> {args.results_dir}")
        return 0
    except Exception as exc:
        write_run_json(args.results_dir / "phase11_run.json", phase=11,
                       phase_name="post_hoc_recalibration", status="failed",
                       payload={"error": f"{type(exc).__name__}: {exc}"})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
