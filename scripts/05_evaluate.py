"""Compute Phase 8 task and calibration tables from saved full-run logits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.phase8 import compute_deltas_vs_fp, evaluate_all_artifacts
from src.utils.logging_utils import write_run_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logits-dir", type=Path, default=ROOT / "outputs/logits")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "outputs/results")
    args = parser.parse_args()
    result_dir = args.results_dir
    result_dir.mkdir(parents=True, exist_ok=True)
    try:
        results = evaluate_all_artifacts(args.logits_dir)
        deltas = compute_deltas_vs_fp(results)
        results.to_csv(result_dir / "all_metrics.csv", index=False, na_rep="")
        results.to_parquet(result_dir / "all_metrics.parquet", index=False)
        deltas.to_csv(result_dir / "delta_vs_fp.csv", index=False)
        deltas.to_parquet(result_dir / "delta_vs_fp.parquet", index=False)
        write_run_json(result_dir / "phase8_run.json", phase=8,
                       phase_name="evaluation_task_performance_calibration", status="success",
                       payload={"logits_dir": str(args.logits_dir), "n_logit_files": len(results),
                                "n_deltas": len(deltas), "excluded": "smoke runs",
                                "single_class_rows": int(results["single_class_eval_set"].sum())})
        print(f"Phase 8 complete: {len(results)} metric rows, {len(deltas)} deltas -> {result_dir}")
        return 0
    except Exception as exc:
        write_run_json(result_dir / "phase8_run.json", phase=8,
                       phase_name="evaluation_task_performance_calibration", status="failed",
                       payload={"error": f"{type(exc).__name__}: {exc}"})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
