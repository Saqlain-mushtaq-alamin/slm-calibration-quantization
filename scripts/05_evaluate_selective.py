"""Compute Phase 9 selective metrics and full risk-coverage curves."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.phase9 import evaluate_selective_prediction
from src.utils.logging_utils import write_run_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logits-dir", type=Path, default=ROOT / "outputs/logits")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "outputs/results")
    args = parser.parse_args()
    results_dir = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    try:
        results = evaluate_selective_prediction(args.logits_dir, results_dir / "rc_curves")
        results.to_csv(results_dir / "selective_metrics.csv", index=False, na_rep="")
        results.to_parquet(results_dir / "selective_metrics.parquet", index=False)
        write_run_json(results_dir / "phase9_run.json", phase=9,
                       phase_name="selective_prediction_evaluation", status="success",
                       payload={"logits_dir": str(args.logits_dir), "n_logit_files": len(results),
                                "n_curves": len(results), "excluded": "smoke runs",
                                "nonmonotonic_accuracy_rows": int((~results["selective_accuracy_monotonic_70_90"]).sum()),
                                "undefined_failure_auroc_rows": int((~results["failure_auroc_defined"]).sum())})
        print(f"Phase 9 complete: {len(results)} selective rows and curves -> {results_dir}")
        return 0
    except Exception as exc:
        write_run_json(results_dir / "phase9_run.json", phase=9,
                       phase_name="selective_prediction_evaluation", status="failed",
                       payload={"error": f"{type(exc).__name__}: {exc}"})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
