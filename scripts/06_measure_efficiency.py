"""Benchmark Phase 10 artifact efficiency with live model inference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.phase10 import discover_artifacts, run_sweep
from src.metrics.efficiency import hardware_summary
from src.utils.logging_utils import write_run_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "outputs/results")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--idle-seconds", type=float, default=60.0)
    parser.add_argument("--limit", type=int, help="Rehearsal: first N artifacts only")
    parser.add_argument("--force", action="store_true", help="Rebenchmark existing per-artifact records")
    args = parser.parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    total = len(discover_artifacts(ROOT))
    try:
        table = run_sweep(ROOT, args.results_dir, warmup=args.warmup,
                          repeats=args.repeats, idle_seconds=args.idle_seconds,
                          limit=args.limit, force=args.force)
        export = table.drop(columns=["block_details"], errors="ignore")
        export.to_csv(args.results_dir / "efficiency_metrics.csv", index=False, na_rep="")
        export.to_parquet(args.results_dir / "efficiency_metrics.parquet", index=False)
        measured = table["status"].isin(["complete", "incomplete"])
        gates = {
            "T10.1_latency_cv": bool(measured.all() and table.loc[measured, "latency_stable"].all()),
            "T10.3_energy_stability": bool(measured.all() and table.loc[measured, "energy_stable"].all()),
            "T10.4_idle_power": bool(measured.all() and table.loc[measured, "idle_power_watts"].notna().all()),
            "T10.5_thermal_stability": bool(measured.all() and table.loc[measured, "thermal_stable"].all()),
            "T10.6_all_artifacts": bool(len(table) == total and measured.all()),
        }
        status = "success" if all(gates.values()) else "incomplete"
        write_run_json(args.results_dir / "phase10_run.json", phase=10,
                       phase_name="efficiency_measurement", status=status,
                       payload={"n_artifacts_expected": total, "n_rows": len(table),
                                "n_measured": int(measured.sum()), "gates": gates,
                                "hardware": hardware_summary(),
                                "protocol": {"warmup": args.warmup, "repeats_per_block": args.repeats,
                                             "blocks": 2, "idle_seconds": args.idle_seconds}})
        print(f"Phase 10 {status}: {int(measured.sum())}/{total} measured; see {args.results_dir}")
        return 0 if status == "success" else 2
    except Exception as exc:
        write_run_json(args.results_dir / "phase10_run.json", phase=10,
                       phase_name="efficiency_measurement", status="failed",
                       payload={"error": f"{type(exc).__name__}: {exc}"})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
