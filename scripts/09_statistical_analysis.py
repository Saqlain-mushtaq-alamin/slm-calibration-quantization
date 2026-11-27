"""Run Phase 13 statistics from immutable Phase 8–12 result tables."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.phase13 import run_phase13  # noqa: E402
from src.utils.atomic import atomic_write_csv, atomic_write_parquet  # noqa: E402
from src.utils.logging_utils import write_run_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "outputs/results")
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    phase12_manifest = args.results_dir / "phase12_run.json"
    if not phase12_manifest.is_file():
        raise FileNotFoundError(f"Phase 12 manifest missing: {phase12_manifest}")
    source_phase12_status = json.loads(phase12_manifest.read_text(encoding="utf-8")).get("status")
    if source_phase12_status not in {"partial", "complete"}:
        raise ValueError(f"Phase 12 results are not ready for analysis: {source_phase12_status}")
    table, interactions, summary = run_phase13(args.results_dir, n_boot=args.n_boot, seed=args.seed)
    atomic_write_csv(args.results_dir / "hypothesis_tests.csv", table)
    atomic_write_parquet(args.results_dir / "hypothesis_tests.parquet", table)
    atomic_write_csv(args.results_dir / "phase13_interactions.csv", interactions)
    write_run_json(args.results_dir / "phase13_run.json", phase=13,
                   phase_name="statistical_analysis_hypothesis_testing", status="partial",
                   seed=args.seed, payload={**summary, "n_boot": args.n_boot,
                                            "source_phase12_status": source_phase12_status})
    print(f"Phase 13 partial: {len(table)} contrasts, {int(interactions.converged.sum())}/2 mixed models converged")
    print(args.results_dir / "hypothesis_tests.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
