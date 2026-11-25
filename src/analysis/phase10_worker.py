"""Isolated process for one Phase 10 artifact's CUDA measurement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.analysis.phase10 import measure_artifact
from src.utils.atomic import atomic_write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--warmup", type=int, required=True)
    parser.add_argument("--repeats", type=int, required=True)
    parser.add_argument("--idle-seconds", type=float, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    item = {"phase": args.manifest.parent.name,
            "artifact": manifest.get("artifact_name") or manifest["run_name"],
            "method": manifest.get("method") or "bf16", "manifest": manifest}
    row = measure_artifact(args.root, item, warmup=args.warmup,
                           repeats=args.repeats, idle_seconds=args.idle_seconds)
    atomic_write_json(args.result, row)


if __name__ == "__main__":
    main()
