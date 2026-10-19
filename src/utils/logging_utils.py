"""Structured logging and run.json manifests."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure root logger for console output."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    return logging.getLogger("slm-calib")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_run_json(
    path: str | Path,
    *,
    phase: int,
    phase_name: str,
    status: str,
    payload: dict[str, Any] | None = None,
    seed: int | None = None,
    config_hash: str | None = None,
    git_commit: str | None = None,
) -> dict[str, Any]:
    """Write a run.json manifest and return the dict."""
    run: dict[str, Any] = {
        "phase": phase,
        "phase_name": phase_name,
        "status": status,
        "timestamp_utc": utc_now_iso(),
        "seed": seed,
        "config_hash": config_hash,
        "git_commit": git_commit,
    }
    if payload:
        run.update(payload)

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(run, indent=2, default=str), encoding="utf-8")
    return run
