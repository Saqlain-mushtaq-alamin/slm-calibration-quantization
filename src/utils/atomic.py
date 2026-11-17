"""Atomic file write helpers and resume / checkpoint validation utilities."""

from __future__ import annotations

import json
import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

LOG = logging.getLogger("slm-calib.atomic")


def atomic_write_text(path: str | Path, content: str, encoding: str = "utf-8") -> Path:
    """Atomically write text to path using a temp file in the same directory, flush, fsync, and replace."""
    dest = Path(path).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest.parent / f".{dest.name}.tmp.{os.getpid()}_{uuid.uuid4().hex[:8]}"
    try:
        with open(tmp_path, "w", encoding=encoding) as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, dest)
        return dest
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def atomic_write_json(path: str | Path, data: Any, indent: int = 2, default: Any = str) -> Path:
    """Atomically serialize data to JSON at path."""
    content = json.dumps(data, indent=indent, default=default)
    return atomic_write_text(path, content, encoding="utf-8")


def atomic_write_parquet(path: str | Path, df: pd.DataFrame, **kwargs) -> Path:
    """Atomically write a DataFrame to Parquet using a temp file, flush, fsync, and replace."""
    dest = Path(path).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest.parent / f".{dest.name}.tmp.{os.getpid()}_{uuid.uuid4().hex[:8]}"
    try:
        df.to_parquet(tmp_path, index=False, **kwargs)
        # Ensure data is flushed to disk
        with open(tmp_path, "a+b") as f:
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, dest)
        return dest
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def atomic_write_csv(path: str | Path, df: pd.DataFrame, index: bool = False, **kwargs) -> Path:
    """Atomically write a DataFrame to CSV using a temp file, flush, fsync, and replace."""
    dest = Path(path).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest.parent / f".{dest.name}.tmp.{os.getpid()}_{uuid.uuid4().hex[:8]}"
    try:
        df.to_csv(tmp_path, index=index, **kwargs)
        with open(tmp_path, "a+b") as f:
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, dest)
        return dest
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def is_valid_parquet(
    path: str | Path,
    expected_rows: int | None = None,
    required_columns: Sequence[str] | None = None,
) -> bool:
    """Check if a parquet file exists, is non-empty, and can be read with valid schema and row count."""
    p = Path(path)
    if not p.is_file() or p.stat().st_size == 0:
        return False
    try:
        df = pd.read_parquet(p)
        if expected_rows is not None and len(df) != expected_rows:
            LOG.warning("Parquet row count mismatch for %s: expected %d, got %d", p.name, expected_rows, len(df))
            return False
        if required_columns is not None:
            missing = set(required_columns) - set(df.columns)
            if missing:
                LOG.warning("Parquet missing columns in %s: %s", p.name, missing)
                return False
        return True
    except Exception as exc:
        LOG.warning("Parquet corrupt or unreadable at %s: %s", p, exc)
        return False


def is_valid_checkpoint(checkpoint_dir: str | Path) -> bool:
    """Verify that a Trainer checkpoint directory contains complete, non-empty state files."""
    cdir = Path(checkpoint_dir)
    if not cdir.is_dir():
        return False

    # Check for trainer_state.json and valid json content
    state_file = cdir / "trainer_state.json"
    if not state_file.is_file() or state_file.stat().st_size == 0:
        return False
    try:
        data = json.loads(state_file.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or "global_step" not in data:
            return False
    except Exception:
        return False

    # Check for adapter weights
    has_weights = (
        (cdir / "adapter_model.safetensors").is_file()
        and (cdir / "adapter_model.safetensors").stat().st_size > 0
    ) or (
        (cdir / "adapter_model.bin").is_file()
        and (cdir / "adapter_model.bin").stat().st_size > 0
    )
    if not has_weights:
        return False

    # Check for adapter config
    cfg_file = cdir / "adapter_config.json"
    if not cfg_file.is_file() or cfg_file.stat().st_size == 0:
        return False

    # Check for optimizer state or training_args
    opt_file = cdir / "optimizer.pt"
    args_file = cdir / "training_args.bin"
    if not (
        (opt_file.is_file() and opt_file.stat().st_size > 0)
        or (args_file.is_file() and args_file.stat().st_size > 0)
    ):
        return False

    return True


def find_last_valid_checkpoint(checkpoints_dir: str | Path) -> Path | None:
    """Find the latest valid checkpoint, skipping corrupted or half-written ones."""
    cdir = Path(checkpoints_dir)
    if not cdir.is_dir():
        return None

    candidate_dirs: list[tuple[int, Path]] = []
    for sub in cdir.iterdir():
        if sub.is_dir() and sub.name.startswith("checkpoint-"):
            try:
                step = int(sub.name.split("-")[1])
                candidate_dirs.append((step, sub))
            except (IndexError, ValueError):
                continue

    candidate_dirs.sort(key=lambda x: x[0], reverse=True)
    for step, p in candidate_dirs:
        if is_valid_checkpoint(p):
            return p
        LOG.warning("Skipping corrupted or incomplete checkpoint at %s (step %d)", p, step)

    return None


def is_valid_adapter(adapter_dir: str | Path) -> bool:
    """Check if a final LoRA adapter directory is complete."""
    adir = Path(adapter_dir)
    if not adir.is_dir():
        return False
    has_weights = (
        (adir / "adapter_model.safetensors").is_file()
        and (adir / "adapter_model.safetensors").stat().st_size > 0
    ) or (
        (adir / "adapter_model.bin").is_file()
        and (adir / "adapter_model.bin").stat().st_size > 0
    )
    has_cfg = (adir / "adapter_config.json").is_file() and (adir / "adapter_config.json").stat().st_size > 0
    return has_weights and has_cfg


def is_valid_merged(merged_dir: str | Path) -> bool:
    """Check if a merged model directory is complete and has its completion sentinel."""
    mdir = Path(merged_dir)
    if not mdir.is_dir():
        return False
    sentinel = mdir / ".completed"
    if not sentinel.is_file():
        return False
    has_cfg = (mdir / "config.json").is_file() and (mdir / "config.json").stat().st_size > 0
    has_weights = (
        (mdir / "model.safetensors").is_file() and (mdir / "model.safetensors").stat().st_size > 0
    ) or (
        (mdir / "model.safetensors.index.json").is_file() and (mdir / "model.safetensors.index.json").stat().st_size > 0
    )
    return has_cfg and has_weights
