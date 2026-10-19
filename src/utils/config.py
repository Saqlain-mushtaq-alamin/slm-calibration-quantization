"""Config loading helpers (YAML via omegaconf / hydra)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


def load_config(path: str | Path, overrides: list[str] | None = None) -> DictConfig:
    """Load a YAML config file with optional dotlist overrides."""
    cfg = OmegaConf.load(str(path))
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return cfg


def config_hash(cfg: DictConfig | dict[str, Any]) -> str:
    """Stable hash of a config object for run manifests."""
    import hashlib
    import json

    container = OmegaConf.to_container(cfg, resolve=True) if isinstance(cfg, DictConfig) else cfg
    payload = json.dumps(container, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
