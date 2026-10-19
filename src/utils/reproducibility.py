"""Reproducibility helpers: seeding and environment capture."""

from __future__ import annotations

import os
import random
import subprocess
from typing import Any


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Seed python, numpy, and torch RNGs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_git_commit(repo_root: str | None = None) -> str | None:
    """Return current git commit hash, or None if unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=repo_root,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (OSError, FileNotFoundError):
        pass
    return None


def collect_env_snapshot() -> dict[str, Any]:
    """Collect lightweight environment info for run manifests."""
    info: dict[str, Any] = {
        "python_version": __import__("platform").python_version(),
        "platform": __import__("platform").platform(),
        "executable": __import__("sys").executable,
    }
    try:
        import torch

        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["gpu_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        info["torch_version"] = "NOT INSTALLED"
    return info
