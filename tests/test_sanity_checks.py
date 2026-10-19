"""Sanity-check unit tests (Phase 0)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.sanity_checks import (  # noqa: E402
    check_gpu_min_vram,
    check_required_imports,
)


def test_gpu_min_vram_check_runs():
    result = check_gpu_min_vram(min_vram_gb=0.0)
    assert result.name == "gpu_vram"
    # On this lab machine CUDA should be available; if not, check must fail cleanly.
    if result.passed:
        assert result.extras["vram_gb"] >= 0.0
    else:
        assert "CUDA" in result.detail or "torch" in result.detail


def test_required_imports_core():
    result = check_required_imports(["json", "pathlib", "math"])
    assert result.passed, result.detail


def test_required_imports_missing_module():
    result = check_required_imports(["definitely_not_a_real_module_xyz"])
    assert not result.passed
    assert "definitely_not_a_real_module_xyz" in result.detail
