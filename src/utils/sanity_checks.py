"""Automated sanity checks used across phases."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""
    extras: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "detail": self.detail,
            **self.extras,
        }


def check_gpu_min_vram(min_vram_gb: float = 7.5) -> CheckResult:
    """Verify CUDA GPU is present with at least min_vram_gb VRAM."""
    try:
        import torch
    except ImportError:
        return CheckResult("gpu_vram", False, "torch not installed")
    if not torch.cuda.is_available():
        return CheckResult("gpu_vram", False, "CUDA not available")
    props = torch.cuda.get_device_properties(0)
    total_bytes = getattr(props, "total_memory", None) or props.total_mem
    vram_gb = round(total_bytes / 1e9, 2)
    name = torch.cuda.get_device_name(0)
    passed = vram_gb >= min_vram_gb
    return CheckResult(
        "gpu_vram",
        passed,
        f"{name}: {vram_gb} GB (need >= {min_vram_gb} GB)",
        {"gpu_name": name, "vram_gb": vram_gb},
    )


def check_required_imports(modules: list[str]) -> CheckResult:
    """Verify a list of import names are importable."""
    missing = []
    versions: dict[str, str] = {}
    import importlib

    for mod in modules:
        try:
            m = importlib.import_module(mod)
            versions[mod] = getattr(m, "__version__", "unknown")
        except ImportError:
            missing.append(mod)
    passed = not missing
    detail = "all present" if passed else f"missing: {missing}"
    return CheckResult("required_imports", passed, detail, {"versions": versions})
