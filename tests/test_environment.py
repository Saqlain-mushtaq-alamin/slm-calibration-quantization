"""Phase 0 environment verification tests (thin wrappers around the script)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_SCRIPT = ROOT / "scripts" / "00_verify_environment.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("verify_env", _SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["verify_env"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def env_mod():
    return _load_script()


def test_verify_hardware(env_mod):
    info = env_mod.verify_hardware()
    assert info["vram_gb"] >= env_mod.MIN_VRAM_GB
    assert info["cuda_available"] if "cuda_available" in info else True
    assert "torch_version" in info


def test_pin_versions_writes_manifest(env_mod, tmp_path, monkeypatch):
    # Redirect ROOT outputs by temporarily pointing write path via env_mod.ROOT
    monkeypatch.setattr(env_mod, "ROOT", tmp_path)
    versions = env_mod.pin_versions()
    assert "torch" in versions
    manifest = tmp_path / "outputs" / "environment_versions.json"
    assert manifest.exists()
    import json

    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert "versions" in data


def test_smoke_test_energy(env_mod):
    detail = env_mod.smoke_test_energy()
    assert detail["power_mw"] > 0
    assert detail["power_w"] > 0


def test_pip_check_runs(env_mod):
    result = env_mod.run_pip_check()
    assert "passed" in result
    assert "returncode" in result
