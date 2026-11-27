"""Run independent, isolated recovery and quantization rehearsals for both encoders."""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from huggingface_hub import try_to_load_from_cache

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.encoder import load_config
from src.utils.atomic import atomic_write_json


def run_one(name: str, allow_download: bool):
    cfg = load_config(f"configs/encoder_{name}.yaml")
    output = Path("outputs/encoders/smoke") / name
    output.mkdir(parents=True, exist_ok=True)
    if not allow_download:
        files = ("config.json", "tokenizer_config.json")
        weights = ("model.safetensors", "pytorch_model.bin")
        if not all(isinstance(try_to_load_from_cache(cfg["model_id"], f), str) for f in files) or not any(
                isinstance(try_to_load_from_cache(cfg["model_id"], f), str) for f in weights):
            result = {"phase": 16, "model": cfg["model_id"], "model_key": name,
                      "fp32": "unavailable", "int8": "unavailable",
                      "reason": "Model files not cached; rerun with --allow-download when Hugging Face is reachable",
                      "smoke_only": True}
            atomic_write_json(output / "availability.json", result)
            return result
    spec = importlib.util.spec_from_file_location("encoder_train", "scripts/10_train_encoders.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        first = module.train(cfg, smoke=True)
        second = module.train(cfg, smoke=True, explicit_resume=True)
        result = {"phase": 16, "model": cfg["model_id"], "model_key": name,
                  "fp32": "passed", "int8": "passed", "first_step": first["global_step"],
                  "resumed_step": second["global_step"], "quantized_modules": second["quantized_modules"],
                  "smoke_only": True}
        if result["resumed_step"] <= result["first_step"]:
            raise RuntimeError("Global step did not increase after resume")
    except Exception as exc:
        result = {"phase": 16, "model": cfg["model_id"], "model_key": name,
                  "fp32": "failed", "int8": "failed", "error": repr(exc), "smoke_only": True}
    atomic_write_json(output / "availability.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-download", action="store_true")
    ns = parser.parse_args()
    results = [run_one(name, ns.allow_download) for name in ("banglabert", "xlmr")]
    print(json.dumps(results, indent=2))
    if any(r["fp32"] == "failed" for r in results):
        raise SystemExit(1)
