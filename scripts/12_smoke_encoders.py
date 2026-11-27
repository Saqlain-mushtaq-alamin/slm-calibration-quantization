"""Run independent, isolated recovery and quantization rehearsals for both encoders."""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import try_to_load_from_cache

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.encoder import load_config, load_frozen
from src.utils.atomic import atomic_write_json


def verify_outputs(cfg: dict, output: Path):
    frames, _ = load_frozen(cfg)
    ids = [set(frame.example_id.astype(str)) for frame in frames.values()]
    if any(a & b for i, a in enumerate(ids) for b in ids[i + 1:]):
        raise ValueError("Frozen split leakage")
    for precision in ("fp32", "int8"):
        for part in ("id_val", "id_test", "ood_test"):
            pred_path = output / "predictions" / f"{precision}_{part}.csv"
            metric_path = output / "metrics" / f"{precision}_{part}.json"
            pred = pd.read_csv(pred_path)
            expected = frames[part].iloc[:8]
            if pred.example_id.astype(str).tolist() != expected.example_id.astype(str).tolist() or not np.array_equal(pred.y_true, expected.label):
                raise ValueError(f"Smoke alignment failed: {precision} {part}")
            if not np.isfinite(pred[["logit_nonhate", "logit_hate", "prob_nonhate", "prob_hate"]]).all().all():
                raise ValueError(f"Nonfinite logits/probabilities: {precision} {part}")
            if not set(pred.prediction) <= {0, 1} or not np.allclose(pred.prob_nonhate + pred.prob_hate, 1):
                raise ValueError(f"Invalid predictions: {precision} {part}")
            metrics = json.loads(metric_path.read_text())
            if not all(np.isfinite(metrics[key]) for key in ("accuracy", "macro_f1", "macro_precision", "macro_recall", "ece", "brier", "nll", "aurc")):
                raise ValueError(f"Invalid metrics: {precision} {part}")


def run_one(name: str, allow_download: bool, verify_only: bool = False):
    cfg = load_config(f"configs/encoder_{name}.yaml")
    output = Path("outputs/encoders/smoke") / name
    output.mkdir(parents=True, exist_ok=True)
    if not allow_download and not verify_only:
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
        if verify_only:
            prior = json.loads((output / "availability.json").read_text())
            first_step, resumed_step = prior["first_step"], prior["resumed_step"]
            modules = prior["quantized_modules"]
        else:
            first = module.train(cfg, smoke=True)
            second = module.train(cfg, smoke=True, explicit_resume=True)
            first_step, resumed_step = first["global_step"], second["global_step"]
            modules = second["quantized_modules"]
        verify_outputs(cfg, output)
        result = {"phase": 16, "model": cfg["model_id"], "model_key": name,
                  "fp32": "passed", "int8": "passed", "first_step": first_step,
                  "resumed_step": resumed_step, "quantized_modules": modules,
                  "smoke_only": True}
        if result["resumed_step"] <= result["first_step"]:
            raise RuntimeError("Global step did not increase after resume")
    except Exception as exc:  # noqa: BLE001 - preserve a per-arm failure manifest
        result = {"phase": 16, "model": cfg["model_id"], "model_key": name,
                  "fp32": "failed", "int8": "failed", "error": repr(exc), "smoke_only": True}
    atomic_write_json(output / "availability.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    ns = parser.parse_args()
    results = [run_one(name, ns.allow_download, ns.verify_only) for name in ("banglabert", "xlmr")]
    print(json.dumps(results, indent=2))
    if any(r["fp32"] == "failed" for r in results):
        raise SystemExit(1)
