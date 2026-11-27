"""Evaluate saved FP32 encoder and dynamic INT8 copy on the same CPU examples."""

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.encoder import (
    checkpoint_valid,
    evaluate_and_save,
    load_config,
    load_frozen,
    make_model,
    quantize_dynamic_int8,
)
from src.utils.atomic import atomic_write_json


def main(config: str):
    started = time.perf_counter()
    cfg = load_config(config)
    output = Path(cfg["output"])
    run_path = output / "run.json"
    if not run_path.exists():
        raise RuntimeError("FP32 run manifest absent")
    run = json.loads(run_path.read_text())
    if run["status"] != "completed" or run.get("smoke_only"):
        raise RuntimeError("FP32 full training is not complete")
    best = Path(run.get("best_checkpoint", output / "best_checkpoint"))
    if not checkpoint_valid(best):
        raise RuntimeError("Best FP32 checkpoint is incomplete")
    frames, sha = load_frozen(cfg)
    if sha != run["split_sha256"]:
        raise ValueError("Frozen split fingerprint changed")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"])
    model = make_model(cfg)
    if (best / "model.safetensors").exists():
        from safetensors.torch import load_file
        state = load_file(best / "model.safetensors")
    else:
        state = torch.load(best / "pytorch_model.bin", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    fp_metrics = evaluate_and_save(model, tokenizer, frames, cfg, sha, "fp32_cpu", output)
    del model
    model = make_model(cfg)
    model.load_state_dict(state)
    del state
    quantized, modules = quantize_dynamic_int8(model)
    int8_metrics = evaluate_and_save(quantized, tokenizer, frames, cfg, sha, "int8", output)
    arm = {"phase": 16, "status": "completed", "model": cfg["model_id"],
           "model_revision": run.get("model_revision"), "precision": "dynamic_int8_cpu",
           "language": cfg["language"], "seed": cfg["seed"], "fold": cfg["fold"],
           "split_sha256": sha, "dataset_fingerprints": run["dataset_fingerprints"],
           "training_configuration": cfg, "package_versions": run["package_versions"],
           "hardware": {"cpu": platform.processor(), "gpu": None},
           "checkpoint": str(best), "global_step": run["global_step"], "epoch": run["epoch"],
           "quantized_modules": modules, "metrics": int8_metrics, "fp32_cpu_metrics": fp_metrics,
           "runtime_seconds": time.perf_counter() - started,
           "note": "CPU dynamic INT8 only; no GPU speed or energy claim"}
    atomic_write_json(output / "int8_run.json", arm)
    return arm


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    ns = p.parse_args()
    print(json.dumps({"status": main(ns.config)["status"]}, indent=2))
