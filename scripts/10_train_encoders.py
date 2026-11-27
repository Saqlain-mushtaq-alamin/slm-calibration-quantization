"""Train or smoke-test Phase 16 FP32 encoders. Never invoked by import."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.metrics.task_metrics import compute_all_task_metrics
from src.models.encoder import (
    EncodedDataset,
    checkpoint_valid,
    evaluate_and_save,
    latest_checkpoint,
    load_config,
    load_frozen,
    make_model,
    manifest_base,
    quantize_dynamic_int8,
)
from src.utils.atomic import atomic_write_json


class StopAndEpochCheckpoint(TrainerCallback):
    def __init__(self, stop_file: Path):
        self.stop_file = stop_file

    def on_step_end(self, args, state, control, **kwargs):
        if self.stop_file.exists():
            control.should_save = True
            control.should_training_stop = True
        return control

    def on_epoch_end(self, args, state, control, **kwargs):
        control.should_save = True
        control.should_evaluate = True
        return control


def train(cfg: dict, smoke: bool = False, explicit_resume: bool = False):
    start = time.perf_counter()
    frames, sha = load_frozen(cfg)
    output = Path("outputs/encoders/smoke") / cfg["model_key"] if smoke else Path(cfg["output"])
    checkpoint_root = output / "checkpoints"
    for name in ("checkpoints", "best_checkpoint", "predictions", "metrics", "logs"):
        (output / name).mkdir(parents=True, exist_ok=True)
    run_path = output / "run.json"
    prior = json.loads(run_path.read_text()) if run_path.exists() else None
    if prior and prior.get("split_sha256") != sha:
        raise ValueError("Frozen split changed since this run began")
    if prior and prior.get("training_configuration") != cfg:
        raise ValueError("Training configuration changed since this run began")
    if not smoke and prior and prior.get("status") == "completed":
        raise RuntimeError("Completed FP32 run exists; refusing to overwrite")
    latest = latest_checkpoint(checkpoint_root)
    if explicit_resume and latest is None:
        raise RuntimeError("--resume requested but no valid checkpoint exists")
    if prior and latest is None and not smoke:
        raise RuntimeError("Existing run has no valid checkpoint; refusing to start over")
    manifest = prior or manifest_base(cfg, sha, frames)
    manifest["status"] = "running"
    manifest["checkpoint"] = str(latest) if latest else None
    atomic_write_json(run_path, manifest)
    set_seed(int(cfg["seed"]))
    if smoke:
        frames = {k: v.iloc[:(16 if k == "train" else 8)].copy() for k, v in frames.items()}
        cfg = {**cfg, "max_length": 32, "batch_size": 2, "eval_batch_size": 2,
               "gradient_accumulation_steps": 1, "save_steps": 1, "epochs": 2}
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"])
    model = make_model(cfg)
    manifest["model_revision"] = getattr(model.config, "_commit_hash", None)
    train_set = EncodedDataset(frames["train"], tokenizer, cfg["language"], int(cfg["max_length"]))
    val_set = EncodedDataset(frames["id_val"], tokenizer, cfg["language"], int(cfg["max_length"]))
    args = TrainingArguments(
        output_dir=str(checkpoint_root), per_device_train_batch_size=int(cfg["batch_size"]),
        per_device_eval_batch_size=int(cfg["eval_batch_size"]),
        gradient_accumulation_steps=int(cfg["gradient_accumulation_steps"]),
        learning_rate=float(cfg["learning_rate"]), num_train_epochs=float(cfg["epochs"]),
        weight_decay=float(cfg["weight_decay"]), seed=int(cfg["seed"]),
        eval_strategy="steps", eval_steps=int(cfg["save_steps"]),
        save_strategy="steps", save_steps=int(cfg["save_steps"]), save_total_limit=4,
        load_best_model_at_end=True, metric_for_best_model="macro_f1", greater_is_better=True,
        logging_dir=str(output / "logs"), logging_steps=1 if smoke else 20,
        report_to="none", fp16=False, bf16=False, dataloader_num_workers=0,
        max_steps=(int(json.loads((latest / "trainer_state.json").read_text())["global_step"]) + 1 if latest else 2) if smoke else -1,
    )

    def metrics(pred):
        values = np.asarray(pred.predictions)
        p = torch.softmax(torch.tensor(values), dim=-1).numpy()[:, 1]
        result = compute_all_task_metrics(pred.label_ids, values.argmax(axis=1), p)
        return {"macro_f1": result["macro_f1"], "accuracy": result["accuracy"]}

    trainer = Trainer(model=model, args=args, train_dataset=train_set, eval_dataset=val_set,
                      data_collator=DataCollatorWithPadding(tokenizer),
                      compute_metrics=metrics, callbacks=[StopAndEpochCheckpoint(output / "STOP")])
    try:
        trainer.train(resume_from_checkpoint=str(latest) if latest else None)
        if latest and trainer.state.global_step <= int(json.loads((latest / "trainer_state.json").read_text())["global_step"]):
            raise RuntimeError("Resume did not advance global step")
        if torch.cuda.is_available():
            manifest["gpu_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            manifest["gpu_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        best = Path(trainer.state.best_model_checkpoint) if trainer.state.best_model_checkpoint else latest_checkpoint(checkpoint_root)
        if best is None or not checkpoint_valid(best):
            raise RuntimeError("No valid best checkpoint")
        best_root = output / "best_checkpoint"
        previous_best = Path(manifest.get("best_checkpoint", str(best_root)))
        if checkpoint_valid(previous_best) and json.loads((previous_best / "trainer_state.json").read_text())["global_step"] == int(best.name.split("-")[-1]):
            best_dest = previous_best
        elif checkpoint_valid(best_root):
            best_dest = best_root / best.name
            if not checkpoint_valid(best_dest):
                shutil.copytree(best, best_dest, dirs_exist_ok=True)
        else:
            best_dest = best_root
            shutil.copytree(best, best_dest, dirs_exist_ok=True)
        manifest["best_checkpoint"] = str(best_dest)
        manifest["checkpoint"] = str(best)
        manifest["global_step"] = trainer.state.global_step
        manifest["epoch"] = trainer.state.epoch
        manifest["runtime_seconds"] = time.perf_counter() - start
        if (output / "STOP").exists():
            manifest["status"] = "interrupted"
        elif smoke:
            manifest["status"] = "completed"
            # Smoke is a recovery/conversion rehearsal, never a research evaluation.
            manifest["smoke_only"] = True
            fp = evaluate_and_save(trainer.model.cpu(), tokenizer, frames, cfg, sha, "fp32", output)
            reloaded = make_model(cfg)
            state = torch.load(best / "pytorch_model.bin", map_location="cpu", weights_only=True) if (best / "pytorch_model.bin").exists() else None
            if state is not None:
                reloaded.load_state_dict(state)
            else:
                from safetensors.torch import load_file
                reloaded.load_state_dict(load_file(best / "model.safetensors"))
            qmodel, modules = quantize_dynamic_int8(reloaded)
            iq = evaluate_and_save(qmodel, tokenizer, frames, cfg, sha, "int8", output)
            manifest["quantized_modules"] = modules
            manifest["metrics"] = {"fp32": fp, "int8": iq}
        else:
            manifest["status"] = "completed"
            manifest["metrics"] = evaluate_and_save(trainer.model.cpu(), tokenizer, frames, cfg, sha, "fp32", output)
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = repr(exc)
        raise
    finally:
        manifest["runtime_seconds"] = time.perf_counter() - start
        atomic_write_json(run_path, manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    ns = parser.parse_args()
    print(json.dumps({k: v for k, v in train(load_config(ns.config), ns.smoke, ns.resume).items()
                      if k in {"status", "global_step", "checkpoint", "runtime_seconds"}}, indent=2))
