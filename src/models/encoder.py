"""Phase 16 encoder data, evaluation, checkpoint, and quantization helpers."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification

from src.analysis.phase12 import frozen_frames
from src.data.loader import normalize_text
from src.metrics.aggregate import evaluate_logits_frame, metrics_to_record
from src.utils.atomic import atomic_write_json


def load_config(path: str | Path) -> dict:
    import yaml

    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    required = {"model_key", "model_id", "language", "data", "splits", "fold", "seed", "output"}
    if required - cfg.keys():
        raise ValueError(f"Missing config keys: {sorted(required - cfg.keys())}")
    if cfg["model_key"] not in {"banglabert", "xlmr"} or cfg["language"] not in {"bn", "en"}:
        raise ValueError("Unsupported encoder configuration")
    if (cfg["model_key"] == "banglabert") != (cfg["language"] == "bn"):
        raise ValueError("Model/language mismatch")
    return cfg


def load_frozen(cfg: dict) -> tuple[dict[str, pd.DataFrame], str]:
    frames, sha = frozen_frames(Path(cfg["data"]), Path(cfg["splits"]), cfg["fold"])
    sidecar = Path(str(cfg["splits"]) + ".sha256")
    if not sidecar.is_file() or sidecar.read_text().strip() != sha:
        raise ValueError("Frozen split checksum does not match registered sidecar")
    for name, frame in frames.items():
        if not set(frame["label"].astype(int)).issubset({0, 1}):
            raise ValueError(f"Invalid labels in {name}")
    return frames, sha


def prepare_text(text: str, language: str) -> str:
    if language not in {"bn", "en"}:
        raise ValueError(language)
    value = normalize_text(text)
    if language == "bn":
        try:
            from normalizer import normalize
        except ImportError as exc:
            raise RuntimeError("Install csebuetnlp/normalizer for BanglaBERT") from exc
        value = normalize(value)
    return value


class EncodedDataset(torch.utils.data.Dataset):
    def __init__(self, frame: pd.DataFrame, tokenizer, language: str, max_length: int):
        self.frame = frame.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.language = language
        self.max_length = max_length

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        item = self.tokenizer(prepare_text(row["text"], self.language),
                              truncation=True, max_length=self.max_length)
        item["labels"] = int(row["label"])
        return item


def make_model(cfg: dict):
    return AutoModelForSequenceClassification.from_pretrained(
        cfg["model_id"], num_labels=2, id2label={0: "non_hate", 1: "hate"},
        label2id={"non_hate": 0, "hate": 1},
    )


def checkpoint_valid(path: Path) -> bool:
    required = ["trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth"]
    if not path.is_dir() or any(not (path / x).is_file() or (path / x).stat().st_size == 0 for x in required):
        return False
    if not any((path / x).is_file() for x in ("model.safetensors", "pytorch_model.bin")):
        return False
    try:
        state = json.loads((path / "trainer_state.json").read_text())
        if int(state["global_step"]) < 0:
            return False
        if not all(zipfile.is_zipfile(path / x) for x in ("optimizer.pt", "scheduler.pt", "rng_state.pth")):
            return False
        if (path / "model.safetensors").exists():
            from safetensors import safe_open

            with safe_open(path / "model.safetensors", framework="pt", device="cpu") as weights:
                if not weights.keys():
                    return False
        elif not zipfile.is_zipfile(path / "pytorch_model.bin"):
            return False
        return True
    except (ValueError, KeyError, OSError):
        return False


def latest_checkpoint(folder: Path) -> Path | None:
    candidates = sorted(folder.glob("checkpoint-*"),
                        key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else -1,
                        reverse=True)
    return next((p for p in candidates if checkpoint_valid(p)), None)


def fingerprint(frame: pd.DataFrame) -> str:
    rows = frame[["example_id", "label"]].astype(str).values.tolist()
    return hashlib.sha256(json.dumps(rows, ensure_ascii=False).encode()).hexdigest()


def manifest_base(cfg: dict, split_sha: str, frames: dict) -> dict:
    import transformers

    try:
        normalizer_version = importlib.metadata.version("normalizer")
    except importlib.metadata.PackageNotFoundError:
        normalizer_version = None
    return {
        "phase": 16, "status": "running", "model": cfg["model_id"],
        "model_revision": None, "precision": "FP32", "language": cfg["language"],
        "seed": cfg["seed"], "fold": cfg["fold"], "split_sha256": split_sha,
        "dataset_fingerprints": {key: fingerprint(value) for key, value in frames.items()},
        "training_configuration": cfg, "package_versions": {
            "torch": torch.__version__, "transformers": transformers.__version__,
            "numpy": np.__version__, "pandas": pd.__version__,
            "normalizer": normalizer_version,
        }, "hardware": {"cpu": platform.processor(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
        "checkpoint": None, "global_step": 0, "epoch": 0,
        "metrics": {}, "runtime_seconds": 0,
    }


def logits_frame(model, tokenizer, frame: pd.DataFrame, cfg: dict, device: str = "cpu") -> tuple[pd.DataFrame, dict]:
    import psutil

    rss_before = psutil.Process().memory_info().rss if device == "cpu" else None
    model.to(device).eval()
    logits = []
    start = time.perf_counter()
    with torch.inference_mode():
        for i in range(0, len(frame), int(cfg["eval_batch_size"])):
            rows = frame.iloc[i:i + int(cfg["eval_batch_size"])]
            batch = tokenizer([prepare_text(x, cfg["language"]) for x in rows["text"].astype(str)],
                              padding=True, truncation=True, max_length=int(cfg["max_length"]), return_tensors="pt")
            batch = {k: v.to(device) for k, v in batch.items()}
            logits.append(model(**batch).logits.float().cpu())
    values = torch.cat(logits).numpy()
    if values.shape != (len(frame), 2) or not np.isfinite(values).all():
        raise ValueError("Invalid logits")
    probs = torch.softmax(torch.from_numpy(values), dim=-1).numpy()
    out = pd.DataFrame({"example_id": frame["example_id"].astype(str).tolist(),
                        "y_true": frame["label"].astype(int).tolist(),
                        "logit_nonhate": values[:, 0], "logit_hate": values[:, 1],
                        "prob_nonhate": probs[:, 0], "prob_hate": probs[:, 1],
                        "prediction": probs.argmax(axis=1)})
    if out.example_id.duplicated().any() or not np.isfinite(probs).all() or not np.allclose(probs.sum(axis=1), 1):
        raise ValueError("Invalid probabilities or IDs")
    elapsed = time.perf_counter() - start
    rss_after = psutil.Process().memory_info().rss if device == "cpu" else None
    return out, {"latency_seconds_per_example": elapsed / len(frame), "total_seconds": elapsed,
                 "cpu_rss_before_bytes": rss_before, "cpu_rss_after_bytes": rss_after}


def evaluate_and_save(model, tokenizer, frames, cfg, split_sha, precision: str, output: Path, device="cpu") -> dict:
    output.joinpath("predictions").mkdir(parents=True, exist_ok=True)
    output.joinpath("metrics").mkdir(parents=True, exist_ok=True)
    result = {}
    for part in ("id_val", "id_test", "ood_test"):
        pred, timing = logits_frame(model, tokenizer, frames[part], cfg, device)
        if pred.example_id.tolist() != frames[part].example_id.astype(str).tolist() or not np.array_equal(pred.y_true, frames[part].label):
            raise ValueError("Example alignment failed")
        metrics = metrics_to_record(evaluate_logits_frame(pred))
        metrics.update({"split": part, "precision": precision, "language": cfg["language"],
                        "fold": cfg["fold"], "dataset_fingerprint": fingerprint(frames[part]),
                        "split_sha256": split_sha, "timing": timing,
                        "ece_comparability_note": "Single-class Bengali OOD ECE is not directly comparable to two-class OOD ECE" if part == "ood_test" and cfg["language"] == "bn" else None})
        pred.to_csv(output / "predictions" / f"{precision.lower()}_{part}.csv", index=False)
        atomic_write_json(output / "metrics" / f"{precision.lower()}_{part}.json", metrics)
        result[part] = metrics
    return result


def quantize_dynamic_int8(model):
    model.cpu().eval()
    quantized = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    names = [name for name, module in quantized.named_modules()
             if isinstance(module, torch.ao.nn.quantized.dynamic.Linear)]
    if not names:
        raise RuntimeError("No Linear modules converted to dynamic INT8")
    return quantized, names
