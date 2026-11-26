"""Run reproducible Phase 12 ablation cells; --dry-run performs no model work."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from src.analysis.phase12 import (
    CALIB_VARIANTS, PROMPT_VARIANTS, calibration_sets, frozen_frames,
    metric_row, temperature_source_rows, validate_ablation_logits,
)
from src.models.finetune import build_prompt, load_tokenizer, subsample
from src.models.logit_extraction import extract_logits_hf, probs_to_frame
from src.models.quantize_track_a import free, load_gptq, load_quantized_hf, quantize_gptq, GPTQUnavailable
from src.utils.atomic import (
    atomic_write_csv, atomic_write_json, atomic_write_parquet,
    is_valid_adapter, is_valid_merged,
)
from src.utils.config import config_hash, load_config
from src.utils.logging_utils import write_run_json


def _path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _cached_model_path(model_id: str) -> str:
    """Prefer a pinned local HF snapshot when network access is unavailable."""
    from huggingface_hub import snapshot_download
    try:
        return snapshot_download(model_id, local_files_only=True)
    except Exception:
        return model_id


def _release(model) -> None:
    del model
    gc.collect()
    free(None)


def _frame_data(cfg, language: str):
    paths = cfg["paths"]
    if language == "bn":
        return frozen_frames(_path(paths["bn_data"]), _path(paths["bn_splits"]), "loco_slander")
    return frozen_frames(_path(paths["en_data"]), _path(paths["en_splits"]), "loco_lgbtq")


def _cells(cfg, models: list[str], languages: list[str], seeds: list[int], ablations: list[str]):
    cells = []
    for language in languages:
        for model in models:
            for seed in seeds:
                base = f"{language}_{model}_lora_bf16_s{seed}"
                merged = _path(cfg["paths"]["phase4_models" if language == "bn" else "phase5_models"]) / base / "merged"
                if any(a in ablations for a in ("A1", "A2", "A3", "A4", "A5", "A6", "A7")) and not (merged / "config.json").is_file():
                    raise FileNotFoundError(f"Missing merged FP checkpoint: {merged}")
                if "A1" in ablations:
                    for variant in CALIB_VARIANTS:
                        for method in ("gptq_int4", "awq_int4"):
                            cells.append(("A1", language, model, seed, base, method, variant, merged))
                if "A2" in ablations:
                    for variant in PROMPT_VARIANTS:
                        cells.append(("A2", language, model, seed, base, "label_token", variant, merged))
                    cells.append(("A2", language, model, seed, base, "linear_head", "baseline", merged))
                if "A5" in ablations:
                    for group in (128, 32):
                        cells.append(("A5", language, model, seed, base, "gptq_int4", f"group_{group}", merged))
                if "A4" in ablations:
                    cells.append(("A4", language, model, seed, base, "gptq_int4", "full", merged))
                    for variant in ("attention_only", "mlp_only"):
                        cells.append(("A4", language, model, seed, base, "gptq_int4", variant, merged))
                if "A7" in ablations:
                    cells.append(("A7", language, model, seed, base, "gptq_int3", "int3", merged))
                    cells.append(("A7", language, model, seed, base, "gguf_q2_k", "q2_k", merged))
                if "A6" in ablations:
                    for fraction in (10, 25, 100):
                        cells.append(("A6", language, model, seed, base, "lora_bf16", f"scale_{fraction}", merged))
                if "A3" in ablations:
                    cells.append(("A3", language, model, seed, base, "lora_then_ptq", "bnb_nf4", merged))
                    cells.append(("A3", language, model, seed, base, "qlora_nf4", "as_trained", merged))
    return cells


def _record_path(out_root: Path, ablation: str, base: str, method: str, variant: str) -> Path:
    return out_root / ablation / base / f"{method}__{variant}.json"


def _hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _stage_matches(path: Path, plan: dict) -> bool:
    try:
        return json.loads(path.read_text(encoding="utf-8")) == plan
    except (OSError, ValueError):
        return False


def _valid_gptq_checkpoint(path: Path) -> bool:
    return ((path / "config.json").is_file()
            and (any(path.glob("*.safetensors")) or any(path.glob("*.bin"))))


def _discard_incomplete_dir(path: Path, root: Path) -> None:
    resolved, boundary = path.resolve(), root.resolve()
    if resolved == boundary or not resolved.is_relative_to(boundary):
        raise ValueError(f"Refusing to remove path outside Phase 12 work root: {path}")
    if path.exists():
        shutil.rmtree(path)


def _saved_eval(path: Path, expected: pd.DataFrame, split: str, plan_hash: str) -> pd.DataFrame | None:
    """Reuse only an atomically written frame that still matches frozen data."""
    try:
        frame = pd.read_parquet(path)
        if ("split" not in frame or not (frame["split"] == split).all()
                or "phase12_eval_plan_hash" not in frame
                or not (frame["phase12_eval_plan_hash"] == plan_hash).all()):
            return None
        validate_ablation_logits(frame, expected)
        return frame
    except (OSError, ValueError, KeyError, ImportError):
        return None


def _complete_manifest_valid(record: dict, cfg, frames: dict, *, smoke: bool) -> bool:
    if record.get("status") != "success" or record.get("config_hash") != config_hash(cfg):
        return False
    rows = record.get("rows", [])
    if len(rows) != len(cfg["eval_sets"]):
        return False
    for row in rows:
        split = row.get("eval_set")
        if split not in cfg["eval_sets"]:
            return False
        expected = frames[split].head(16).copy() if smoke else frames[split]
        path = _path(row["logit_file"])
        try:
            frame = pd.read_parquet(path)
            if smoke and len(frame) > len(expected):
                frame = frame.head(len(expected)).copy()
            validate_ablation_logits(frame, expected)
        except (OSError, ValueError, KeyError, ImportError):
            return False
    return True


def _train_or_resume(train_cfg: dict, train: pd.DataFrame, val: pd.DataFrame,
                     output_dir: Path, work_root: Path, *, qlora: bool, force: bool):
    """Resume a complete adapter or the latest valid Trainer checkpoint."""
    from src.models.finetune import finetune_lora

    train_cfg["train"]["save_strategy"] = "steps"
    train_cfg["train"]["save_steps"] = 10
    train_cfg["train"]["save_total_limit"] = 2
    plan = {"recipe_hash": _hash({"model": train_cfg["model"], "train": train_cfg["train"],
                                    "data": train_cfg["data"], "seed": train_cfg["experiment"]["seed"]}),
            "train_hash": _hash(train[["example_id", "label"]].astype(str).values.tolist()),
            "val_hash": _hash(val[["example_id", "label"]].astype(str).values.tolist()),
            "qlora": qlora}
    marker = output_dir / ".phase12_train_plan.json"
    if output_dir.exists():
        if force:
            _discard_incomplete_dir(output_dir, work_root)
        elif not _stage_matches(marker, plan):
            raise RuntimeError(f"Training directory has no matching resume plan: {output_dir}; use --force to replace it")
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(marker, plan)
    train_cfg["experiment"]["output_dir"] = str(output_dir)
    adapter = output_dir / "adapter"
    merged = output_dir / "merged"
    metrics_file = output_dir / "train_metrics.json"
    if not force and metrics_file.is_file() and is_valid_adapter(adapter) and (qlora or is_valid_merged(merged)):
        metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
        if metrics.get("n_train") == len(train) and metrics.get("n_id_val") == len(val):
            losses = [x["loss"] for x in metrics.get("log_history", []) if "loss" in x]
            print(f"reuse completed training: {output_dir}")
            return SimpleNamespace(adapter_path=str(adapter), merged_path="" if qlora else str(merged),
                                   loss_decreased=len(losses) >= 2 and losses[-1] < losses[0],
                                   wall_s=metrics.get("wall_s"))
    # finetune_lora finds the latest valid checkpoint, skipping a checkpoint
    # left half-written by a sudden power cut.
    return finetune_lora(train_cfg, train, val)


def _hidden_features(model, tokenizer, frame: pd.DataFrame, template: str, max_len: int,
                     cache_dir: Path | None = None) -> np.ndarray:
    """Last real-token hidden state for a frozen, unmodified causal backbone."""
    model.eval()
    original_side = tokenizer.truncation_side
    tokenizer.truncation_side = "left"
    device = next(model.parameters()).device
    features = []
    if cache_dir is not None:
        source = {"ids_and_text": _hash(frame[["example_id", "text"]].astype(str).values.tolist()),
                  "template": template, "max_len": max_len,
                  "model": str(getattr(model.config, "_name_or_path", ""))}
        cache_dir = cache_dir / _hash(source)[:16]
        cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        with torch.no_grad():
            for block_start in range(0, len(frame), 128):
                block_end = min(len(frame), block_start + 128)
                saved = cache_dir / f"{block_start}_{block_end}.npy" if cache_dir else None
                block = None
                if saved is not None and saved.is_file():
                    try:
                        candidate = np.load(saved, allow_pickle=False)
                        if candidate.shape[0] == block_end - block_start and np.isfinite(candidate).all():
                            block = candidate
                    except (OSError, ValueError):
                        pass
                if block is None:
                    pieces = []
                    for start in range(block_start, block_end, 4):
                        prompts = [build_prompt(t, template) for t in frame["text"].iloc[start:min(start + 4, block_end)]]
                        inputs = tokenizer(prompts, padding=True, truncation=True,
                                           max_length=max_len, return_tensors="pt")
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                        output = model(**inputs, output_hidden_states=True, use_cache=False)
                        hidden = output.hidden_states[-1]
                        if tokenizer.padding_side == "left":
                            positions = torch.full((len(prompts),), hidden.shape[1] - 1, device=device)
                        else:
                            positions = inputs["attention_mask"].sum(dim=1) - 1
                        vectors = hidden[torch.arange(len(prompts), device=device), positions]
                        pieces.append(vectors.float().cpu().numpy())
                    block = np.concatenate(pieces)
                    if saved is not None:
                        tmp = saved.with_name(saved.name + f".{os.getpid()}.part")
                        with open(tmp, "wb") as file:
                            np.save(file, block)
                        os.replace(tmp, saved)
                features.append(block)
    finally:
        tokenizer.truncation_side = original_side
    result = np.concatenate(features)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite hidden features for A2 classification head")
    return result


def _evaluate(model, tokenizer, frames, *, cfg, template, ablation, base, method, variant, split_sha,
              logits_root: Path, smoke: bool, force: bool = False):
    rows = []
    eval_dir = logits_root / ablation / base / method / variant
    plan = {"config_hash": config_hash(cfg), "ablation": ablation, "artifact": base,
            "method": method, "variant": variant, "split_sha256": split_sha,
            "template": template, "smoke": smoke}
    plan_path = eval_dir / ".eval_plan.json"
    plan_hash = _hash(plan)
    resume = not force and _stage_matches(plan_path, plan)
    eval_dir.mkdir(parents=True, exist_ok=True)
    if not resume:
        atomic_write_json(plan_path, plan)
    for split in cfg["eval_sets"]:
        expected = frames[str(split)]
        if smoke:
            expected = expected.head(16).copy()
        path = eval_dir / f"{split}.parquet"
        logit_frame = _saved_eval(path, expected, str(split), plan_hash) if resume else None
        if logit_frame is None:
            logit_frame = extract_logits_hf(
                model, tokenizer, expected, verbalizer=dict(cfg["verbalizer"]),
                batch_size=int(cfg["eval_batch_size"]), split=str(split),
                max_len=int(cfg["max_len"]), template=template, model_id=base,
            )
            validate_ablation_logits(logit_frame, expected)
            logit_frame["phase12_eval_plan_hash"] = plan_hash
            atomic_write_parquet(path, logit_frame)
        else:
            print(f"reuse evaluated split: {path}")
        rows.append(metric_row(logit_frame, ablation_id=ablation, artifact=base,
                               language=base[:2], model=base[3:].rsplit("_lora_bf16_s", 1)[0],
                               seed=int(base.rsplit("_s", 1)[1]), method=method,
                               variant=variant, eval_set=str(split), split_sha256=split_sha,
                               logit_file=str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                               status="success"))
    return rows


def _evaluate_custom(frames, *, cfg, template, ablation, language, model_key, seed,
                     base, method, variant, split_sha, logits_root: Path, smoke: bool,
                     make_frame, force: bool = False, extra: dict | None = None):
    eval_dir = logits_root / ablation / base / method / variant
    plan = {"config_hash": config_hash(cfg), "ablation": ablation, "artifact": base,
            "method": method, "variant": variant, "split_sha256": split_sha,
            "template": template, "smoke": smoke}
    marker = eval_dir / ".eval_plan.json"
    resume = not force and _stage_matches(marker, plan)
    eval_dir.mkdir(parents=True, exist_ok=True)
    if not resume:
        atomic_write_json(marker, plan)
    plan_hash = _hash(plan)
    rows = []
    for split in cfg["eval_sets"]:
        expected = frames[str(split)].head(16).copy() if smoke else frames[str(split)]
        path = eval_dir / f"{split}.parquet"
        frame = _saved_eval(path, expected, str(split), plan_hash) if resume else None
        if frame is None:
            frame = make_frame(str(split), expected)
            validate_ablation_logits(frame, expected)
            frame["phase12_eval_plan_hash"] = plan_hash
            atomic_write_parquet(path, frame)
        else:
            print(f"reuse evaluated split: {path}")
        rows.append(metric_row(frame, ablation_id=ablation, artifact=base,
                               language=language, model=model_key, seed=seed,
                               method=method, variant=variant, eval_set=str(split),
                               split_sha256=split_sha, logit_file=str(path.relative_to(ROOT)),
                               status="success", **(extra or {})))
    return rows


def _run_cell(cell, cfg, *, cache, output_root: Path, smoke: bool, force: bool = False):
    ablation, language, model_key, seed, base, method, variant, merged = cell
    frames, split_sha = cache[language]
    template = str(cfg["prompt_templates"][language]["baseline"])
    model = None
    try:
        if (ablation == "A3" and method == "lora_then_ptq") or (ablation == "A4" and variant == "full"):
            source_method = "bnb_nf4" if ablation == "A3" else "gptq_int4"
            rows = []
            for split in cfg["eval_sets"]:
                path = ROOT / "outputs/logits/phase6" / f"{base}__{source_method}" / f"{source_method}__{split}.parquet"
                if not path.is_file():
                    raise FileNotFoundError(f"Missing comparator logits: {path}")
                expected = frames[str(split)]
                original = pd.read_parquet(path)
                validate_ablation_logits(original, expected)
                if smoke:
                    original = original.head(16).copy()
                rows.append(metric_row(original, ablation_id=ablation, artifact=base,
                                       language=language, model=model_key, seed=seed,
                                       method=method, variant=variant, eval_set=str(split),
                                       split_sha256=split_sha, logit_file=str(path.relative_to(ROOT)),
                                       source="phase6_saved_logits", status="success"))
            return rows, "success", None
        if ablation in ("A1", "A4", "A5", "A7") and method != "gguf_q2_k":
            if method == "awq_int4":
                return [], "unavailable", "AWQ quantization driver is not implemented in the Track A stack"
            if ablation == "A1":
                bn, _ = cache["bn"]
                en, _ = cache["en"]
                forbidden_bn = set().union(*(set(bn[x]["example_id"].astype(str)) for x in ("id_val", "id_test", "ood_test")))
                forbidden_en = set().union(*(set(en[x]["example_id"].astype(str)) for x in ("id_val", "id_test", "ood_test")))
                sets = calibration_sets(bn["train"], en["train"],
                                        n=8 if smoke else int(cfg["calibration_samples"]),
                                        seed=seed, template=template,
                                        forbidden_bn=forbidden_bn, forbidden_en=forbidden_en)
                calib = sets[variant]
                group = 128
            else:
                # A4/A5/A7 hold the calibration data fixed while changing only
                # the declared quantization setting.
                train = frames["train"]
                count = 8 if smoke else int(cfg["calibration_samples"])
                if len(train) < count:
                    raise ValueError("Insufficient frozen training samples")
                selected = train.sample(n=count, random_state=seed)
                from src.models.finetune import build_prompt
                calib = {"prompts": [build_prompt(t, template) for t in selected["text"].astype(str)],
                         "ids": selected["example_id"].astype(str).tolist()}
                group = int(variant.removeprefix("group_")) if ablation == "A5" else 128
            dynamic = None
            if ablation == "A4":
                dynamic = ({"-:.*\\.mlp\\..*": {}} if variant == "attention_only"
                           else {"-:.*\\.self_attn\\..*": {}})
            quant_plan = {"config_hash": config_hash(cfg), "ablation": ablation,
                          "artifact": base, "method": method, "variant": variant,
                          "source": str(merged), "calibration_hash": _hash(calib),
                          "bits": 3 if method == "gptq_int3" else 4,
                          "group_size": group, "dynamic": dynamic, "smoke": smoke}
            quant_dir = output_root / "quantized" / ablation / base / variant / _hash(quant_plan)[:16]
            marker = quant_dir / ".phase12_quant.json"
            quant_info = None
            if not force and marker.is_file():
                try:
                    saved = json.loads(marker.read_text(encoding="utf-8"))
                    if saved.get("plan") == quant_plan and _valid_gptq_checkpoint(quant_dir):
                        model, tokenizer = load_gptq(str(quant_dir))
                        quant_info = saved["quant_info"]
                        print(f"reuse quantized model: {quant_dir}")
                except (OSError, ValueError, KeyError, RuntimeError):
                    quant_info = None
            if quant_info is None:
                _discard_incomplete_dir(quant_dir, output_root)
                quant_info = quantize_gptq(str(merged), calib["prompts"], str(quant_dir),
                                           bits=quant_plan["bits"], group_size=group,
                                           max_len=int(cfg["max_len"]), dynamic=dynamic)
                if not _valid_gptq_checkpoint(quant_dir):
                    raise RuntimeError(f"GPTQ save incomplete: {quant_dir}")
                atomic_write_json(marker, {"plan": quant_plan, "quant_info": quant_info})
                model, tokenizer = load_gptq(str(quant_dir))
            rows = _evaluate(model, tokenizer, frames, cfg=cfg, template=template,
                             ablation=ablation, base=base, method=method, variant=variant,
                             split_sha=split_sha, logits_root=output_root / "logits", smoke=smoke, force=force)
            for row in rows:
                row.update({"calibration_source": variant if ablation == "A1" else language,
                            "calibration_n": len(calib["prompts"]), "calibration_ids_sha256":
                            __import__("hashlib").sha256(json.dumps(calib["ids"]).encode()).hexdigest(),
                            "group_size": group, "quantize_s": quant_info["quantize_s"]})
            return rows, "success", None
        if ablation == "A7" and method == "gguf_q2_k":
            from src.models.quantize_track_b import (
                extract_logits_llamacpp, free_gguf, is_valid_gguf_file,
                load_gguf_model, quantize_gguf,
            )
            source = ROOT / "outputs/quantized/track_b" / f"{base}__f16.gguf"
            if not is_valid_gguf_file(source):
                raise FileNotFoundError(f"A7 requires Phase 7 F16 GGUF: {source}")
            quant_plan = {"config_hash": config_hash(cfg), "ablation": "A7",
                          "artifact": base, "source": str(source), "method": "q2_k", "smoke": smoke}
            target = output_root / "quantized/A7" / base / _hash(quant_plan)[:16] / "q2_k.gguf"
            marker = target.with_suffix(".phase12_quant.json")
            if not force and is_valid_gguf_file(target) and _stage_matches(marker, quant_plan):
                info = {"quantize_s": 0.0, "reused": True}
                print(f"reuse quantized GGUF: {target}")
            else:
                info = quantize_gguf(source, target, "q2_k")
                atomic_write_json(marker, quant_plan)
            llm = load_gguf_model(target, n_ctx=512, n_gpu_layers=-1, verbose=False, logits_all=True)
            try:
                tokenizer = load_tokenizer(str(merged))
                def make_frame(split, expected):
                    return extract_logits_llamacpp(
                        llm, expected, verbalizer=dict(cfg["verbalizer"]),
                        split=split, max_len=int(cfg["max_len"]),
                        template=template, model_id=base, tokenizer=tokenizer,
                    )
                rows = _evaluate_custom(frames, cfg=cfg, template=template, ablation="A7",
                                        language=language, model_key=model_key, seed=seed,
                                        base=base, method=method, variant=variant, split_sha=split_sha,
                                        logits_root=output_root / "logits", smoke=smoke,
                                        make_frame=make_frame, force=force,
                                        extra={"quantize_s": info["quantize_s"]})
                return rows, "success", None
            finally:
                free_gguf(llm)
        if ablation == "A2":
            from src.models.finetune import load_base_model
            template = str(cfg["prompt_templates"][language][variant])
            model = load_base_model(str(merged))
            tokenizer = load_tokenizer(str(merged))
            if method == "linear_head":
                from sklearn.linear_model import LogisticRegression
                n_probe = 32 if smoke else 2048
                train = subsample(frames["train"], min(len(frames["train"]), n_probe), seed=seed)
                head_path = output_root / "heads/A2" / base / "linear_head.npz"
                head_path.parent.mkdir(parents=True, exist_ok=True)
                head_plan = {"config_hash": config_hash(cfg), "artifact": base, "seed": seed,
                             "template": template, "train_hash": _hash(train[["example_id", "label"]].astype(str).values.tolist()),
                             "smoke": smoke}
                head_marker = head_path.with_suffix(".plan.json")
                head = None
                if not force and _stage_matches(head_marker, head_plan) and head_path.is_file():
                    try:
                        with np.load(head_path, allow_pickle=False) as saved:
                            coef, intercept, classes = saved["coef"], saved["intercept"], saved["classes"]
                        if coef.ndim == 2 and coef.shape[0] == 1 and intercept.shape == (1,) and np.array_equal(classes, [0, 1]) and np.isfinite(coef).all():
                            head = SimpleNamespace(decision_function=lambda x: (x @ coef.T + intercept).ravel())
                            print(f"reuse trained linear head: {head_path}")
                    except (OSError, ValueError, KeyError):
                        pass
                if head is None:
                    features = _hidden_features(model, tokenizer, train, template, int(cfg["max_len"]),
                                                cache_dir=output_root / "features/A2" / base / "train")
                    head = LogisticRegression(max_iter=300, random_state=seed)
                    head.fit(features, train["label"].to_numpy(dtype=int))
                    tmp = head_path.with_name(head_path.name + f".{os.getpid()}.part")
                    with open(tmp, "wb") as file:
                        np.savez(file, coef=head.coef_, intercept=head.intercept_, classes=head.classes_)
                    os.replace(tmp, head_path)
                    atomic_write_json(head_marker, head_plan)
                def make_frame(split, expected):
                    values = _hidden_features(model, tokenizer, expected, template, int(cfg["max_len"]),
                                              cache_dir=output_root / "features/A2" / base / split)
                    margin = head.decision_function(values)
                    return probs_to_frame(expected["example_id"].astype(str), split,
                                          expected["label"], margin, np.zeros(len(margin)))
                rows = _evaluate_custom(frames, cfg=cfg, template=template, ablation="A2",
                                        language=language, model_key=model_key, seed=seed,
                                        base=base, method=method, variant=variant, split_sha=split_sha,
                                        logits_root=output_root / "logits", smoke=smoke,
                                        make_frame=make_frame, force=force,
                                        extra={"n_head_train": len(train),
                                               "head_file": str(head_path.relative_to(ROOT))})
                return rows, "success", None
            rows = _evaluate(model, tokenizer, frames, cfg=cfg, template=template,
                             ablation=ablation, base=base, method=method, variant=variant,
                             split_sha=split_sha, logits_root=output_root / "logits", smoke=smoke, force=force)
            return rows, "success", None
        if ablation == "A6":
            from omegaconf import OmegaConf
            from src.models.finetune import load_base_model
            fraction = int(variant.removeprefix("scale_"))
            if fraction == 100 and not smoke:
                rows = []
                original_phase = "phase4" if language == "bn" else "phase5"
                for split in cfg["eval_sets"]:
                    path = ROOT / "outputs/logits" / original_phase / base / f"bf16__{split}.parquet"
                    if not path.is_file():
                        raise FileNotFoundError(f"Missing 100% baseline logits: {path}")
                    expected = frames[str(split)]
                    original = pd.read_parquet(path)
                    validate_ablation_logits(original, expected)
                    rows.append(metric_row(original, ablation_id="A6", artifact=base,
                                           language=language, model=model_key, seed=seed,
                                           method=method, variant=variant, eval_set=str(split),
                                           split_sha256=split_sha, logit_file=str(path.relative_to(ROOT)),
                                           train_fraction=1.0, n_train=len(frames["train"]),
                                           status="success"))
                return rows, "success", None
            baseline_path = ROOT / ("configs/baseline/default.yaml" if language == "bn"
                                    else "configs/baseline/english_control.yaml")
            train_cfg = OmegaConf.to_container(load_config(baseline_path), resolve=True)
            run_manifest = ROOT / "outputs/runs" / ("phase4" if language == "bn" else "phase5") / f"{base}.json"
            if not run_manifest.is_file():
                raise FileNotFoundError(f"Missing baseline model provenance: {run_manifest}")
            resolved = _cached_model_path(json.loads(run_manifest.read_text(encoding="utf-8"))["resolved_hf_id"])
            train_cfg["model"]["hf_id"] = resolved
            entry = next(x for x in train_cfg["models"] if x["key"] == model_key)
            train_cfg["model"]["attn_implementation"] = entry.get("attn_implementation")
            train_cfg["experiment"] = {"seed": seed, "name": f"{base}__{variant}",
                                       "output_dir": str(output_root / "models/A6" / base / variant)}
            base_train = frames["train"]
            if smoke:
                base_train = subsample(base_train, min(len(base_train), 640), seed=seed)
            n_train = max(2, round(len(base_train) * fraction / 100))
            selected = subsample(base_train, n_train, seed=seed)
            val = frames["id_val"].head(32).copy() if smoke else frames["id_val"]
            trained = _train_or_resume(train_cfg, selected, val,
                                       output_root / "models/A6" / base / variant,
                                       output_root, qlora=False, force=force)
            if not trained.loss_decreased:
                raise RuntimeError(f"A6 training loss did not decrease for {base}/{variant}")
            model = load_base_model(trained.merged_path)
            tokenizer = load_tokenizer(trained.merged_path)
            rows = _evaluate(model, tokenizer, frames, cfg=cfg, template=template,
                             ablation=ablation, base=base, method=method, variant=variant,
                             split_sha=split_sha, logits_root=output_root / "logits", smoke=smoke, force=force)
            for row in rows:
                row.update({"train_fraction": fraction / 100, "n_train": len(selected),
                            "train_loss_decreased": trained.loss_decreased,
                            "train_wall_s": trained.wall_s})
            return rows, "success", None
        if ablation == "A3":
            from omegaconf import OmegaConf
            from peft import PeftModel
            baseline_path = ROOT / ("configs/baseline/default.yaml" if language == "bn"
                                    else "configs/baseline/english_control.yaml")
            train_cfg = OmegaConf.to_container(load_config(baseline_path), resolve=True)
            run_manifest = ROOT / "outputs/runs" / ("phase4" if language == "bn" else "phase5") / f"{base}.json"
            if not run_manifest.is_file():
                raise FileNotFoundError(f"Missing baseline model provenance: {run_manifest}")
            resolved = _cached_model_path(json.loads(run_manifest.read_text(encoding="utf-8"))["resolved_hf_id"])
            train_cfg["model"]["hf_id"] = resolved
            train_cfg["train"]["quantization"] = "nf4"
            train_cfg["experiment"] = {"seed": seed, "name": f"{base}__qlora_nf4",
                                       "output_dir": str(output_root / "models/A3" / base)}
            train = subsample(frames["train"], min(len(frames["train"]), 160), seed=seed) if smoke else frames["train"]
            val = frames["id_val"].head(32).copy() if smoke else frames["id_val"]
            trained = _train_or_resume(train_cfg, train, val,
                                       output_root / "models/A3" / base,
                                       output_root, qlora=True, force=force)
            if not trained.loss_decreased:
                raise RuntimeError(f"A3 QLoRA training loss did not decrease for {base}")
            model = load_quantized_hf(resolved, "bnb_nf4")
            model = PeftModel.from_pretrained(model, trained.adapter_path)
            tokenizer = load_tokenizer(resolved)
            rows = _evaluate(model, tokenizer, frames, cfg=cfg, template=template,
                             ablation=ablation, base=base, method=method, variant=variant,
                             split_sha=split_sha, logits_root=output_root / "logits", smoke=smoke, force=force)
            for row in rows:
                row.update({"n_train": len(train), "train_loss_decreased": trained.loss_decreased,
                            "train_wall_s": trained.wall_s, "base_precision": "bnb_nf4"})
            return rows, "success", None
        raise ValueError(f"Unsupported ablation {ablation}")
    finally:
        if model is not None:
            del model
            _release(None)


def _rebuild_results(manifests: list[Path], a8: pd.DataFrame | None, results_path: Path):
    rows = []
    for path in manifests:
        if not path.is_file():
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("status") == "success":
            if any(not _path(row["logit_file"]).is_file() for row in record["rows"]):
                raise FileNotFoundError(f"Successful manifest has missing logits: {path}")
            rows.extend(record["rows"])
    result = pd.DataFrame(rows)
    if a8 is not None:
        result = pd.concat([result, a8], ignore_index=True)
    if not result.empty:
        key = ["ablation_id", "artifact", "method", "variant", "eval_set"] if "method" in result else ["ablation_id", "artifact", "variant", "eval_set"]
        if result.duplicated(key).any():
            raise ValueError("Duplicate ablation result rows")
        results_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_csv(results_path, result, index=False)
        atomic_write_parquet(results_path.with_suffix(".parquet"), result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/ablation/default.yaml")
    parser.add_argument("--ablations", nargs="+", choices=["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"])
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--languages", nargs="+", choices=["bn", "en"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--variants", nargs="+", help="Restrict A1/A2/A5 variants for a targeted run")
    parser.add_argument("--methods", nargs="+", help="Restrict quantization/readout methods for a targeted run")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    ablations = args.ablations or list(cfg["ablations"])
    models = args.models or list(cfg["models"])
    languages = args.languages or list(cfg["languages"])
    cells = _cells(cfg, models, languages, args.seeds, ablations)
    if args.variants:
        cells = [cell for cell in cells if cell[6] in args.variants]
    if args.methods:
        cells = [cell for cell in cells if cell[5] in args.methods]
    if not cells and "A8" not in ablations:
        raise ValueError("No ablation cells match the requested filters")
    if args.dry_run:
        for cell in cells:
            print(" ".join(str(x) for x in cell[:7]))
        if "A8" in ablations:
            print("A8: reuse Phase 11 fitted temperature results")
        print(f"{len(cells)} model cells + {'A8' if 'A8' in ablations else 'no A8'}")
        return 0
    root = ROOT / "outputs/phase12_smoke" if args.smoke else ROOT / "outputs"
    runs_root = root / "runs/phase12"
    out_root = root / "phase12_work"
    results_path = root / "results/ablation_results.csv"
    manifests = [_record_path(runs_root, c[0], c[4], c[5], c[6]) for c in cells]
    cache = {}
    if any(cell[0] in ("A1", "A2", "A3", "A4", "A5", "A6", "A7") for cell in cells):
        for lang in ({cell[1] for cell in cells} | ({"bn", "en"} if "A1" in ablations else set())):
            cache[lang] = _frame_data(cfg, lang)
    a8 = None
    if "A8" in ablations:
        a8 = temperature_source_rows(_path(cfg["paths"]["phase11_results"]),
                                     models=set(models), languages=set(languages), seeds=set(args.seeds))
        if args.smoke:
            a8 = a8[a8["artifact"] == sorted(a8["artifact"].unique())[0]].copy()
    failures = []
    for cell in cells:
        ablation, language, model_key, seed, base, method, variant, merged = cell
        manifest_path = _record_path(runs_root, ablation, base, method, variant)
        if manifest_path.is_file() and not args.force:
            old = json.loads(manifest_path.read_text(encoding="utf-8"))
            valid = (old.get("status") == "unavailable" and old.get("config_hash") == config_hash(cfg))
            valid = valid or _complete_manifest_valid(old, cfg, cache[language][0], smoke=args.smoke)
            if valid:
                print(f"skip {ablation} {base} {method} {variant}: {old['status']}")
                continue
        start = time.perf_counter()
        try:
            rows, status, reason = _run_cell(cell, cfg, cache=cache, output_root=out_root,
                                              smoke=args.smoke, force=args.force)
        except (GPTQUnavailable, ImportError, NotImplementedError) as exc:
            rows, status, reason = [], "unavailable", f"{type(exc).__name__}: {exc}"
        except Exception as exc:
            rows, status, reason = [], "failed", f"{type(exc).__name__}: {exc}"
            failures.append(f"{ablation}/{base}/{method}/{variant}: {reason}")
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(manifest_path, {"phase": 12, "ablation_id": ablation, "artifact": base,
                                         "method": method, "variant": variant, "status": status,
                                         "reason": reason, "rows": rows, "config_hash": config_hash(cfg),
                                         "wall_s": round(time.perf_counter() - start, 2)})
        print(f"{status} {ablation} {base} {method} {variant}: {reason or len(rows)}")
    result = _rebuild_results(manifests, a8, results_path)
    unavailable = [str(p) for p in manifests if p.is_file() and json.loads(p.read_text(encoding="utf-8")).get("status") == "unavailable"]
    status = "failed" if failures else ("smoke" if args.smoke else "partial")
    write_run_json(root / "results/phase12_run.json", phase=12, phase_name="ablation_studies",
                   status=status, payload={"n_result_rows": len(result), "n_cells": len(cells),
                                           "unavailable": unavailable, "failures": failures,
                                           "smoke": args.smoke, "ablations_requested": ablations})
    print(f"Phase 12 {status}: {len(result)} result rows -> {results_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
