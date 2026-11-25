"""Run reproducible Phase 12 ablation cells; --dry-run performs no model work."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

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
from src.models.finetune import load_tokenizer
from src.models.logit_extraction import extract_logits_hf, probs_to_frame
from src.models.quantize_track_a import free, load_gptq, load_quantized_hf, quantize_gptq, GPTQUnavailable
from src.utils.atomic import atomic_write_csv, atomic_write_json, atomic_write_parquet
from src.utils.config import config_hash, load_config
from src.utils.logging_utils import write_run_json


def _path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


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
                if "A5" in ablations:
                    for group in (128, 32):
                        cells.append(("A5", language, model, seed, base, "gptq_int4", f"group_{group}", merged))
                if "A4" in ablations:
                    for variant in ("attention_only", "mlp_only"):
                        cells.append(("A4", language, model, seed, base, "gptq_int4", variant, merged))
                if "A7" in ablations:
                    cells.append(("A7", language, model, seed, base, "gptq_int3", "int3", merged))
                    cells.append(("A7", language, model, seed, base, "gguf_q2_k", "q2_k", merged))
                if "A6" in ablations:
                    for fraction in (10, 25, 100):
                        cells.append(("A6", language, model, seed, base, "lora_bf16", f"scale_{fraction}", merged))
                if "A3" in ablations:
                    cells.append(("A3", language, model, seed, base, "qlora_nf4", "as_trained", merged))
    return cells


def _record_path(out_root: Path, ablation: str, base: str, method: str, variant: str) -> Path:
    return out_root / ablation / base / f"{method}__{variant}.json"


def _evaluate(model, tokenizer, frames, *, cfg, template, ablation, base, method, variant, split_sha,
              logits_root: Path, smoke: bool):
    rows = []
    for split in cfg["eval_sets"]:
        expected = frames[str(split)]
        if smoke:
            expected = expected.head(16).copy()
        logit_frame = extract_logits_hf(
            model, tokenizer, expected, verbalizer=dict(cfg["verbalizer"]),
            batch_size=int(cfg["eval_batch_size"]), split=str(split),
            max_len=int(cfg["max_len"]), template=template, model_id=base,
        )
        validate_ablation_logits(logit_frame, expected)
        path = logits_root / ablation / base / method / variant / f"{split}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_parquet(path, logit_frame)
        rows.append(metric_row(logit_frame, ablation_id=ablation, artifact=base,
                               language=base[:2], model=base[3:].rsplit("_lora_bf16_s", 1)[0],
                               seed=int(base.rsplit("_s", 1)[1]), method=method,
                               variant=variant, eval_set=str(split), split_sha256=split_sha,
                               logit_file=str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                               status="success"))
    return rows


def _run_cell(cell, cfg, *, cache, output_root: Path, smoke: bool):
    ablation, language, model_key, seed, base, method, variant, merged = cell
    frames, split_sha = cache[language]
    template = str(cfg["prompt_templates"][language]["baseline"])
    model = None
    try:
        if ablation in ("A1", "A4", "A5", "A7") and method != "gguf_q2_k":
            if method == "awq_int4":
                return [], "unavailable", "AWQ driver is absent in the current Track A stack"
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
            quant_dir = output_root / "quantized" / ablation / base / variant
            quant_info = quantize_gptq(str(merged), calib["prompts"], str(quant_dir),
                                       bits=3 if method == "gptq_int3" else 4,
                                       group_size=group, max_len=int(cfg["max_len"]), dynamic=dynamic)
            model, tokenizer = load_gptq(str(quant_dir))
            rows = _evaluate(model, tokenizer, frames, cfg=cfg, template=template,
                             ablation=ablation, base=base, method=method, variant=variant,
                             split_sha=split_sha, logits_root=output_root / "logits", smoke=smoke)
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
            target = output_root / "quantized/A7" / base / "q2_k.gguf"
            info = quantize_gguf(source, target, "q2_k")
            llm = load_gguf_model(target, n_ctx=512, n_gpu_layers=-1, verbose=False, logits_all=True)
            try:
                tokenizer = load_tokenizer(str(merged))
                rows = []
                for split in cfg["eval_sets"]:
                    expected = frames[str(split)].head(16).copy() if smoke else frames[str(split)]
                    logit_frame = extract_logits_llamacpp(
                        llm, expected, verbalizer=dict(cfg["verbalizer"]),
                        split=str(split), max_len=int(cfg["max_len"]),
                        template=template, model_id=base, tokenizer=tokenizer,
                    )
                    validate_ablation_logits(logit_frame, expected)
                    path = output_root / "logits/A7" / base / method / variant / f"{split}.parquet"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    atomic_write_parquet(path, logit_frame)
                    rows.append(metric_row(logit_frame, ablation_id="A7", artifact=base,
                                           language=language, model=model_key, seed=seed,
                                           method=method, variant=variant, eval_set=str(split),
                                           split_sha256=split_sha, logit_file=str(path.relative_to(ROOT)),
                                           quantize_s=info["quantize_s"], status="success"))
                return rows, "success", None
            finally:
                free_gguf(llm)
        if ablation == "A2":
            from src.models.finetune import load_base_model
            template = str(cfg["prompt_templates"][language][variant])
            model = load_base_model(str(merged))
            tokenizer = load_tokenizer(str(merged))
            rows = _evaluate(model, tokenizer, frames, cfg=cfg, template=template,
                             ablation=ablation, base=base, method=method, variant=variant,
                             split_sha=split_sha, logits_root=output_root / "logits", smoke=smoke)
            return rows, "success", None
        if ablation == "A6":
            from omegaconf import OmegaConf
            from src.models.finetune import finetune_lora, load_base_model, subsample
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
            resolved = json.loads(run_manifest.read_text(encoding="utf-8"))["resolved_hf_id"]
            train_cfg["model"]["hf_id"] = resolved
            entry = next(x for x in train_cfg["models"] if x["key"] == model_key)
            train_cfg["model"]["attn_implementation"] = entry.get("attn_implementation")
            train_cfg["experiment"] = {"seed": seed, "name": f"{base}__{variant}",
                                       "output_dir": str(output_root / "models/A6" / base / variant)}
            base_train = frames["train"]
            if smoke:
                base_train = subsample(base_train, min(len(base_train), 160), seed=seed)
            n_train = max(2, round(len(base_train) * fraction / 100))
            selected = subsample(base_train, n_train, seed=seed)
            val = frames["id_val"].head(32).copy() if smoke else frames["id_val"]
            trained = finetune_lora(train_cfg, selected, val)
            if not trained.loss_decreased:
                raise RuntimeError(f"A6 training loss did not decrease for {base}/{variant}")
            model = load_base_model(trained.merged_path)
            tokenizer = load_tokenizer(trained.merged_path)
            rows = _evaluate(model, tokenizer, frames, cfg=cfg, template=template,
                             ablation=ablation, base=base, method=method, variant=variant,
                             split_sha=split_sha, logits_root=output_root / "logits", smoke=smoke)
            for row in rows:
                row.update({"train_fraction": fraction / 100, "n_train": len(selected),
                            "train_loss_decreased": trained.loss_decreased,
                            "train_wall_s": trained.wall_s})
            return rows, "success", None
        if ablation == "A3":
            from omegaconf import OmegaConf
            from peft import PeftModel
            from src.models.finetune import finetune_lora, subsample
            baseline_path = ROOT / ("configs/baseline/default.yaml" if language == "bn"
                                    else "configs/baseline/english_control.yaml")
            train_cfg = OmegaConf.to_container(load_config(baseline_path), resolve=True)
            run_manifest = ROOT / "outputs/runs" / ("phase4" if language == "bn" else "phase5") / f"{base}.json"
            if not run_manifest.is_file():
                raise FileNotFoundError(f"Missing baseline model provenance: {run_manifest}")
            resolved = json.loads(run_manifest.read_text(encoding="utf-8"))["resolved_hf_id"]
            train_cfg["model"]["hf_id"] = resolved
            train_cfg["train"]["quantization"] = "nf4"
            train_cfg["experiment"] = {"seed": seed, "name": f"{base}__qlora_nf4",
                                       "output_dir": str(output_root / "models/A3" / base)}
            train = subsample(frames["train"], min(len(frames["train"]), 160), seed=seed) if smoke else frames["train"]
            val = frames["id_val"].head(32).copy() if smoke else frames["id_val"]
            trained = finetune_lora(train_cfg, train, val)
            if not trained.loss_decreased:
                raise RuntimeError(f"A3 QLoRA training loss did not decrease for {base}")
            model = load_quantized_hf(resolved, "bnb_nf4")
            model = PeftModel.from_pretrained(model, trained.adapter_path)
            tokenizer = load_tokenizer(resolved)
            rows = _evaluate(model, tokenizer, frames, cfg=cfg, template=template,
                             ablation=ablation, base=base, method=method, variant=variant,
                             split_sha=split_sha, logits_root=output_root / "logits", smoke=smoke)
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
        result.to_parquet(results_path.with_suffix(".parquet"), index=False)
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
    failures = []
    for cell in cells:
        ablation, language, model_key, seed, base, method, variant, merged = cell
        manifest_path = _record_path(runs_root, ablation, base, method, variant)
        if manifest_path.is_file() and not args.force:
            old = json.loads(manifest_path.read_text(encoding="utf-8"))
            logit_files_exist = all(_path(row["logit_file"]).is_file() for row in old.get("rows", []))
            if old.get("status") in ("success", "unavailable") and old.get("config_hash") == config_hash(cfg) and logit_files_exist:
                print(f"skip {ablation} {base} {method} {variant}: {old['status']}")
                continue
        start = time.perf_counter()
        try:
            rows, status, reason = _run_cell(cell, cfg, cache=cache, output_root=out_root, smoke=args.smoke)
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
    a8 = None
    if "A8" in ablations:
        a8 = temperature_source_rows(_path(cfg["paths"]["phase11_results"]),
                                     models=set(models), languages=set(languages), seeds=set(args.seeds))
        if args.smoke:
            a8 = a8[a8["artifact"] == sorted(a8["artifact"].unique())[0]].copy()
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
