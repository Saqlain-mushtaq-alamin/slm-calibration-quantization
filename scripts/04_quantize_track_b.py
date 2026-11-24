#!/usr/bin/env python
"""Phase 7: Post-Training Quantization — Track B (GGUF / llama.cpp).

Implements T7.1-T7.7 from
planning/phases/phase_07_post_training_quantization_track_b_gguf_llama_cpp.md.

Converts merged BF16 checkpoints from Phase 4 (Bengali) and Phase 5 (English)
to GGUF format and quantizes to F16, Q8_0, Q4_K_M, and Q3_K_M (optional Q2_K).
Extracts per-example logits on id_val, id_test, and ood_test via llama-cpp-python.
Validates F16 agreement with HF BF16 (T7.5 critical gate) and checks file size
compression ratios (T7.2, T7.3, T7.4).

Global Rule 4: Track A and Track B results are kept strictly separate.

Usage:
    python scripts/04_quantize_track_b.py                     # full Track B sweep
    python scripts/04_quantize_track_b.py --smoke             # 32 eval rehearsal
    python scripts/04_quantize_track_b.py --methods f16 q8_0
    python scripts/04_quantize_track_b.py --models qwen2_5_0_5b_instruct
    python scripts/04_quantize_track_b.py --languages bn
    python scripts/04_quantize_track_b.py --status            # print status matrix and exit
    python scripts/04_quantize_track_b.py --dry-run           # display matrix and exit
    python scripts/04_quantize_track_b.py --force             # re-run existing successes
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import re
import shutil
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import llama_cpp
from omegaconf import OmegaConf

from src.data.assemble import SPLIT_PARTS, build_split_frames
from src.data.splits import load_splits
from src.metrics.aggregate import evaluate_logits_frame, metrics_frame
from src.metrics.efficiency import Stopwatch
from src.metrics.validation import validate_logits_frame
from src.models import finetune as ft
from src.models import logit_extraction as lx
from src.models.quantize_track_b import (
    ALL_TRACK_B_METHODS,
    GGUFConversionUnavailable,
    GGUFQuantizeUnavailable,
    TRACK_B_METHODS,
    convert_to_gguf,
    extract_logits_llamacpp,
    free_gguf,
    gguf_file_size_mb,
    is_valid_gguf_file,
    load_gguf_model,
    quantize_gguf,
    sanity_check_gguf_f16,
)
from src.utils.atomic import (
    atomic_write_csv,
    atomic_write_json,
    atomic_write_parquet,
    is_valid_merged,
    is_valid_parquet,
)
from src.utils.config import config_hash, load_config
from src.utils.logging_utils import setup_logging, write_run_json
from src.utils.reproducibility import collect_env_snapshot, get_git_commit, set_seed

LOG = setup_logging()

CFG_PATH = ROOT / "configs" / "quantization" / "track_b.yaml"
EVAL_SPLITS: tuple[str, ...] = ("id_val", "id_test", "ood_test")

_INTERRUPTED = False


def _handle_interrupt(signum: int, frame: Any) -> None:
    global _INTERRUPTED
    LOG.warning("Received signal %s (interrupted). Stopping gracefully...", signum)
    _INTERRUPTED = True
    ft.set_interrupted(True)


signal.signal(signal.SIGINT, _handle_interrupt)
if hasattr(signal, "SIGBREAK"):
    signal.signal(signal.SIGBREAK, _handle_interrupt)


def is_interrupted() -> bool:
    return _INTERRUPTED or ft.is_interrupted()


def _resolve(path_like: str | Path) -> Path:
    p = Path(path_like)
    return p if p.is_absolute() else ROOT / p


def _dev(rel: Path) -> str:
    try:
        return str(rel.relative_to(ROOT))
    except ValueError:
        return str(rel)


# ---------------------------------------------------------------------------
# Matrix Builder
# ---------------------------------------------------------------------------

def parse_run_name(run_name: str) -> dict[str, Any]:
    """Parse canonical run directory name into components."""
    m = re.match(r"^(bn|en)_(.+)_(?:lora_bf16|lora-bf16)_s(\d+)$", run_name)
    if not m:
        lang = "en" if run_name.startswith("en_") else "bn"
        return {"language": lang, "model_key": run_name, "seed": 0}
    lang, model_key, seed_str = m.groups()
    return {"language": lang, "model_key": model_key, "seed": int(seed_str)}


def build_quant_matrix_b(
    cfg: Any,
    *,
    models: Sequence[str] | None = None,
    methods: Sequence[str] | None = None,
    languages: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    """Scan fine-tuned source models and expand across Track B quantization methods."""
    allowed_methods = list(methods) if methods else list(cfg["track_b"]["methods"])
    sources = cfg["sources"]
    matrix: list[dict[str, Any]] = []

    for source_key, source_cfg in sources.items():
        lang = str(source_cfg["language"])
        if languages and lang not in languages:
            continue

        base_models_dir = _resolve(source_cfg["path"])
        if not base_models_dir.is_dir():
            LOG.warning("Source directory %s not found; skipping source %s", base_models_dir, source_key)
            continue

        for run_dir in sorted(base_models_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            run_name = run_dir.name
            merged_dir = run_dir / "merged"

            # Must have a valid merged directory (is_valid_merged); do not fall back to adapter dir
            if not is_valid_merged(merged_dir):
                LOG.warning("Skipping source run %s: no valid merged directory found at %s", run_name, merged_dir)
                continue

            parsed = parse_run_name(run_name)
            model_key = parsed["model_key"]
            seed = parsed["seed"]

            if models and model_key not in models:
                continue
            if seeds and seed not in seeds:
                continue

            for method in allowed_methods:
                artifact_name = f"{run_name}__{method}"
                matrix.append(
                    {
                        "artifact_name": artifact_name,
                        "base_run_name": run_name,
                        "source_key": source_key,
                        "source_phase": 4 if lang == "bn" else 5,
                        "language": lang,
                        "model_key": model_key,
                        "seed": seed,
                        "method": method,
                        "merged_dir": str(merged_dir),
                        "source_cfg": source_cfg,
                    }
                )

    return matrix


# ---------------------------------------------------------------------------
# Status Display
# ---------------------------------------------------------------------------

def show_status(cfg: Any, matrix: list[dict[str, Any]]) -> None:
    """Print the Track B quantization sweep status table."""
    runs_dir = _resolve(cfg["paths"]["runs"])
    logits_dir = _resolve(cfg["paths"]["logits"])

    rows: list[dict[str, str]] = []
    for cell in matrix:
        name = cell["artifact_name"]
        method = cell["method"]
        run_json = runs_dir / f"{name}.json"
        run_logits = logits_dir / name

        status = "pending"
        size = "-"
        wall = "-"
        agree = "-"
        f1_test = "-"

        if run_json.is_file():
            try:
                data = json.loads(run_json.read_text(encoding="utf-8"))
                status = str(data.get("status", "unknown"))
                size_val = data.get("file_size_mb")
                size = f"{size_val:.1f} MB" if size_val is not None else "-"
                wall_val = data.get("wall_s")
                wall = f"{wall_val:.1f}s" if wall_val is not None else "-"
                f1_val = (data.get("metrics") or {}).get("macro_f1")
                f1_test = f"{f1_val:.4f}" if f1_val is not None else "-"
                t7_5 = data.get("T7.5_detail") or {}
                if t7_5.get("agreement") is not None:
                    agree = f"{t7_5['agreement'] * 100:.1f}%"
            except Exception:
                status = "corrupted_json"
        elif run_logits.is_dir():
            parquets = list(run_logits.glob("*.parquet"))
            if len(parquets) == len(EVAL_SPLITS):
                status = "logits_extracted"
            elif parquets:
                status = "partial_logits"

        rows.append(
            {
                "artifact_name": name,
                "method": method,
                "status": status,
                "size": size,
                "f16_agree": agree,
                "id_test_f1": f1_test,
                "wall_time": wall,
            }
        )

    headers = ["Artifact Name", "Method", "Status", "Size", "F16 Agree", "id_test F1", "Wall Time"]
    col_keys = ["artifact_name", "method", "status", "size", "f16_agree", "id_test_f1", "wall_time"]
    col_widths = [len(h) for h in headers]
    for r in rows:
        for idx, k in enumerate(col_keys):
            col_widths[idx] = max(col_widths[idx], len(r[k]))

    sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
    hdr_line = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"

    print("\nPhase 7 Post-Training Quantization (Track B GGUF) Status:")
    print(sep)
    print(hdr_line)
    print(sep)
    for r in rows:
        line = "| " + " | ".join(r[k].ljust(col_widths[i]) for i, k in enumerate(col_keys)) + " |"
        print(line)
    print(sep + "\n")


# ---------------------------------------------------------------------------
# Dataset and Splits Caching
# ---------------------------------------------------------------------------

_DATASET_CACHE: dict[str, tuple[pd.DataFrame, dict[str, Any], str]] = {}


def get_dataset_and_splits(source_cfg: Any) -> tuple[pd.DataFrame, dict[str, Any], str]:
    cache_key = str(source_cfg["splits_path"])
    if cache_key in _DATASET_CACHE:
        return _DATASET_CACHE[cache_key]

    proc_path = _resolve(source_cfg["data_path"])
    splits_path = _resolve(source_cfg["splits_path"])

    df = pd.read_csv(proc_path)
    df["example_id"] = df["example_id"].astype(str)

    splits, sha = load_splits(splits_path)
    split_name = str(source_cfg["split"])
    if split_name not in splits:
        raise KeyError(f"split {split_name!r} not in {splits_path}; available: {list(splits)}")

    res = (df, splits[split_name], sha)
    _DATASET_CACHE[cache_key] = res
    return res


# ---------------------------------------------------------------------------
# Per-Artifact Execution (T7.1 - T7.7)
# ---------------------------------------------------------------------------

def run_one_quant_b(
    cell: dict[str, Any],
    cfg: Any,
    *,
    force: bool = False,
    caps: dict[str, int | None] | None = None,
    n_gpu_layers: int | None = None,
    n_threads: int | None = None,
) -> dict[str, Any]:
    """Execute GGUF conversion, quantization, logit extraction, and gate validation."""
    if is_interrupted():
        raise KeyboardInterrupt("Stopped before starting artifact")

    name = cell["artifact_name"]
    base_run = cell["base_run_name"]
    method = cell["method"]
    source_cfg = cell["source_cfg"]
    lang = cell["language"]
    seed = cell["seed"]
    merged_dir = cell["merged_dir"]

    runs_dir = _resolve(cfg["paths"]["runs"])
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_json_path = runs_dir / f"{name}.json"

    quant_models_dir = _resolve(cfg["paths"]["quantized_models"])
    quant_models_dir.mkdir(parents=True, exist_ok=True)

    logits_dir = _resolve(cfg["paths"]["logits"]) / name
    logits_dir.mkdir(parents=True, exist_ok=True)
    expected_logits = {s: logits_dir / f"{method}__{s}.parquet" for s in EVAL_SPLITS}

    manifest: dict[str, Any] = {
        "artifact_name": name,
        "phase": 7,
        "phase_name": "post_training_quantization_track_b_gguf_llama_cpp",
        "track": "B",
        "base_run_name": base_run,
        "model_key": cell["model_key"],
        "method": method,
        "language": lang,
        "seed": seed,
        "source_phase": cell["source_phase"],
        "merged_dir": _dev(Path(merged_dir)),
        "backend": "llama.cpp",
        "llama_cpp_version": getattr(llama_cpp, "__version__", "unknown"),
        "n_gpu_layers": n_gpu_layers if n_gpu_layers is not None else int(cfg["track_b"]["runtime"].get("n_gpu_layers", -1)),
        "gpu_offload_supported": bool(llama_cpp.llama_supports_gpu_offload()),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
        "config_hash": config_hash(cfg),
        "git_commit": get_git_commit(str(ROOT)),
    }

    # Load dataset & materialize evaluation frames
    df, spec, split_sha = get_dataset_and_splits(source_cfg)
    eval_caps = caps or {}
    frames = build_split_frames(df, spec, caps=eval_caps, seed=seed, parts=SPLIT_PARTS)

    # Resumability check
    curr_cfg_hash = config_hash(cfg)
    target_gguf_file = quant_models_dir / f"{name}.gguf"

    if not force and run_json_path.is_file():
        try:
            prev = json.loads(run_json_path.read_text(encoding="utf-8"))
            parquets_valid = all(
                is_valid_parquet(expected_logits[s], expected_rows=len(frames[s]))
                for s in EVAL_SPLITS
            )
            hash_matches = prev.get("config_hash") == curr_cfg_hash
            gguf_valid = is_valid_gguf_file(target_gguf_file)
            if prev.get("status") == "success" and parquets_valid and hash_matches and gguf_valid:
                LOG.info("SKIP %s (already completed with matching config, valid gguf, and valid logits)", name)
                prev["skipped"] = True
                return prev
        except Exception as exc:
            LOG.warning("Failed to parse existing manifest for %s (%s); re-running", name, exc)

    # Stop quantizing after a T7.5 failure (Critical Gate)
    if method != "f16":
        f16_artifact = f"{base_run}__f16"
        f16_json_path = runs_dir / f"{f16_artifact}.json"
        if f16_json_path.is_file():
            try:
                f16_prev = json.loads(f16_json_path.read_text(encoding="utf-8"))
                t7_5 = f16_prev.get("T7.5") or {}
                if t7_5.get("passed") is False:
                    LOG.error(
                        "HALT: Skipping %s because base GGUF F16 (%s) failed critical gate T7.5",
                        name,
                        f16_artifact,
                    )
                    manifest["status"] = "skipped_f16_gate_failed"
                    manifest["error"] = f"Base GGUF F16 ({f16_artifact}) failed critical gate T7.5"
                    manifest["wall_s"] = 0.0
                    manifest["environment"] = collect_env_snapshot()
                    write_run_json(
                        run_json_path,
                        phase=7,
                        phase_name="post_training_quantization_track_b_gguf_llama_cpp",
                        status=manifest["status"],
                        payload={k: v for k, v in manifest.items() if k not in {"phase", "phase_name", "status"}},
                        seed=seed,
                        config_hash=manifest["config_hash"],
                        git_commit=manifest["git_commit"],
                    )
                    return manifest
            except Exception as exc:
                LOG.warning("Failed to inspect base F16 manifest %s: %s", f16_json_path, exc)

    t_start = time.perf_counter()
    f16_gguf_path = quant_models_dir / f"{base_run}__f16.gguf"
    llm = None
    tokenizer = None

    try:
        sw = Stopwatch(f"quant_b::{name}")
        with sw:
            # ---- STAGE 1: GGUF F16 Conversion (T7.1) ----------------------
            convert_script = cfg["paths"].get("convert_script")
            if not is_valid_gguf_file(f16_gguf_path) or (method == "f16" and force):
                LOG.info("Converting base checkpoint to GGUF F16: %s -> %s", merged_dir, f16_gguf_path.name)
                conv_info = convert_to_gguf(
                    merged_dir,
                    f16_gguf_path,
                    outtype="f16",
                    convert_script=convert_script,
                )
                manifest["f16_conversion"] = conv_info
            else:
                LOG.info("Reusing existing base GGUF F16 model: %s", f16_gguf_path.name)
                manifest["f16_conversion"] = {
                    "reused": True,
                    "output_path": str(f16_gguf_path),
                    "file_size_mb": gguf_file_size_mb(f16_gguf_path),
                }

            f16_size_mb = gguf_file_size_mb(f16_gguf_path)
            manifest["f16_size_mb"] = f16_size_mb

            # ---- STAGE 2: Quantize GGUF (T7.2, T7.3, T7.4) ----------------
            if method == "f16":
                artifact_file = f16_gguf_path
                manifest["quant_info"] = {
                    "method": "f16",
                    "output_path": str(artifact_file),
                    "file_size_mb": f16_size_mb,
                    "quantize_s": 0.0,
                }
                manifest["file_size_mb"] = f16_size_mb
            else:
                artifact_file = target_gguf_file
                if not is_valid_gguf_file(artifact_file) or force:
                    q_threads = n_threads if n_threads is not None else cfg["track_b"]["runtime"].get("n_threads")
                    LOG.info("Quantizing GGUF: %s -> %s (%s)", f16_gguf_path.name, artifact_file.name, method)
                    q_info = quantize_gguf(
                        f16_gguf_path,
                        artifact_file,
                        quant_type=method,
                        n_threads=q_threads,
                    )
                    manifest["quant_info"] = q_info
                    manifest["file_size_mb"] = q_info["output_size_mb"]
                else:
                    LOG.info("Reusing existing quantized GGUF model: %s", artifact_file.name)
                    q_size = gguf_file_size_mb(artifact_file)
                    manifest["quant_info"] = {
                        "reused": True,
                        "method": method,
                        "output_path": str(artifact_file),
                        "file_size_mb": q_size,
                        "quantize_s": 0.0,
                    }
                    manifest["file_size_mb"] = q_size

            # Record relative size metrics for verification
            art_size = manifest["file_size_mb"]
            manifest["compression_ratio_vs_f16"] = round(art_size / f16_size_mb, 4) if f16_size_mb > 0 else 0.0

            # ---- STAGE 3: Inference & Logit Extraction (T7.6, T7.7) --------
            gpu_layers = n_gpu_layers if n_gpu_layers is not None else int(cfg["track_b"]["runtime"]["n_gpu_layers"])
            threads = n_threads if n_threads is not None else cfg["track_b"]["runtime"].get("n_threads")
            n_ctx = int(cfg["track_b"]["runtime"].get("n_ctx", 512))

            LOG.info(
                "Loading GGUF model for inference: %s (n_ctx=%d, n_gpu_layers=%s)",
                artifact_file.name,
                n_ctx,
                gpu_layers,
            )
            llm = load_gguf_model(
                artifact_file,
                n_ctx=n_ctx,
                n_gpu_layers=gpu_layers,
                n_threads=threads,
                verbose=bool(cfg["track_b"]["runtime"].get("verbose", False)),
                logits_all=True,
            )

            # Load HF tokenizer from merged directory to ensure 100% vocabulary fidelity
            try:
                tokenizer = ft.load_tokenizer(merged_dir)
            except Exception as exc:
                LOG.warning("Could not load HF tokenizer from %s (%s); will fall back to GGUF tokenizer", merged_dir, exc)
                tokenizer = None

            t_eval_start = time.perf_counter()
            for split in EVAL_SPLITS:
                if is_interrupted():
                    raise KeyboardInterrupt(f"Interrupted during eval split {split}")

                target_parquet = expected_logits[split]
                expected_rows = len(frames[split])
                if not force and is_valid_parquet(target_parquet, expected_rows=expected_rows):
                    LOG.info("Skip split %s for %s (valid parquet exists: %d rows)", split, name, expected_rows)
                    continue

                frame = extract_logits_llamacpp(
                    llm,
                    frames[split],
                    verbalizer=cfg["verbalizer"],
                    split=split,
                    max_len=256,
                    template=str(source_cfg["prompt_template"]),
                    model_id=f"{base_run}|{method}",
                    tokenizer=tokenizer,
                    should_stop=is_interrupted,
                )
                lx.save_logits(
                    frame,
                    target_parquet,
                    model=cell["model_key"],
                    canonical_model=cell["model_key"],
                    model_key=cell["model_key"],
                    precision=method,
                    seed=seed,
                    language=lang,
                    split_name=str(source_cfg["split"]),
                    run_name=name,
                )
            manifest["eval_s"] = round(time.perf_counter() - t_eval_start, 2)

        manifest["wall_s"] = round(sw.wall_s, 2)

        # ---- STAGE 4: Evaluate Metrics & Validation Gates (T7.1 - T7.7) ----
        validation: dict[str, Any] = {}
        split_metrics: dict[str, Any] = {}

        for split in EVAL_SPLITS:
            frame = pd.read_parquet(expected_logits[split])
            validation[split] = validate_logits_frame(
                frame, min_macro_f1=0.45, require_both_classes=False
            )
            split_metrics[split] = evaluate_logits_frame(
                frame,
                ece_bins=int(cfg["track_b"]["eval"]["ece_bins"]),
                ece_strategy=str(cfg["track_b"]["eval"]["ece_strategy"]),
            )

        manifest["validation"] = validation
        manifest["split_metrics"] = split_metrics
        manifest["metrics"] = split_metrics.get("id_test", {})

        # T7.1: GGUF F16 loads and is valid
        manifest["T7.1"] = {
            "passed": is_valid_gguf_file(f16_gguf_path),
            "f16_size_mb": f16_size_mb,
        }

        # T7.2: Q8_0 quantization completed and size < F16
        if method == "q8_0":
            manifest["T7.2"] = {
                "passed": is_valid_gguf_file(artifact_file) and (art_size < f16_size_mb),
                "q8_size_mb": art_size,
                "f16_size_mb": f16_size_mb,
            }

        # T7.3: Q4_K_M quantization completed and size ~25-30% of F16
        if method == "q4_k_m":
            ratio = art_size / f16_size_mb if f16_size_mb > 0 else 1.0
            manifest["T7.3"] = {
                "passed": is_valid_gguf_file(artifact_file) and (0.15 <= ratio <= 0.45),
                "q4_size_mb": art_size,
                "f16_size_mb": f16_size_mb,
                "ratio": round(ratio, 4),
            }

        # T7.4: Q3_K_M quantization completed
        if method == "q3_k_m":
            manifest["T7.4"] = {
                "passed": is_valid_gguf_file(artifact_file) and (art_size < f16_size_mb),
                "q3_size_mb": art_size,
            }

        # T7.5: Critical Gate — GGUF F16 approximately matches HF BF16
        if method == "f16":
            fp_logits_dir = _resolve(source_cfg["fp_logits_dir"]) / base_run
            fp_id_test_file = fp_logits_dir / "bf16__id_test.parquet"
            if fp_id_test_file.is_file():
                fp_logits = pd.read_parquet(fp_id_test_file)
                gguf_logits = pd.read_parquet(expected_logits["id_test"])
                tol = OmegaConf.to_container(cfg["track_b"]["f16_tolerance"], resolve=True)
                t7_5_res = sanity_check_gguf_f16(fp_logits, gguf_logits, tolerance=tol)
                manifest["T7.5_detail"] = t7_5_res
                manifest["T7.5"] = {"passed": t7_5_res["passed"]}
                if not t7_5_res["passed"]:
                    LOG.error("CRITICAL: T7.5 GGUF F16 agreement gate failed for %s: %s", name, t7_5_res["checks"])
                    manifest["status"] = "failed"
                    manifest["error"] = "T7.5 GGUF F16 agreement gate failed"
            else:
                LOG.warning("FP reference logits not found at %s; skipping T7.5 gate", fp_id_test_file)
                manifest["T7.5"] = {"passed": None, "note": "fp_reference_missing"}

        # T7.6: Logits parsed correctly, probabilities valid, no NaNs
        manifest["T7.6"] = {
            "passed": all(validation[s]["checks"]["probs_sum_to_one"] for s in EVAL_SPLITS),
            "no_nan_logits": all(validation[s]["checks"]["no_nan_logits"] for s in EVAL_SPLITS),
        }

        # T7.7: Parquet files saved for all 3 splits
        manifest["T7.7"] = {
            "expected": [f"{method}__{s}.parquet" for s in EVAL_SPLITS],
            "passed": all(expected_logits[s].is_file() for s in EVAL_SPLITS),
        }

        if manifest.get("status") != "failed":
            manifest["status"] = "success"

    except KeyboardInterrupt:
        LOG.warning("Artifact %s interrupted by user", name)
        raise
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        manifest["traceback"] = traceback.format_exc()
        manifest["wall_s"] = round(time.perf_counter() - t_start, 2)
        LOG.error("FAILED %s: %s", name, exc)
    finally:
        if llm is not None:
            free_gguf(llm)
        llm = None
        tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    manifest["environment"] = collect_env_snapshot()
    write_run_json(
        run_json_path,
        phase=7,
        phase_name="post_training_quantization_track_b_gguf_llama_cpp",
        status=manifest["status"],
        payload={k: v for k, v in manifest.items() if k not in {"phase", "phase_name", "status"}},
        seed=seed,
        config_hash=manifest["config_hash"],
        git_commit=manifest["git_commit"],
    )
    return manifest


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> dict[str, Any]:
    cfg_path = Path(args.config) if getattr(args, "config", None) else CFG_PATH
    overrides: list[str] = []
    cfg = load_config(cfg_path, overrides)

    is_smoke = bool(args.smoke or args.max_eval)
    if is_smoke:
        cfg["paths"]["runs"] = "outputs/runs/phase7_smoke"
        cfg["paths"]["logits"] = "outputs/logits/phase7_smoke"
        cfg["paths"]["quantized_models"] = "outputs/quantized/track_b_smoke"
        cfg["paths"]["results"] = "outputs/results/phase7_smoke"
        cfg["paths"]["metrics_csv"] = "outputs/results/phase7_smoke/phase7_metrics.csv"
        cfg["paths"]["summary"] = "outputs/results/phase7_smoke/phase7_summary.json"

    matrix = build_quant_matrix_b(
        cfg,
        models=args.models or None,
        methods=args.methods or None,
        languages=args.languages or None,
        seeds=args.seeds or None,
    )

    if not matrix:
        raise ValueError(
            "Track B quantization matrix is empty! Verify source directories contain valid 'merged' folders."
        )

    if args.dry_run:
        print(f"\nTrack B Quantization Matrix ({len(matrix)} scheduled artifacts):")
        for idx, item in enumerate(matrix, 1):
            print(f"  [{idx:02d}] {item['artifact_name']} (lang={item['language']}, seed={item['seed']})")
        return {"matrix": matrix, "count": len(matrix)}

    if args.status:
        show_status(cfg, matrix)
        return {"matrix": matrix, "count": len(matrix)}

    LOG.info("Starting Phase 7 Post-Training Quantization (Track B GGUF): %d artifacts", len(matrix))
    if is_smoke:
        LOG.info("Running in SMOKE mode (eval capped at %d samples)", args.max_eval or 32)

    eval_caps = {"id_val": args.max_eval or 32, "id_test": args.max_eval or 32, "ood_test": args.max_eval or 32} if is_smoke else None

    results: list[dict[str, Any]] = []
    t_start_all = time.perf_counter()
    interrupted = False

    try:
        for idx, cell in enumerate(matrix, 1):
            if is_interrupted():
                LOG.warning("Execution interrupted before artifact %d (%s)", idx, cell["artifact_name"])
                interrupted = True
                break

            LOG.info("\n[%d / %d] Processing Track B artifact: %s", idx, len(matrix), cell["artifact_name"])
            res = run_one_quant_b(
                cell,
                cfg,
                force=args.force,
                caps=eval_caps,
                n_gpu_layers=args.n_gpu_layers,
                n_threads=args.n_threads,
            )
            results.append(res)
    except KeyboardInterrupt:
        LOG.warning("\n[STOPPED] Track B sweep interrupted. Re-run the same command to resume.\n")
        interrupted = True

    wall_total = round(time.perf_counter() - t_start_all, 2)

    # Compile metrics CSV across all completed runs
    metrics_records: list[dict[str, Any]] = []
    for r in results:
        if r.get("status") == "success" and "split_metrics" in r:
            for s, s_metrics in r["split_metrics"].items():
                record = {
                    "artifact_name": r["artifact_name"],
                    "phase": 7,
                    "track": "B",
                    "base_run_name": r["base_run_name"],
                    "model_key": r["model_key"],
                    "method": r["method"],
                    "language": r["language"],
                    "seed": r["seed"],
                    "split": s,
                    "file_size_mb": r.get("file_size_mb"),
                    **s_metrics,
                }
                metrics_records.append(record)

    if metrics_records:
        csv_path = _resolve(cfg["paths"]["metrics_csv"])
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(metrics_records).to_csv(csv_path, index=False)
        LOG.info("Wrote Phase 7 metrics to %s (%d rows)", csv_path, len(metrics_records))

    # Build comprehensive T7.1 - T7.7 tests verdict
    f16_results = [r for r in results if r.get("method") == "f16"]
    q8_results = [r for r in results if r.get("method") == "q8_0"]
    q4_results = [r for r in results if r.get("method") == "q4_k_m"]
    q3_results = [r for r in results if r.get("method") == "q3_k_m"]
    success_results = [r for r in results if r.get("status") == "success"]

    tests_verdict: dict[str, Any] = {
        "T7.1": all((r.get("T7.1") or {}).get("passed", False) for r in f16_results) if f16_results else None,
        "T7.2": all((r.get("T7.2") or {}).get("passed", False) for r in q8_results) if q8_results else None,
        "T7.3": all((r.get("T7.3") or {}).get("passed", False) for r in q4_results) if q4_results else None,
        "T7.4": all((r.get("T7.4") or {}).get("passed", False) for r in q3_results) if q3_results else None,
        "T7.5": all((r.get("T7.5") or {}).get("passed") is True for r in f16_results if (r.get("T7.5") or {}).get("passed") is not None) if f16_results else None,
        "T7.6": all((r.get("T7.6") or {}).get("passed", False) for r in success_results) if success_results else None,
        "T7.7": all((r.get("T7.7") or {}).get("passed", False) for r in success_results) if success_results else None,
    }

    failed_count = sum(1 for r in results if r.get("status") == "failed")
    success_count = sum(1 for r in results if r.get("status") == "success")
    skipped_count = sum(1 for r in results if str(r.get("status", "")).startswith("skipped"))

    if interrupted:
        overall_status = "stopped"
    elif failed_count > 0:
        overall_status = "failed"
    elif success_count > 0 and all(v is not False for v in tests_verdict.values() if v is not None):
        overall_status = "passed"
    else:
        overall_status = "pending"

    # Compile comprehensive summary JSON
    summary_path = _resolve(cfg["paths"]["summary"])
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_data = {
        "phase": 7,
        "phase_name": "post_training_quantization_track_b_gguf_llama_cpp",
        "track": "B",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": overall_status,
        "total_artifacts": len(matrix),
        "executed_count": len(results),
        "success_count": success_count,
        "failed_count": failed_count,
        "skipped_count": skipped_count,
        "total_wall_s": wall_total,
        "tests": tests_verdict,
        "results": results,
    }
    atomic_write_json(summary_path, summary_data)
    LOG.info("Wrote Phase 7 summary manifest to %s (status: %s)", summary_path, overall_status)

    return summary_data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 7: Post-Training Quantization — Track B (GGUF / llama.cpp)"
    )
    parser.add_argument("--config", type=str, default=str(CFG_PATH), help="Path to track_b.yaml config")
    parser.add_argument("--smoke", action="store_true", help="Run rapid smoke test with 32 samples")
    parser.add_argument("--dry-run", action="store_true", help="Print experiment matrix and exit")
    parser.add_argument("--status", action="store_true", help="Print status of all planned runs and exit")
    parser.add_argument("--force", action="store_true", help="Re-run completed artifacts")
    parser.add_argument("--models", nargs="+", help="Filter to specific models (e.g. qwen2_5_0_5b_instruct)")
    parser.add_argument("--methods", nargs="+", help="Filter to specific methods (e.g. f16 q8_0 q4_k_m q3_k_m)")
    parser.add_argument("--languages", nargs="+", choices=["bn", "en"], help="Filter by language")
    parser.add_argument("--seeds", nargs="+", type=int, help="Filter by seeds")
    parser.add_argument("--max-eval", type=int, default=None, help="Cap eval samples per split")
    parser.add_argument("--n-gpu-layers", type=int, default=None, help="GPU layers to offload (-1 all, 0 CPU)")
    parser.add_argument("--n-threads", type=int, default=None, help="CPU threads to use")

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
