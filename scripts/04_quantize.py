#!/usr/bin/env python
"""Phase 6: Post-Training Quantization — Track A (GPU / Hugging Face).

Implements T6.1-T6.9 from
planning/phases/phase_06_post_training_quantization_track_a_gpu_hf.md.

Quantizes fine-tuned BF16 checkpoints from Phase 4 (Bengali) and Phase 5 (English)
to BNB INT8, BNB NF4, GPTQ-4, and AWQ-4. Extracts per-example logits on id_val,
id_test, and ood_test. Validates near-lossless INT8 fidelity (T6.5 critical gate)
and verifies zero calibration data leakage (T6.8, T6.9).

Usage:
    python scripts/04_quantize.py                     # full Track A sweep
    python scripts/04_quantize.py --smoke             # 32 eval / 32 calib rehearsal
    python scripts/04_quantize.py --methods bnb_int8 bnb_nf4
    python scripts/04_quantize.py --models qwen2_5_0_5b_instruct
    python scripts/04_quantize.py --languages bn
    python scripts/04_quantize.py --status            # print status matrix and exit
    python scripts/04_quantize.py --dry-run           # display matrix and exit
    python scripts/04_quantize.py --force             # re-run existing successes
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
from omegaconf import OmegaConf

from src.data.assemble import SPLIT_PARTS, build_split_frames
from src.data.splits import load_splits
from src.metrics.aggregate import evaluate_logits_frame, metrics_frame
from src.metrics.efficiency import Stopwatch
from src.metrics.validation import validate_logits_frame
from src.models import finetune as ft
from src.models import logit_extraction as lx
from src.models.quantize_track_a import (
    AWQUnavailable,
    GPTQUnavailable,
    TRACK_A_METHODS,
    artifact_size_mb,
    build_calibration_texts,
    calibration_provenance,
    free,
    gptq_backend_status,
    load_gptq,
    load_quantized_hf,
    quant_summary,
    quantize_awq,
    quantize_gptq,
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
from src.utils.sanity_checks import sanity_check_int8

LOG = setup_logging()

CFG_PATH = ROOT / "configs" / "quantization" / "default.yaml"
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
    """Parse canonical run directory name into components.

    Example: ``bn_qwen2_5_0_5b_instruct_lora_bf16_s13``
    """
    m = re.match(r"^(bn|en)_(.+)_(?:lora_bf16|lora-bf16)_s(\d+)$", run_name)
    if not m:
        # Fallback heuristic
        lang = "en" if run_name.startswith("en_") else "bn"
        return {"language": lang, "model_key": run_name, "seed": 0}
    lang, model_key, seed_str = m.groups()
    return {"language": lang, "model_key": model_key, "seed": int(seed_str)}


def build_quant_matrix(
    cfg: Any,
    *,
    models: Sequence[str] | None = None,
    methods: Sequence[str] | None = None,
    languages: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    """Scan fine-tuned source models and expand across Track A quantization methods."""
    allowed_methods = list(methods) if methods else list(cfg["track_a"]["methods"])
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
            adapter_dir = run_dir / "adapter"

            # Must have merged model or adapter
            if not (merged_dir.is_dir() or adapter_dir.is_dir()):
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
                        "model_path": str(merged_dir if merged_dir.is_dir() else adapter_dir),
                        "merged_dir": str(merged_dir),
                        "adapter_dir": str(adapter_dir),
                        "source_cfg": source_cfg,
                    }
                )

    return matrix


# ---------------------------------------------------------------------------
# Status Display
# ---------------------------------------------------------------------------

def show_status(cfg: Any, matrix: list[dict[str, Any]]) -> None:
    """Print the Track A quantization sweep status table."""
    runs_dir = _resolve(cfg["paths"]["runs"])
    logits_dir = _resolve(cfg["paths"]["logits"])

    rows: list[dict[str, str]] = []
    for cell in matrix:
        name = cell["artifact_name"]
        method = cell["method"]
        run_json = runs_dir / f"{name}.json"
        run_logits = logits_dir / name

        status = "pending"
        vram = "-"
        wall = "-"
        int8_agree = "-"
        f1_test = "-"

        if run_json.is_file():
            try:
                data = json.loads(run_json.read_text(encoding="utf-8"))
                status = str(data.get("status", "unknown"))
                vram_val = data.get("peak_vram_gb")
                vram = f"{vram_val:.2f} GB" if vram_val is not None else "-"
                wall_val = data.get("wall_s")
                wall = f"{wall_val:.1f}s" if wall_val is not None else "-"
                f1_val = (data.get("metrics") or {}).get("macro_f1")
                f1_test = f"{f1_val:.4f}" if f1_val is not None else "-"
                int8_chk = data.get("T6.5_detail") or {}
                if int8_chk.get("agreement") is not None:
                    int8_agree = f"{int8_chk['agreement'] * 100:.1f}%"
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
                "int8_agree": int8_agree,
                "id_test_f1": f1_test,
                "peak_vram": vram,
                "wall_time": wall,
            }
        )

    headers = ["Artifact Name", "Method", "Status", "INT8 Agree", "id_test F1", "Peak VRAM", "Wall Time"]
    col_keys = ["artifact_name", "method", "status", "int8_agree", "id_test_f1", "peak_vram", "wall_time"]
    col_widths = [len(h) for h in headers]
    for r in rows:
        for idx, k in enumerate(col_keys):
            col_widths[idx] = max(col_widths[idx], len(r[k]))

    sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
    hdr_line = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"

    print("\nPhase 6 Post-Training Quantization (Track A) Status:")
    print(sep)
    print(hdr_line)
    print(sep)
    for r in rows:
        line = "| " + " | ".join(r[k].ljust(col_widths[i]) for i, k in enumerate(col_keys)) + " |"
        print(line)
    print(sep + "\n")


# ---------------------------------------------------------------------------
# Calibration Dataset Loader & Provenance Guard (T6.8, T6.9)
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


def prepare_calibration_data(
    source_cfg: Any,
    *,
    n_samples: int = 512,
    max_len: int = 256,
    seed: int = 42,
) -> tuple[list[str], list[str], dict[str, Any]]:
    """Sample calibration texts strictly from the training split (T6.8, T6.9).

    Returns:
        calib_texts: formatted prompt strings
        calib_ids: list of example_ids used for calibration
        prov: provenance metadata confirming zero eval leakage
    """
    df, spec, split_sha = get_dataset_and_splits(source_cfg)
    train_ids = spec["train_ids"]
    train_id_set = set(train_ids)

    # Forbidden IDs: any validation or test example (T6.8)
    forbidden_ids = set(spec.get("id_val_ids", [])) | set(spec.get("id_test_ids", [])) | set(spec.get("ood_test_ids", []))

    train_df = df[df["example_id"].isin(train_id_set)].reset_index(drop=True)
    if train_df.empty:
        raise ValueError(f"Training split is empty for {source_cfg['split']}")

    n_sample = min(n_samples, len(train_df))
    sampled_df = train_df.sample(n=n_sample, random_state=seed).reset_index(drop=True)
    calib_ids = sampled_df["example_id"].astype(str).tolist()
    calib_id_set = set(calib_ids)

    # T6.8 Check: programmatic assertion of zero leakage
    leaked_ids = list(calib_id_set.intersection(forbidden_ids))
    assert not leaked_ids, f"CRITICAL LEAKAGE (T6.8): calibration set contains eval IDs {leaked_ids[:5]}!"
    assert calib_id_set.issubset(train_id_set), "Calibration IDs are not a subset of train IDs!"

    # T6.9 Check: calibration language matches source language
    expected_lang = str(source_cfg["language"])
    if "language" in sampled_df.columns:
        calib_langs = sampled_df["language"].unique().tolist()
        assert all(l == expected_lang for l in calib_langs), (
            f"Language mismatch (T6.9): expected {expected_lang}, got {calib_langs}"
        )

    template = str(source_cfg["prompt_template"])
    calib_texts = [template.format(text=" ".join(str(t).split())) for t in sampled_df["text"].tolist()]

    prov = {
        "n_samples": len(calib_texts),
        "source_split": source_cfg["split"],
        "language": expected_lang,
        "split_sha256": split_sha,
        "calib_ids_sample": calib_ids[:5],
        "leaked_into_eval": False,
        "verified_train_only": True,
    }
    return calib_texts, calib_ids, prov


# ---------------------------------------------------------------------------
# Per-Artifact Execution
# ---------------------------------------------------------------------------

def run_one_quant(
    cell: dict[str, Any],
    cfg: Any,
    *,
    force: bool = False,
    caps: dict[str, int | None] | None = None,
) -> dict[str, Any]:
    """Execute quantization, logit extraction, INT8 sanity check, and manifest writing."""
    if is_interrupted():
        raise KeyboardInterrupt("Stopped before starting artifact")

    name = cell["artifact_name"]
    base_run = cell["base_run_name"]
    method = cell["method"]
    source_cfg = cell["source_cfg"]
    lang = cell["language"]
    seed = cell["seed"]

    runs_dir = _resolve(cfg["paths"]["runs"])
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_json_path = runs_dir / f"{name}.json"

    logits_dir = _resolve(cfg["paths"]["logits"]) / name
    logits_dir.mkdir(parents=True, exist_ok=True)
    expected_logits = {s: logits_dir / f"{method}__{s}.parquet" for s in EVAL_SPLITS}

    manifest: dict[str, Any] = {
        "artifact_name": name,
        "phase": 6,
        "phase_name": "post_training_quantization_track_a_gpu_hf",
        "base_run_name": base_run,
        "model_key": cell["model_key"],
        "method": method,
        "language": lang,
        "seed": seed,
        "source_phase": cell["source_phase"],
        "model_path": _dev(Path(cell["model_path"])),
        "config_hash": config_hash(cfg),
        "git_commit": get_git_commit(str(ROOT)),
    }

    # Load dataset & materialize evaluation frames
    df, spec, split_sha = get_dataset_and_splits(source_cfg)
    eval_caps = caps or {}
    frames = build_split_frames(df, spec, caps=eval_caps, seed=seed, parts=SPLIT_PARTS)

    # Resumability check (A7)
    if not force and run_json_path.is_file():
        try:
            prev = json.loads(run_json_path.read_text(encoding="utf-8"))
            parquets_valid = all(
                is_valid_parquet(expected_logits[s], expected_rows=len(frames[s]))
                for s in EVAL_SPLITS
            )
            if prev.get("status") == "success" and parquets_valid:
                LOG.info("SKIP %s (already completed with valid logits)", name)
                prev["skipped"] = True
                return prev
        except Exception as exc:
            LOG.warning("Failed to parse existing manifest for %s (%s); re-running", name, exc)

    t_start = time.perf_counter()
    model = None
    tokenizer = None

    try:
        sw = Stopwatch(f"quant::{name}")
        with sw:
            # ---- STAGE 1: Quantize / Load Model ---------------------------
            if method in {"bnb_int8", "bnb_nf4"}:
                LOG.info("Loading %s with %s", base_run, method)
                model = load_quantized_hf(cell["model_path"], method)
                tokenizer = ft.load_tokenizer(cell["model_path"])
                manifest["quant_info"] = {
                    "method": method,
                    "mode": "on_load",
                    "model_summary": quant_summary(model),
                }
            elif method == "gptq_int4":
                status = gptq_backend_status()
                if status["driver_api"] is None:
                    raise GPTQUnavailable("; ".join(status["problems"]))

                calib_samples = int(cfg["track_a"]["calibration"]["n_samples"])
                calib_max_len = int(cfg["track_a"]["calibration"]["max_len"])
                calib_seed = int(cfg["track_a"]["calibration"]["seed"])
                if eval_caps.get("train"):
                    calib_samples = min(calib_samples, int(eval_caps["train"]))

                calib_texts, calib_ids, prov = prepare_calibration_data(
                    source_cfg, n_samples=calib_samples, max_len=calib_max_len, seed=calib_seed
                )
                manifest["calibration_provenance"] = prov

                quant_out_dir = _resolve(cfg["paths"]["quantized_models"]) / name
                LOG.info("Running GPTQ-4 quantization -> %s (n_calib=%d)", quant_out_dir, len(calib_texts))
                gptq_res = quantize_gptq(
                    cell["model_path"],
                    calib_texts,
                    str(quant_out_dir),
                    bits=4,
                    group_size=128,
                    max_len=calib_max_len,
                )
                manifest["quant_info"] = gptq_res
                manifest["artifact_dir"] = _dev(quant_out_dir)

                # Reload saved GPTQ model for evaluation
                model, tokenizer = load_gptq(str(quant_out_dir))

            elif method == "awq_int4":
                calib_samples = int(cfg["track_a"]["calibration"]["n_samples"])
                calib_texts, calib_ids, prov = prepare_calibration_data(
                    source_cfg, n_samples=calib_samples
                )
                manifest["calibration_provenance"] = prov
                quant_out_dir = _resolve(cfg["paths"]["quantized_models"]) / name
                quantize_awq(cell["model_path"], calib_texts, str(quant_out_dir))
                raise NotImplementedError("AWQ quantization driver not available")
            else:
                raise ValueError(f"Unknown Track A method {method!r}")

            # ---- STAGE 2: Logit Extraction --------------------------------
            for split in EVAL_SPLITS:
                if is_interrupted():
                    raise KeyboardInterrupt(f"Interrupted during eval split {split}")

                frame = lx.extract_logits_hf(
                    model,
                    tokenizer,
                    frames[split],
                    verbalizer=cfg["verbalizer"],
                    batch_size=int(cfg["track_a"]["eval"]["batch_size"]),
                    split=split,
                    max_len=int(cfg["track_a"]["calibration"]["max_len"]),
                    template=str(source_cfg["prompt_template"]),
                    model_id=f"{base_run}|{method}",
                )
                lx.save_logits(
                    frame,
                    expected_logits[split],
                    model=cell["model_key"],
                    canonical_model=cell["model_key"],
                    model_key=cell["model_key"],
                    precision=method,
                    seed=seed,
                    language=lang,
                    split_name=str(source_cfg["split"]),
                    run_name=name,
                )

        manifest["wall_s"] = round(sw.wall_s, 2)
        manifest["peak_vram_gb"] = None if sw.peak_vram_gb is None else round(sw.peak_vram_gb, 3)

        # ---- STAGE 3: Evaluate Metrics & Validation Gates -----------------
        validation: dict[str, Any] = {}
        split_metrics: dict[str, Any] = {}
        min_f1 = 0.50

        for split in EVAL_SPLITS:
            frame = pd.read_parquet(expected_logits[split])
            validation[split] = lx.validate_logits_frame(
                frame, min_macro_f1=min_f1, require_both_classes=False
            )
            split_metrics[split] = evaluate_logits_frame(
                frame,
                ece_bins=int(cfg["track_a"]["eval"]["ece_bins"]),
                ece_strategy=str(cfg["track_a"]["eval"]["ece_strategy"]),
            )

        manifest["validation"] = validation
        manifest["split_metrics"] = split_metrics
        manifest["metrics"] = split_metrics.get("id_test", {})

        # T6.6 Logits saved gate
        manifest["T6.6"] = {
            "expected": [f"{method}__{s}.parquet" for s in EVAL_SPLITS],
            "passed": all(expected_logits[s].is_file() for s in EVAL_SPLITS),
        }

        # T6.7 Valid probabilities gate
        manifest["T6.7"] = {
            "passed": all(validation[s]["checks"]["probs_sum_to_one"] for s in EVAL_SPLITS),
            "no_nan_logits": all(validation[s]["checks"]["no_nan_logits"] for s in EVAL_SPLITS),
        }

        # T6.5 INT8 Sanity Check Gate (when method == bnb_int8)
        if method == "bnb_int8":
            fp_logits_dir = _resolve(source_cfg["fp_logits_dir"]) / base_run
            fp_id_test_file = fp_logits_dir / "bf16__id_test.parquet"
            if fp_id_test_file.is_file():
                fp_logits = pd.read_parquet(fp_id_test_file)
                int8_logits = pd.read_parquet(expected_logits["id_test"])
                tol = OmegaConf.to_container(cfg["track_a"]["int8_tolerance"], resolve=True)
                t6_5_res = sanity_check_int8(fp_logits, int8_logits, tolerance=tol)
                manifest["T6.5_detail"] = t6_5_res
                manifest["T6.5"] = {"passed": t6_5_res["passed"]}
                if not t6_5_res["passed"]:
                    LOG.error("CRITICAL: INT8 sanity check failed for %s: %s", name, t6_5_res["checks"])
            else:
                LOG.warning("FP reference logits not found at %s; skipping T6.5 gate", fp_id_test_file)
                manifest["T6.5"] = {"passed": None, "note": "fp_reference_missing"}

        # Method-specific load gates
        if method == "bnb_int8":
            manifest["T6.1"] = {"passed": True}
        elif method == "bnb_nf4":
            manifest["T6.2"] = {"passed": True}
        elif method == "gptq_int4":
            manifest["T6.3"] = {"passed": True}
        elif method == "awq_int4":
            manifest["T6.4"] = {"passed": True}

        manifest["status"] = "success"

    except (GPTQUnavailable, AWQUnavailable) as exc:
        manifest["status"] = "unavailable"
        manifest["error"] = str(exc)
        manifest["wall_s"] = round(time.perf_counter() - t_start, 2)
        LOG.warning("Backend unavailable for %s: %s", name, exc)
    except KeyboardInterrupt:
        LOG.warning("Artifact %s interrupted by user", name)
        raise
    except Exception as exc:  # noqa: BLE001
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        manifest["traceback"] = traceback.format_exc()
        manifest["wall_s"] = round(time.perf_counter() - t_start, 2)
        LOG.error("FAILED %s: %s", name, exc)
    finally:
        if model is not None:
            free(model)
        gc.collect()

    manifest["environment"] = collect_env_snapshot()
    write_run_json(
        run_json_path,
        phase=6,
        phase_name="post_training_quantization_track_a_gpu_hf",
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
    overrides: list[str] = []
    cfg = load_config(CFG_PATH, overrides)

    is_smoke = bool(args.smoke or args.max_eval)
    if is_smoke:
        cfg["paths"]["runs"] = "outputs/runs/phase6_smoke"
        cfg["paths"]["logits"] = "outputs/logits/phase6_smoke"
        cfg["paths"]["quantized_models"] = "outputs/quantized/track_a_smoke"
        cfg["paths"]["results"] = "outputs/results/phase6_smoke"
        cfg["paths"]["metrics_csv"] = "outputs/results/phase6_smoke/phase6_metrics.csv"
        cfg["paths"]["summary"] = "outputs/results/phase6_smoke/phase6_summary.json"

    matrix = build_quant_matrix(
        cfg,
        models=args.models or None,
        methods=args.methods or None,
        languages=args.languages or None,
        seeds=args.seeds or None,
    )

    if not matrix:
        raise ValueError(
            "Quantization matrix is empty! Ensure Phase 4/5 models exist under outputs/models/ "
            "or adjust your --models / --languages / --methods filters."
        )

    if args.status:
        show_status(cfg, matrix)
        return {"status": "status_displayed", "matrix_size": len(matrix)}

    LOG.info("Phase 6 Track A Quantization Matrix (%d artifacts):", len(matrix))
    for cell in matrix:
        LOG.info("  %-50s method=%-10s lang=%s", cell["artifact_name"], cell["method"], cell["language"])

    if args.dry_run:
        return {"status": "dry_run", "matrix": [c["artifact_name"] for c in matrix]}

    if not torch.cuda.is_available():
        LOG.warning("CUDA unavailable; Track A quantization requires GPU acceleration.")

    caps: dict[str, int | None] = {}
    if is_smoke:
        max_eval = args.max_eval or 32
        caps = {"train": 32, "id_val": max_eval, "id_test": max_eval, "ood_test": max_eval}

    records: list[dict[str, Any]] = []
    stopped_early = False

    try:
        for idx, cell in enumerate(matrix, start=1):
            if is_interrupted():
                stopped_early = True
                break
            LOG.info("--- [%d/%d] %s ---", idx, len(matrix), cell["artifact_name"])
            records.append(run_one_quant(cell, cfg, force=args.force, caps=caps))
    except KeyboardInterrupt:
        stopped_early = True
        LOG.warning("Quantization execution stopped by user; state preserved for resume.")

    if stopped_early:
        print("\n[STOPPED] Execution stopped; re-run the same command to resume.\n")
        return {"status": "stopped", "n_processed": len(records)}

    succeeded = [r for r in records if r.get("status") == "success"]
    failed = [r for r in records if r.get("status") == "failed"]
    unavailable = [r for r in records if r.get("status") == "unavailable"]

    # Compile evaluation metrics across all artifacts
    metric_rows: list[dict[str, Any]] = []
    for rec in succeeded:
        split_metrics = rec.get("split_metrics") or {}
        validation = rec.get("validation") or {}
        for split in EVAL_SPLITS:
            val = validation.get(split, {})
            s_metrics = split_metrics.get(split, {})
            metric_rows.append(
                {
                    "artifact_name": rec["artifact_name"],
                    "base_run_name": rec["base_run_name"],
                    "model_key": rec["model_key"],
                    "precision": rec["method"],
                    "language": rec["language"],
                    "split": split,
                    "seed": rec["seed"],
                    "n": val.get("n_rows"),
                    "macro_f1": val.get("macro_f1"),
                    "accuracy": val.get("accuracy"),
                    "single_class": val.get("single_class"),
                    **s_metrics,
                }
            )

    table = metrics_frame(metric_rows)
    results_dir = _resolve(cfg["paths"]["results"])
    results_dir.mkdir(parents=True, exist_ok=True)
    table_path = _resolve(cfg["paths"]["metrics_csv"])
    atomic_write_csv(table_path, table)

    # Compile Phase 6 Acceptance Tests (T6.1-T6.9)
    int8_records = [r for r in succeeded if r.get("method") == "bnb_int8"]
    nf4_records = [r for r in succeeded if r.get("method") == "bnb_nf4"]
    gptq_records = [r for r in succeeded if r.get("method") == "gptq_int4"]
    awq_records = [r for r in succeeded if r.get("method") == "awq_int4"]

    tests_verdict = {
        "T6.1": bool(int8_records) and all((r.get("T6.1") or {}).get("passed") for r in int8_records),
        "T6.2": bool(nf4_records) and all((r.get("T6.2") or {}).get("passed") for r in nf4_records),
        "T6.3": bool(gptq_records) and all((r.get("T6.3") or {}).get("passed") for r in gptq_records),
        "T6.4": (bool(awq_records) and all((r.get("T6.4") or {}).get("passed") for r in awq_records))
        if any(c["method"] == "awq_int4" for c in matrix)
        else None,
        "T6.5": bool(int8_records) and all((r.get("T6.5") or {}).get("passed") is True for r in int8_records),
        "T6.6": bool(succeeded) and all((r.get("T6.6") or {}).get("passed") for r in succeeded),
        "T6.7": bool(succeeded) and all((r.get("T6.7") or {}).get("passed") for r in succeeded),
        "T6.8": bool(records) and all(
            (r.get("calibration_provenance") or {}).get("verified_train_only", True) for r in records
        ),
        "T6.9": bool(records) and all(
            (r.get("calibration_provenance") or {}).get("language") in {None, r.get("language")} for r in records
        ),
    }

    summary = {
        "phase": 6,
        "phase_name": "post_training_quantization_track_a_gpu_hf",
        "status": "passed" if not failed and bool(succeeded) else ("passed_with_failures" if succeeded else "failed"),
        "n_artifacts": len(records),
        "n_success": len(succeeded),
        "n_failed": len(failed),
        "n_unavailable": len(unavailable),
        "tests": tests_verdict,
        "metrics_csv": _dev(table_path),
        "per_artifact": [
            {
                "artifact_name": r["artifact_name"],
                "method": r["method"],
                "status": r.get("status"),
                "macro_f1_id_test": (r.get("metrics") or {}).get("macro_f1"),
                "int8_agreement": (r.get("T6.5_detail") or {}).get("agreement"),
                "wall_s": r.get("wall_s"),
                "peak_vram_gb": r.get("peak_vram_gb"),
            }
            for r in records
        ],
    }

    summary_path = _resolve(cfg["paths"]["summary"])
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(summary_path, summary)

    print(
        json.dumps(
            {
                "status": summary["status"],
                "n_success": summary["n_success"],
                "n_failed": summary["n_failed"],
                "n_unavailable": summary["n_unavailable"],
                "tests": summary["tests"],
                "summary": _dev(summary_path),
            },
            indent=2,
        )
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 6: Post-Training Quantization Track A (GPU / Hugging Face)")
    parser.add_argument("--smoke", action="store_true", help="fast rehearsal with capped eval/calibration")
    parser.add_argument("--models", nargs="+", default=[], help="filter by base model key")
    parser.add_argument("--methods", nargs="+", default=[], help="filter by method (bnb_int8, bnb_nf4, gptq_int4)")
    parser.add_argument("--languages", nargs="+", default=[], help="filter by language (bn, en)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[], help="filter by seed")
    parser.add_argument("--max-eval", type=int, default=0, help="cap eval examples per set")
    parser.add_argument("--force", action="store_true", help="re-run completed artifacts")
    parser.add_argument("--status", action="store_true", help="show quantization status table and exit")
    parser.add_argument("--dry-run", action="store_true", help="print planned artifacts and exit")
    args = parser.parse_args()

    result = run(args)
    return 0 if result.get("status") in {"passed", "passed_with_failures", "dry_run", "status_displayed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

