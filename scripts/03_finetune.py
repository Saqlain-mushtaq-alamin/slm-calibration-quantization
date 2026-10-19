#!/usr/bin/env python
"""Phase 4: Baseline Fine-Tuning (all Bengali models, LoRA BF16, multi-seed).

Implements T4.1-T4.9 from
planning/phases/phase_04_baseline_fine_tuning_all_models_bengali.md.

Every (model, seed) cell of the run matrix produces a LoRA BF16 adapter, a merged
bf16 checkpoint, and per-example logits for id_val / id_test / ood_test.  Those BF16
logits are the FP reference for every later phase, so each run also gets its own
run.json carrying the git commit, config hash, seed, timing and VRAM.

Failed runs are recorded with ``status: "failed"`` and the sweep continues; they are
never silently dropped (global rule 6).

Usage:
    python scripts/03_finetune.py                      # full 16-run matrix
    python scripts/03_finetune.py --smoke              # 64/32 rehearsal, all models
    python scripts/03_finetune.py --models qwen2_5_1_5b_instruct
    python scripts/03_finetune.py --seeds 42
    python scripts/03_finetune.py --max-train 4000 --max-eval 1200
    python scripts/03_finetune.py --dry-run            # print the matrix, touch nothing
    python scripts/03_finetune.py --force              # ignore existing successes
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
import traceback
from pathlib import Path
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from sklearn.metrics import f1_score

from src.data.assemble import (
    SPLIT_PARTS,
    build_split_frames,
    resolve_model_id,
    split_frame_meta,
)
from src.data.splits import load_splits
from src.metrics.aggregate import evaluate_logits_frame, metrics_frame
from src.models import finetune as ft
from src.models import logit_extraction as lx
from src.utils.config import config_hash, load_config
from src.utils.logging_utils import setup_logging, write_run_json
from src.utils.reproducibility import collect_env_snapshot, get_git_commit, set_seed

LOG = setup_logging()

CFG_PATH = ROOT / "configs" / "baseline" / "default.yaml"

REQUIRED_LOGIT_COLUMNS = [
    "example_id",
    "y_true",
    "logit_hate",
    "logit_nonhate",
    "prob_hate",
    "prob_nonhate",
    "prediction",
]

# T4.4: logits must exist for every eval set.
EVAL_SPLITS: tuple[str, ...] = ("id_val", "id_test", "ood_test")

# Library versions recorded in every run manifest (global rule 10).
LIBRARY_VERSIONS = [
    "torch",
    "transformers",
    "peft",
    "accelerate",
    "bitsandbytes",
    "numpy",
    "scipy",
    "pandas",
    "scikit-learn",
    "tokenizers",
    "safetensors",
    "pynvml",
    "omegaconf",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve(path_like: str | Path) -> Path:
    p = Path(path_like)
    return p if p.is_absolute() else ROOT / p


def _dev(rel: Path) -> str:
    try:
        return str(rel.relative_to(ROOT))
    except ValueError:
        return str(rel)


def _library_versions() -> dict[str, str]:
    import importlib.metadata as md

    out: dict[str, str] = {}
    for name in LIBRARY_VERSIONS:
        try:
            out[name] = md.version(name)
        except Exception:
            out[name] = "MISSING"
    return out


def run_name(model_key: str, seed: int) -> str:
    """Canonical run id, e.g. ``bn_qwen2_5_1_5b_instruct_lora_bf16_s42``."""
    return f"bn_{model_key}_lora_bf16_s{int(seed)}"


def build_run_matrix(cfg: Any, models: Sequence[str] | None, seeds: Sequence[int] | None) -> list[dict[str, Any]]:
    """Expand the config into the (model, seed) cells of the plan's run matrix."""
    seed_cfg = cfg["seeds"]
    matrix: list[dict[str, Any]] = []
    for entry in cfg["models"]:
        key = str(entry["key"])
        if models and key not in models:
            continue
        tier = str(entry.get("tier", "secondary"))
        use_seeds = [int(s) for s in (seeds if seeds else seed_cfg[tier])]
        for seed in use_seeds:
            matrix.append(
                {
                    "model_key": key,
                    "hf_id": str(entry["hf_id"]),
                    "mirrors": [str(m) for m in (entry.get("mirrors") or [])],
                    "tier": tier,
                    "seed": seed,
                    "attn_implementation": entry.get("attn_implementation") or None,
                    "run_name": run_name(key, seed),
                }
            )
    return matrix


def validate_logits_frame(frame: pd.DataFrame, *, min_macro_f1: float, require_both_classes: bool) -> dict[str, Any]:
    """Phase 4 plan 4.3 / T4.2, T4.3, T4.5, T4.6.

    ``single_class`` marks a set whose ground truth has one class only.  The bd_shs
    LOCO axis makes every ``ood_test`` single-class, so macro-F1 and the
    both-classes-predicted gate are undefined there and are reported as ``null``
    instead of failing the run.  Structural checks always apply.
    """
    missing = [c for c in REQUIRED_LOGIT_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"logits frame is missing columns {missing}")
    if frame.empty:
        raise ValueError("logits frame is empty")

    y_true = frame["y_true"].to_numpy(dtype=np.int64)
    y_pred = frame["prediction"].to_numpy(dtype=np.int64)
    prob_sum = frame["prob_hate"].to_numpy(dtype=np.float64) + frame[
        "prob_nonhate"
    ].to_numpy(dtype=np.float64)
    single_class = bool(np.unique(y_true).size < 2)

    macro_f1 = (
        float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        if not single_class
        else None
    )
    n_pred_classes = int(np.unique(y_pred).size)

    checks: dict[str, bool | None] = {
        "has_all_columns": True,
        "probs_sum_to_one": bool(np.allclose(prob_sum, 1.0, atol=1e-5)),
        "no_nan_logits": not bool(
            frame[["logit_hate", "logit_nonhate"]].isna().any().any()
            or not np.isfinite(
                frame[["logit_hate", "logit_nonhate", "prob_hate", "prob_nonhate"]]
                .to_numpy(dtype=np.float64)
            ).all()
        ),
        "f1_above_random": None if single_class else bool(macro_f1 > min_macro_f1),
        "covers_both_classes": None
        if single_class or not require_both_classes
        else bool(n_pred_classes == 2),
    }
    return {
        "n_rows": int(len(frame)),
        "n_true_classes": int(np.unique(y_true).size),
        "n_pred_classes": n_pred_classes,
        "single_class": single_class,
        "macro_f1": macro_f1,
        "accuracy": float((y_pred == y_true).mean()),
        "max_prob_sum_deviation": float(np.max(np.abs(prob_sum - 1.0))),
        "checks": checks,
        "passed": all(v for v in checks.values() if v is not None),
        "note": (
            "ground truth is single-class; macro-F1 and the both-classes-predicted gate "
            "are undefined and are reported as null"
            if single_class
            else ""
        ),
    }


# ---------------------------------------------------------------------------
# Per-run execution
# ---------------------------------------------------------------------------

def _load_dataset(cfg: Any) -> tuple[pd.DataFrame, dict[str, Any]]:
    processed = _resolve(cfg["data"]["processed"])
    splits_path = _resolve(cfg["data"]["splits"])
    df = pd.read_csv(processed)
    df["example_id"] = df["example_id"].astype(str)
    splits, sha = load_splits(splits_path)
    split_name = str(cfg["split"])
    if split_name not in splits:
        raise KeyError(f"split {split_name!r} not in {_dev(splits_path)}; have {list(splits)}")
    return df, {"spec": splits[split_name], "sha": sha, "path": _dev(splits_path), "name": split_name}


def _release(model: Any) -> None:
    """Drop a model and reclaim VRAM before the next cell of the matrix runs."""
    del model
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def run_one(cell: dict[str, Any], cfg: Any, dataset: dict[str, Any], *, force: bool) -> dict[str, Any]:
    """Train, extract logits for every eval set, validate, and write one run manifest."""
    from src.metrics.efficiency import Stopwatch

    seed = int(cell["seed"])
    name = cell["run_name"]
    runs_dir = _resolve(cfg["paths"]["runs"])
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_json_path = runs_dir / f"{name}.json"

    ft_config = OmegaConf.to_container(cfg, resolve=True)
    ft_config.pop("models", None)
    ft_config.pop("seeds", None)
    ft_config.pop("caps", None)
    ft_config.pop("eval", None)
    ft_config.pop("sanity", None)
    ft_config.pop("paths", None)
    ft_config.pop("phase", None)
    ft_config.pop("keep_merged", None)
    ft_config["model"] = {
        **ft_config.get("model", {}),
        "hf_id": cell["resolved_hf_id"],
        "attn_implementation": cell.get("attn_implementation") or None,
    }

    manifest: dict[str, Any] = {
        "run_name": name,
        "phase": 4,
        "phase_name": "baseline_fine_tuning_all_models_bengali",
        "model_key": cell["model_key"],
        "canonical_hf_id": cell["hf_id"],
        "resolved_hf_id": cell["resolved_hf_id"],
        "mirror_notes": cell.get("mirror_notes", []),
        "tier": cell["tier"],
        "seed": seed,
        "attn_implementation": cell.get("attn_implementation"),
        "language": str(cfg["language"]),
        "split": dataset["name"],
        "config_path": _dev(CFG_PATH),
        "config_hash": config_hash(cfg),
        "git_commit": get_git_commit(str(ROOT)),
    }

    # ---- resumability: a completed cell is not re-trained -------------------
    logits_dir = _resolve(cfg["paths"]["logits"]) / name
    expected_logits = [logits_dir / f"bf16__{s}.parquet" for s in EVAL_SPLITS]
    if not force and run_json_path.is_file():
        try:
            previous = json.loads(run_json_path.read_text(encoding="utf-8"))
            complete = previous.get("status") == "success" and all(p.is_file() for p in expected_logits)
        except Exception:
            complete = False
        if complete:
            LOG.info("SKIP %s (already successful)", name)
            previous["skipped"] = True
            return previous

    set_seed(seed)
    t_start = time.perf_counter()
    try:
        spec = dataset["spec"]
        caps_cfg = cfg.get("caps", {}) or {}
        caps = {
            "train": int(caps_cfg.get("max_train_samples") or 0) or None,
            "id_val": int(caps_cfg.get("max_val_samples") or 0) or None,
            "id_test": int(caps_cfg.get("max_eval_samples") or 0) or None,
            "ood_test": int(caps_cfg.get("max_eval_samples") or 0) or None,
        }
        frames = build_split_frames(
            dataset["df"], spec, caps=caps, seed=seed, parts=SPLIT_PARTS
        )
        manifest["split_meta"] = split_frame_meta(frames, spec)
        manifest["split_sha256"] = dataset["sha"]
        manifest["split_file"] = dataset["path"]

        output_dir = _resolve(cfg["paths"]["models"]) / name
        ft_config["experiment"] = {
            "name": name,
            "seed": seed,
            "output_dir": str(output_dir),
        }

        LOG.info("=== %s | %s | seed=%d | train=%d ===", name, cell["resolved_hf_id"], seed, len(frames["train"]))
        sw_train = Stopwatch(f"finetune::{name}")
        with sw_train:
            result = ft.finetune_lora(ft_config, frames["train"], frames["id_val"])
        merged_path = result.merged_path

        train_losses = [float(h["loss"]) for h in result.history if "loss" in h]
        manifest["finetune"] = {
            "n_train": int(result.n_train),
            "n_id_val": int(result.n_id_val),
            "epochs": float(cfg["train"]["epochs"]),
            "first_train_loss": train_losses[0] if train_losses else None,
            "final_train_loss": train_losses[-1] if train_losses else None,
            "loss_decreased": bool(result.loss_decreased),
            "any_nan": bool(train_losses) and not all(np.isfinite(v) for v in train_losses),
            "best_eval_loss": result.best_eval_loss,
            "best_checkpoint": result.best_checkpoint,
            "trainable_params": int(result.trainable_params),
            "total_params": int(result.total_params),
            "label_token_ids": result.label_token_ids,
            "max_seq_len": int(result.max_seq_len),
            "wall_s": round(sw_train.wall_s, 2),
            "peak_vram_gb": None if sw_train.peak_vram_gb is None else round(sw_train.peak_vram_gb, 3),
            "energy_wh": sw_train.energy_wh,
        }

        # ---- T4.1 -----------------------------------------------------------
        max_loss = float(cfg["sanity"]["max_final_train_loss"])
        t4_1 = {
            "loss_decreased": bool(result.loss_decreased),
            "no_nan": not manifest["finetune"]["any_nan"],
            "checkpoint_saved": bool(result.best_checkpoint) and Path(result.adapter_path).is_dir(),
            "merged_saved": Path(merged_path).is_dir(),
            "loss_exploded": bool(train_losses and train_losses[-1] > max_loss),
            "passed": False,
        }
        t4_1["passed"] = bool(
            t4_1["loss_decreased"]
            and t4_1["no_nan"]
            and t4_1["checkpoint_saved"]
            and not t4_1["loss_exploded"]
        )
        manifest["T4.1"] = t4_1

        # ---- logits for every eval set ------------------------------------
        from transformers import AutoModelForCausalLM

        sw_eval = Stopwatch(f"eval::{name}")
        logits_dir.mkdir(parents=True, exist_ok=True)
        validation: dict[str, Any] = {}
        metrics_rows: list[dict[str, Any]] = []
        min_f1 = float(cfg["sanity"]["min_macro_f1_id_test"])
        require_both = bool(cfg["sanity"]["require_both_classes_predicted"])
        with sw_eval:
            tokenizer = ft.load_tokenizer(merged_path)
            model = ft.load_base_model(
                merged_path,
                str(cfg["model"]["dtype"]),
                attn_implementation=cell.get("attn_implementation") or None,
            )
            for split in EVAL_SPLITS:
                frame = lx.extract_logits_hf(
                    model,
                    tokenizer,
                    frames[split],
                    verbalizer=cfg["model"]["verbalizer"],
                    batch_size=int(cfg["eval"]["batch_size"]),
                    split=split,
                    max_len=int(cfg["data"]["max_len"]),
                    template=str(cfg["data"]["prompt_template"]),
                    model_id=f"{cell['resolved_hf_id']}|bf16",
                )
                lx.save_logits(
                    frame,
                    logits_dir / f"bf16__{split}.parquet",
                    model=cell["resolved_hf_id"],
                    canonical_model=cell["hf_id"],
                    model_key=cell["model_key"],
                    precision="bf16",
                    seed=seed,
                    language=str(cfg["language"]),
                    split_name=dataset["name"],
                    run_name=name,
                )
                validation[split] = validate_logits_frame(
                    frame, min_macro_f1=min_f1, require_both_classes=require_both
                )
                metrics_rows.append(
                    {
                        "model_key": cell["model_key"],
                        "model": cell["resolved_hf_id"],
                        "precision": "bf16",
                        "split": split,
                        "language": str(cfg["language"]),
                        "loco": dataset["name"],
                        "seed": seed,
                        "run_name": name,
                        **evaluate_logits_frame(
                            frame,
                            ece_bins=int(cfg["eval"]["ece_bins"]),
                            ece_strategy=str(cfg["eval"]["ece_strategy"]),
                        ),
                    }
                )
            _release(model)
        manifest["eval"] = {
            "wall_s": round(sw_eval.wall_s, 2),
            "peak_vram_gb": None if sw_eval.peak_vram_gb is None else round(sw_eval.peak_vram_gb, 3),
            "energy_wh": sw_eval.energy_wh,
            "n_examples": {s: int(len(frames[s])) for s in EVAL_SPLITS},
        }
        manifest["validation"] = validation
        manifest["metrics"] = {k: v for k, v in metrics_rows[0].items() if k in {"macro_f1", "accuracy", "auroc", "ece", "brier", "nll", "aurc"}}

        # ---- T4.2 / T4.3 / T4.5 / T4.6 -------------------------------------
        id_test = validation["id_test"]
        manifest["T4.2"] = {
            "macro_f1": id_test["macro_f1"],
            "min_required": min_f1,
            "passed": bool(id_test["checks"]["f1_above_random"]),
        }
        manifest["T4.3"] = {
            "n_pred_classes": id_test["n_pred_classes"],
            "passed": bool(id_test["checks"]["covers_both_classes"]),
        }
        manifest["T4.4"] = {
            "expected": [f"bf16__{s}.parquet" for s in EVAL_SPLITS],
            "found": [p.name for p in expected_logits if p.is_file()],
            "passed": all(p.is_file() for p in expected_logits),
        }
        manifest["T4.5"] = {
            "max_dev": max(validation[s]["max_prob_sum_deviation"] for s in EVAL_SPLITS),
            "passed": all(validation[s]["checks"]["probs_sum_to_one"] for s in EVAL_SPLITS),
        }
        manifest["T4.6"] = {
            "passed": all(validation[s]["checks"]["no_nan_logits"] for s in EVAL_SPLITS),
        }

        manifest["artifacts"] = {
            "adapter_path": _dev(Path(result.adapter_path)),
            "merged_path": _dev(Path(merged_path)),
            "logits_dir": _dev(logits_dir),
        }
        manifest["status"] = (
            "success"
            if all(manifest[k]["passed"] for k in ("T4.1", "T4.2", "T4.3", "T4.4", "T4.5", "T4.6"))
            else "failed"
        )
        manifest["wall_s"] = round(time.perf_counter() - t_start, 2)

        if not bool(cfg.get("keep_merged", True)):
            shutil.rmtree(merged_path, ignore_errors=True)
            manifest["artifacts"]["merged_path"] = None
            manifest["artifacts"]["merged_deleted"] = True

    except Exception as exc:  # noqa: BLE001 — a failed cell must be recorded, not dropped
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        manifest["traceback"] = traceback.format_exc()
        manifest["wall_s"] = round(time.perf_counter() - t_start, 2)
        LOG.error("FAILED %s: %s", name, exc)

    manifest["library_versions"] = _library_versions()
    manifest["environment"] = collect_env_snapshot()
    write_run_json(
        run_json_path,
        phase=4,
        phase_name="baseline_fine_tuning_all_models_bengali",
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

def _seed_variance(records: list[dict[str, Any]], max_std: float) -> dict[str, Any]:
    """T4.9: macro-F1 standard deviation across seeds of one model, on id_test."""
    by_model: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        if rec.get("status") != "success":
            continue
        f1 = (rec.get("T4.2") or {}).get("macro_f1")
        if f1 is None:
            continue
        by_model.setdefault(rec["model_key"], []).append({"seed": rec["seed"], "macro_f1": float(f1)})

    out: dict[str, Any] = {}
    for model_key, rows in sorted(by_model.items()):
        values = [r["macro_f1"] for r in sorted(rows, key=lambda r: r["seed"])]
        std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        out[model_key] = {
            "n_seeds": len(values),
            "seeds": [r["seed"] for r in sorted(rows, key=lambda r: r["seed"])],
            "macro_f1_by_seed": values,
            "macro_f1_mean": float(np.mean(values)),
            "macro_f1_std": std,
            "max_allowed_std": max_std,
            # A single seed cannot demonstrate stability; that is not a failure.
            "passed": bool(std < max_std) if len(values) > 1 else None,
        }
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    overrides: list[str] = []
    if args.max_train:
        overrides.append(f"caps.max_train_samples={args.max_train}")
    if args.max_eval:
        overrides.append(f"caps.max_eval_samples={args.max_eval}")
        overrides.append(f"caps.max_val_samples={max(args.max_eval // 2, 1)}")
    if args.split:
        overrides.append(f"split={args.split}")
    if args.no_keep_merged:
        overrides.append("keep_merged=false")

    cfg = load_config(CFG_PATH, overrides)
    matrix = build_run_matrix(cfg, args.models or None, args.seeds or None)
    if not matrix:
        raise ValueError("run matrix is empty; check --models / --seeds filters")

    LOG.info("Phase 4 run matrix (%d cells):", len(matrix))
    for cell in matrix:
        LOG.info("  %-46s seed=%-5d tier=%s", cell["run_name"], cell["seed"], cell["tier"])

    if args.dry_run:
        return {"status": "dry_run", "matrix": [c["run_name"] for c in matrix]}

    if not torch.cuda.is_available():
        LOG.warning("CUDA unavailable; Phase 4 is GPU-bound and will be extremely slow")

    df, dataset = _load_dataset(cfg)
    dataset["df"] = df

    # Resolve canonical -> reachable model id once per model, not once per seed.
    resolved: dict[str, tuple[str, list[str]]] = {}
    for cell in matrix:
        if cell["model_key"] not in resolved:
            try:
                resolved[cell["model_key"]] = resolve_model_id(cell["hf_id"], cell["mirrors"])
                model_id, notes = resolved[cell["model_key"]]
                LOG.info("resolved %s -> %s", cell["hf_id"], model_id)
                for note in notes:
                    LOG.warning("mirror substitution: %s", note)
            except Exception as exc:  # noqa: BLE001
                LOG.error("unresolvable model %s: %s", cell["hf_id"], exc)
                resolved[cell["model_key"]] = (cell["hf_id"], [f"unresolvable: {exc}"])

    records: list[dict[str, Any]] = []
    for i, cell in enumerate(matrix, start=1):
        cell = dict(cell)
        cell["resolved_hf_id"], cell["mirror_notes"] = resolved[cell["model_key"]]
        LOG.info("--- [%d/%d] %s ---", i, len(matrix), cell["run_name"])
        records.append(run_one(cell, cfg, dataset, force=args.force))

    # ---- aggregate ---------------------------------------------------------
    succeeded = [r for r in records if r.get("status") == "success"]
    failed = [r for r in records if r.get("status") != "success"]
    variance = _seed_variance(succeeded, float(cfg["sanity"]["max_seed_f1_std"]))

    # T4.8: every model saw the same frozen data under the same recipe.  The gates above
    # must not read "true" just because nothing succeeded, hence the explicit guard.
    split_hashes = sorted({r.get("split_sha256") for r in succeeded if r.get("split_sha256")})
    train_sizes = sorted({(r.get("finetune") or {}).get("n_train") for r in succeeded})
    t4_8_detail = {
        "n_succeeded": len(succeeded),
        "split_sha256": split_hashes,
        "same_split_for_all": len(split_hashes) == 1,
        "n_train_values": train_sizes,
        "same_train_size_for_all": len(train_sizes) <= 1,
        "languages": sorted({r.get("language") for r in succeeded if r.get("language")}),
    }

    metric_rows: list[dict[str, Any]] = []
    for rec in succeeded:
        for split, val in (rec.get("validation") or {}).items():
            metric_rows.append(
                {
                    "model_key": rec["model_key"],
                    "model": rec["resolved_hf_id"],
                    "precision": "bf16",
                    "split": split,
                    "seed": rec["seed"],
                    "language": rec["language"],
                    "loco": rec["split"],
                    "run_name": rec["run_name"],
                    "n": val["n_rows"],
                    "macro_f1": val["macro_f1"],
                    "accuracy": val["accuracy"],
                    "single_class": val["single_class"],
                }
            )
    table = metrics_frame(metric_rows)
    results_dir = _resolve(cfg["paths"]["results"])
    results_dir.mkdir(parents=True, exist_ok=True)
    table_path = _resolve(cfg["paths"]["metrics_csv"])
    table.to_csv(table_path, index=False)

    summary = {
        "phase": 4,
        "status": "passed" if not failed else ("passed_with_failures" if succeeded else "failed"),
        "n_cells": len(records),
        "n_success": len(succeeded),
        "n_failed": len(failed),
        "failed_runs": {r["run_name"]: r.get("error") for r in failed},
        "tests": {
            "T4.1": bool(succeeded) and all((r.get("T4.1") or {}).get("passed") for r in succeeded),
            "T4.2": bool(succeeded) and all((r.get("T4.2") or {}).get("passed") for r in succeeded),
            "T4.3": bool(succeeded) and all((r.get("T4.3") or {}).get("passed") for r in succeeded),
            "T4.4": bool(succeeded) and all((r.get("T4.4") or {}).get("passed") for r in succeeded),
            "T4.5": bool(succeeded) and all((r.get("T4.5") or {}).get("passed") for r in succeeded),
            "T4.6": bool(succeeded) and all((r.get("T4.6") or {}).get("passed") for r in succeeded),
            "T4.7": bool(succeeded) and all(
                (r.get("git_commit") is not None and r.get("config_hash") and r.get("seed") is not None)
                for r in succeeded
            ),
            "T4.8": bool(succeeded) and t4_8_detail["same_split_for_all"] and t4_8_detail["same_train_size_for_all"],
            "T4.9": bool(variance)
            and all(v["passed"] for v in variance.values() if v["passed"] is not None)
            and all(v["n_seeds"] > 1 for v in variance.values()),
        },
        "T4.8_detail": t4_8_detail,
        "seed_variance": variance,
        "metrics_csv": _dev(table_path),
        "per_run": [
            {
                "run_name": r["run_name"],
                "model_key": r["model_key"],
                "resolved_hf_id": r.get("resolved_hf_id"),
                "seed": r.get("seed"),
                "status": r.get("status"),
                "macro_f1_id_test": (r.get("T4.2") or {}).get("macro_f1"),
                "wall_s": r.get("wall_s"),
                "peak_vram_gb": (r.get("finetune") or {}).get("peak_vram_gb"),
            }
            for r in records
        ],
    }
    summary_path = _resolve(cfg["paths"]["summary"])
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print(
        json.dumps(
            {
                "status": summary["status"],
                "n_success": summary["n_success"],
                "n_failed": summary["n_failed"],
                "failed_runs": summary["failed_runs"],
                "tests": summary["tests"],
                "summary": _dev(summary_path),
            },
            indent=2,
        )
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 4 baseline fine-tuning (Bengali, all models)")
    parser.add_argument("--smoke", action="store_true", help="64 train / 32 eval rehearsal")
    parser.add_argument("--models", nargs="+", default=[], help="restrict to these model keys")
    parser.add_argument("--seeds", nargs="+", type=int, default=[], help="restrict to these seeds")
    parser.add_argument("--max-train", type=int, default=0, help="cap training examples (0 = config)")
    parser.add_argument("--max-eval", type=int, default=0, help="cap eval examples per set (0 = config)")
    parser.add_argument("--split", type=str, default="", help="override the LOCO split name")
    parser.add_argument("--no-keep-merged", action="store_true", help="delete merged bf16 after logits")
    parser.add_argument("--force", action="store_true", help="re-run cells that already succeeded")
    parser.add_argument("--dry-run", action="store_true", help="print the run matrix and exit")
    args = parser.parse_args()
    if args.smoke:
        args.max_train = args.max_train or 64
        args.max_eval = args.max_eval or 32
        args.models = args.models or None
    result = run(args)
    return 0 if result.get("status") in {"passed", "passed_with_failures", "dry_run"} else 1


if __name__ == "__main__":
    raise SystemExit(main())