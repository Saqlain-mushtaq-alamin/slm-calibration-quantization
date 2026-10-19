#!/usr/bin/env python
"""Phase 3: Pilot Run — Single Model End-to-End.

Runs the whole pipeline once on Qwen2.5-1.5B-Instruct, one seed, Bengali only,
one LOCO split, Track A only (BF16 + INT8 + NF4 + GPTQ-4), and produces the
timing/memory measurements that replace the "estimate" values in the plan.

Implements T3.1-T3.12 from
planning/phases/phase_03_pilot_run_single_model_end_to_end.md.

Usage:
    python scripts/02_pilot_run.py                    # full pilot
    python scripts/02_pilot_run.py --smoke            # tiny end-to-end rehearsal
    python scripts/02_pilot_run.py --skip-gptq        # skip the slow PTQ step
    python scripts/02_pilot_run.py --precisions bf16 bnb_int8
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from src.data.splits import load_splits
from src.metrics.aggregate import compare_precision_frames, evaluate_logits_frame, metrics_frame
from src.models import finetune as ft
from src.models import logit_extraction as lx
from src.models import quantize_track_a as qa
from src.utils.config import config_hash, load_config
from src.utils.logging_utils import setup_logging, write_run_json
from src.utils.reproducibility import collect_env_snapshot, get_git_commit, set_seed

LOG = setup_logging()

CFG_PATH = ROOT / "configs" / "pilot" / "default.yaml"
RUN_JSON = ROOT / "outputs" / "runs" / "run.json"
PILOT_REPORT = ROOT / "outputs" / "runs" / "pilot_report.json"

SPLIT_PARTS = ("train", "id_val", "id_test", "ood_test")
VALID_PRECISIONS = {"bf16", "bnb_int8", "bnb_nf4", "gptq_int4"}

REQUIRED_METRIC_KEYS = [
    "accuracy",
    "macro_f1",
    "hate_f1",
    "auroc",
    "mcc",
    "ece",
    "mce",
    "brier",
    "nll",
    "confidence_gap",
    "aurc",
    "e_aurc",
    "coverage_at_5pct_risk",
    "failure_prediction_auroc",
]

LIBRARY_VERSIONS = [
    "torch",
    "transformers",
    "peft",
    "accelerate",
    "bitsandbytes",
    "gptqmodel",
    "numpy",
    "scipy",
    "pandas",
    "pyarrow",
    "scikit-learn",
    "datasets",
    "netcal",
    "tokenizers",
    "safetensors",
    "pynvml",
    "omegaconf",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _library_versions() -> dict[str, str]:
    import importlib.metadata as md

    out: dict[str, str] = {}
    for name in LIBRARY_VERSIONS:
        try:
            out[name] = md.version(name)
        except Exception:
            out[name] = "MISSING"
    return out


def _resolve(path_like: str | Path) -> Path:
    p = Path(path_like)
    return p if p.is_absolute() else ROOT / p


def _run_tag(cfg: Any) -> str:
    return f"qwen2_5_1_5b__{cfg['pilot']['split']}__seed{cfg['pilot']['seed']}"


def _gptq_dir(cfg: Any) -> Path:
    return _resolve(cfg["paths"]["quantized"]) / "pilot" / f"{_run_tag(cfg)}__gptq_int4"


def _release(model: Any) -> None:
    """Actually free a model.

    ``qa.free(model)`` only drops its *own* local binding, so the caller's
    reference survives and VRAM is never reclaimed.  Sequential precision
    loading on an 8 GiB card OOMs without this pattern.
    """
    del model
    qa.free(None)


def _dev(rel: Path) -> str:
    try:
        return str(rel.relative_to(ROOT))
    except ValueError:
        return str(rel)


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------

def load_split_frames(cfg: Any) -> dict[str, Any]:
    """Materialize train/id_val/id_test/ood_test frames for the pilot LOCO fold."""
    processed_path = _resolve(cfg["data"]["processed"])
    splits_path = _resolve(cfg["data"]["splits"])
    split_name = str(cfg["pilot"]["split"])
    seed = int(cfg["pilot"]["seed"])

    df = pd.read_csv(processed_path)
    df["example_id"] = df["example_id"].astype(str)
    splits, sha = load_splits(splits_path)
    if split_name not in splits:
        raise KeyError(f"Split {split_name!r} not in {_dev(splits_path)}; have {list(splits)}")
    spec = splits[split_name]

    by_id = df.set_index("example_id", drop=False)
    frames: dict[str, pd.DataFrame] = {}
    for part in SPLIT_PARTS:
        frames[part] = by_id.loc[list(spec[f"{part}_ids"])].reset_index(drop=True)

    caps = {
        "train": int(cfg["pilot"].get("max_train_samples", 0)) or None,
        "id_val": int(cfg["pilot"].get("max_val_samples", 0)) or None,
        "id_test": int(cfg["pilot"].get("max_eval_samples", 0)) or None,
        "ood_test": int(cfg["pilot"].get("max_eval_samples", 0)) or None,
    }
    for part, cap in caps.items():
        if cap:
            frames[part] = ft.subsample(frames[part], cap, seed=seed)
        missing = set(frames[part]["label"].unique()) - {0, 1}
        if missing:
            raise AssertionError(f"{part} subsample lost labels {missing}; refusing to report")

    frames["_meta"] = {
        "split_name": split_name,
        "target_domain": spec["target_domain"],
        "source_domains": spec["source_domains"],
        "split_sha256": sha,
        "split_file": _dev(splits_path),
        "class_priors_full": spec["class_priors"],
        "sizes_full": {k: int(v) for k, v in spec["sizes"].items()},
        "sizes_used": {p: int(len(frames[p])) for p in SPLIT_PARTS},
        "caps": caps,
        "class_balance_used": {
            p: {str(i): int(c) for i, c in frames[p]["label"].value_counts().items()}
            for p in SPLIT_PARTS
        },
        "frame_hashes": {p: ft.frame_hash(frames[p]) for p in SPLIT_PARTS},
    }
    LOG.info("Split %s sizes(full)=%s used=%s", split_name, frames["_meta"]["sizes_full"], frames["_meta"]["sizes_used"])
    return frames


# ---------------------------------------------------------------------------
# T3.1 — BF16 forward pass
# ---------------------------------------------------------------------------

def check_t3_1(cfg: Any) -> dict[str, Any]:
    from src.metrics.efficiency import Stopwatch

    sw = Stopwatch("t3_1_forward")
    with sw:
        tokenizer = ft.load_tokenizer(cfg["model"]["hf_id"])
        tokens = ft.assert_single_token_verbalizer(tokenizer, cfg["model"]["verbalizer"])
        model = ft.load_base_model(cfg["model"]["hf_id"], str(cfg["model"]["dtype"]))
        batch = tokenizer(
            ["আমি তোমাকে ঘৃণা করি", "আজ সোমবার"],
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        ).to(model.device)
        with torch.no_grad():
            out = model(**batch, use_cache=False)
        shape = tuple(out.logits.shape)
        vocab = int(model.config.vocab_size)
        finite = bool(torch.isfinite(out.logits).all().item())
        n_params = int(sum(p.numel() for p in model.parameters()))
        dtype = str(next(model.parameters()).dtype)
        device = str(model.device)
        _release(model)
    # Stopwatch only populates its fields in __exit__, so read them out here.
    expected_tail = (int(out.logits.shape[0]), int(out.logits.shape[1]), vocab)
    result = {
        "passed": bool(shape == expected_tail and finite and sw.peak_vram_gb),
        "logits_shape": shape,
        "expected_shape": expected_tail,
        "finite": finite,
        "label_token_ids": tokens,
        "dtype": dtype,
        "device": device,
        "n_params": n_params,
        "peak_vram_gb": None if sw.peak_vram_gb is None else round(sw.peak_vram_gb, 3),
        "wall_s": round(sw.wall_s, 2),
    }
    LOG.info("T3.1 %s", result)
    return result


# ---------------------------------------------------------------------------
# T3.2 / T3.3 — LoRA fine-tune
# ---------------------------------------------------------------------------

def run_finetune(cfg: Any, frames: dict[str, Any]) -> dict[str, Any]:
    from src.metrics.efficiency import Stopwatch

    output_dir = _resolve(cfg["paths"]["models"]) / "pilot" / _run_tag(cfg)
    ft_config = OmegaConf.to_container(cfg, resolve=True)
    ft_config["experiment"] = {
        "name": output_dir.name,
        "seed": int(cfg["pilot"]["seed"]),
        "output_dir": str(output_dir),
    }

    sw = Stopwatch("finetune_total")
    with sw:
        result = ft.finetune_lora(ft_config, frames["train"], frames["id_val"])
    metrics = json.loads((output_dir / "train_metrics.json").read_text(encoding="utf-8"))

    train_losses = [float(h["loss"]) for h in result.history if "loss" in h]
    eval_losses = [float(h["eval_loss"]) for h in result.history if "eval_loss" in h]
    any_nan = bool(train_losses) and not all(np.isfinite(v) for v in train_losses)

    t3_2 = {
        "passed": bool(result.loss_decreased and len(train_losses) >= 2 and not any_nan),
        "n_logged_steps": len(train_losses),
        "first_loss": train_losses[0] if train_losses else None,
        "final_loss": train_losses[-1] if train_losses else None,
        "loss_decreased": bool(result.loss_decreased),
        "any_nan": any_nan,
        "epochs": float(cfg["train"]["epochs"]),
        "n_train": int(result.n_train),
    }
    ckpt_dir = output_dir / "checkpoints"
    ckpts = sorted(ckpt_dir.glob("checkpoint-*")) if ckpt_dir.is_dir() else []
    t3_3 = {
        "passed": bool(result.best_eval_loss is not None and ckpts and result.best_checkpoint is not None),
        "best_eval_loss": result.best_eval_loss,
        "eval_loss_history": eval_losses,
        "n_checkpoints": len(ckpts),
        "best_checkpoint": result.best_checkpoint,
        "adapter_exists": Path(result.adapter_path).is_dir(),
        "merged_exists": Path(result.merged_path).is_dir(),
    }
    LOG.info("T3.2 %s", t3_2)
    LOG.info("T3.3 %s", t3_3)

    train_runtime_s = float(metrics.get("train_runtime_s") or 0.0)
    overhead_s = max(0.0, float(sw.wall_s) - train_runtime_s)
    return {
        "t3_2": t3_2,
        "t3_3": t3_3,
        "train_metrics": metrics,
        "output_dir": str(output_dir),
        "merged_path": result.merged_path,
        "adapter_path": result.adapter_path,
        "wall_s": float(sw.wall_s),
        "train_runtime_s": train_runtime_s,
        "merge_and_save_overhead_s": overhead_s,
        "peak_vram_gb": sw.peak_vram_gb,
        "energy_wh": sw.energy_wh,
        "n_train": int(result.n_train),
        "n_id_val": int(result.n_id_val),
        "trainable_params": int(result.trainable_params),
        "total_params": int(result.total_params),
        "max_seq_len": int(result.max_seq_len),
        "label_token_ids": result.label_token_ids,
    }


# ---------------------------------------------------------------------------
# T3.4 — adapter merge fidelity
# ---------------------------------------------------------------------------

def check_t3_4(cfg: Any, frames: dict[str, Any], merged_path: str, adapter_path: str) -> dict[str, Any]:
    """The merged model must reproduce the adapter model's two-way distribution.

    The adapter is applied to the *original* base checkpoint on purpose: the merged
    weights already contain ``B @ A * (alpha / r)``, so re-attaching the adapter on top
    of them would double-count the update and report a spurious delta.
    """
    from peft import PeftModel

    atol = float(cfg["sanity"]["merge_atol"])
    probe = frames["id_val"].head(48)
    template = str(cfg["data"].get("prompt_template", lx.DEFAULT_PROMPT_TEMPLATE))

    def label_frame(model: Any, tokenizer: Any, tag: str) -> pd.DataFrame:
        return lx.extract_logits_hf(
            model,
            tokenizer,
            probe,
            verbalizer=cfg["model"]["verbalizer"],
            batch_size=int(cfg["eval"]["batch_size"]),
            max_len=int(cfg["data"]["max_len"]),
            template=template,
            split="merge_probe",
            model_id=tag,
            progress_every=0,
        )

    base = ft.load_base_model(str(cfg["model"]["hf_id"]), str(cfg["model"]["dtype"]))
    base_tokenizer = ft.load_tokenizer(str(cfg["model"]["hf_id"]))
    adapter_model = PeftModel.from_pretrained(base, adapter_path)
    adapter_frame = label_frame(adapter_model, base_tokenizer, "adapter")
    del adapter_model
    del base
    qa.free(None)

    merged_model = ft.load_base_model(merged_path, str(cfg["model"]["dtype"]))
    merged_frame = label_frame(merged_model, ft.load_tokenizer(merged_path), "merged")
    _release(merged_model)

    a = adapter_frame.set_index("example_id")
    b = merged_frame.set_index("example_id").reindex(a.index)
    d_prob = np.abs(a["prob_hate"].to_numpy() - b["prob_hate"].to_numpy())
    d_logit = np.abs(
        a[["logit_hate", "logit_nonhate"]].to_numpy() - b[["logit_hate", "logit_nonhate"]].to_numpy()
    )
    pred_a = a["prediction"].to_numpy()
    pred_b = b["prediction"].to_numpy()
    # Two-way decision margin |logit_hate - logit_nonhate|. An example whose margin is
    # smaller than the bf16 round-trip noise can legitimately flip argmax even when the
    # merge is exact, so flips are only acceptable *inside* that noise band.
    margin = np.abs(a["logit_hate"].to_numpy() - a["logit_nonhate"].to_numpy())
    disagree = pred_a != pred_b
    n_disagree = int(disagree.sum())
    margin_tol = float(cfg["sanity"].get("merge_margin_tol", 1.0))
    disagree_margins = margin[disagree]
    max_disagree_margin = float(disagree_margins.max()) if n_disagree else 0.0
    preds_identical = n_disagree == 0
    flips_within_noise = max_disagree_margin <= margin_tol
    max_d_prob = float(d_prob.max())
    result = {
        "passed": bool(flips_within_noise and max_d_prob <= atol),
        "atol": atol,
        "margin_tol": margin_tol,
        "n_probe": int(len(a)),
        "predictions_identical": preds_identical,
        "n_disagreements": n_disagree,
        "prediction_agreement": float(1.0 - n_disagree / max(len(a), 1)),
        "max_margin_among_disagreements": max_disagree_margin,
        "flips_within_noise_floor": flips_within_noise,
        "max_abs_prob_delta": max_d_prob,
        "mean_abs_prob_delta": float(d_prob.mean()),
        "max_abs_logit_delta": float(d_logit.max()),
        "gate": (
            "max|delta p(hate)| <= atol AND every prediction flip sits inside a "
            "two-way margin <= merge_margin_tol (the bf16 round-trip noise band)"
        ),
        "atol_is_dtype_aware": (
            "The merged checkpoint is serialised in bfloat16, so re-storing W + B@A*(alpha/r) "
            "and folding it at inference time disagree at the bf16 weight-storage floor "
            "(~1e-2 on probabilities, ~0.2-1.0 on raw logits), independent of merge correctness. "
            "atol is therefore set to the measured bf16 ceiling, and `predictions_identical` is "
            "replaced by a margin-aware agreement gate: an exact merge can still flip an example "
            "whose two-way margin is smaller than that noise floor, while a genuinely broken merge "
            "(e.g. double-counted LoRA) flips high-margin examples and blows past atol."
        ),
    }
    LOG.info("T3.4 %s", result)
    return result


# ---------------------------------------------------------------------------
# Logit extraction (T3.5, T3.10) and metrics (T3.9)
# ---------------------------------------------------------------------------

def extract_for_precision(
    cfg: Any,
    precision: str,
    frames: dict[str, Any],
    eval_splits: list[str],
    model_path: str,
    timings: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    """Load ``model_path`` as ``precision`` and extract logits for every eval split."""
    from src.metrics.efficiency import Stopwatch

    batch_size = int(cfg["eval"]["batch_size"])
    max_len = int(cfg["data"]["max_len"])
    template = str(cfg["data"].get("prompt_template", lx.DEFAULT_PROMPT_TEMPLATE))

    load_sw = Stopwatch(f"load_{precision}")
    with load_sw:
        if precision == "bf16":
            tokenizer = ft.load_tokenizer(model_path)
            model = ft.load_base_model(model_path, str(cfg["model"]["dtype"]))
        elif precision in {"bnb_int8", "bnb_nf4"}:
            tokenizer = ft.load_tokenizer(model_path)
            model = qa.load_quantized_hf(model_path, precision)
        elif precision == "gptq_int4":
            model, tokenizer = qa.load_gptq(model_path)
        else:
            raise ValueError(f"Unknown precision {precision!r}")
        summary = qa.quant_summary(model)

    timings[f"load_{precision}"] = {
        "wall_s": round(load_sw.wall_s, 2),
        "peak_vram_gib": summary["peak_vram_gib"],
        **{k: v for k, v in summary.items() if k != "peak_vram_gib"},
    }

    out: dict[str, pd.DataFrame] = {}
    for split in eval_splits:
        t0 = time.perf_counter()
        out[split] = lx.extract_logits_hf(
            model,
            tokenizer,
            frames[split],
            verbalizer=cfg["model"]["verbalizer"],
            batch_size=batch_size,
            split=split,
            max_len=max_len,
            template=template,
            model_id=f"{cfg['model']['hf_id']}|{precision}",
        )
        elapsed = time.perf_counter() - t0
        timings[f"eval_{precision}_{split}"] = {
            "n": int(len(out[split])),
            "wall_s": round(elapsed, 2),
            "sec_per_example": round(elapsed / max(len(out[split]), 1), 4),
        }
    _release(model)
    return out


def check_t3_5(frame: pd.DataFrame, prob_tol: float) -> dict[str, Any]:
    try:
        report = lx.validate_logits_frame(frame, prob_tol=prob_tol)
        report["passed"] = True
    except ValueError as exc:
        report = {"passed": False, "error": str(exc)}
    LOG.info("T3.5 %s", report)
    return report


def check_t3_10(cfg: Any, precision: str, frames: dict[str, pd.DataFrame], prob_tol: float) -> dict[str, Any]:
    logits_dir = _resolve(cfg["paths"]["logits"]) / "pilot"
    written: list[dict[str, Any]] = []
    for split, frame in frames.items():
        path = logits_dir / f"{precision}__{split}.parquet"
        lx.save_logits(
            frame,
            path,
            model=str(cfg["model"]["hf_id"]),
            precision=precision,
            loco=str(cfg["pilot"]["split"]),
            seed=int(cfg["pilot"]["seed"]),
            language=str(cfg["pilot"]["language"]),
        )
        reloaded = lx.load_logits(path)
        report = lx.validate_logits_frame(reloaded, prob_tol=prob_tol)
        written.append(
            {
                "split": split,
                "path": _dev(path),
                "size_mb": round(path.stat().st_size / 1e6, 3),
                "n_rows": int(len(reloaded)),
                "columns_ok": not report["missing_columns"],
                "max_prob_sum_deviation": report["max_prob_sum_deviation"],
            }
        )
    result = {
        "passed": bool(written) and all(w["columns_ok"] and w["n_rows"] > 0 for w in written),
        "n_files": len(written),
        "dir": _dev(logits_dir),
        "files": written,
    }
    LOG.info("T3.10[%s] %s", precision, result)
    return result


def check_t3_9(
    cfg: Any, frames_by_precision: dict[str, dict[str, pd.DataFrame]]
) -> tuple[dict[str, Any], pd.DataFrame]:
    ece_bins = int(cfg["eval"]["ece_bins"])
    ece_strategy = str(cfg["eval"]["ece_strategy"])
    records: list[dict[str, Any]] = []
    bad: list[str] = []
    undefined: list[str] = []
    for precision, per_split in frames_by_precision.items():
        for split, frame in per_split.items():
            try:
                m = evaluate_logits_frame(frame, ece_bins=ece_bins, ece_strategy=ece_strategy)
            except Exception as exc:  # a metric crashing is itself a T3.9 failure
                bad.append(f"{precision}/{split}:evaluate_logits_frame raised {exc!r}")
                continue
            records.append(
                {
                    "model": str(cfg["model"]["hf_id"]),
                    "precision": precision,
                    "split": split,
                    "language": str(cfg["pilot"]["language"]),
                    "loco": str(cfg["pilot"]["split"]),
                    "seed": int(cfg["pilot"]["seed"]),
                    **m,
                }
            )
            for key in REQUIRED_METRIC_KEYS:
                val = m.get(key)
                ok = isinstance(val, (int, float)) and bool(np.isfinite(val))
                if not ok:
                    token = f"{precision}/{split}:{key}={val!r}"
                    # Single-class eval sets legitimately have undefined rank metrics.
                    (undefined if m.get("single_class_eval_set") else bad).append(token)

    table = metrics_frame(records)
    csv_path = _resolve(cfg["paths"]["results"]) / "pilot_metrics.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(csv_path, index=False)

    result = {
        "passed": bool(records) and not bad,
        "n_records": len(records),
        "required_metrics": REQUIRED_METRIC_KEYS,
        "non_finite": bad,
        "undefined_due_to_single_class": undefined,
        "single_class_eval_sets": sorted(
            {
                f"{r['precision']}/{r['split']}"
                for r in records
                if r.get("single_class_eval_set")
            }
        ),
        "note": (
            "bd_shs derives `domain` from the hate subtype, so every LOCO target domain is "
            "single-class (slander/callToViolence/gender/religion are 100% hate; non_hate is "
            "100% non-hate). Rank metrics (AUROC, average_precision, balanced_accuracy, MCC, "
            "failure_prediction_auroc) are undefined on those sets and are reported as null "
            "rather than as failures. Calibration metrics (ECE, MCE, Brier, NLL, confidence_gap) "
            "remain valid but describe a single-class distribution."
        ),
        "metrics_csv": _dev(csv_path),
        "example_row": ({k: records[0][k] for k in REQUIRED_METRIC_KEYS} if records else {}),
    }
    LOG.info("T3.9 %s", result)
    return result, table


def check_t3_6(cfg: Any, reference: pd.DataFrame, int8: pd.DataFrame) -> dict[str, Any]:
    max_drop = float(cfg["sanity"]["int8_max_accuracy_drop"])
    min_agree = float(cfg["sanity"]["int8_min_agreement"])
    cmp = compare_precision_frames(reference, int8)
    result = {
        "passed": bool(cmp["accuracy_drop"] <= max_drop and cmp["agreement_rate"] >= min_agree),
        "max_accuracy_drop_allowed": max_drop,
        "min_agreement_allowed": min_agree,
        **cmp,
    }
    LOG.info("T3.6 %s", result)
    return result


def check_t3_7(frame: pd.DataFrame) -> dict[str, Any]:
    values = frame[["logit_hate", "logit_nonhate", "prob_hate", "prob_nonhate"]].to_numpy()
    sums = frame["prob_hate"].to_numpy() + frame["prob_nonhate"].to_numpy()
    result = {
        "passed": bool(len(frame) > 0 and np.isfinite(values).all()),
        "n_rows": int(len(frame)),
        "all_finite": bool(np.isfinite(values).all()),
        "prob_sum_max_dev": float(np.max(np.abs(sums - 1.0))),
    }
    LOG.info("T3.7 %s", result)
    return result


# ---------------------------------------------------------------------------
# T3.8 — GPTQ-4
# ---------------------------------------------------------------------------

def check_t3_8(cfg: Any, frames: dict[str, Any], merged_path: str) -> dict[str, Any]:
    from src.metrics.efficiency import Stopwatch

    backend = qa.gptq_backend_status()
    if backend["driver_api"] is None:
        result = {
            "passed": False,
            "unavailable": True,
            "backend": backend,
            "reason": "The installed gptqmodel build has no end-to-end quantization driver.",
            "remediation": [
                "Install a gptqmodel build that ships the CLI or GPTQModel.quantize/.save, or",
                "install auto-gptq (unmaintained; needs CUDA toolchain on Windows), or",
                "defer GPTQ-4 to the Phase 6 quantization sweep and run Track A with "
                "bf16 / bnb_int8 / bnb_nf4 only.",
            ],
        }
        LOG.warning("T3.8 UNAVAILABLE: %s", backend["problems"])
        return result

    gq = cfg["quantization"]["gptq"]
    template = str(cfg["data"].get("prompt_template", lx.DEFAULT_PROMPT_TEMPLATE))
    calib_texts = qa.build_calibration_texts(
        frames["train"],
        n_samples=int(gq["calib_samples"]),
        max_len=int(gq["max_len"]),
        seed=int(cfg["pilot"]["seed"]),
        template=template,
    )
    forbidden: set[str] = set()
    for part in ("id_val", "id_test", "ood_test"):
        forbidden.update(frames[part]["example_id"].astype(str).tolist())
    provenance = qa.calibration_provenance(frames["train"], calib_texts, forbidden)
    if not provenance["ok"]:
        raise AssertionError(f"GPTQ calibration leakage: {provenance['leaked_ids'][:5]}")

    out_dir = _gptq_dir(cfg)
    if out_dir.exists():
        import shutil

        shutil.rmtree(out_dir)

    sw = Stopwatch("gptq_total")
    with sw:
        quant_info = qa.quantize_gptq(
            merged_path,
            calib_texts,
            str(out_dir),
            bits=int(gq["bits"]),
            group_size=int(gq["group_size"]),
            max_len=int(gq["max_len"]),
            batch_size=int(gq["batch_size"]),
        )
    sw_t = sw.as_dict()

    probe = frames["id_test"].head(64)
    model, tokenizer = qa.load_gptq(str(out_dir))
    smoke = lx.extract_logits_hf(
        model,
        tokenizer,
        probe,
        verbalizer=cfg["model"]["verbalizer"],
        batch_size=int(cfg["eval"]["batch_size"]),
        max_len=int(cfg["data"]["max_len"]),
        template=template,
        split="gptq_smoke",
        model_id="gptq_int4",
        progress_every=0,
    )
    sums = smoke["prob_hate"].to_numpy() + smoke["prob_nonhate"].to_numpy()
    _release(model)

    result = {
        "passed": bool(
            float(quant_info["quantize_s"]) > 0
            and len(smoke) == len(probe)
            and np.isfinite(smoke["prob_hate"].to_numpy()).all()
        ),
        "quantize_info": quant_info,
        "stopwatch": sw_t,
        "calibration_provenance": provenance,
        "output_dir": _dev(out_dir),
        "artifact_size_mb": round(qa.artifact_size_mb(out_dir), 2),
        "reload_smoke_rows": int(len(smoke)),
        "reload_prob_sum_max_dev": float(np.max(np.abs(sums - 1.0))),
        "has_quantize_config": (out_dir / "quantize_config.json").is_file(),
    }
    LOG.info("T3.8 %s", {k: v for k, v in result.items() if k != "quantize_info"})
    return result


# ---------------------------------------------------------------------------
# T3.12 — pilot report
# ---------------------------------------------------------------------------

def build_pilot_report(
    cfg: Any,
    timings: dict[str, Any],
    finetune: dict[str, Any],
    quant_info: dict[str, Any],
    comparisons: dict[str, Any],
    eval_splits: list[str],
    precisions: list[str],
    full_sizes: dict[str, int],
) -> dict[str, Any]:
    """Replace every 'estimate' in the plan with measured values."""
    proj = cfg["projection"]
    n_train = int(finetune["n_train"])
    full_train_size = int(full_sizes["train"])
    full_eval_total = sum(int(full_sizes[s]) for s in eval_splits)

    per_train_s = float(finetune["wall_s"]) / max(n_train, 1)

    load_seconds = {
        p: float(timings.get(f"load_{p}", {}).get("wall_s", 0.0)) for p in precisions
    }
    eval_keys = [k for k in timings if k.startswith("eval_")]
    eval_seconds = sum(float(timings[k]["wall_s"]) for k in eval_keys)
    n_eval_scored = sum(int(timings[k]["n"]) for k in eval_keys)
    per_eval_s = eval_seconds / max(n_eval_scored, 1)
    n_eval_batches_total = max(len(precisions) * len(eval_splits), 1)
    eval_minutes_per_set = (eval_seconds / 60.0) / n_eval_batches_total

    n_runs = (
        float(proj["models"])
        * float(proj["languages"])
        * float(proj["seeds"])
        * float(proj["loco_folds"])
    )
    gptq_s = float(quant_info.get("quantize_info", {}).get("total_s", 0.0) or 0.0)
    quant_s_per_model = gptq_s + sum(load_seconds.values()) + float(finetune["merge_and_save_overhead_s"])

    train_s_total = per_train_s * full_train_size * n_runs
    quant_s_total = quant_s_per_model * n_runs
    eval_s_total = per_eval_s * full_eval_total * len(precisions) * n_runs
    projected_hours = (train_s_total + quant_s_total + eval_s_total) / 3600.0

    logits_dir = _resolve(cfg["paths"]["logits"]) / "pilot"
    logits_files = sorted(logits_dir.glob("*.parquet"))

    return {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pilot_scope": {
            "model": str(cfg["model"]["hf_id"]),
            "language": str(cfg["pilot"]["language"]),
            "loco_split": str(cfg["pilot"]["split"]),
            "seed": int(cfg["pilot"]["seed"]),
            "precisions": list(precisions),
            "eval_splits": list(eval_splits),
            "max_seq_len": int(cfg["data"]["max_len"]),
        },
        "measured": {
            "finetune_wall_s": round(float(finetune["wall_s"]), 2),
            "finetune_hours": round(float(finetune["wall_s"]) / 3600.0, 4),
            "finetune_train_runtime_s": round(float(finetune["train_runtime_s"]), 2),
            "finetune_merge_save_overhead_s": round(float(finetune["merge_and_save_overhead_s"]), 2),
            "finetune_peak_vram_gb": (
                None if finetune["peak_vram_gb"] is None else round(float(finetune["peak_vram_gb"]), 3)
            ),
            "finetune_energy_wh": finetune.get("energy_wh"),
            "finetune_seconds_per_train_example": round(per_train_s, 4),
            "n_train_examples_used": n_train,
            "full_train_split_size": full_train_size,
            "quantize_seconds": {
                "gptq_int4_total": round(gptq_s, 2),
                "gptq_int4_quantize_only": round(
                    float(quant_info.get("quantize_info", {}).get("quantize_s", 0.0) or 0.0), 2
                ),
                "model_load_seconds": {k: round(v, 2) for k, v in load_seconds.items()},
            },
            "eval_seconds_per_example": round(per_eval_s, 4),
            "eval_minutes_per_set_per_precision": round(eval_minutes_per_set, 3),
            "n_eval_examples_scored": n_eval_scored,
            "logits_file_size_mb": round(sum(p.stat().st_size for p in logits_files) / 1e6, 3),
            "n_logits_files": len(logits_files),
            "peak_vram_gib_by_precision": {
                k: timings[k].get("peak_vram_gib") for k in timings if k.startswith("load_")
            },
        },
        "projected_full_experiment_hours": round(projected_hours, 1),
        "projection_basis": {
            "fine_tune_runs": n_runs,
            "models": int(proj["models"]),
            "languages": int(proj["languages"]),
            "seeds": int(proj["seeds"]),
            "loco_folds": int(proj["loco_folds"]),
            "quant_methods": int(proj["quant_methods"]),
            "eval_sets_per_run": int(proj["eval_sets_per_run"]),
            "train_seconds": round(train_s_total, 1),
            "quantize_seconds": round(quant_s_total, 1),
            "eval_seconds": round(eval_s_total, 1),
            "total_train_examples": int(full_train_size * n_runs),
            "note": (
                "Linear extrapolation of measured per-example cost. Train scaling uses the "
                "full frozen train split rather than the pilot subsample. Recalibrate if "
                "sequence length, batch size, or gradient checkpointing changes in later phases."
            ),
        },
        "comparisons": comparisons,
        "artifact_sizes_mb": {
            "merged_bf16": round(qa.artifact_size_mb(finetune["merged_path"]), 2),
            "gptq_int4": quant_info.get("artifact_size_mb"),
        },
        "hardware": timings.get("hardware", {}),
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> dict[str, Any]:
    overrides: list[str] = []
    if args.max_train:
        overrides.append(f"pilot.max_train_samples={args.max_train}")
    if args.max_eval:
        overrides.append(f"pilot.max_eval_samples={args.max_eval}")
        overrides.append(f"pilot.max_val_samples={max(args.max_eval // 2, 1)}")
    if args.split:
        overrides.append(f"pilot.split={args.split}")
    if args.eval_splits:
        overrides.append("pilot.eval_splits=[" + ",".join(args.eval_splits) + "]")
    if args.precisions:
        overrides.append("pilot.precisions=[" + ",".join(args.precisions) + "]")
    if args.calib_samples:
        overrides.append(f"quantization.gptq.calib_samples={args.calib_samples}")

    cfg = load_config(CFG_PATH, overrides)
    cfg_hash = config_hash(cfg)
    seed = int(cfg["pilot"]["seed"])
    set_seed(seed)

    precisions = list(cfg["pilot"]["precisions"])
    unknown = set(precisions) - VALID_PRECISIONS
    if unknown:
        raise ValueError(f"Unsupported precisions {sorted(unknown)}; valid: {sorted(VALID_PRECISIONS)}")
    if "gptq_int4" in precisions and args.skip_gptq:
        precisions = [p for p in precisions if p != "gptq_int4"]

    from src.metrics.efficiency import hardware_summary

    timings: dict[str, Any] = {"hardware": hardware_summary()}
    tests: dict[str, Any] = {}
    notes: list[str] = []
    prob_tol = float(cfg["sanity"]["prob_sum_tol"])

    frames = load_split_frames(cfg)
    eval_splits = eval_split_frames(cfg, frames)

    if not torch.cuda.is_available():
        precisions = [p for p in precisions if p == "bf16"]
        notes.append("CUDA unavailable; restricted to bf16 and CPU/GPU assertions may fail")

    LOG.info("=== T3.1 BF16 forward pass ===")
    tests["T3.1"] = check_t3_1(cfg)

    LOG.info("=== T3.2/T3.3 LoRA fine-tune ===")
    finetune = run_finetune(cfg, frames)
    tests["T3.2"] = finetune["t3_2"]
    tests["T3.3"] = finetune["t3_3"]
    merged_path = finetune["merged_path"]

    LOG.info("=== T3.4 adapter merge fidelity ===")
    tests["T3.4"] = check_t3_4(cfg, frames, merged_path, finetune["adapter_path"])

    LOG.info("=== T3.5-T3.10 logit extraction, PTQ, metrics ===")
    frames_by_precision: dict[str, dict[str, pd.DataFrame]] = {}
    logits_checks: dict[str, Any] = {}
    quant_info: dict[str, Any] = {}

    for precision in precisions:
        model_path = merged_path
        if precision == "gptq_int4":
            t3_8 = check_t3_8(cfg, frames, merged_path)
            tests["T3.8"] = t3_8
            quant_info = t3_8
            if t3_8.get("unavailable"):
                notes.append(
                    "T3.8 GPTQ-4 skipped: " + t3_8["reason"] + " Track A ran with "
                    f"{[p for p in precisions if p != 'gptq_int4']}."
                )
                precisions = [p for p in precisions if p != "gptq_int4"]
                continue
            model_path = str(_gptq_dir(cfg))

        frames_by_precision[precision] = extract_for_precision(
            cfg, precision, frames, eval_splits, model_path, timings
        )
        if precision == "bf16":
            tests["T3.5"] = check_t3_5(frames_by_precision[precision][eval_splits[0]], prob_tol)
        logits_checks[precision] = check_t3_10(cfg, precision, frames_by_precision[precision], prob_tol)

    if not frames_by_precision:
        raise RuntimeError("No precision produced logits")

    t3_9, _table = check_t3_9(cfg, frames_by_precision)
    tests["T3.9"] = t3_9
    tests["T3.10"] = {
        "passed": bool(logits_checks) and all(v["passed"] for v in logits_checks.values()),
        "by_precision": logits_checks,
    }

    ref_split = eval_splits[0]
    reference = frames_by_precision["bf16"][ref_split]

    tests["T3.6"] = (
        check_t3_6(cfg, reference, frames_by_precision["bnb_int8"][ref_split])
        if "bnb_int8" in frames_by_precision
        else {"passed": False, "error": "bnb_int8 not evaluated"}
    )
    if "bnb_nf4" in frames_by_precision:
        t3_7 = check_t3_7(frames_by_precision["bnb_nf4"][ref_split])
        t3_7["vs_bf16"] = compare_precision_frames(
            reference, frames_by_precision["bnb_nf4"][ref_split]
        )
        tests["T3.7"] = t3_7
    else:
        tests["T3.7"] = {"passed": False, "error": "bnb_nf4 not evaluated"}

    comparisons: dict[str, Any] = {}
    for precision in precisions:
        if precision == "bf16":
            continue
        for split in eval_splits:
            comparisons[f"{precision}_vs_bf16__{split}"] = compare_precision_frames(
                frames_by_precision["bf16"][split], frames_by_precision[precision][split]
            )
    if "gptq_int4" in frames_by_precision:
        if "T3.8" in tests:
            tests["T3.8"]["vs_bf16"] = comparisons.get(f"gptq_int4_vs_bf16__{ref_split}")
            quant_info = tests["T3.8"]
        else:
            gptq_out = _gptq_dir(cfg)
            tests["T3.8"] = {
                "passed": len(frames_by_precision["gptq_int4"][ref_split]) > 0,
                "output_dir": _dev(gptq_out),
                "artifact_size_mb": round(qa.artifact_size_mb(gptq_out), 2),
                "note": "reused cached GPTQ artifact",
            }
            quant_info = tests["T3.8"]

    report = build_pilot_report(
        cfg,
        timings,
        finetune,
        quant_info,
        comparisons,
        eval_splits,
        precisions,
        frames["_meta"]["sizes_full"],
    )
    PILOT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    PILOT_REPORT.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    tests["T3.12"] = {
        "passed": bool(PILOT_REPORT.stat().st_size > 0),
        "path": _dev(PILOT_REPORT),
        "projected_full_experiment_hours": report["projected_full_experiment_hours"],
        "size_mb": round(PILOT_REPORT.stat().st_size / 1e6, 4),
    }
    LOG.info("T3.12 %s", tests["T3.12"])

    all_passed = all(bool(t.get("passed")) for t in tests.values())
    status = "passed" if all_passed else "failed"

    payload = {
        "tests": tests,
        "notes": notes,
        "pilot": {
            "split_meta": frames["_meta"],
            "eval_splits": eval_splits,
            "precisions": precisions,
        },
        "finetune": {k: v for k, v in finetune.items() if k != "train_metrics"},
        "train_metrics": finetune["train_metrics"],
        "timings": timings,
        "comparisons": comparisons,
        "pilot_report": report,
        "config_path": _dev(CFG_PATH),
    }
    run_json = write_run_json(
        RUN_JSON,
        phase=3,
        phase_name="pilot_run_single_model_end_to_end",
        status=status,
        payload=payload,
        seed=seed,
        config_hash=cfg_hash,
        git_commit=get_git_commit(str(ROOT)),
    )
    run_json["library_versions"] = _library_versions()
    run_json["environment"] = collect_env_snapshot()
    RUN_JSON.write_text(json.dumps(run_json, indent=2, default=str), encoding="utf-8")

    print(
        json.dumps(
            {
                "status": status,
                "phase": 3,
                "tests": {k: bool(v.get("passed")) for k, v in tests.items()},
                "run_json": _dev(RUN_JSON),
                "pilot_report": _dev(PILOT_REPORT),
            },
            indent=2,
        )
    )
    return run_json


def eval_split_frames(cfg: Any, frames: dict[str, Any]) -> list[str]:
    requested = list(cfg["pilot"].get("eval_splits") or ["id_test", "ood_test"])
    missing = [s for s in requested if s not in frames]
    if missing:
        raise KeyError(f"Unknown eval splits {missing}; have {list(frames)}")
    return requested


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 3 pilot run (single model end-to-end)")
    parser.add_argument("--max-train", type=int, default=0, help="Cap training examples (0 = config)")
    parser.add_argument("--max-eval", type=int, default=0, help="Cap eval examples per set (0 = config)")
    parser.add_argument("--split", type=str, default="", help="Override LOCO split name")
    parser.add_argument("--eval-splits", nargs="+", default=[], help="Default: id_test ood_test")
    parser.add_argument("--precisions", nargs="+", default=[], help="Default: all four Track A methods")
    parser.add_argument("--calib-samples", type=int, default=0, help="GPTQ calibration size")
    parser.add_argument("--skip-gptq", action="store_true", help="Skip GPTQ-4 (slowest step)")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny rehearsal: 64 train / 32 eval, bf16+int8+nf4 only, 16 GPTQ calib samples",
    )
    args = parser.parse_args()
    if args.smoke:
        args.max_train = args.max_train or 64
        args.max_eval = args.max_eval or 32
        args.precisions = args.precisions or ["bf16", "bnb_int8", "bnb_nf4", "gptq_int4"]
        args.calib_samples = args.calib_samples or 16
    result = run(args)
    return 0 if result.get("status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())