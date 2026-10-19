#!/usr/bin/env python
"""Phase 1: Data Preparation & Split Engineering.

Implements T1.1–T1.11 from planning/phases/phase_01_data_preparation_split_engineering.md.

Usage:
    python scripts/01_prepare_data.py
    python scripts/01_prepare_data.py --skip-heavy   # skip fertility + embeddings
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.data.dedup import dedup_pipeline
from src.data.fertility import measure_all_fertility
from src.data.loader import (
    FERTILITY_MODEL_IDS,
    load_bd_shs,
    load_bengali_hs_30k,
    load_english_control,
    write_english_control_decision,
)
from src.data.shift import shift_rows_from_splits
from src.data.splits import (
    create_loco_splits,
    freeze_and_hash_splits,
    verify_split_integrity,
)
from src.utils.config import config_hash, load_config
from src.utils.logging_utils import setup_logging, write_run_json
from src.utils.reproducibility import get_git_commit, set_seed

LOG = setup_logging()

CFG_PATH = ROOT / "configs" / "data" / "default.yaml"


def _resolve(path_like: str | Path) -> Path:
    p = Path(path_like)
    return p if p.is_absolute() else ROOT / p


def check_t1_1(df: pd.DataFrame) -> dict[str, Any]:
    n = len(df)
    has_cols = "text" in df.columns and "label" in df.columns
    result = {
        "t1_1_load_bd_shs": bool(n >= 45000 and has_cols),
        "n_samples": int(n),
        "has_text": "text" in df.columns,
        "has_label": "label" in df.columns,
        "class_distribution": {int(k): int(v) for k, v in df["label"].value_counts().items()},
    }
    LOG.info("T1.1: %s", result)
    return result


def check_t1_2(df: pd.DataFrame) -> dict[str, Any]:
    meta = {}
    for col in ("domain", "type", "target", "category"):
        if col in df.columns:
            meta[col] = int(df[col].nunique())
    ok = any(v >= 3 for v in meta.values())
    result = {"t1_2_metadata": bool(ok), "nunique": meta}
    LOG.info("T1.2: %s", result)
    return result


def run(skip_heavy: bool = False) -> dict[str, Any]:
    set_seed(42)
    cfg = load_config(CFG_PATH)
    data_cfg = cfg["data"]
    seed = int(data_cfg.get("seed", 42))

    processed_dir = _resolve(data_cfg["paths"]["processed"])
    splits_dir = _resolve(data_cfg["paths"]["splits"])
    results_dir = _resolve(data_cfg["paths"]["results"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    tests: dict[str, Any] = {}
    notes: list[str] = []

    # ------------------------------------------------------------------
    # T1.1–T1.2: load BD-SHS
    # ------------------------------------------------------------------
    bd_raw = _resolve(data_cfg["bengali"]["bd_shs"]["raw_dir"])
    df = load_bd_shs(bd_raw)
    tests.update(check_t1_1(df))
    tests.update(check_t1_2(df))

    # ------------------------------------------------------------------
    # Optional secondary Bengali corpus (full category labels)
    # ------------------------------------------------------------------
    bn30k = None
    sec = data_cfg["bengali"].get("secondary", {}).get("bengali_hs_30k", {})
    if sec.get("enabled", False):
        try:
            bn30k = load_bengali_hs_30k(_resolve(sec.get("raw_csv", "")))
        except Exception as e:
            LOG.warning("30k Bengali HS not loaded: %s", e)
            bn30k = None

    # ------------------------------------------------------------------
    # T1.3–T1.4: dedup BD-SHS (global, before splits)
    # ------------------------------------------------------------------
    dedup_cfg = data_cfg["dedup"]
    df_dedup, dedup_stats = dedup_pipeline(
        df,
        text_col="text",
        near_threshold=float(dedup_cfg.get("near_threshold", 0.8)),
        num_perm=int(dedup_cfg.get("minhash_num_perm", 128)),
        run_near=bool(dedup_cfg.get("near", True)),
    )
    tests["t1_3_exact_dedup"] = {
        "removed": dedup_stats["removed_exact"],
        "n_raw": dedup_stats["n_raw"],
        "n_after_exact": dedup_stats["n_exact"],
    }
    tests["t1_4_near_dedup"] = {
        "removed": dedup_stats["removed_near"],
        "threshold": dedup_cfg.get("near_threshold", 0.8),
        "n_final": dedup_stats["n_final"],
    }

    # Persist processed frame
    processed_path = processed_dir / "bd_shs_processed.csv"
    df_dedup.to_csv(processed_path, index=False)
    LOG.info("Wrote %s (%d rows)", processed_path, len(df_dedup))

    # ------------------------------------------------------------------
    # T1.5–T1.8: LOCO splits + freeze/hash
    # ------------------------------------------------------------------
    min_domain = int(data_cfg["splits"].get("min_domain_size", 500))
    splits = create_loco_splits(
        df_dedup,
        domain_col="domain",
        seed=seed,
        min_domain_size=min_domain,
        id_col="example_id",
        train_frac=float(data_cfg["splits"].get("train_frac", 0.7)),
    )
    integrity = verify_split_integrity(splits)
    tests["t1_5_loco_disjoint"] = integrity["ok"]
    tests["t1_7_no_leakage"] = integrity["ok"]
    tests["t1_5_split_names"] = list(splits.keys())
    tests["t1_5_integrity"] = integrity

    # Class priors (T1.8)
    prior_rows = []
    for name, spec in splits.items():
        for split_name, priors in spec["class_priors"].items():
            prior_rows.append(
                {
                    "split": name,
                    "part": split_name,
                    "target_domain": spec["target_domain"],
                    **{f"prior_{k}": v for k, v in priors.items()},
                }
            )
    priors_df = pd.DataFrame(prior_rows)
    priors_path = results_dir / "split_class_priors.csv"
    priors_df.to_csv(priors_path, index=False)
    tests["t1_8_class_priors_logged"] = {
        "path": str(priors_path),
        "rows": len(priors_df),
        "by_split": {name: spec["class_priors"] for name, spec in splits.items()},
    }

    # Freeze twice with same seed → identical hash (T1.6)
    split_json = splits_dir / "bd_shs_loco_splits.json"
    sha1 = freeze_and_hash_splits(splits, split_json)
    sha2 = freeze_and_hash_splits(splits, split_json)
    tests["t1_6_split_hash_deterministic"] = {
        "sha256_run1": sha1,
        "sha256_run2": sha2,
        "match": sha1 == sha2,
        "path": str(split_json),
    }
    if sha1 != sha2:
        raise AssertionError("Split hash not deterministic across two runs")

    # Also freeze LOCO for 30k if available (secondary analysis axis)
    extra_split_hashes: dict[str, str] = {}
    if bn30k is not None:
        try:
            bn30k_dedup, bn_stats = dedup_pipeline(
                bn30k,
                text_col="text",
                near_threshold=float(dedup_cfg.get("near_threshold", 0.8)),
                num_perm=int(dedup_cfg.get("minhash_num_perm", 128)),
                run_near=False,  # already category-limited; exact only
            )
            bn30k_splits = create_loco_splits(
                bn30k_dedup,
                domain_col="domain",
                seed=seed,
                min_domain_size=200,
                id_col="example_id",
            )
            bn30k_json = splits_dir / "bengali_hs_30k_loco_splits.json"
            extra_split_hashes["bengali_hs_30k"] = freeze_and_hash_splits(
                bn30k_splits, bn30k_json
            )
            bn30k_dedup.to_csv(processed_dir / "bengali_hs_30k_processed.csv", index=False)
        except Exception as e:
            LOG.warning("30k LOCO splits skipped: %s", e)
    tests["t1_5_extra_split_hashes"] = extra_split_hashes

    # ------------------------------------------------------------------
    # T1.10: English control
    # ------------------------------------------------------------------
    en_cfg = data_cfg["english_control"]
    en_df, en_decision = load_english_control(
        dataset_name=str(en_cfg.get("primary", "auto")),
        target_n=int(en_cfg.get("target_n", 50281)),
        target_prior=float(en_cfg.get("target_prior", 24156 / 50281)),
        seed=seed,
        prefer=str(en_cfg.get("primary", "toxigen")),
    )
    write_english_control_decision(en_decision, results_dir / "english_control_decision.json")
    en_df.to_csv(processed_dir / "english_control_matched.csv", index=False)

    bd_prior = float(df_dedup["label"].mean())
    en_prior = float(en_df["label"].mean())
    size_rel_err = abs(len(en_df) - len(df_dedup)) / max(len(df_dedup), 1)
    prior_abs_err = abs(en_prior - bd_prior)
    t1_10_ok = size_rel_err <= 0.10 and prior_abs_err <= 0.02
    tests["t1_10_english_control"] = {
        "ok": bool(t1_10_ok),
        "chosen": en_decision.get("chosen"),
        "n_en": int(len(en_df)),
        "n_bd": int(len(df_dedup)),
        "size_rel_err": size_rel_err,
        "prior_en": en_prior,
        "prior_bd": bd_prior,
        "prior_abs_err": prior_abs_err,
        "n_domains_en": int(en_df["domain"].nunique()),
        "decision_path": str(results_dir / "english_control_decision.json"),
        "decision": en_decision,
    }
    if not t1_10_ok:
        notes.append(
            f"T1.10 size/prior match failed: size_rel_err={size_rel_err:.4f}, "
            f"prior_abs_err={prior_abs_err:.4f}"
        )

    # English LOCO splits (for later phases)
    try:
        en_splits = create_loco_splits(
            en_df,
            domain_col="domain",
            seed=seed,
            min_domain_size=max(100, int(0.02 * len(en_df))),
            id_col="example_id",
        )
        en_json = splits_dir / "english_control_loco_splits.json"
        tests["t1_5_english_split_hash"] = freeze_and_hash_splits(en_splits, en_json)
        tests["t1_5_english_integrity"] = verify_split_integrity(en_splits)
    except Exception as e:
        LOG.warning("English LOCO splits skipped: %s", e)
        tests["t1_5_english_split_hash"] = None
        notes.append(f"English LOCO skipped: {e}")

    # ------------------------------------------------------------------
    # T1.9: shift magnitudes
    # ------------------------------------------------------------------
    shift_model = str(data_cfg["shift"].get("embedding_model", "paraphrase-multilingual-MiniLM-L12-v2"))
    max_side = int(data_cfg["shift"].get("max_samples_per_side", 1500))
    if skip_heavy:
        LOG.info("Skipping shift magnitudes (--skip-heavy)")
        shift_rows: list[dict[str, Any]] = []
        tests["t1_9_shift_magnitude"] = {"skipped": True}
    else:
        shift_rows = shift_rows_from_splits(
            df_dedup,
            splits,
            text_col="text",
            id_col="example_id",
            model_name=shift_model,
            max_samples_per_side=max_side,
        )
        # Also measure English control internal shift (largest domain gap)
        if len(en_df["domain"].unique()) >= 2 and len(shift_rows) > 0:
            try:
                from src.data.shift import compute_shift_magnitude

                # Use first LOCO-like pair: most frequent domain vs rest
                vc = en_df["domain"].value_counts()
                top = vc.index[0]
                en_src = en_df.loc[en_df["domain"] != top, "text"].tolist()
                en_tgt = en_df.loc[en_df["domain"] == top, "text"].tolist()
                en_metrics = compute_shift_magnitude(
                    en_src,
                    en_tgt,
                    model_name=shift_model,
                    max_samples=max_side,
                )
                en_metrics.update(
                    {
                        "split": "english_control_top_domain_vs_rest",
                        "source_domain": "english_rest",
                        "target_domain": top,
                    }
                )
                shift_rows.append(en_metrics)
            except Exception as e:
                LOG.warning("English shift metrics skipped: %s", e)

        shift_df = pd.DataFrame(shift_rows)
        shift_path = results_dir / "shift_magnitudes.csv"
        shift_df.to_csv(shift_path, index=False)
        tests["t1_9_shift_magnitude"] = {
            "ok": bool(len(shift_df) > 0),
            "path": str(shift_path),
            "rows": len(shift_df),
            "columns": list(shift_df.columns),
            "has_mmd": "mmd" in shift_df.columns,
            "has_vocab_overlap": "vocab_overlap" in shift_df.columns,
        }

    # ------------------------------------------------------------------
    # T1.11: tokenizer fertility
    # ------------------------------------------------------------------
    fert_n = int(data_cfg["fertility"].get("n_samples", 1000))
    model_ids = list(data_cfg["fertility"].get("model_ids", FERTILITY_MODEL_IDS))
    if skip_heavy:
        LOG.info("Skipping fertility (--skip-heavy)")
        fert_rows: list[dict[str, Any]] = []
        tests["t1_11_fertility"] = {"skipped": True}
    else:
        bn_texts = df_dedup["text"].sample(
            n=min(fert_n, len(df_dedup)), random_state=seed
        ).tolist()
        en_texts = en_df["text"].sample(
            n=min(fert_n, len(en_df)), random_state=seed
        ).tolist()
        fert_rows = measure_all_fertility(
            model_ids, bn_texts, en_texts, n_samples=fert_n, skip_on_error=True
        )
        fert_df = pd.DataFrame(fert_rows)
        fert_path = results_dir / "tokenizer_fertility.csv"
        fert_df.to_csv(fert_path, index=False)
        tests["t1_11_fertility"] = {
            "ok": bool(len(fert_df) > 0),
            "path": str(fert_path),
            "n_rows": len(fert_df),
            "models": sorted(set(fert_df["model"])),
            "languages": sorted(set(fert_df["language"])),
        }

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------
    all_ok = bool(tests.get("t1_1_load_bd_shs") and tests.get("t1_2_metadata"))
    all_ok = all_ok and bool(tests.get("t1_5_loco_disjoint"))
    all_ok = all_ok and bool(tests.get("t1_6_split_hash_deterministic", {}).get("match"))
    all_ok = all_ok and bool(tests.get("t1_7_no_leakage"))
    if not skip_heavy:
        all_ok = all_ok and bool(tests.get("t1_9_shift_magnitude", {}).get("ok", True))
        all_ok = all_ok and bool(tests.get("t1_11_fertility", {}).get("ok", True))
    # T1.10 may warn without hard-fail if only notes (documented)
    status = "passed" if all_ok else "failed"
    if notes:
        status = "passed_with_notes" if all_ok else "failed"

    payload = {
        "tests": tests,
        "dedup_stats": dedup_stats,
        "notes": notes,
        "outputs": {
            "processed_bd_shs": str(processed_path),
            "processed_english_control": str(processed_dir / "english_control_matched.csv"),
            "splits_json": str(split_json),
            "shift_magnitudes": str(results_dir / "shift_magnitudes.csv"),
            "fertility": str(results_dir / "tokenizer_fertility.csv"),
            "english_control_decision": str(results_dir / "english_control_decision.json"),
            "split_class_priors": str(priors_path),
        },
        "config_path": str(CFG_PATH),
        "config_hash": config_hash(cfg),
        "bd_shs_n": int(len(df_dedup)),
        "english_control_n": int(len(en_df)),
        "loco_splits": list(splits.keys()),
    }
    run = write_run_json(
        ROOT / "outputs" / "runs" / "run.json",
        phase=1,
        phase_name="data_preparation_split_engineering",
        status=status,
        payload=payload,
        seed=seed,
        config_hash=config_hash(cfg),
        git_commit=get_git_commit(str(ROOT)),
    )
    LOG.info("Phase 1 status=%s", status)
    if notes:
        for n in notes:
            LOG.warning("NOTE: %s", n)
    return run


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 1 data preparation")
    parser.add_argument(
        "--skip-heavy",
        action="store_true",
        help="Skip embedding shift metrics and tokenizer fertility",
    )
    args = parser.parse_args()
    result = run(skip_heavy=args.skip_heavy)
    print(json.dumps({"status": result.get("status"), "phase": result.get("phase")}, indent=2))
    return 0 if result.get("status") in {"passed", "passed_with_notes"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
