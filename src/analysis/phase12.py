"""Phase 12 ablation planning, validation, and result assembly."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.splits import load_splits
from src.metrics.calibration import brier_score, expected_calibration_error, negative_log_likelihood
from src.metrics.task_metrics import compute_all_task_metrics
from src.models.finetune import build_prompt


CALIB_VARIANTS = ("bn", "en", "mixed", "random")
PROMPT_VARIANTS = ("baseline", "direct", "instruction")


def frozen_frames(data_path: Path, split_path: Path, split_name: str) -> tuple[dict[str, pd.DataFrame], str]:
    """Load exactly the frozen split IDs and fail on missing/overlapping rows."""
    data = pd.read_csv(data_path)
    data["example_id"] = data["example_id"].astype(str)
    if data["example_id"].duplicated().any() or data["text"].isna().any():
        raise ValueError(f"Invalid source data: {data_path}")
    specs, sha = load_splits(split_path)
    if split_name not in specs:
        raise KeyError(f"Missing frozen split {split_name} in {split_path}")
    spec = specs[split_name]
    indexed = data.set_index("example_id", drop=False)
    out = {}
    seen = set()
    for part in ("train", "id_val", "id_test", "ood_test"):
        ids = [str(x) for x in spec[f"{part}_ids"]]
        if len(ids) != len(set(ids)) or seen.intersection(ids):
            raise ValueError(f"Duplicate or overlapping IDs in {part}")
        seen.update(ids)
        missing = set(ids) - set(indexed.index)
        if missing:
            raise ValueError(f"{len(missing)} missing IDs in {part}")
        out[part] = indexed.loc[ids].reset_index(drop=True)
    return out, sha


def calibration_sets(bn_train: pd.DataFrame, en_train: pd.DataFrame, *, n: int,
                     seed: int, template: str, forbidden_bn: set[str],
                     forbidden_en: set[str]) -> dict[str, dict]:
    """Construct four deterministic A1 sets with train-only provenance."""
    if n < 2 or n % 2:
        raise ValueError("calibration sample count must be even and at least two")
    rng = np.random.default_rng(seed)

    def sample(frame: pd.DataFrame, count: int, forbidden: set[str]):
        if len(frame) < count:
            raise ValueError(f"Need {count} training examples; found {len(frame)}")
        chosen = frame.sample(n=count, random_state=seed)
        ids = chosen["example_id"].astype(str).tolist()
        if set(ids) & forbidden:
            raise ValueError("Calibration IDs overlap validation or test IDs")
        return chosen["text"].astype(str).tolist(), ids

    bn_text, bn_ids = sample(bn_train, n, forbidden_bn)
    en_text, en_ids = sample(en_train, n, forbidden_en)
    # Synthetic random Unicode is a negative-control calibration input. It has
    # no dataset IDs and is never interpreted as training or evaluation data.
    alphabet = list("অআইউকগচজটতদনপবমরলসহ") + list("abcdefghijklmnopqrstuvwxyz")
    random_text = ["".join(rng.choice(alphabet, size=64)) for _ in range(n)]
    raw = {
        "bn": (bn_text, bn_ids),
        "en": (en_text, en_ids),
        "mixed": (bn_text[: n // 2] + en_text[: n // 2], bn_ids[: n // 2] + en_ids[: n // 2]),
        "random": (random_text, []),
    }
    result = {}
    for name, (texts, ids) in raw.items():
        prompts = [build_prompt(text, template) for text in texts]
        digest = hashlib.sha256(json.dumps({"prompts": prompts, "ids": ids}, ensure_ascii=False).encode()).hexdigest()
        result[name] = {"prompts": prompts, "ids": ids, "sha256": digest,
                        "n": len(prompts), "source": "synthetic" if name == "random" else "frozen_train_only"}
    return result


def validate_ablation_logits(frame: pd.DataFrame, expected: pd.DataFrame) -> None:
    required = {"example_id", "y_true", "logit_nonhate", "logit_hate", "prob_nonhate", "prob_hate", "prediction"}
    if required - set(frame):
        raise ValueError(f"Missing logit columns: {sorted(required - set(frame))}")
    if frame.empty or frame["example_id"].duplicated().any():
        raise ValueError("Empty or duplicate logits")
    ids = expected["example_id"].astype(str).tolist()
    if frame["example_id"].astype(str).tolist() != ids or not np.array_equal(
        frame["y_true"].to_numpy(dtype=int), expected["label"].to_numpy(dtype=int)
    ):
        raise ValueError("Ablation logits do not match the frozen evaluation examples")
    values = frame[list(required - {"example_id"})].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite ablation logits or probabilities")
    p = frame["prob_hate"].to_numpy(dtype=float)
    if (p < 0).any() or (p > 1).any() or not np.allclose(p + frame["prob_nonhate"], 1, atol=1e-6):
        raise ValueError("Invalid ablation probabilities")
    if not np.array_equal((p >= .5).astype(int), frame["prediction"].to_numpy(dtype=int)):
        raise ValueError("Ablation predictions disagree with probabilities")


def metric_row(frame: pd.DataFrame, **context) -> dict:
    y = frame["y_true"].to_numpy(dtype=int)
    p = frame["prob_hate"].to_numpy(dtype=float)
    pred = frame["prediction"].to_numpy(dtype=int)
    task = compute_all_task_metrics(y, pred, p)
    if np.unique(y).size < 2:
        for key in ("balanced_accuracy", "auroc", "average_precision", "mcc"):
            task[key] = None
    return {**context, "n_samples": len(frame), "accuracy": task["accuracy"],
            "macro_f1": task["macro_f1"], "ece_equal_mass_15": expected_calibration_error(y, p, 15, "equal_mass"),
            "ece_equal_width_15": expected_calibration_error(y, p, 15, "equal_width"),
            "brier": brier_score(y, p), "nll": negative_log_likelihood(y, p),
            "single_class_eval_set": bool(np.unique(y).size < 2)}


def temperature_source_rows(path: Path, *, models: set[str], languages: set[str], seeds: set[int]) -> pd.DataFrame:
    """A8 reuses Phase 11 held-out fits; missing oracle remains missing."""
    if not path.is_file():
        raise FileNotFoundError(f"A8 requires Phase 11 results: {path}")
    frame = pd.read_parquet(path)
    required = {"artifact", "model", "lang", "seed", "eval_set", "arm", "temperature", "fit_split", "fit_file",
                "raw_ece_equal_mass_15", "ece_equal_mass_15", "raw_brier", "brier", "raw_nll", "nll", "deployable"}
    if required - set(frame):
        raise ValueError("Phase 11 results lack A8 provenance or metrics")
    frame = frame[frame["model"].isin(models) & frame["lang"].isin(languages) & frame["seed"].isin(seeds)].copy()
    if frame.empty:
        raise ValueError("No Phase 11 rows match A8 selection")
    if frame.duplicated(["artifact", "eval_set", "arm"]).any():
        raise ValueError("Duplicate Phase 11 arm/evaluation rows")
    if (frame.loc[frame.arm.isin(["T_own", "T_fp"]), "fit_split"] != "id_val").any():
        raise ValueError("A8 non-oracle fits must use ID validation only")
    if (frame.loc[frame.arm == "T_oracle", "fit_split"] != "ood_val").any():
        raise ValueError("A8 oracle fits must use OOD validation only")
    frame["ablation_id"] = "A8"
    frame["variant"] = frame["arm"]
    frame["language"] = frame["lang"]
    frame["status"] = "success"
    frame["source"] = "phase11_recalibration_results"
    return frame
