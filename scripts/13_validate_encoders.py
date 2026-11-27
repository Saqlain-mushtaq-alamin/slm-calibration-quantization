"""Validate all four Phase 16 arms against frozen IDs and saved manifests."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.encoder import fingerprint, load_config, load_frozen
from src.utils.atomic import atomic_write_json


def check_arm(cfg, precision, frames, sha):
    output = Path(cfg["output"])
    manifest_path = output / ("run.json" if precision == "fp32" else "int8_run.json")
    if not manifest_path.exists():
        return {"status": "missing", "reason": str(manifest_path)}
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "completed" or manifest.get("smoke_only"):
        return {"status": "incomplete", "reason": manifest.get("status")}
    if manifest["split_sha256"] != sha:
        return {"status": "failed", "reason": "Split fingerprint changed"}
    if manifest["dataset_fingerprints"] != {key: fingerprint(frame) for key, frame in frames.items()}:
        return {"status": "failed", "reason": "Dataset fingerprint changed"}
    for part in ("id_test", "ood_test"):
        pred_path = output / "predictions" / f"{precision}_{part}.csv"
        metric_path = output / "metrics" / f"{precision}_{part}.json"
        if not pred_path.exists() or not metric_path.exists():
            return {"status": "failed", "reason": f"Missing {precision} {part} artifact"}
        pred = pd.read_csv(pred_path)
        expected = frames[part]
        if pred.example_id.astype(str).tolist() != expected.example_id.astype(str).tolist() or not np.array_equal(pred.y_true, expected.label):
            return {"status": "failed", "reason": f"Alignment mismatch: {part}"}
        if not np.isfinite(pred[["logit_nonhate", "logit_hate", "prob_nonhate", "prob_hate"]]).all().all():
            return {"status": "failed", "reason": f"Nonfinite output: {part}"}
        metrics = json.loads(metric_path.read_text())
        if metrics["dataset_fingerprint"] != fingerprint(expected):
            return {"status": "failed", "reason": f"Metric fingerprint mismatch: {part}"}
        if not all(np.isfinite(metrics[key]) for key in ("accuracy", "macro_f1", "macro_precision", "macro_recall", "ece", "brier", "nll", "aurc")):
            return {"status": "failed", "reason": f"Invalid metrics: {part}"}
    return {"status": "completed"}


def main():
    arms = {}
    for name in ("banglabert", "xlmr"):
        cfg = load_config(f"configs/encoder_{name}.yaml")
        frames, sha = load_frozen(cfg)
        for precision in ("fp32", "int8"):
            arms[f"{name}_{precision}"] = check_arm(cfg, precision, frames, sha)
    status = "COMPLETE" if all(v["status"] == "completed" for v in arms.values()) else "PARTIAL"
    result = {"phase": 16, "status": status, "arms": arms}
    atomic_write_json("outputs/encoders/phase16_status.json", result)
    print(json.dumps(result, indent=2))
    return status


if __name__ == "__main__":
    raise SystemExit(0 if main() == "COMPLETE" else 1)
