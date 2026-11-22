# check_fit.py  (run: python check_fit.py)
import json, glob, numpy as np, pandas as pd
from pathlib import Path

rows = []
for phase in (4, 5):
    for rj in sorted(glob.glob(f"outputs/runs/phase{phase}/*.json")):
        m = json.load(open(rj, encoding="utf-8"))
        if m.get("status") != "success":
            continue
        name = m["run_name"]
        tm = json.load(
            open(
                f"outputs/models/phase{phase}/{name}/train_metrics.json",
                encoding="utf-8",
            )
        )
        losses = np.array([h["loss"] for h in tm["log_history"] if "loss" in h])
        k = max(3, len(losses) // 5)
        first, last = losses[:k].mean(), losses[-k:].mean()
        tail = losses[-max(5, len(losses) // 4) :]
        slope = np.polyfit(np.arange(len(tail)), tail, 1)[0]  # <0 = still improving
        val_loss = tm["eval"].get("eval_loss")
        sm = m["split_metrics"]
        rows.append(
            {
                "run": name.replace("_lora_bf16", ""),
                "loss_first": round(first, 3),
                "loss_last": round(last, 3),
                "val_loss": round(val_loss, 3),
                "val-train": round(val_loss - last, 3),
                "tail_slope": round(slope, 4),
                "acc_val": round(sm["id_val"]["accuracy"], 3),
                "acc_test": round(sm["id_test"]["accuracy"], 3),
                "acc_ood": round(sm["ood_test"]["accuracy"], 3),
                "ece_test": round(sm["id_test"]["ece"], 3),
                "ece_ood": round(sm["ood_test"]["ece"], 3),
                "conf_gap_test": round(sm["id_test"]["confidence_gap"], 3),
                "conf_gap_ood": round(sm["ood_test"]["confidence_gap"], 3),
            }
        )
df = pd.DataFrame(rows)
pd.set_option("display.width", 250, "display.max_columns", None)
print(df.to_string(index=False))
df.to_csv("outputs/results/fit_diagnostics.csv", index=False)
