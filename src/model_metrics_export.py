from __future__ import annotations

import json
import os
import pandas as pd

METRICS_DIR = os.path.join("outputs", "metrics")
IN_PATH = os.path.join(METRICS_DIR, "metrics.json")
OUT_PATH = os.path.join(METRICS_DIR, "model_metrics.csv")

def main():
    if not os.path.exists(IN_PATH):
        raise FileNotFoundError(
            f"Missing {IN_PATH}. Run:\n"
            f"  python src/evaluate.py"
        )

    with open(IN_PATH, "r", encoding="utf-8") as f:
        m = json.load(f)

    df = pd.DataFrame([{
        "join_key": 1,
        "evaluated_at_utc": m.get("evaluated_at_utc"),
        "threshold_used": m.get("threshold_used"),
        "accuracy": m.get("accuracy"),
        "f1": m.get("f1"),
        "precision": m.get("precision"),
        "recall": m.get("recall"),
        "roc_auc": m.get("roc_auc"),
    }])

    df.to_csv(OUT_PATH, index=False)

    print("model_metrics export created")
    print(f"Saved: {OUT_PATH}")

if __name__ == "__main__":
    main()
