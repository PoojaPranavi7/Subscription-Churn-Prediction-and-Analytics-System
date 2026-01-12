from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_curve,
)

@dataclass(frozen=True)
class Config:
    processed_dir: str = os.path.join("outputs", "processed")
    models_dir: str = os.path.join("outputs", "models")
    metrics_dir: str = os.path.join("outputs", "metrics")

CFG = Config()

def ensure_dirs() -> None:
    os.makedirs(CFG.metrics_dir, exist_ok=True)

def load_X(processed_dir: str, base: str):
    npz_path = os.path.join(processed_dir, f"{base}.npz")
    npy_path = os.path.join(processed_dir, f"{base}.npy")

    if os.path.exists(npz_path):
        return sparse.load_npz(npz_path)
    if os.path.exists(npy_path):
        return np.load(npy_path)

    raise FileNotFoundError(f"Neither {npy_path} nor {npz_path} found")

def load_threshold(meta_path: str) -> float:
    if not os.path.exists(meta_path):
        return 0.50

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    return float(meta.get("threshold_optimization", {}).get("chosen_threshold", 0.50))

def main() -> None:
    ensure_dirs()

    # Load artifacts
    X_test = load_X(CFG.processed_dir, "X_test")
    y_test = pd.read_csv(os.path.join(CFG.processed_dir, "y_test.csv"))["y"].astype(int).values

    model_path = os.path.join(CFG.models_dir, "best_model.joblib")
    meta_path = os.path.join(CFG.models_dir, "best_model_meta.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Best model not found at {model_path}. Run: python src/train.py")

    model = joblib.load(model_path)
    threshold = load_threshold(meta_path)

    # Probabilities for ROC + thresholding
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_score = model.decision_function(X_test)
        y_proba = (y_score - y_score.min()) / (y_score.max() - y_score.min() + 1e-9)

    # Threshold-based prediction
    y_pred = (y_proba >= threshold).astype(int)

    # Metrics
    metrics = {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "threshold_used": float(threshold),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }

    # Save metrics.json
    metrics_path = os.path.join(CFG.metrics_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Confusion matrix CSV
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    cm_df = pd.DataFrame(cm, index=["actual_0", "actual_1"], columns=["pred_0", "pred_1"])
    cm_df.to_csv(os.path.join(CFG.metrics_dir, "confusion_matrix.csv"), index=True)

    # ROC curve points
    fpr, tpr, thr = roc_curve(y_test, y_proba)
    roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thr})
    roc_df.to_csv(os.path.join(CFG.metrics_dir, "roc_curve_points.csv"), index=False)

    print("Evaluation complete (threshold-aware)")
    print(json.dumps(metrics, indent=2))
    print(f"Saved: {metrics_path}")

if __name__ == "__main__":
    main()