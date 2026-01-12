from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

@dataclass(frozen=True)
class Config:
    processed_dir: str = os.path.join("outputs", "processed")
    models_dir: str = os.path.join("outputs", "models")
    metrics_dir: str = os.path.join("outputs", "metrics")
    experiments_log: str = os.path.join("outputs", "metrics", "experiments.jsonl")

    random_state: int = 42

    # Threshold optimization settings
    threshold_min: float = 0.10
    threshold_max: float = 0.90
    threshold_step: float = 0.01

    # Choose what to optimize. Options: "accuracy" or "f1"
    threshold_objective: str = "accuracy"

    min_recall_constraint: float = 0.60

CFG = Config()

def ensure_dirs() -> None:
    os.makedirs(CFG.models_dir, exist_ok=True)
    os.makedirs(CFG.metrics_dir, exist_ok=True)

def get_git_commit_hash() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"

def load_artifacts():
    def load_X(base: str):
        npy_path = os.path.join(CFG.processed_dir, f"{base}.npy")
        npz_path = os.path.join(CFG.processed_dir, f"{base}.npz")

        if os.path.exists(npz_path):
            return sparse.load_npz(npz_path)
        if os.path.exists(npy_path):
            return np.load(npy_path)
        raise FileNotFoundError(f"Neither {npy_path} nor {npz_path} found")

    X_train = load_X("X_train")
    X_test = load_X("X_test")

    y_train = pd.read_csv(os.path.join(CFG.processed_dir, "y_train.csv"))["y"].astype(int).values
    y_test = pd.read_csv(os.path.join(CFG.processed_dir, "y_test.csv"))["y"].astype(int).values

    return X_train, X_test, y_train, y_test

def _oof_predictions(model, X_train, y_train) -> Tuple[np.ndarray, np.ndarray]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=CFG.random_state)

    y_pred = cross_val_predict(model, X_train, y_train, cv=cv, method="predict")

    if hasattr(model, "predict_proba"):
        y_proba = cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    else:
        y_score = cross_val_predict(model, X_train, y_train, cv=cv, method="decision_function")
        y_proba = (y_score - y_score.min()) / (y_score.max() - y_score.min() + 1e-9)

    return y_pred, y_proba

def evaluate_model_cv(model, X_train, y_train) -> Dict[str, float]:
    y_pred, y_proba = _oof_predictions(model, X_train, y_train)

    metrics = {
        "accuracy": float(accuracy_score(y_train, y_pred)),
        "f1": float(f1_score(y_train, y_pred)),
        "precision": float(precision_score(y_train, y_pred, zero_division=0)),
        "recall": float(recall_score(y_train, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_train, y_proba)),
    }
    return metrics

def threshold_sweep(y_true: np.ndarray, y_proba: np.ndarray) -> pd.DataFrame:
    thresholds = np.arange(CFG.threshold_min, CFG.threshold_max + 1e-9, CFG.threshold_step)

    rows = []
    for t in thresholds:
        y_hat = (y_proba >= t).astype(int)
        rows.append(
            {
                "threshold": float(t),
                "accuracy": float(accuracy_score(y_true, y_hat)),
                "f1": float(f1_score(y_true, y_hat)),
                "precision": float(precision_score(y_true, y_hat, zero_division=0)),
                "recall": float(recall_score(y_true, y_hat, zero_division=0)),
            }
        )

    return pd.DataFrame(rows)

def choose_best_threshold(sweep_df: pd.DataFrame) -> Tuple[float, Dict[str, float]]:

    df = sweep_df.copy()

    # apply recall constraint (business rule)
    df = df[df["recall"] >= CFG.min_recall_constraint].copy()
    if df.empty:
        # if constraint too strict, fall back to unconstrained best
        df = sweep_df.copy()

    objective = CFG.threshold_objective.lower()
    if objective not in {"accuracy", "f1"}:
        objective = "accuracy"

    best_row = df.sort_values([objective, "recall"], ascending=False).iloc[0]

    best_threshold = float(best_row["threshold"])
    best_metrics = {
        "accuracy": float(best_row["accuracy"]),
        "f1": float(best_row["f1"]),
        "precision": float(best_row["precision"]),
        "recall": float(best_row["recall"]),
    }
    return best_threshold, best_metrics

def log_experiment(record: Dict[str, Any]) -> None:
    with open(CFG.experiments_log, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

def main() -> None:
    ensure_dirs()

    X_train, X_test, y_train, y_test = load_artifacts()

    run_ts = datetime.now(timezone.utc).isoformat()
    git_hash = get_git_commit_hash()

    # Candidate models (2–3)
    candidates: list[Dict[str, Any]] = []

    candidates.append(
        {
            "name": "logreg",
            "model": LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                solver="liblinear",
                random_state=CFG.random_state,
            ),
            "params": {
                "max_iter": 2000,
                "class_weight": "balanced",
                "solver": "liblinear",
                "random_state": CFG.random_state,
            },
        }
    )

    candidates.append(
        {
            "name": "random_forest",
            "model": RandomForestClassifier(
                n_estimators=400,
                max_depth=None,
                min_samples_split=4,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=CFG.random_state,
                n_jobs=-1,
            ),
            "params": {
                "n_estimators": 400,
                "max_depth": None,
                "min_samples_split": 4,
                "min_samples_leaf": 2,
                "class_weight": "balanced",
                "random_state": CFG.random_state,
            },
        }
    )

    try:
        from xgboost import XGBClassifier  # type: ignore

        candidates.append(
            {
                "name": "xgboost",
                "model": XGBClassifier(
                    n_estimators=700,
                    learning_rate=0.05,
                    max_depth=4,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    reg_lambda=1.0,
                    random_state=CFG.random_state,
                    eval_metric="logloss",
                    n_jobs=-1,
                ),
                "params": {
                    "n_estimators": 700,
                    "learning_rate": 0.05,
                    "max_depth": 4,
                    "subsample": 0.9,
                    "colsample_bytree": 0.9,
                    "reg_lambda": 1.0,
                    "random_state": CFG.random_state,
                    "eval_metric": "logloss",
                },
            }
        )
    except Exception:
        pass

    # Train + CV evaluate each candidate on TRAIN
    best = None
    best_score = -1.0

    for cand in candidates:
        name = cand["name"]
        model = cand["model"]
        params = cand["params"]

        metrics = evaluate_model_cv(model, X_train, y_train)

        record = {
            "timestamp_utc": run_ts,
            "git_commit": git_hash,
            "model_name": name,
            "hyperparameters": params,
            "cv_metrics": metrics,
        }
        log_experiment(record)

        print(f"[{name}] CV metrics: {metrics}")

        # Select best by ROC-AUC (primary) + small F1 tie-breaker
        score = metrics["roc_auc"] + 0.001 * metrics["f1"]
        if score > best_score:
            best_score = score
            best = cand

    if best is None:
        raise RuntimeError("No models were evaluated.")

    # Threshold optimization using OOF probabilities on TRAIN 
    best_template = best["model"]
    _, best_oof_proba = _oof_predictions(best_template, X_train, y_train)

    sweep_df = threshold_sweep(y_train, best_oof_proba)
    sweep_path = os.path.join(CFG.metrics_dir, "threshold_sweep.csv")
    sweep_df.to_csv(sweep_path, index=False)

    chosen_threshold, thresh_metrics = choose_best_threshold(sweep_df)

    # Fit the best model on full training set
    best_model = best["model"]
    best_model.fit(X_train, y_train)

    # Save best model
    model_path = os.path.join(CFG.models_dir, "best_model.joblib")
    joblib.dump(best_model, model_path)

    meta = {
        "trained_at_utc": run_ts,
        "git_commit": git_hash,
        "selected_model": best["name"],
        "hyperparameters": best["params"],
        "selection_rule": "max(roc_auc + 0.001*f1) on 5-fold stratified CV (train only)",
        "threshold_optimization": {
            "objective": CFG.threshold_objective,
            "min_recall_constraint": CFG.min_recall_constraint,
            "chosen_threshold": chosen_threshold,
            "oof_metrics_at_threshold": thresh_metrics,
            "threshold_sweep_path": "outputs/metrics/threshold_sweep.csv",
        },
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_features": int(X_train.shape[1]) if hasattr(X_train, "shape") else None,
    }

    meta_path = os.path.join(CFG.models_dir, "best_model_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\nTraining complete")
    print(f"Best model: {best['name']}")
    print(f"Saved: {model_path}")
    print(f"Meta: {meta_path}")
    print(f"Experiments log: {CFG.experiments_log}")
    print(f"Threshold sweep saved: {sweep_path}")
    print(f"Chosen threshold ({CFG.threshold_objective}, recall>={CFG.min_recall_constraint}): {chosen_threshold:.2f}")
    print(f"OOF metrics at chosen threshold: {thresh_metrics}")

if __name__ == "__main__":
    main()
