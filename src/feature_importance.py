from __future__ import annotations

import os
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

@dataclass(frozen=True)
class Config:
    processed_dir: str = os.path.join("outputs", "processed")
    models_dir: str = os.path.join("outputs", "models")
    metrics_dir: str = os.path.join("outputs", "metrics")

    scoring: str = "roc_auc"  # best for imbalance
    n_repeats: int = 15
    random_state: int = 42
    top_k: int = 10

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

def main() -> None:
    ensure_dirs()

    # Load model
    model_path = os.path.join(CFG.models_dir, "best_model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing model: {model_path}. Run: python src/train.py")

    model = joblib.load(model_path)

    # Load test data
    X_test = load_X(CFG.processed_dir, "X_test")
    y_test = pd.read_csv(os.path.join(CFG.processed_dir, "y_test.csv"))["y"].astype(int).values

    # Load feature names
    fn_path = os.path.join(CFG.processed_dir, "feature_names.csv")
    if not os.path.exists(fn_path):
        raise FileNotFoundError(f"Missing feature names: {fn_path}. Ensure Phase 4 wrote feature_names.csv")

    feature_names = pd.read_csv(fn_path)["feature_name"].astype(str).tolist()

    # Safety check: feature count
    if hasattr(X_test, "shape") and X_test.shape[1] != len(feature_names):
        raise ValueError(
            f"Feature mismatch: X_test has {X_test.shape[1]} cols but feature_names has {len(feature_names)}"
        )

    # Compute permutation importance (model-agnostic)
    result = permutation_importance(
        model,
        X_test,
        y_test,
        scoring=CFG.scoring,
        n_repeats=CFG.n_repeats,
        random_state=CFG.random_state,
        n_jobs=-1,
    )

    importances_mean = result.importances_mean
    importances_std = result.importances_std

    df_imp = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_mean": importances_mean,
            "importance_std": importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    # Save Top K
    top_df = df_imp.head(CFG.top_k).copy()
    out_csv = os.path.join(CFG.metrics_dir, "feature_importance.csv")
    top_df.to_csv(out_csv, index=False)

    # Plot Top K
    plt.figure(figsize=(10, 6))

    plot_df = top_df.iloc[::-1]
    plt.barh(plot_df["feature"], plot_df["importance_mean"])
    plt.xlabel(f"Permutation Importance (Δ {CFG.scoring})")
    plt.title("Top Feature Drivers of Churn (Permutation Importance)")
    plt.tight_layout()

    out_png = os.path.join(CFG.metrics_dir, "feature_importance.png")
    plt.savefig(out_png, dpi=200)
    plt.close()

    print("Feature importance complete")
    print(f"Saved: {out_csv}")
    print(f"Saved: {out_png}")
    print("\nTop drivers:")
    print(top_df.to_string(index=False))

if __name__ == "__main__":
    main()
