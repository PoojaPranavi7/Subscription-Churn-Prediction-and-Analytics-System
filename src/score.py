"""Score customers and assign risk buckets.

Role in the pipeline (step 9, final output):
    Loads the selected model pipeline (preprocessing + Random Forest) and the tuned
    decision threshold, scores every customer in the modeling table, and assigns a
    risk bucket. Produces the analytics-ready file for BI tools.

Risk buckets (Section 10), based on churn_probability:
    - Low        (< 0.30)
    - Medium     (0.30 - 0.60)
    - High       (0.60 - 0.80)
    - Very High  (0.80+)

Output:
    - outputs/predictions/customer_scored.csv with columns:
      customer_id, churn_probability, predicted_churn, risk_bucket
      (actual churn label is also included for convenience/auditing).

Run standalone (requires `python -m src.train` and `python -m src.evaluate`):
    python -m src.score
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

try:
    from src.config import MODELS_DIR, PREDICTIONS_DIR, PROCESSED_DATA_DIR
    from src import preprocess as P
    from src import train as T
    from src import evaluate as E
except ImportError:  # allow running as a plain script
    from config import MODELS_DIR, PREDICTIONS_DIR, PROCESSED_DATA_DIR
    import preprocess as P
    import train as T
    import evaluate as E


SCORED_ARTIFACT = "customer_scored.csv"

# Risk-bucket cut points (upper-exclusive except the open-ended top bucket).
RISK_BINS = [-np.inf, 0.30, 0.60, 0.80, np.inf]
RISK_LABELS = ["Low", "Medium", "High", "Very High"]


def assign_risk_bucket(proba: np.ndarray | pd.Series) -> pd.Categorical:
    """Map churn probabilities to ordered risk buckets (Section 10)."""
    return pd.cut(proba, bins=RISK_BINS, labels=RISK_LABELS, right=False, ordered=True)


def load_model_and_threshold() -> tuple:
    """Load the fitted pipeline and the tuned decision threshold."""
    model_path = MODELS_DIR / T.FINAL_MODEL_ARTIFACT
    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing model: {model_path}. Run `python -m src.train` first."
        )
    model = joblib.load(model_path)

    thr_path = MODELS_DIR / E.THRESHOLD_ARTIFACT
    if not thr_path.exists():
        raise FileNotFoundError(
            f"Missing threshold: {thr_path}. Run `python -m src.evaluate` first."
        )
    threshold = float(json.loads(thr_path.read_text())["threshold"])
    return model, threshold


def score_customers(model, threshold: float, df: pd.DataFrame) -> pd.DataFrame:
    """Produce the customer-level scored table."""
    proba = model.predict_proba(df[P.all_feature_columns()])[:, 1]
    out = pd.DataFrame({
        "customer_id": df["customer_id"].to_numpy(),
        "churn_probability": np.round(proba, 6),
        "predicted_churn": (proba >= threshold).astype(int),
        "risk_bucket": assign_risk_bucket(proba),
    })
    if "churn" in df.columns:
        out["actual_churn"] = df["churn"].to_numpy()
    return out.sort_values("churn_probability", ascending=False).reset_index(drop=True)


def run_scoring() -> pd.DataFrame:
    """Full Phase-10 scoring flow: score all customers and persist the file."""
    path = PROCESSED_DATA_DIR / "modeling_table.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing modeling table: {path}. Run `python -m src.features` first."
        )
    df = pd.read_csv(path)

    model, threshold = load_model_and_threshold()
    scored = score_customers(model, threshold, df)

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    scored.to_csv(PREDICTIONS_DIR / SCORED_ARTIFACT, index=False)

    # --- summary ---
    bucket_counts = scored["risk_bucket"].value_counts().reindex(RISK_LABELS, fill_value=0)
    print("=" * 60)
    print("Customer scoring summary")
    print("=" * 60)
    print(f"customers scored      : {len(scored):,}")
    print(f"decision threshold    : {threshold:.2f}")
    print(f"predicted churn rate  : {scored['predicted_churn'].mean():.1%}")
    print("-" * 60)
    print("risk buckets:")
    for label in RISK_LABELS:
        n = int(bucket_counts[label])
        print(f"  {label:<10}: {n:>6,}  ({n / len(scored):.1%})")
    if "actual_churn" in scored.columns:
        print("-" * 60)
        print("mean actual churn by risk bucket (calibration sanity):")
        cal = scored.groupby("risk_bucket", observed=False)["actual_churn"].mean()
        for label in RISK_LABELS:
            print(f"  {label:<10}: {cal.get(label, float('nan')):.1%}")
    print("-" * 60)
    print(f"saved: {PREDICTIONS_DIR / SCORED_ARTIFACT}")
    print("=" * 60)

    return scored


def main() -> None:
    run_scoring()


if __name__ == "__main__":
    main()
