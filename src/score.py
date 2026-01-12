from __future__ import annotations

import json
import os
import sqlite3

import joblib
import numpy as np
import pandas as pd


DB_PATH = "churn.db"

MODELS_DIR = os.path.join("outputs", "models")
METRICS_DIR = os.path.join("outputs", "metrics")
PRED_DIR = os.path.join("outputs", "predictions")
OUT_PATH = os.path.join(PRED_DIR, "customer_scored.csv")

MODEL_PATH = os.path.join(MODELS_DIR, "best_model.joblib")
PIPELINE_PATH = os.path.join(MODELS_DIR, "preprocess_pipeline.joblib")
META_PATH = os.path.join(MODELS_DIR, "best_model_meta.json")
FEATURE_IMPORTANCE_PATH = os.path.join(METRICS_DIR, "feature_importance.csv")

def ensure_dirs() -> None:
    os.makedirs(PRED_DIR, exist_ok=True)

def connect_db():
    return sqlite3.connect(DB_PATH)

def list_tables(conn) -> set[str]:
    q = "SELECT name FROM sqlite_master WHERE type='table';"
    return set(pd.read_sql(q, conn)["name"].tolist())

def load_threshold() -> float:
    if not os.path.exists(META_PATH):
        return 0.50
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return float(meta.get("threshold_optimization", {}).get("chosen_threshold", 0.50))

def load_top3_global_drivers() -> list[str]:
    if not os.path.exists(FEATURE_IMPORTANCE_PATH):
        return ["(missing_feature_importance)", "", ""]
    df = pd.read_csv(FEATURE_IMPORTANCE_PATH)
    imp_col = "importance_mean" if "importance_mean" in df.columns else ("importance" if "importance" in df.columns else None)
    if imp_col is None:
        top = df.head(3)["feature"].astype(str).tolist()
    else:
        top = df.sort_values(imp_col, ascending=False).head(3)["feature"].astype(str).tolist()
    while len(top) < 3:
        top.append("")
    return top[:3]

def compute_tenure_band(tenure) -> str:
    if pd.isna(tenure):
        return "Unknown"
    t = float(tenure)
    if t <= 6:
        return "0–6"
    if t <= 12:
        return "7–12"
    if t <= 24:
        return "13–24"
    if t <= 48:
        return "25–48"
    return "49+"

def risk_bucket(p: float) -> str:
    if p < 0.30:
        return "Low (<0.30)"
    if p < 0.60:
        return "Medium (0.30–0.60)"
    if p < 0.80:
        return "High (0.60–0.80)"
    return "Very High (0.80+)"

def load_customer_table() -> pd.DataFrame:
    conn = connect_db()
    tables = list_tables(conn)

    if "customers_clean" in tables:
        df = pd.read_sql("SELECT * FROM customers_clean", conn)
        conn.close()
        return df

    if "customers_raw" in tables:
        df = pd.read_sql("SELECT * FROM customers_raw", conn)
        conn.close()
        return df

    conn.close()
    raise FileNotFoundError("No customers_clean/customers_raw tables found in churn.db")

def normalize_target(series: pd.Series) -> pd.Series:
    if series.dtype == "O":
        s = series.astype(str).str.strip().str.lower()
        return (s == "yes").astype(int)
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(int)

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    monthly = pd.to_numeric(out.get("MonthlyCharges"), errors="coerce")
    total = pd.to_numeric(out.get("TotalCharges"), errors="coerce")
    tenure = pd.to_numeric(out.get("tenure"), errors="coerce")

    out["log_monthly_charges"] = np.log1p(monthly)
    out["log_total_charges"] = np.log1p(total)
    out["charges_per_tenure"] = total / (tenure.fillna(0) + 1.0)

    return out

def main() -> None:
    ensure_dirs()

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Missing model at {MODEL_PATH}. Run: python src/train.py")

    if not os.path.exists(PIPELINE_PATH):
        raise FileNotFoundError(
            f"Missing preprocessing pipeline at {PIPELINE_PATH}. Run: python src/preprocess.py"
        )

    model = joblib.load(MODEL_PATH)
    preprocess_pipe = joblib.load(PIPELINE_PATH)
    threshold = load_threshold()
    d1, d2, d3 = load_top3_global_drivers()

    df = load_customer_table()

    # Identify ID + target columns (Telco uses customerID and Churn)
    id_col = "customerID" if "customerID" in df.columns else None
    churn_col = "Churn" if "Churn" in df.columns else ("churn" if "churn" in df.columns else None)

    if id_col is None:
        df["customerID"] = np.arange(len(df))
        id_col = "customerID"

    # Add engineered features BEFORE preprocess transform
    df_eng = add_engineered_features(df)

    # Build X_raw: drop target if present (keep customerID in X if your pipeline expects it? Usually NO.)
    X_raw = df_eng.copy()
    if churn_col and churn_col in X_raw.columns:
        X_raw = X_raw.drop(columns=[churn_col])

    # If pipeline was trained without customerID (likely), drop it for safety:
    if "customerID" in X_raw.columns:
        X_raw = X_raw.drop(columns=["customerID"])

    # Transform using the saved preprocess pipeline
    X_proc = preprocess_pipe.transform(X_raw)

    # Score
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_proc)[:, 1]
    else:
        score = model.decision_function(X_proc)
        proba = (score - score.min()) / (score.max() - score.min() + 1e-9)

    pred = (proba >= threshold).astype(int)

    # Assemble Tableau dataset
    out = pd.DataFrame({
        "customer_id": df[id_col].astype(str),
        "churn_actual": normalize_target(df[churn_col]) if churn_col else np.nan,
        "churn_probability": proba,
        "predicted_label": pred,
        "threshold_used": float(threshold),
        "key_driver_1": d1,
        "key_driver_2": d2,
        "key_driver_3": d3,
    })

    # Add segment fields for Tableau story
    for col in ["Contract", "tenure", "MonthlyCharges", "TotalCharges", "PaymentMethod",
                "InternetService", "PaperlessBilling", "SeniorCitizen", "gender", "Partner", "Dependents"]:
        if col in df.columns:
            out[col] = df[col]

    if "tenure" in out.columns:
        out["tenure_band"] = out["tenure"].apply(compute_tenure_band)

    out["risk_bucket"] = [risk_bucket(p) for p in out["churn_probability"]]

    out.to_csv(OUT_PATH, index=False)

    print("customer_scored export created")
    print(f"Saved: {OUT_PATH}")
    print(f"Rows: {len(out)}")
    print(f"Threshold used: {threshold:.2f}")

if __name__ == "__main__":
    main()
