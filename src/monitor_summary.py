from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone

import numpy as np
import pandas as pd

DB_PATH = "churn.db"
TABLE = "customer_features"

METRICS_DIR = os.path.join("outputs", "metrics")
OUT_PATH = os.path.join(METRICS_DIR, "monitor_summary.csv")
REPORT_PATH = os.path.join(METRICS_DIR, "monitor_report.json")

def load_features_table() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(f"SELECT * FROM {TABLE}", conn)
    conn.close()
    return df

def drift_flag() -> str:
    if not os.path.exists(REPORT_PATH):
        return "NO_REPORT"
    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        rep = json.load(f)
    return "DRIFT_DETECTED" if not rep.get("success", True) else "OK"

def main():
    os.makedirs(METRICS_DIR, exist_ok=True)

    df = load_features_table()

    run_date = datetime.now(timezone.utc).date().isoformat()

    row_count = int(df.shape[0])
    churn_rate = float((pd.to_numeric(df.get("churn"), errors="coerce") == 1).mean()) if "churn" in df.columns else np.nan
    avg_monthly = float(pd.to_numeric(df.get("MonthlyCharges"), errors="coerce").mean()) if "MonthlyCharges" in df.columns else np.nan
    null_rate_total = float(df["TotalCharges"].isna().mean()) if "TotalCharges" in df.columns else np.nan

    row = pd.DataFrame([{
        "run_date": run_date,
        "row_count": row_count,
        "churn_rate": churn_rate,
        "avg_monthly_charges": avg_monthly,
        "null_rate_total_charges": null_rate_total,
        "drift_flags": drift_flag(),
    }])

    if os.path.exists(OUT_PATH):
        hist = pd.read_csv(OUT_PATH)
        combined = pd.concat([hist, row], ignore_index=True)
        combined = combined.drop_duplicates(subset=["run_date"], keep="last")
    else:
        combined = row

    combined.to_csv(OUT_PATH, index=False)
    print("Monitoring summary export created")
    print(f"Saved: {OUT_PATH}")

if __name__ == "__main__":
    main()
