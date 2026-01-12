from __future__ import annotations

import argparse
import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List

import numpy as np
import pandas as pd

@dataclass(frozen=True)
class Config:
    db_path: str = "churn.db"
    table_name: str = "customer_features"

    out_dir: str = os.path.join("outputs", "metrics")
    baseline_path: str = os.path.join("outputs", "metrics", "monitor_baseline.json")
    report_json: str = os.path.join("outputs", "metrics", "monitor_report.json")
    report_csv: str = os.path.join("outputs", "metrics", "monitor_report.csv")

    # Thresholds (tune to your liking)
    max_row_count_pct_change: float = 0.03     # 3%
    max_churn_rate_abs_change: float = 0.02    # 2 percentage points (absolute)
    max_numeric_mean_pct_change: float = 0.05  # 5% relative
    max_numeric_median_pct_change: float = 0.05
    max_null_rate_abs_increase: float = 0.01   # 1 percentage point (absolute)

    # Which numeric columns to monitor (must exist in table)
    numeric_cols: tuple[str, ...] = (
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "charges_per_tenure",
    )

    # Which columns to monitor null rates for (critical columns)
    critical_null_cols: tuple[str, ...] = (
        "TotalCharges",
        "MonthlyCharges",
        "tenure",
        "Contract",
        "PaymentMethod",
        "InternetService",
        "PaperlessBilling",
    )

    target_col: str = "churn"

CFG = Config()

def ensure_dirs() -> None:
    os.makedirs(CFG.out_dir, exist_ok=True)

def load_table() -> pd.DataFrame:
    conn = sqlite3.connect(CFG.db_path)
    df = pd.read_sql(f"SELECT * FROM {CFG.table_name}", conn)
    conn.close()
    return df

def safe_pct_change(curr: float, base: float) -> float:
    eps = 1e-9
    denom = max(abs(base), eps)
    return float((curr - base) / denom)

def compute_snapshot(df: pd.DataFrame) -> Dict[str, Any]:
    snap: Dict[str, Any] = {}

    snap["row_count"] = int(df.shape[0])

    # Churn rate
    if CFG.target_col in df.columns:
        churn = pd.to_numeric(df[CFG.target_col], errors="coerce")
        snap["churn_rate"] = float((churn == 1).mean())
    else:
        snap["churn_rate"] = None

    # Numeric summaries
    numeric_stats: Dict[str, Any] = {}
    for col in CFG.numeric_cols:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            numeric_stats[col] = {
                "mean": float(s.mean()),
                "median": float(s.median()),
            }
    snap["numeric_stats"] = numeric_stats

    # Null rates
    null_rates: Dict[str, float] = {}
    for col in CFG.critical_null_cols:
        if col in df.columns:
            null_rates[col] = float(df[col].isna().mean())
    snap["null_rates"] = null_rates

    snap["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    snap["source"] = {"db_path": CFG.db_path, "table_name": CFG.table_name}

    return snap

def compare_snapshots(baseline: Dict[str, Any], current: Dict[str, Any]) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []

    # Row count % change
    base_rows = float(baseline.get("row_count", 0))
    curr_rows = float(current.get("row_count", 0))
    row_pct = safe_pct_change(curr_rows, base_rows)
    checks.append(
        {
            "check": "row_count_pct_change",
            "passed": abs(row_pct) <= CFG.max_row_count_pct_change,
            "baseline": base_rows,
            "current": curr_rows,
            "delta": float(curr_rows - base_rows),
            "pct_change": row_pct,
            "threshold": CFG.max_row_count_pct_change,
        }
    )

    # Churn rate absolute change
    base_churn = baseline.get("churn_rate", None)
    curr_churn = current.get("churn_rate", None)
    if base_churn is not None and curr_churn is not None:
        churn_abs = float(curr_churn - base_churn)
        checks.append(
            {
                "check": "churn_rate_abs_change",
                "passed": abs(churn_abs) <= CFG.max_churn_rate_abs_change,
                "baseline": float(base_churn),
                "current": float(curr_churn),
                "delta": churn_abs,
                "threshold": CFG.max_churn_rate_abs_change,
            }
        )

    # Numeric mean/median drift
    base_num = baseline.get("numeric_stats", {}) or {}
    curr_num = current.get("numeric_stats", {}) or {}

    for col in CFG.numeric_cols:
        if col in base_num and col in curr_num:
            for stat_name, thresh in [
                ("mean", CFG.max_numeric_mean_pct_change),
                ("median", CFG.max_numeric_median_pct_change),
            ]:
                b = float(base_num[col].get(stat_name, np.nan))
                c = float(curr_num[col].get(stat_name, np.nan))
                pct = safe_pct_change(c, b)
                checks.append(
                    {
                        "check": f"{col}_{stat_name}_pct_change",
                        "passed": abs(pct) <= thresh,
                        "baseline": b,
                        "current": c,
                        "delta": float(c - b),
                        "pct_change": pct,
                        "threshold": thresh,
                    }
                )

    # Null rate increases (absolute)
    base_null = baseline.get("null_rates", {}) or {}
    curr_null = current.get("null_rates", {}) or {}

    for col in CFG.critical_null_cols:
        if col in base_null and col in curr_null:
            b = float(base_null[col])
            c = float(curr_null[col])
            inc = float(c - b)
            checks.append(
                {
                    "check": f"{col}_null_rate_increase",
                    "passed": inc <= CFG.max_null_rate_abs_increase,
                    "baseline": b,
                    "current": c,
                    "delta": inc,
                    "threshold": CFG.max_null_rate_abs_increase,
                }
            )

    success = all(ch["passed"] for ch in checks)

    report = {
        "success": success,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_path": CFG.baseline_path,
        "thresholds": {
            "max_row_count_pct_change": CFG.max_row_count_pct_change,
            "max_churn_rate_abs_change": CFG.max_churn_rate_abs_change,
            "max_numeric_mean_pct_change": CFG.max_numeric_mean_pct_change,
            "max_numeric_median_pct_change": CFG.max_numeric_median_pct_change,
            "max_null_rate_abs_increase": CFG.max_null_rate_abs_increase,
        },
        "checks": checks,
    }
    return report

def report_to_csv(report: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for ch in report.get("checks", []):
        rows.append(
            {
                "check": ch.get("check"),
                "passed": ch.get("passed"),
                "baseline": ch.get("baseline"),
                "current": ch.get("current"),
                "delta": ch.get("delta"),
                "pct_change": ch.get("pct_change"),
                "threshold": ch.get("threshold"),
            }
        )
    return pd.DataFrame(rows)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="store_true", help="Write baseline snapshot and exit")
    args = parser.parse_args()

    ensure_dirs()

    df = load_table()
    current = compute_snapshot(df)

    if args.baseline:
        with open(CFG.baseline_path, "w", encoding="utf-8") as f:
            json.dump(current, f, indent=2)
        print("Baseline snapshot saved")
        print(f"Baseline: {CFG.baseline_path}")
        return

    if not os.path.exists(CFG.baseline_path):
        raise FileNotFoundError(
            f"Baseline not found at {CFG.baseline_path}. Create it first:\n"
            f"  python src/monitor.py --baseline"
        )

    with open(CFG.baseline_path, "r", encoding="utf-8") as f:
        baseline = json.load(f)

    report = compare_snapshots(baseline, current)

    # Write report JSON
    with open(CFG.report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Write report CSV
    df_csv = report_to_csv(report)
    df_csv.to_csv(CFG.report_csv, index=False)

    # Console summary
    print("=== Monitor Summary ===")
    print(f"Success: {report['success']}")
    print(f"Report JSON: {CFG.report_json}")
    print(f"Report CSV:  {CFG.report_csv}")

    failed = [c for c in report["checks"] if not c["passed"]]
    if failed:
        print("\nFailed checks:")
        for c in failed:
            print(f"- {c['check']} | baseline={c.get('baseline')} current={c.get('current')} delta={c.get('delta')}")

        raise SystemExit(1)

    print("Monitoring checks passed (no significant drift detected).")

if __name__ == "__main__":
    main()
