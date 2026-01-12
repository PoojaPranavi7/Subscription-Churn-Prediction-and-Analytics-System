from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd

@dataclass(frozen=True)
class Config:
    db_path: str = "churn.db"
    table_name: str = "customer_features"
    out_dir: str = os.path.join("outputs", "validation")

    # Null thresholds (fail if exceeded)
    max_null_rate_totalcharges: float = 0.01          # 1% allowed (Telco has small known issue)
    max_null_rate_monthlycharges: float = 0.0         # should not be null
    max_null_rate_tenure: float = 0.0                 # should not be null
    max_null_rate_contract: float = 0.0               # should not be null
    max_null_rate_paymentmethod: float = 0.0          # should not be null

    # Basic sanity on row count
    min_rows: int = 6500
    max_rows: int = 8000

CFG = Config()

EXPECTED_COLUMNS = [
    "customerID",
    "churn",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "charges_per_tenure",
    "Contract",
    "PaymentMethod",
    "InternetService",
    "PaperlessBilling",
    "SeniorCitizen",
]

def ensure_dirs() -> None:
    os.makedirs(CFG.out_dir, exist_ok=True)

def load_table() -> pd.DataFrame:
    conn = sqlite3.connect(CFG.db_path)
    df = pd.read_sql(f"SELECT * FROM {CFG.table_name}", conn)
    conn.close()
    return df

def null_rate(series: pd.Series) -> float:
    return float(series.isna().mean())

def add_check(results: list[dict[str, Any]], name: str, passed: bool, details: dict[str, Any] | None = None) -> None:
    results.append(
        {
            "check": name,
            "passed": bool(passed),
            "details": details or {},
        }
    )

def validate(df: pd.DataFrame) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    # Row count sanity
    n_rows = int(df.shape[0])
    add_check(
        checks,
        "row_count_between_min_max",
        CFG.min_rows <= n_rows <= CFG.max_rows,
        {"rows": n_rows, "min_rows": CFG.min_rows, "max_rows": CFG.max_rows},
    )

    # Required columns present (and no accidental drops)
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    add_check(
        checks,
        "required_columns_present",
        len(missing_cols) == 0,
        {"missing_columns": missing_cols, "expected_columns": EXPECTED_COLUMNS},
    )

    if missing_cols:
        return {"success": False, "checks": checks}

    # Schema / types (lightweight but meaningful)
    numeric_cols = ["churn", "tenure", "MonthlyCharges", "TotalCharges", "charges_per_tenure", "SeniorCitizen"]
    numeric_cast_fail = {}
    for col in numeric_cols:
        coerced = pd.to_numeric(df[col], errors="coerce")
        if col == "TotalCharges":
            fail_rate = float((coerced.isna() & df[col].notna()).mean())
        else:
            fail_rate = float((coerced.isna() & df[col].notna()).mean())
        numeric_cast_fail[col] = fail_rate

    add_check(
        checks,
        "numeric_columns_castable",
        all(v < 0.001 for k, v in numeric_cast_fail.items() if k != "TotalCharges") and numeric_cast_fail["TotalCharges"] < 0.01,
        {"cast_fail_rate": numeric_cast_fail},
    )

    # Constraints
    churn_set_ok = df["churn"].dropna().isin([0, 1]).all()
    add_check(checks, "constraint_churn_binary_0_1", churn_set_ok, {"unique_values": sorted(df["churn"].dropna().unique().tolist())})

    tenure_num = pd.to_numeric(df["tenure"], errors="coerce")
    add_check(checks, "constraint_tenure_gte_0", bool((tenure_num.dropna() >= 0).all()), {"min_tenure": float(tenure_num.min())})

    monthly_num = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
    add_check(checks, "constraint_monthlycharges_gt_0", bool((monthly_num.dropna() > 0).all()), {"min_monthlycharges": float(monthly_num.min())})

    total_num = pd.to_numeric(df["TotalCharges"], errors="coerce")
    add_check(
        checks,
        "constraint_totalcharges_gte_0_when_present",
        bool((total_num.dropna() >= 0).all()),
        {"min_totalcharges_nonnull": float(total_num.dropna().min()) if total_num.dropna().shape[0] else None},
    )

    # Null thresholds (fail fast on critical columns)
    nr_total = null_rate(df["TotalCharges"])
    add_check(
        checks,
        "null_rate_totalcharges_le_threshold",
        nr_total <= CFG.max_null_rate_totalcharges,
        {"null_rate": nr_total, "threshold": CFG.max_null_rate_totalcharges},
    )

    nr_monthly = null_rate(df["MonthlyCharges"])
    add_check(
        checks,
        "null_rate_monthlycharges_le_threshold",
        nr_monthly <= CFG.max_null_rate_monthlycharges,
        {"null_rate": nr_monthly, "threshold": CFG.max_null_rate_monthlycharges},
    )

    nr_tenure = null_rate(df["tenure"])
    add_check(
        checks,
        "null_rate_tenure_le_threshold",
        nr_tenure <= CFG.max_null_rate_tenure,
        {"null_rate": nr_tenure, "threshold": CFG.max_null_rate_tenure},
    )

    nr_contract = null_rate(df["Contract"])
    add_check(
        checks,
        "null_rate_contract_le_threshold",
        nr_contract <= CFG.max_null_rate_contract,
        {"null_rate": nr_contract, "threshold": CFG.max_null_rate_contract},
    )

    nr_payment = null_rate(df["PaymentMethod"])
    add_check(
        checks,
        "null_rate_paymentmethod_le_threshold",
        nr_payment <= CFG.max_null_rate_paymentmethod,
        {"null_rate": nr_payment, "threshold": CFG.max_null_rate_paymentmethod},
    )

    # Uniqueness of customerID (key quality signal)
    id_null = df["customerID"].isna().sum()
    id_dup = int(df["customerID"].duplicated().sum())
    add_check(checks, "customerID_not_null", id_null == 0, {"null_count": int(id_null)})
    add_check(checks, "customerID_unique", id_dup == 0, {"duplicate_count": id_dup})

    success = all(c["passed"] for c in checks)

    return {
        "success": success,
        "checks": checks,
    }

def main() -> None:
    ensure_dirs()

    df = load_table()
    report = validate(df)

    ts = datetime.now(timezone.utc).isoformat()

    full_report = {
        "run_utc": ts,
        "table": CFG.table_name,
        "rows": int(df.shape[0]),
        "columns": list(df.columns),
        **report,
    }

    out_path = os.path.join(CFG.out_dir, "validation_report.json")
    with open(out_path, "w") as f:
        json.dump(full_report, f, indent=2)

    # Print a crisp console summary
    passed = sum(1 for c in report["checks"] if c["passed"])
    total = len(report["checks"])
    print("=== Validation Summary ===")
    print(f"Table: {CFG.table_name}")
    print(f"Success: {report['success']}")
    print(f"Passed: {passed}/{total}")
    print(f"Report: {out_path}")

    if not report["success"]:
        print("\nFailed checks:")
        for c in report["checks"]:
            if not c["passed"]:
                print(f"- {c['check']} | {c['details']}")
        raise SystemExit(1)

    print("Data validation passed. Safe to proceed to preprocessing/training.")


if __name__ == "__main__":
    main()
