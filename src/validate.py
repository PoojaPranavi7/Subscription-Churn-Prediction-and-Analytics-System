"""Data-quality validation checks (run before training).

Role in the pipeline (step 5, gate before training):
    Validates the modeling table (and the customer base) against data-quality rules
    and writes a pass/fail report. Critical failures (broken schema, duplicate keys,
    impossible values, mis-ordered dates) raise loudly so a bad table never reaches
    training; softer issues are recorded as warnings.

Checks (Section 9):
    - Schema / column presence.
    - Row-count and duplicate-key checks.
    - Null-rate thresholds per column.
    - Range / sanity checks (no negative fees/counts, churn in {0,1},
      bounded ratios, dates ordered).

Severity:
    - "critical" -> recorded AND raises RuntimeError at the end of the run.
    - "warning"  -> recorded only (does not stop the pipeline).

Output:
    - outputs/metrics/validation_report.csv  (one row per check)

Run standalone (requires `python -m src.features`):
    python -m src.validate
"""

from __future__ import annotations

import pandas as pd

try:
    from src.config import METRICS_DIR, PROCESSED_DATA_DIR
    from src import preprocess as P
except ImportError:  # allow running as a plain script
    from config import METRICS_DIR, PROCESSED_DATA_DIR
    import preprocess as P


REPORT_ARTIFACT = "validation_report.csv"

MIN_ROWS = 100
NULL_RATE_WARN = 0.20       # > this share of nulls in a column -> warning
NULL_RATE_CRITICAL = 0.50   # > this -> critical
CHURN_RATE_PLAUSIBLE = (0.05, 0.40)  # expected band (project targets ~15-20%)

# Columns that must never be negative (trends are allowed to be negative).
NON_NEGATIVE_COLUMNS = [
    "monthly_fee", "total_logins", "logins_last_30d", "logins_prev_30d",
    "avg_session_minutes", "active_days", "days_since_last_activity",
    "total_purchases", "purchases_last_30d", "purchases_prev_30d",
    "purchase_frequency", "avg_order_value", "total_spend",
    "distinct_product_categories", "activity_gap_std",
    "coefficient_of_variation_sessions", "active_weeks_ratio",
    "tenure_months", "recency_days", "support_tickets_count",
    "late_or_missed_payments",
]
BOUNDED_0_1_COLUMNS = ["active_weeks_ratio"]


class _Checks:
    """Accumulates check results into rows for the report."""

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(self, name: str, level: str, passed: bool, message: str,
            target: str = "", value=None, threshold=None) -> None:
        self.rows.append({
            "check": name, "target": target, "level": level,
            "passed": bool(passed), "value": value, "threshold": threshold,
            "message": message,
        })

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=[
            "check", "target", "level", "passed", "value", "threshold", "message",
        ])


def _validate_modeling_table(df: pd.DataFrame, checks: _Checks) -> None:
    required = ["customer_id", "churn", *P.all_feature_columns()]
    missing = [c for c in required if c not in df.columns]
    checks.add("schema.column_presence", "critical", not missing,
               "all required columns present" if not missing else f"missing columns: {missing}",
               target="modeling_table", value=len(df.columns), threshold=len(required))

    checks.add("row_count.min", "critical", len(df) >= MIN_ROWS,
               f"{len(df):,} rows", target="modeling_table",
               value=len(df), threshold=MIN_ROWS)

    if "customer_id" in df.columns:
        dups = int(df["customer_id"].duplicated().sum())
        checks.add("duplicate_key.customer_id", "critical", dups == 0,
                   f"{dups} duplicate customer_id(s)", target="modeling_table", value=dups, threshold=0)

    # Null-rate thresholds (aggregate; offenders listed in the message).
    null_rates = df.isna().mean()
    warn = null_rates[(null_rates > NULL_RATE_WARN) & (null_rates <= NULL_RATE_CRITICAL)]
    crit = null_rates[null_rates > NULL_RATE_CRITICAL]
    checks.add("null_rate.critical", "critical", crit.empty,
               "no column exceeds critical null rate" if crit.empty
               else f"columns over {NULL_RATE_CRITICAL:.0%} null: {dict(crit.round(3))}",
               target="modeling_table", value=float(null_rates.max()), threshold=NULL_RATE_CRITICAL)
    checks.add("null_rate.warn", "warning", warn.empty,
               "no column exceeds warning null rate" if warn.empty
               else f"columns over {NULL_RATE_WARN:.0%} null: {dict(warn.round(3))}",
               target="modeling_table", value=float(null_rates.max()), threshold=NULL_RATE_WARN)

    # churn in {0,1}
    if "churn" in df.columns:
        bad_churn = int((~df["churn"].isin([0, 1])).sum())
        checks.add("range.churn_binary", "critical", bad_churn == 0,
                   f"{bad_churn} churn values outside {{0,1}}", target="churn",
                   value=bad_churn, threshold=0)
        rate = float(df["churn"].mean())
        in_band = CHURN_RATE_PLAUSIBLE[0] <= rate <= CHURN_RATE_PLAUSIBLE[1]
        checks.add("sanity.churn_rate", "warning", in_band,
                   f"churn rate {rate:.1%} (expected {CHURN_RATE_PLAUSIBLE[0]:.0%}-{CHURN_RATE_PLAUSIBLE[1]:.0%})",
                   target="churn", value=round(rate, 4), threshold=str(CHURN_RATE_PLAUSIBLE))

    # Non-negativity (no negative fees/counts/values).
    present_nn = [c for c in NON_NEGATIVE_COLUMNS if c in df.columns]
    neg = {c: int((df[c] < 0).sum()) for c in present_nn}
    offenders = {c: n for c, n in neg.items() if n > 0}
    checks.add("range.non_negative", "critical", not offenders,
               "no negative values in non-negative columns" if not offenders
               else f"negative values found: {offenders}",
               target="modeling_table", value=sum(neg.values()), threshold=0)

    # Bounded [0,1] ratios.
    for col in (c for c in BOUNDED_0_1_COLUMNS if c in df.columns):
        out_of_range = int(((df[col] < 0) | (df[col] > 1)).sum())
        checks.add(f"range.bounded_0_1.{col}", "critical", out_of_range == 0,
                   f"{out_of_range} values outside [0,1]", target=col,
                   value=out_of_range, threshold=0)


def _validate_customer_base(base: pd.DataFrame, checks: _Checks) -> None:
    for col in ["signup_date", "cancel_date", "cutoff_date", "observation_window_end"]:
        if col in base.columns:
            base[col] = pd.to_datetime(base[col], errors="coerce")

    if "customer_id" in base.columns:
        dups = int(base["customer_id"].duplicated().sum())
        checks.add("duplicate_key.customer_base", "critical", dups == 0,
                   f"{dups} duplicate customer_id(s)", target="customer_base", value=dups, threshold=0)

    if "is_active" in base.columns:
        bad = int((~base["is_active"].isin([0, 1])).sum())
        checks.add("range.is_active_binary", "critical", bad == 0,
                   f"{bad} is_active values outside {{0,1}}", target="customer_base",
                   value=bad, threshold=0)

    if {"signup_date", "cutoff_date"}.issubset(base.columns):
        bad = int((base["cutoff_date"] < base["signup_date"]).sum())
        checks.add("dates.signup_le_cutoff", "critical", bad == 0,
                   f"{bad} rows with cutoff_date before signup_date", target="customer_base",
                   value=bad, threshold=0)

    if {"signup_date", "cancel_date"}.issubset(base.columns):
        has_cancel = base["cancel_date"].notna()
        bad = int((base.loc[has_cancel, "cancel_date"] < base.loc[has_cancel, "signup_date"]).sum())
        checks.add("dates.signup_le_cancel", "critical", bad == 0,
                   f"{bad} rows with cancel_date before signup_date", target="customer_base",
                   value=bad, threshold=0)

    if {"cancel_date", "churn"}.issubset(base.columns):
        mismatch = int((base["cancel_date"].notna() != base["churn"].eq(1)).sum())
        checks.add("consistency.cancel_matches_churn", "warning", mismatch == 0,
                   f"{mismatch} rows where cancel_date presence != churn==1",
                   target="customer_base", value=mismatch, threshold=0)


def run_validation(raise_on_critical: bool = True) -> tuple[pd.DataFrame, bool]:
    """Run all checks, write the report, and (optionally) raise on critical failures."""
    checks = _Checks()

    modeling_path = PROCESSED_DATA_DIR / "modeling_table.csv"
    if not modeling_path.exists():
        raise FileNotFoundError(
            f"Missing modeling table: {modeling_path}. Run `python -m src.features` first."
        )
    _validate_modeling_table(pd.read_csv(modeling_path), checks)

    base_path = PROCESSED_DATA_DIR / "customer_base.csv"
    if base_path.exists():
        _validate_customer_base(pd.read_csv(base_path), checks)

    report = checks.to_frame()
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    report.to_csv(METRICS_DIR / REPORT_ARTIFACT, index=False)

    critical_failures = report[(report["level"] == "critical") & (~report["passed"])]
    warnings_failed = report[(report["level"] == "warning") & (~report["passed"])]
    all_critical_passed = critical_failures.empty

    # --- report ---
    print("=" * 64)
    print("Data-quality validation")
    print("=" * 64)
    for _, r in report.iterrows():
        status = "PASS" if r["passed"] else ("FAIL" if r["level"] == "critical" else "WARN")
        print(f"  [{status:>4}] ({r['level']:<8}) {r['check']:<32} {r['message']}")
    print("-" * 64)
    print(f"checks: {len(report)} | critical failed: {len(critical_failures)} | "
          f"warnings: {len(warnings_failed)}")
    print(f"saved: {METRICS_DIR / REPORT_ARTIFACT}")
    print("=" * 64)

    if raise_on_critical and not all_critical_passed:
        names = ", ".join(critical_failures["check"])
        raise RuntimeError(f"Validation failed on critical checks: {names}")

    return report, all_critical_passed


def main() -> None:
    run_validation(raise_on_critical=True)


if __name__ == "__main__":
    main()
