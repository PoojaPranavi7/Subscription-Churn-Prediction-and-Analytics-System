"""Behavioral feature engineering (core of the project).

Role in the pipeline (step 4):
    Aggregates the cleaned/joined processed layer into a single customer-level
    modeling table. Every feature is computed per customer using ONLY pre-cutoff
    data: ``feature_window_events.csv`` already excludes anything at/after each
    customer's labeling cutoff (see src/join_clean.py), and all rolling windows
    below are measured backwards from that same ``cutoff_date`` -- so there is no
    leakage by construction.

Window convention:
    For a customer with cutoff C, an event that occurred ``d`` days before C
    (d = (C - event_date).days, always >= 1 since events are strictly pre-cutoff):
        last_30d  ->  1 <= d <= 30
        prev_30d  -> 31 <= d <= 60
    Trends compare the most recent 30 days against the preceding 30 days, so a
    negative trend means declining activity heading into the cutoff.

Feature families and EXACT names (Section 4):
    Engagement      : total_logins, logins_last_30d, logins_prev_30d,
                      engagement_trend, avg_session_minutes, session_minutes_trend,
                      active_days, days_since_last_activity
    Purchase pattern: total_purchases, purchases_last_30d, purchases_prev_30d,
                      purchase_frequency, purchase_frequency_trend, avg_order_value,
                      total_spend, spend_trend, distinct_product_categories
    Consistency     : activity_gap_std, coefficient_of_variation_sessions,
                      active_weeks_ratio
    Retention/tenure: tenure_months, recency_days, support_tickets_count,
                      late_or_missed_payments
    Contractual     : plan_type, contract_type, monthly_fee, payment_method, region

Missing-value policy here (the reusable imputation pipeline in src/preprocess.py
owns the rest, per Section 5):
    - Count/sum/avg-over-empty features where "missing" genuinely means *no
      activity* are filled with 0 (e.g. a customer with no logins -> 0).
    - Stats that are mathematically undefined for sparse customers
      (activity_gap_std needs >= 2 active days; coefficient_of_variation_sessions
      needs >= 2 logins) are left as NaN and imputed downstream (median).

Output:
    - data/processed/modeling_table.csv  (one row per customer, incl. churn target)

Run standalone:
    python -m src.features
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from src.config import PROCESSED_DATA_DIR
except ImportError:  # allow running as a plain script (python src/features.py)
    from config import PROCESSED_DATA_DIR


DAYS_PER_MONTH = 30.44

# Features filled with 0 when a customer simply had no activity of that kind.
ZERO_FILL_FEATURES = [
    "total_logins", "logins_last_30d", "logins_prev_30d", "engagement_trend",
    "avg_session_minutes", "session_minutes_trend", "active_days",
    "total_purchases", "purchases_last_30d", "purchases_prev_30d",
    "purchase_frequency", "purchase_frequency_trend", "avg_order_value",
    "total_spend", "spend_trend", "distinct_product_categories",
    "support_tickets_count", "active_weeks_ratio",
]

# Final column order for the modeling table.
CONTRACTUAL_FEATURES = ["plan_type", "contract_type", "monthly_fee", "payment_method", "region"]
FEATURE_ORDER = [
    # engagement
    "total_logins", "logins_last_30d", "logins_prev_30d", "engagement_trend",
    "avg_session_minutes", "session_minutes_trend", "active_days", "days_since_last_activity",
    # purchase patterns
    "total_purchases", "purchases_last_30d", "purchases_prev_30d", "purchase_frequency",
    "purchase_frequency_trend", "avg_order_value", "total_spend", "spend_trend",
    "distinct_product_categories",
    # consistency / volatility
    "activity_gap_std", "coefficient_of_variation_sessions", "active_weeks_ratio",
    # retention / tenure
    "tenure_months", "recency_days", "support_tickets_count", "late_or_missed_payments",
    # contractual / demographic
    *CONTRACTUAL_FEATURES,
]


def _trend(recent: pd.Series, prior: pd.Series) -> pd.Series:
    """Normalized change: (recent - prior) / (prior + 1). Negative => decline."""
    recent = recent.fillna(0.0)
    prior = prior.fillna(0.0)
    return (recent - prior) / (prior + 1.0)


def compute_features(base: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Compute all Section 4 features at customer grain and attach the churn target."""
    cust_index = base["customer_id"]

    # Attach each event to its customer's cutoff and compute days-before-cutoff.
    ev = events.merge(base[["customer_id", "cutoff_date"]], on="customer_id", how="left")
    ev["days_before_cutoff"] = (ev["cutoff_date"] - ev["event_date"]).dt.days
    ev["in_last_30d"] = ev["days_before_cutoff"].between(1, 30)
    ev["in_prev_30d"] = ev["days_before_cutoff"].between(31, 60)

    logins = ev[ev["event_type"] == "login"]
    purchases = ev[ev["event_type"] == "purchase"]
    support = ev[ev["event_type"] == "support_ticket"]

    g = ev.groupby("customer_id")
    gl = logins.groupby("customer_id")
    gp = purchases.groupby("customer_id")

    feat = pd.DataFrame(index=pd.Index(cust_index, name="customer_id"))

    # --- Engagement ---------------------------------------------------------
    feat["total_logins"] = gl.size()
    feat["logins_last_30d"] = logins[logins["in_last_30d"]].groupby("customer_id").size()
    feat["logins_prev_30d"] = logins[logins["in_prev_30d"]].groupby("customer_id").size()
    feat["engagement_trend"] = _trend(feat["logins_last_30d"], feat["logins_prev_30d"])

    feat["avg_session_minutes"] = gl["session_minutes"].mean()
    sess_last = logins[logins["in_last_30d"]].groupby("customer_id")["session_minutes"].mean()
    sess_prev = logins[logins["in_prev_30d"]].groupby("customer_id")["session_minutes"].mean()
    feat["session_minutes_trend"] = _trend(sess_last, sess_prev)

    feat["active_days"] = g["event_date"].nunique()
    feat["days_since_last_activity"] = g["days_before_cutoff"].min()

    # --- Purchase patterns --------------------------------------------------
    feat["total_purchases"] = gp.size()
    feat["purchases_last_30d"] = purchases[purchases["in_last_30d"]].groupby("customer_id").size()
    feat["purchases_prev_30d"] = purchases[purchases["in_prev_30d"]].groupby("customer_id").size()

    tenure_months = (base.set_index("customer_id")["tenure_days"] / DAYS_PER_MONTH)
    tenure_months = tenure_months.reindex(feat.index)
    total_purchases = feat["total_purchases"].fillna(0)
    feat["purchase_frequency"] = np.where(tenure_months > 0, total_purchases / tenure_months, 0.0)

    feat["purchase_frequency_trend"] = _trend(feat["purchases_last_30d"], feat["purchases_prev_30d"])

    feat["avg_order_value"] = gp["amount"].mean()
    feat["total_spend"] = gp["amount"].sum()
    spend_last = purchases[purchases["in_last_30d"]].groupby("customer_id")["amount"].sum()
    spend_prev = purchases[purchases["in_prev_30d"]].groupby("customer_id")["amount"].sum()
    feat["spend_trend"] = _trend(spend_last, spend_prev)

    feat["distinct_product_categories"] = g["product_category"].nunique()

    # --- Consistency / volatility ------------------------------------------
    # Gaps (in days) between consecutive *active days* per customer.
    active_days = (
        ev[["customer_id", "event_date"]]
        .drop_duplicates()
        .sort_values(["customer_id", "event_date"])
    )
    active_days["gap"] = active_days.groupby("customer_id")["event_date"].diff().dt.days
    feat["activity_gap_std"] = active_days.groupby("customer_id")["gap"].std()  # NaN if < 2 days

    # Coefficient of variation of session lengths (std / mean); NaN if < 2 logins.
    sess_mean = gl["session_minutes"].mean()
    sess_std = gl["session_minutes"].std()
    feat["coefficient_of_variation_sessions"] = sess_std / sess_mean

    # Active weeks ratio: distinct active ISO weeks / total weeks observed.
    # Build an integer ISO-week id (year*100 + week) vectorized -- far faster than
    # per-element strftime over ~1.5M rows.
    iso = ev["event_date"].dt.isocalendar()
    ev["iso_week"] = iso["year"].astype("int64") * 100 + iso["week"].astype("int64")
    active_weeks = g["iso_week"].nunique()
    total_weeks = (base.set_index("customer_id")["tenure_days"] / 7.0).reindex(feat.index)
    total_weeks = total_weeks.clip(lower=1.0)
    feat["active_weeks_ratio"] = (active_weeks / total_weeks).clip(upper=1.0)

    # --- Retention / tenure -------------------------------------------------
    feat["tenure_months"] = tenure_months
    feat["recency_days"] = feat["days_since_last_activity"]
    feat["support_tickets_count"] = support.groupby("customer_id").size()
    # No billing/payment event stream exists in the source, so this Section 4
    # "(if derivable)" feature is not derivable; kept as a 0 placeholder so the
    # schema is complete for a future payments feed.
    feat["late_or_missed_payments"] = 0

    # --- Fill "no activity -> 0" features -----------------------------------
    for col in ZERO_FILL_FEATURES:
        feat[col] = feat[col].fillna(0)
    # Recency is always defined (every customer has >= 1 event); fall back to tenure.
    feat["days_since_last_activity"] = feat["days_since_last_activity"].fillna(
        base.set_index("customer_id")["tenure_days"].reindex(feat.index)
    )
    feat["recency_days"] = feat["recency_days"].fillna(feat["days_since_last_activity"])

    # --- Contractual / demographic + target ---------------------------------
    feat = feat.reset_index()
    out = base[["customer_id", "churn", *CONTRACTUAL_FEATURES]].merge(feat, on="customer_id", how="left")

    ordered_cols = ["customer_id", *FEATURE_ORDER, "churn"]
    return out[ordered_cols]


def _print_summary(modeling: pd.DataFrame) -> None:
    n = len(modeling)
    churn_rate = modeling["churn"].mean()
    print("=" * 60)
    print("Behavioral feature engineering summary")
    print("=" * 60)
    print(f"modeling_table.csv : {n:,} rows x {modeling.shape[1]} cols")
    print(f"churn rate         : {churn_rate:.1%}")
    print("-" * 60)
    print("null counts per feature (NaN -> imputed in preprocess):")
    nulls = modeling.isna().sum()
    for col, cnt in nulls[nulls > 0].items():
        print(f"  {col:<34}: {cnt:,}")
    if int(nulls.sum()) == 0:
        print("  (none)")
    print("-" * 60)
    print("mean of key behavioral signals by churn (sanity check):")
    cols = ["engagement_trend", "purchase_frequency_trend", "activity_gap_std", "recency_days"]
    print(modeling.groupby("churn")[cols].mean().round(3).to_string())
    print("=" * 60)


def main() -> None:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    base = pd.read_csv(
        PROCESSED_DATA_DIR / "customer_base.csv",
        parse_dates=["signup_date", "cancel_date", "cutoff_date", "observation_window_end"],
    )
    events = pd.read_csv(
        PROCESSED_DATA_DIR / "feature_window_events.csv",
        parse_dates=["event_date"],
    )

    modeling = compute_features(base, events)
    modeling.to_csv(PROCESSED_DATA_DIR / "modeling_table.csv", index=False)

    _print_summary(modeling)


if __name__ == "__main__":
    main()
