"""Clean and join the three raw tables into the processed base layer.

Role in the pipeline (step 3):
    Reads the raw relational tables from ``churn.db``, cleans each one (dtypes,
    date parsing, duplicate handling, sanity fixes), derives the churn label from a
    documented labeling cutoff, and joins subscriptions + products + transactions
    into the processed layer that feature engineering (Phase 4) consumes.

Outputs (written to data/processed/):
    - customer_base.csv          one row per customer: cleaned subscription
                                 attributes + churn label + cutoff metadata. This is
                                 the customer-grained base table the modeling table
                                 is built on.
    - feature_window_events.csv  cleaned transactions joined with the product
                                 catalog, restricted to each customer's FEATURE
                                 WINDOW (events strictly before the labeling cutoff).
                                 Phase 4 aggregates this into behavioral features.

==============================================================================
LABELING CUTOFF & NO-LEAKAGE DESIGN  (Section 3: define churn from a cutoff so
features come only from data BEFORE the label window)
==============================================================================
The observation window ends at OBSERVATION_WINDOW_END (the latest event date in
the dataset). Churn is defined over this window:

    churn = 1  if the subscription was cancelled within the observation window
               (is_active == 0 and cancel_date is present)
    churn = 0  otherwise (still active at OBSERVATION_WINDOW_END)

To guarantee no leakage we attach a per-customer ``cutoff_date`` that separates
the feature window (the past we are allowed to learn from) from the label/outcome:

    * Churned customer : cutoff_date = cancel_date.
        Features use ONLY events with event_date < cancel_date. The cancellation
        event itself (and any same-day account teardown activity) lives in the
        label window and is never fed to the model — otherwise the model would
        "see" the churn it is supposed to predict.

    * Active customer  : cutoff_date = OBSERVATION_WINDOW_END.
        Features use events with event_date <= OBSERVATION_WINDOW_END (the full
        observed history, since there is no cancellation to guard against).

The pre-cancel engagement decline that the synthetic data encodes happens BEFORE
cancel_date, so it remains available as a (leak-free) predictive signal. Recency
for every customer is measured relative to its own cutoff, so churned and active
customers are not trivially separable by "time since cutoff".

Run standalone:
    python -m src.join_clean
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

try:
    from src.config import DB_PATH, PROCESSED_DATA_DIR
except ImportError:  # allow running as a plain script (python src/join_clean.py)
    from config import DB_PATH, PROCESSED_DATA_DIR


VALID_EVENT_TYPES = {"login", "feature_use", "purchase", "support_ticket"}


# --- Loading -----------------------------------------------------------------
def _load_tables(conn: sqlite3.Connection) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    subs = pd.read_sql("SELECT * FROM subscriptions", conn)
    products = pd.read_sql("SELECT * FROM products", conn)
    tx = pd.read_sql("SELECT * FROM transactions", conn)
    return subs, products, tx


# --- Cleaning ----------------------------------------------------------------
def clean_subscriptions(subs: pd.DataFrame) -> pd.DataFrame:
    """Fix dtypes, parse dates, dedup PKs, and apply basic sanity fixes."""
    df = subs.copy()

    # Strip stray whitespace on categorical/text fields.
    for col in ["plan_type", "contract_type", "payment_method", "region"]:
        df[col] = df[col].astype("string").str.strip()

    # Dates -> datetime (cancel_date is nullable -> NaT).
    df["signup_date"] = pd.to_datetime(df["signup_date"], errors="coerce")
    df["cancel_date"] = pd.to_datetime(df["cancel_date"], errors="coerce")

    df["monthly_fee"] = pd.to_numeric(df["monthly_fee"], errors="coerce")
    df["is_active"] = pd.to_numeric(df["is_active"], errors="coerce").astype("Int64")

    # Duplicate primary keys: keep first, report the rest.
    n_dup = int(df.duplicated(subset=["customer_id"]).sum())
    if n_dup:
        df = df.drop_duplicates(subset=["customer_id"], keep="first")

    # Sanity fixes.
    neg_fee = int((df["monthly_fee"] < 0).sum())
    df.loc[df["monthly_fee"] < 0, "monthly_fee"] = np.nan          # no negative fees
    df.loc[~df["is_active"].isin([0, 1]), "is_active"] = pd.NA      # is_active in {0,1}

    # cancel_date must not precede signup_date; if it does, treat as invalid (NaT).
    bad_cancel = int((df["cancel_date"].notna() & (df["cancel_date"] < df["signup_date"])).sum())
    df.loc[df["cancel_date"].notna() & (df["cancel_date"] < df["signup_date"]), "cancel_date"] = pd.NaT

    df.attrs["clean_report"] = {
        "dup_customer_ids": n_dup,
        "negative_fees_fixed": neg_fee,
        "cancel_before_signup_fixed": bad_cancel,
    }
    return df


def clean_products(products: pd.DataFrame) -> pd.DataFrame:
    """Fix dtypes, dedup PKs, and sanity-check prices."""
    df = products.copy()
    for col in ["product_name", "product_category"]:
        df[col] = df[col].astype("string").str.strip()
    df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce")

    n_dup = int(df.duplicated(subset=["product_id"]).sum())
    if n_dup:
        df = df.drop_duplicates(subset=["product_id"], keep="first")

    neg_price = int((df["unit_price"] < 0).sum())
    df.loc[df["unit_price"] < 0, "unit_price"] = np.nan

    df.attrs["clean_report"] = {"dup_product_ids": n_dup, "negative_prices_fixed": neg_price}
    return df


def clean_transactions(tx: pd.DataFrame) -> pd.DataFrame:
    """Fix dtypes, parse dates, drop dup/invalid rows, sanity-fix numeric fields."""
    df = tx.copy()

    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    df["event_type"] = df["event_type"].astype("string").str.strip().str.lower()
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["session_minutes"] = pd.to_numeric(df["session_minutes"], errors="coerce")
    df["product_id"] = pd.to_numeric(df["product_id"], errors="coerce").astype("Int64")

    n_exact_dup = int(df.duplicated().sum())
    if n_exact_dup:
        df = df.drop_duplicates(keep="first")

    n_dup_pk = int(df.duplicated(subset=["transaction_id"]).sum())
    if n_dup_pk:
        df = df.drop_duplicates(subset=["transaction_id"], keep="first")

    # Drop rows that fail hard validity rules.
    bad_type = int((~df["event_type"].isin(VALID_EVENT_TYPES)).sum())
    bad_date = int(df["event_date"].isna().sum())
    df = df[df["event_type"].isin(VALID_EVENT_TYPES) & df["event_date"].notna()]

    # Negative monetary / session values are impossible -> null them out.
    neg_amount = int((df["amount"] < 0).sum())
    neg_sess = int((df["session_minutes"] < 0).sum())
    df.loc[df["amount"] < 0, "amount"] = np.nan
    df.loc[df["session_minutes"] < 0, "session_minutes"] = np.nan

    df.attrs["clean_report"] = {
        "exact_duplicate_rows": n_exact_dup,
        "dup_transaction_ids": n_dup_pk,
        "invalid_event_types_dropped": bad_type,
        "unparseable_dates_dropped": bad_date,
        "negative_amounts_fixed": neg_amount,
        "negative_sessions_fixed": neg_sess,
    }
    return df


# --- Labeling & cutoff -------------------------------------------------------
def build_customer_base(subs: pd.DataFrame, observation_window_end: pd.Timestamp) -> pd.DataFrame:
    """Derive the churn label and per-customer labeling cutoff (see module docstring)."""
    df = subs.copy()

    # churn = cancelled within the observation window.
    df["churn"] = (df["is_active"].eq(0) & df["cancel_date"].notna()).astype(int)

    # Per-customer cutoff: cancel_date for churned, observation end for active.
    df["cutoff_date"] = df["cancel_date"].where(df["churn"].eq(1), observation_window_end)
    df["observation_window_end"] = observation_window_end

    # Tenure measured up to the cutoff (no future information).
    df["tenure_days"] = (df["cutoff_date"] - df["signup_date"]).dt.days.clip(lower=0)

    return df


def build_feature_window_events(tx: pd.DataFrame, products: pd.DataFrame,
                                customer_base: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Join transactions with the product catalog and clip to the feature window.

    Returns the leakage-safe event log (event_date strictly before each customer's
    cutoff_date) plus the number of events dropped by that clip.
    """
    cutoffs = customer_base[["customer_id", "cutoff_date", "churn"]]
    df = tx.merge(cutoffs, on="customer_id", how="inner")

    # Feature window = events strictly before the cutoff. For churned customers this
    # excludes the cancel day; for active customers cutoff = observation end, so the
    # only events excluded would be exactly on/after that end (none in practice).
    before_n = len(df)
    df = df[df["event_date"] < df["cutoff_date"]]
    dropped = before_n - len(df)

    # Enrich events with product attributes (login/support have no product_id -> NaN).
    df = df.merge(
        products[["product_id", "product_category", "unit_price"]],
        on="product_id",
        how="left",
    )

    out_cols = [
        "customer_id", "event_date", "event_type",
        "amount", "session_minutes",
        "product_id", "product_category", "unit_price",
    ]
    df = df[out_cols].sort_values(["customer_id", "event_date"]).reset_index(drop=True)
    return df, dropped


# --- Orchestration -----------------------------------------------------------
def _print_report(title: str, report: dict) -> None:
    print(f"  {title}:")
    for k, v in report.items():
        print(f"    - {k}: {v:,}")


def main() -> None:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    try:
        subs_raw, products_raw, tx_raw = _load_tables(conn)
    finally:
        conn.close()

    print("=" * 60)
    print("Cleaning raw tables")
    print("=" * 60)
    subs = clean_subscriptions(subs_raw)
    products = clean_products(products_raw)
    tx = clean_transactions(tx_raw)
    _print_report("subscriptions", subs.attrs["clean_report"])
    _print_report("products", products.attrs["clean_report"])
    _print_report("transactions", tx.attrs["clean_report"])

    # Observation window end = latest event date in the cleaned data.
    observation_window_end = tx["event_date"].max()

    customer_base = build_customer_base(subs, observation_window_end)
    events, dropped = build_feature_window_events(tx, products, customer_base)

    # Persist. Dates are written as ISO strings for portability.
    base_out = customer_base.copy()
    for col in ["signup_date", "cancel_date", "cutoff_date", "observation_window_end"]:
        base_out[col] = base_out[col].dt.strftime("%Y-%m-%d")
    base_out.to_csv(PROCESSED_DATA_DIR / "customer_base.csv", index=False)

    events_out = events.copy()
    events_out["event_date"] = events_out["event_date"].dt.strftime("%Y-%m-%d")
    events_out.to_csv(PROCESSED_DATA_DIR / "feature_window_events.csv", index=False)

    # --- Summary ---
    n_cust = len(customer_base)
    n_churn = int(customer_base["churn"].sum())
    print("-" * 60)
    print(f"observation_window_end : {observation_window_end.date()}")
    print(f"labeling cutoff        : cancel_date (churned) / window end (active)")
    print("-" * 60)
    print(f"customer_base.csv          : {n_cust:>9,} rows (one per customer)")
    print(f"  churned                  : {n_churn:,} / {n_cust:,}  ({n_churn / n_cust:.1%})")
    print(f"feature_window_events.csv  : {len(events):>9,} rows")
    print(f"  events dropped by cutoff : {dropped:,}  (post-cutoff / cancel-day -> no leakage)")
    print(f"  feature-window date range: {events['event_date'].min().date()} -> {events['event_date'].max().date()}")
    print("=" * 60)
    print("Clean & join complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
