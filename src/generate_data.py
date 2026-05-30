"""Generate the raw dataset: Kaggle-style seed + synthetic supplementation.

Role in the pipeline (step 1):
    Produces the raw relational CSVs under ``data/raw/`` that everything else is
    built on. Starts from a public-style Kaggle subscription seed (customer-level
    subscription records) and supplements it with synthetic transactional and
    engagement history so each customer has a realistic time series of logins,
    purchases, feature-use, and support-ticket events.

Outputs:
    - data/raw/subscriptions.csv  (one row per customer subscription)
    - data/raw/products.csv       (product/feature catalog)
    - data/raw/transactions.csv   (time-series activity/purchase events)

Design intent (Sections 2-4 of the context document):
    Churned customers (~15-20% of the base) are generated so that, in the period
    leading up to their cancellation, they show *declining engagement* (fewer
    logins, shorter sessions), *falling purchase frequency*, and *more erratic
    activity gaps*. Contractual fields (plan/contract) carry only a weak signal.
    This makes the behavioral signals end up as the top churn drivers once features
    are engineered downstream. Noise/overlap is intentionally retained so the data
    is not perfectly separable (target ~85% accuracy, not 100%).

Determinism:
    All randomness is driven by a single ``numpy`` generator seeded with
    ``RANDOM_STATE`` so the dataset is reproducible.

Run standalone:
    python -m src.generate_data
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

try:
    from src.config import RANDOM_STATE, RAW_DATA_DIR
except ImportError:  # allow running as a plain script (python src/generate_data.py)
    from config import RANDOM_STATE, RAW_DATA_DIR


# --- Generation parameters ---------------------------------------------------
N_CUSTOMERS = 2500

# Observation window: features will later be computed from data before a labeling
# cutoff; here we just anchor all generated history to a fixed end date so the
# dataset is stable and reproducible.
OBSERVATION_END = date(2025, 12, 31)
MIN_TENURE_DAYS = 60          # earliest signups are ~2 months before the window end
MAX_TENURE_DAYS = 365 * 3     # oldest signups are ~3 years before the window end

# Categorical vocabularies for the subscription seed.
PLAN_TYPES = ["Basic", "Standard", "Premium"]
PLAN_WEIGHTS = [0.45, 0.35, 0.20]
PLAN_MONTHLY_FEE = {"Basic": 9.99, "Standard": 19.99, "Premium": 39.99}

CONTRACT_TYPES = ["Monthly", "Annual"]
CONTRACT_WEIGHTS = [0.7, 0.3]

PAYMENT_METHODS = ["credit_card", "paypal", "bank_transfer", "apple_pay"]
PAYMENT_WEIGHTS = [0.5, 0.25, 0.15, 0.10]

REGIONS = ["North America", "Europe", "Asia", "South America", "Oceania"]
REGION_WEIGHTS = [0.4, 0.3, 0.18, 0.08, 0.04]

EVENT_TYPES = ["login", "feature_use", "purchase", "support_ticket"]

# Product catalog (feature modules the customer can use / buy add-ons for).
PRODUCT_CATALOG = [
    ("Dashboard Analytics", "Analytics", 12.00),
    ("Custom Reports", "Analytics", 18.00),
    ("Data Export", "Analytics", 8.00),
    ("Cloud Storage 100GB", "Storage", 6.00),
    ("Cloud Storage 1TB", "Storage", 15.00),
    ("File Versioning", "Storage", 5.00),
    ("Team Workspaces", "Collaboration", 10.00),
    ("Shared Calendars", "Collaboration", 4.00),
    ("Real-time Chat", "Collaboration", 7.00),
    ("Single Sign-On", "Security", 20.00),
    ("Audit Logs", "Security", 14.00),
    ("2FA Enforcement", "Security", 3.00),
    ("API Access", "Integrations", 25.00),
    ("Zapier Connector", "Integrations", 9.00),
    ("Webhooks", "Integrations", 6.00),
    ("Priority Support", "Support", 30.00),
    ("Onboarding Session", "Support", 50.00),
    ("Dedicated Manager", "Support", 99.00),
]

# Base churn rate plus weak adjustments from contractual/plan fields. Behavior is
# the dominant signal; these keep contractual fields only mildly informative.
BASE_CHURN_RATE = 0.15
CONTRACT_CHURN_ADJ = {"Monthly": 0.05, "Annual": -0.06}
PLAN_CHURN_ADJ = {"Basic": 0.03, "Standard": 0.0, "Premium": -0.03}


def _to_iso(d: date) -> str:
    return d.isoformat()


def build_products(rng: np.random.Generator) -> pd.DataFrame:
    """Build the static product catalog table."""
    rows = []
    for i, (name, category, price) in enumerate(PRODUCT_CATALOG, start=1):
        rows.append(
            {
                "product_id": i,
                "product_name": name,
                "product_category": category,
                "unit_price": round(float(price), 2),
            }
        )
    return pd.DataFrame(rows)


def build_subscriptions(rng: np.random.Generator) -> pd.DataFrame:
    """Build the customer-level subscription seed and decide churn + cancel dates.

    Returns one row per customer with all Section 3 ``subscriptions`` columns. The
    returned frame also drives transaction generation (signup/cancel define each
    customer's active span and whether engagement should decay toward the end).
    """
    rows = []
    for cid in range(1, N_CUSTOMERS + 1):
        plan = rng.choice(PLAN_TYPES, p=PLAN_WEIGHTS)
        contract = rng.choice(CONTRACT_TYPES, p=CONTRACT_WEIGHTS)
        payment = rng.choice(PAYMENT_METHODS, p=PAYMENT_WEIGHTS)
        region = rng.choice(REGIONS, p=REGION_WEIGHTS)

        tenure_days = int(rng.integers(MIN_TENURE_DAYS, MAX_TENURE_DAYS + 1))
        signup = OBSERVATION_END - timedelta(days=tenure_days)

        churn_p = (
            BASE_CHURN_RATE
            + CONTRACT_CHURN_ADJ[str(contract)]
            + PLAN_CHURN_ADJ[str(plan)]
        )
        churn_p = float(np.clip(churn_p, 0.05, 0.45))
        churned = bool(rng.random() < churn_p)

        if churned:
            # Cancel somewhere between ~25% and 100% of the tenure span, but never
            # in the first 45 days (gives room for a pre-cancel decline window).
            min_offset = max(45, int(0.25 * tenure_days))
            if min_offset >= tenure_days:
                min_offset = max(30, tenure_days - 15)
            cancel_offset = int(rng.integers(min_offset, tenure_days + 1))
            cancel = signup + timedelta(days=cancel_offset)
            cancel_date = _to_iso(cancel)
            is_active = 0
        else:
            cancel_date = ""  # nullable -> empty in CSV
            is_active = 1

        rows.append(
            {
                "customer_id": cid,
                "signup_date": _to_iso(signup),
                "plan_type": plan,
                "contract_type": contract,
                "monthly_fee": PLAN_MONTHLY_FEE[str(plan)],
                "payment_method": payment,
                "region": region,
                "is_active": is_active,
                "cancel_date": cancel_date,
                # internal helpers (dropped before writing the CSV); no leading
                # underscore so they survive DataFrame.itertuples().
                "churn_flag": int(churned),
                "signup_obj": signup,
                "end_obj": cancel if churned else OBSERVATION_END,
            }
        )

    return pd.DataFrame(rows)


def _health_curve(span: int, churned: bool, rng: np.random.Generator) -> np.ndarray:
    """Per-day engagement 'health' multiplier over a customer's active span.

    Active customers stay near 1.0 (stable). Churned customers decay toward ~0 over
    a decline window just before cancellation, which drives the falling logins,
    shorter sessions, and lower purchase frequency that the project relies on.
    """
    n = span + 1
    health = np.ones(n)

    if churned:
        decline_window = int(rng.integers(45, 91))
        decline_window = min(decline_window, max(15, n - 5))
        ramp = np.linspace(1.0, 0.06, decline_window)
        health[n - decline_window:] = ramp

    # Mild day-to-day wobble for everyone (kept small so active customers remain
    # consistent and churned customers' decline still dominates).
    health *= rng.normal(1.0, 0.05, n).clip(0.6, 1.4)
    return health.clip(0.02, 1.5)


def _erratic_sigma(span: int, churned: bool) -> np.ndarray:
    """Per-day noise sigma controlling gap irregularity.

    Higher sigma -> more lognormal spikes/dips -> more erratic activity gaps. Ramps
    up during the churn decline window so churned customers have a higher
    ``activity_gap_std`` downstream.
    """
    n = span + 1
    sigma = np.full(n, 0.25)
    if churned:
        decline_window = min(75, max(15, n - 5))
        sigma[n - decline_window:] = np.linspace(0.35, 1.0, decline_window)
    return sigma


def build_transactions(subscriptions: pd.DataFrame, products: pd.DataFrame,
                       rng: np.random.Generator) -> pd.DataFrame:
    """Generate the synthetic event time series for every customer.

    Emits ``login``, ``feature_use``, ``purchase``, and ``support_ticket`` events
    across each customer's active span, conditioned on the engagement health curve
    so churned customers show declining/erratic behavior before they cancel.
    """
    product_ids = products["product_id"].to_numpy()
    product_price = dict(zip(products["product_id"], products["unit_price"]))

    records: list[dict] = []
    tx_id = 0

    for row in subscriptions.itertuples(index=False):
        churned = bool(row.churn_flag)
        signup: date = row.signup_obj
        end: date = row.end_obj
        span = (end - signup).days
        if span < 1:
            span = 1

        # Per-customer baseline propensities (heterogeneous across the base).
        base_login = float(rng.uniform(0.25, 0.7))       # P(login) per day at full health
        base_purchase = float(rng.uniform(0.02, 0.08))   # P(purchase) per day at full health
        base_feature_lambda = float(rng.uniform(0.5, 2.0))
        session_mean = float(rng.uniform(12.0, 45.0))
        support_base = float(rng.uniform(0.002, 0.012))

        health = _health_curve(span, churned, rng)
        sigma = _erratic_sigma(span, churned)
        day_noise = rng.lognormal(mean=0.0, sigma=sigma)

        days = np.arange(span + 1)

        # --- logins ---
        p_login = np.clip(base_login * health * day_noise, 0.0, 0.97)
        login_mask = rng.random(span + 1) < p_login
        login_days = days[login_mask]

        # --- purchases (correlated with engagement, independent draw) ---
        p_purchase = np.clip(base_purchase * health * day_noise, 0.0, 0.6)
        purchase_mask = rng.random(span + 1) < p_purchase
        purchase_days = days[purchase_mask]

        # --- support tickets (slightly elevated for churned; rare) ---
        support_rate = support_base * (1.6 if churned else 1.0)
        support_mask = rng.random(span + 1) < support_rate
        support_days = days[support_mask]

        # Emit login + feature_use events.
        for d in login_days:
            h = float(health[d])
            event_date = signup + timedelta(days=int(d))
            session = float(rng.normal(session_mean * (0.4 + 0.6 * h),
                                       session_mean * 0.25 * (1.0 + (1.0 - h))))
            session = max(1.0, session)
            tx_id += 1
            records.append({
                "transaction_id": tx_id,
                "customer_id": row.customer_id,
                "product_id": "",
                "event_date": _to_iso(event_date),
                "event_type": "login",
                "amount": "",
                "session_minutes": round(session, 1),
            })

            n_features = rng.poisson(base_feature_lambda * h)
            for _ in range(int(n_features)):
                pid = int(rng.choice(product_ids))
                tx_id += 1
                records.append({
                    "transaction_id": tx_id,
                    "customer_id": row.customer_id,
                    "product_id": pid,
                    "event_date": _to_iso(event_date),
                    "event_type": "feature_use",
                    "amount": "",
                    "session_minutes": "",
                })

        # Emit purchase events.
        for d in purchase_days:
            event_date = signup + timedelta(days=int(d))
            pid = int(rng.choice(product_ids))
            qty = int(rng.integers(1, 4))
            amount = round(product_price[pid] * qty * float(rng.uniform(0.9, 1.1)), 2)
            tx_id += 1
            records.append({
                "transaction_id": tx_id,
                "customer_id": row.customer_id,
                "product_id": pid,
                "event_date": _to_iso(event_date),
                "event_type": "purchase",
                "amount": amount,
                "session_minutes": "",
            })

        # Emit support_ticket events.
        for d in support_days:
            event_date = signup + timedelta(days=int(d))
            tx_id += 1
            records.append({
                "transaction_id": tx_id,
                "customer_id": row.customer_id,
                "product_id": "",
                "event_date": _to_iso(event_date),
                "event_type": "support_ticket",
                "amount": "",
                "session_minutes": "",
            })

    transactions = pd.DataFrame.from_records(records)
    # Order chronologically then by id for a clean, realistic event log.
    transactions = transactions.sort_values(["event_date", "transaction_id"]).reset_index(drop=True)
    return transactions


def _print_summary(subscriptions: pd.DataFrame, products: pd.DataFrame,
                   transactions: pd.DataFrame) -> None:
    n_cust = len(subscriptions)
    n_churn = int(subscriptions["is_active"].eq(0).sum())
    churn_rate = n_churn / n_cust if n_cust else 0.0

    print("=" * 60)
    print("Raw data generation summary")
    print("=" * 60)
    print(f"subscriptions.csv : {n_cust:>8,} rows")
    print(f"products.csv      : {len(products):>8,} rows")
    print(f"transactions.csv  : {len(transactions):>8,} rows")
    print("-" * 60)
    print(f"churned customers : {n_churn:,} / {n_cust:,}  ({churn_rate:.1%})")
    print(f"avg events/customer: {len(transactions) / n_cust:,.1f}")
    print("-" * 60)
    print("event_type distribution:")
    for etype, cnt in transactions["event_type"].value_counts().items():
        print(f"  {etype:<16}: {cnt:>9,}  ({cnt / len(transactions):.1%})")
    print("-" * 60)
    print(f"event_date range  : {transactions['event_date'].min()} -> {transactions['event_date'].max()}")
    print("=" * 60)


def main() -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_STATE)

    products = build_products(rng)
    subscriptions = build_subscriptions(rng)
    transactions = build_transactions(subscriptions, products, rng)

    # Drop internal helper columns before persisting the seed table.
    subs_out = subscriptions.drop(columns=["churn_flag", "signup_obj", "end_obj"])

    products.to_csv(RAW_DATA_DIR / "products.csv", index=False)
    subs_out.to_csv(RAW_DATA_DIR / "subscriptions.csv", index=False)
    transactions.to_csv(RAW_DATA_DIR / "transactions.csv", index=False)

    _print_summary(subscriptions, products, transactions)


if __name__ == "__main__":
    main()
