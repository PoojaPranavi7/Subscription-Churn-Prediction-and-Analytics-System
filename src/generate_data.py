"""Generate the raw dataset: a synthetic dataset modeled on real subscription churn data.

Role in the pipeline (step 1):
    Produces the raw relational CSVs under ``data/raw/`` that everything else is
    built on. The data is fully synthetic, generated deterministically with a seeded
    NumPy generator: customer-level subscription records plus synthetic transactional
    and engagement history so each customer has a realistic time series of logins,
    purchases, feature-use, and support-ticket events.

Outputs:
    - data/raw/subscriptions.csv  (one row per customer subscription)
    - data/raw/products.csv       (product/feature catalog)
    - data/raw/transactions.csv   (time-series activity/purchase events)

Design intent (Sections 2-4 of the context document):
    Churned customers (~15-20% of the base) tend to show, before cancelling,
    *declining engagement* (fewer logins, shorter sessions), *falling purchase
    frequency*, and *more erratic activity gaps*. Contractual fields (plan/contract)
    carry only a weak signal, so behavioral features end up as the top churn drivers.

    To keep the problem realistic (target ~85% accuracy, NOT a perfectly separable
    ~100%), deliberate class overlap is built in via four behavior styles:
      * churned + "decline" : the classic pre-cancel decay (softened to a random
                              floor, so churners still show some late activity).
      * churned + "sudden"  : cancels with little/no prior decline -> looks like an
                              active customer (a source of false negatives).
      * active  + "dip"     : a temporary slump near the cutoff that mimics churn
                              signals even though the customer stays (false positives).
      * active  + "stable"  : steady engagement.
    The decline is also no longer driven to ~0, so recency/trend signals overlap
    between classes instead of separating them cleanly.

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

# Behavior-style mix that controls class overlap (and therefore the achievable
# accuracy). Tuned so a Random Forest lands near the ~85% target with strong-but-
# imperfect recall on churn.
#   - Of churned customers, this fraction show a pre-cancel decline; the rest churn
#     "suddenly" and look like active customers (false negatives).
CHURN_DECLINE_FRAC = 0.85
#   - Of active customers, this fraction show a misleading temporary dip near the
#     cutoff that resembles a churn signal (false positives).
ACTIVE_DIP_FRAC = 0.12

# Purchase-frequency decline for "decline" churners. Their purchases follow a
# dedicated curve (independent of the engagement health curve): a normal rate until
# a ~60-day window before cancel, then a softened ramp down to a random floor. This
# makes falling purchase frequency a genuine, strong-but-imperfect churn signal
# (purchase_frequency_trend, purchases_last_30d vs prev_30d) without altering the
# sudden/dip/stable styles. The floor stays well above zero (realistic, not
# perfectly separating) and is kept shallower than the engagement floor so the
# headline #1 driver remains an engagement signal.
PURCHASE_DECLINE_WINDOW = (55, 71)   # ~60-day pre-cancel drop (randint range)
PURCHASE_DECLINE_FLOOR = (0.05, 0.18)


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
            style = "decline" if rng.random() < CHURN_DECLINE_FRAC else "sudden"
        else:
            cancel = None
            cancel_date = ""  # nullable -> empty in CSV
            is_active = 1
            style = "dip" if rng.random() < ACTIVE_DIP_FRAC else "stable"

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
                "behavior_style": style,
                "signup_obj": signup,
                "end_obj": cancel if churned else OBSERVATION_END,
            }
        )

    return pd.DataFrame(rows)


def _health_curve(span: int, style: str, rng: np.random.Generator) -> np.ndarray:
    """Per-day engagement 'health' multiplier over a customer's active span.

    Styles (see module docstring):
      * "decline" : decay to a random floor over a window before cancel (churn signal).
      * "dip"     : milder, shallower slump near the cutoff (active but looks risky).
      * "sudden"  : no decay -> churner that looks like an active customer.
      * "stable"  : steady engagement.
    Floors are kept well above 0 so churners still show some late activity, which
    makes recency/trend features overlap between classes instead of separating them.
    """
    n = span + 1
    health = np.ones(n)

    if style == "decline":
        window = min(int(rng.integers(30, 91)), max(15, n - 5))
        floor = float(rng.uniform(0.06, 0.25))
        health[n - window:] = np.linspace(1.0, floor, window)
    elif style == "dip":
        window = min(int(rng.integers(25, 71)), max(12, n - 5))
        floor = float(rng.uniform(0.45, 0.75))
        health[n - window:] = np.linspace(1.0, floor, window)
    # "sudden" and "stable" keep health ~1 (no engineered slump).

    # Mild day-to-day wobble for everyone.
    health *= rng.normal(1.0, 0.06, n).clip(0.55, 1.45)
    return health.clip(0.02, 1.6)


def _erratic_sigma(span: int, style: str, rng: np.random.Generator) -> np.ndarray:
    """Per-day noise sigma controlling gap irregularity.

    Higher sigma -> more lognormal spikes/dips -> more erratic activity gaps. Ramps
    up during a decline/dip window so those customers have a higher
    ``activity_gap_std`` downstream; "sudden"/"stable" stay at the calm baseline.
    """
    n = span + 1
    sigma = np.full(n, 0.25)
    if style == "decline":
        window = min(75, max(15, n - 5))
        sigma[n - window:] = np.linspace(0.35, float(rng.uniform(0.85, 1.10)), window)
    elif style == "dip":
        window = min(60, max(12, n - 5))
        sigma[n - window:] = np.linspace(0.30, float(rng.uniform(0.45, 0.65)), window)
    return sigma


def _purchase_intensity(span: int, style: str, health: np.ndarray,
                        rng: np.random.Generator) -> np.ndarray:
    """Per-day purchase-intensity multiplier.

    For "decline" churners, purchases follow a dedicated curve: a normal rate until a
    ~60-day window before cancel, then a softened linear ramp down to a random floor.
    This gives them a clear fall in purchase frequency versus their own earlier
    history (so purchase_frequency_trend and purchases_last_30d vs prev_30d separate
    churners from non-churners) while keeping their pre-decline history normal.

    All other styles ("sudden", "dip", "stable") keep using the engagement health
    curve exactly as before, preserving the existing class overlap and accuracy.
    """
    if style != "decline":
        return health
    n = span + 1
    curve = np.ones(n)
    window = min(int(rng.integers(*PURCHASE_DECLINE_WINDOW)), max(20, n - 5))
    floor = float(rng.uniform(*PURCHASE_DECLINE_FLOOR))

    # The fall is concentrated in the early part of the ~60-day window (roughly the
    # 60->30-day-before-cancel stretch) and then held at the floor for the final
    # ~30 days. This drives purchases_last_30d down near the floor (a clean recent
    # purchase-frequency signal) while purchases_prev_30d stays higher -- yielding a
    # strong, consistently negative purchase_frequency_trend without forcing it to
    # zero.
    flat_tail = min(30, window - 5)
    ramp_len = window - flat_tail
    curve[n - window:n - flat_tail] = np.linspace(1.0, floor, ramp_len)
    curve[n - flat_tail:] = floor
    return curve


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
        style = row.behavior_style
        signup: date = row.signup_obj
        end: date = row.end_obj
        span = (end - signup).days
        if span < 1:
            span = 1

        # Per-customer baseline propensities (heterogeneous across the base).
        base_login = float(rng.uniform(0.25, 0.7))       # P(login) per day at full health
        # Higher than the original 0.02-0.08 so per-30-day purchase counts are less
        # Poisson-noisy, which lets the purchase-frequency *trend* read as a clean
        # signal for decline churners (counts stay realistic for a subscription with
        # add-on purchases).
        base_purchase = float(rng.uniform(0.05, 0.14))   # P(purchase) per day at full health
        base_feature_lambda = float(rng.uniform(0.5, 2.0))
        session_mean = float(rng.uniform(12.0, 45.0))
        support_base = float(rng.uniform(0.002, 0.012))

        health = _health_curve(span, style, rng)
        sigma = _erratic_sigma(span, style, rng)
        day_noise = rng.lognormal(mean=0.0, sigma=sigma)

        days = np.arange(span + 1)

        # --- logins ---
        p_login = np.clip(base_login * health * day_noise, 0.0, 0.97)
        login_mask = rng.random(span + 1) < p_login
        login_days = days[login_mask]

        # --- purchases ---
        # Decline churners use a dedicated purchase-decline curve (clear ~60-day drop
        # in purchase frequency); all other styles keep tracking engagement health.
        purchase_curve = _purchase_intensity(span, style, health, rng)
        p_purchase = np.clip(base_purchase * purchase_curve * day_noise, 0.0, 0.6)
        purchase_mask = rng.random(span + 1) < p_purchase
        purchase_days = days[purchase_mask]

        # --- support tickets (rare; mildly elevated for declining/dipping users,
        # so they are a weak behavioral correlate that also overlaps across classes) ---
        support_mult = {"decline": 1.6, "dip": 1.3}.get(style, 1.0)
        support_rate = support_base * support_mult
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
    print("behavior styles   :", dict(subscriptions["behavior_style"].value_counts()))
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
    subs_out = subscriptions.drop(columns=["churn_flag", "behavior_style", "signup_obj", "end_obj"])

    products.to_csv(RAW_DATA_DIR / "products.csv", index=False)
    subs_out.to_csv(RAW_DATA_DIR / "subscriptions.csv", index=False)
    transactions.to_csv(RAW_DATA_DIR / "transactions.csv", index=False)

    _print_summary(subscriptions, products, transactions)


if __name__ == "__main__":
    main()
