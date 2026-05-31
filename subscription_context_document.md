# Subscription Churn Prediction & Analytics System — Project Context

> This document is the single source of truth for the project. It is written to be
> loaded into Cursor as persistent context so every generated file stays consistent
> with the intended architecture, data model, feature set, and modeling choices.
> When in doubt, follow this document over any default assumption.

---

## 1. Project Goal

Build an **end-to-end subscription churn prediction and analytics system** in Python and SQL.
The system turns raw subscription, product, and transaction data into **behavioral signals**
that predict which customers are at risk of churning early enough for the business to act.

The project must demonstrate, concretely and verifiably in code:

- Cleaning and **joining subscription + product + transaction data** from a relational source.
- **Reusable preprocessing workflows** (not one-off scripts).
- **Engineered behavioral features** centered on engagement, purchase patterns, and retention signals.
- Evaluation of **multiple models** with performance comparison.
- **Feature importance analysis** to surface key churn drivers.
- Handling of **class imbalance** with metrics beyond accuracy (precision, recall, F1, ROC-AUC).
- **Validation checks and lightweight monitoring** to mimic a production ML workflow.
- A final model reaching **~85% accuracy** with a strong recall on the churn class.

---

## 2. Narrative / Intended Story (must hold true in the code)

These statements describe how the system is meant to work. The implementation must make
each of them literally true so the project can be walked through end to end.

1. The data is **a synthetic dataset modeled on real subscription churn data**,
   generated deterministically with a seeded NumPy generator, including realistic
   transactional and engagement history.
2. The raw data is **relational**: separate `subscriptions`, `products`, and `transactions`
   tables that are cleaned and **joined** into a single modeling table.
3. The strongest predictive signals are **behavioral**:
   - **Engagement decline** over time
   - **Changes in purchase frequency**
   - **Usage patterns becoming less consistent** over time
   - **Retention signals** (tenure, recency, gaps in activity)
4. Three models are compared — **Logistic Regression, Random Forest, Gradient Boosting** —
   and **Random Forest is selected** for the best balance of performance and interpretability.
5. Churn is **imbalanced** (~15–20% positive class). Accuracy alone is not trusted;
   **recall on the churn class is prioritized** because missing an at-risk customer is more
   costly than over-flagging. Imbalance is handled via **class weighting and/or resampling**,
   and the **decision threshold is tuned to a business tradeoff**.
6. Random Forest is used partly because it lets us **rank feature importance** and see which
   **combinations of behavioral signals** show up most often for customers who eventually churn.

---

## 3. Data Model

### Source tables (raw, stored unchanged)

**`subscriptions`** — one row per customer subscription
- `customer_id` (PK)
- `signup_date`
- `plan_type` (e.g. Basic / Standard / Premium)
- `contract_type` (Monthly / Annual)
- `monthly_fee`
- `payment_method`
- `region`
- `is_active`
- `cancel_date` (nullable)

**`products`** — catalog of products/features the customer can use
- `product_id` (PK)
- `product_name`
- `product_category`
- `unit_price`

**`transactions`** — one row per customer activity/purchase event (time series)
- `transaction_id` (PK)
- `customer_id` (FK)
- `product_id` (FK)
- `event_date`
- `event_type` (purchase / login / feature_use / support_ticket)
- `amount` (nullable for non-purchase events)
- `session_minutes` (nullable)

### Target

- `churn` = 1 if the subscription was cancelled within the observation window, else 0.
- Churn must be **defined from a labeling cutoff date** so features are computed from data
  **before** the label window (no leakage). Document the cutoff in code.

### Modeling table

A single customer-level table produced by joining the three sources and aggregating
transaction history into behavioral features (see Section 4).

---

## 4. Behavioral Feature Engineering (core of the project)

All features are computed **per customer, using only data before the label cutoff.**
Group features into these families and name them clearly:

### Engagement features
- `total_logins`, `logins_last_30d`, `logins_prev_30d`
- `engagement_trend` = (logins_last_30d − logins_prev_30d) / (logins_prev_30d + 1)  → captures **engagement decline**
- `avg_session_minutes`, `session_minutes_trend`
- `active_days`, `days_since_last_activity` (recency)

### Purchase pattern features
- `total_purchases`, `purchases_last_30d`, `purchases_prev_30d`
- `purchase_frequency` = total_purchases / tenure_months
- `purchase_frequency_trend` → captures **changes in purchase frequency**
- `avg_order_value`, `total_spend`, `spend_trend`
- `distinct_product_categories`

### Consistency / volatility features
- `activity_gap_std` = standard deviation of gaps (in days) between consecutive events
  → captures **usage becoming less consistent over time**
- `coefficient_of_variation_sessions`
- `active_weeks_ratio` = active weeks / total weeks observed

### Retention / tenure features
- `tenure_months`
- `recency_days`
- `support_tickets_count`
- `late_or_missed_payments` (if derivable)

### Contractual / demographic (supporting, not headline)
- `plan_type`, `contract_type`, `monthly_fee`, `payment_method`, `region`

> The headline insight of the project must be that **behavioral signals (engagement decline,
> purchase frequency change, usage inconsistency, recency)** rank at the top of feature
> importance — not contractual fields. Tune the synthetic data generation so this is true.

---

## 5. Imputation Strategy (must be implemented exactly as below)

Apply a consistent, documented rule per feature type:

- **Numerical features** (usage counts, transaction values): **median imputation** when
  missingness is random / missing-at-random.
- **Behavioral features**: impute **0** when a missing value genuinely means **no activity**
  (e.g., no logins in window → 0 logins, not median).
- **Categorical features**: **most frequent** value, or an explicit **`"unknown"`** category
  when missingness may be informative.

Imputation must live inside the reusable preprocessing pipeline, not be hardcoded ad hoc.

---

## 6. Modeling Approach

- Train and compare three models on the same train/test split:
  1. **Logistic Regression** (baseline, interpretable)
  2. **Random Forest** (selected final model)
  3. **Gradient Boosting** (e.g., sklearn `GradientBoostingClassifier`)
- **Random Forest is selected** for best balance of performance and interpretability.
  The selection must be made by comparing metrics in code, not hardcoded — but the data
  and config should make RF the reasonable winner.
- **Imbalance handling**: use `class_weight="balanced"` and/or resampling
  (e.g., `RandomOverSampler` / SMOTE if `imbalanced-learn` is added). Document which is used.
- **Threshold tuning**: after training, sweep thresholds and pick one that maximizes recall
  on the churn class subject to a reasonable precision floor (business tradeoff). Save the
  chosen threshold.
- Determinism: set and reuse a fixed `RANDOM_STATE`.

---

## 7. Evaluation

Report all of the following (never accuracy alone):
- Accuracy (target **~85%**)
- Precision, Recall, F1 — **per class**, with **recall on churn highlighted**
- ROC-AUC
- Confusion matrix
- Threshold sweep table (precision/recall/F1 across thresholds)

Persist metrics to disk as CSV/JSON so they are reproducible and auditable.

---

## 8. Feature Importance & Churn Drivers

- Extract Random Forest feature importances (and optionally permutation importance).
- Save a ranked table and a bar chart of top drivers.
- Write a short generated insights file stating the top behavioral drivers in plain language
  (e.g., "engagement decline and falling purchase frequency are the strongest churn signals").

---

## 9. Validation Checks & Lightweight Monitoring

**Validation (data quality, run before training):**
- Schema/column presence checks
- Null-rate thresholds per column
- Range/sanity checks (no negative fees, dates ordered, churn ∈ {0,1})
- Row-count and duplicate-key checks
- Fail loudly (raise) or write a validation report with pass/fail per check.

**Monitoring (simulate production):**
- Capture a **baseline** feature distribution (means/stds, churn rate, prediction rate).
- On each subsequent run, compare current distribution vs baseline and flag **drift**
  (simple z-score / PSI-style check is fine).
- Write a monitoring report (`monitor_report.csv`, `monitor_summary.csv`).

---

## 10. Scoring / Output

- Produce a customer-level scored file with: `customer_id`, `churn_probability`,
  `predicted_churn` (using the tuned threshold), and a **risk bucket**:
  - Low (< 0.30), Medium (0.30–0.60), High (0.60–0.80), Very High (0.80+)
- Output `outputs/predictions/customer_scored.csv` — analytics-ready for BI tools.

---

## 11. Tech Stack

- **Python 3.10+**
- **SQL** via **SQLite** (file: `churn.db`), Postgres-compatible SQL where reasonable
- pandas, numpy, scikit-learn
- (optional) imbalanced-learn for resampling
- matplotlib for the feature-importance chart
- joblib for artifact persistence

---

## 12. Repository Structure

```
subscription-churn-system/
├── data/
│   ├── raw/                  # generated synthetic CSVs
│   └── processed/            # joined/modeling table
├── sql/
│   ├── create_tables.sql
│   └── features.sql
├── src/
│   ├── generate_data.py      # synthetic data generation
│   ├── ingest.py             # load CSVs -> churn.db
│   ├── join_clean.py         # clean + join subscriptions/products/transactions
│   ├── features.py           # behavioral feature engineering
│   ├── preprocess.py         # reusable sklearn Pipeline (imputation/encoding/scaling)
│   ├── train.py              # train + compare 3 models, select RF
│   ├── evaluate.py           # metrics, confusion matrix, threshold sweep
│   ├── feature_importance.py # ranked importances + chart + insights
│   ├── validate.py           # data quality validation checks
│   ├── monitor.py            # baseline capture + drift checks
│   └── score.py              # scoring + risk buckets
├── outputs/
│   ├── models/
│   ├── metrics/
│   └── predictions/
├── notebooks/
│   └── 01_eda.ipynb
├── churn.db
├── requirements.txt
└── README.md
```

---

## 13. Coding Conventions

- Keep preprocessing **reusable**: one fitted `Pipeline` / `ColumnTransformer`, saved with joblib.
- No data leakage: fit preprocessing on train only; compute features strictly before label cutoff.
- Every script runnable standalone and as a step in the documented run order.
- Set `RANDOM_STATE = 42` everywhere.
- Save intermediate artifacts so each phase is independently re-runnable.
- Add concise docstrings and a short top-of-file comment explaining each script's role.

---

## 14. Definition of Done

- All scripts run in order without manual edits.
- Joined modeling table exists and is reproducible.
- Three models trained and compared; RF selected with metrics saved.
- Accuracy ≈ 85%, recall on churn reported and reasonable.
- Behavioral features dominate the feature-importance ranking.
- Validation + monitoring reports generated.
- `customer_scored.csv` produced with risk buckets.
- README documents the story, data model, features, and run order.
