# Subscription Churn Prediction & Analytics System

An end-to-end subscription churn prediction and analytics system in **Python + SQL**.
It turns raw subscription, product, and transaction data into **behavioral signals**
that predict which customers are at risk of churning — early enough for the business
to act.

> The full project specification lives in
> [`subscription_context_document.md`](subscription_context_document.md), the single
> source of truth for the data model, features, and modeling choices.

## The story

1. Raw data is **relational** — separate `subscriptions`, `products`, and
   `transactions` tables (a synthetic dataset modeled on real subscription churn
   data, generated deterministically with a seeded NumPy generator, including
   transactional/engagement history).
2. The tables are cleaned and **joined**, and a churn label is defined from a
   **labeling cutoff** so features use only pre-cutoff data (no leakage).
3. Customer-level **behavioral features** are engineered: engagement, purchase
   patterns, usage consistency, and retention/recency.
4. Three models are compared on the same split — **Logistic Regression, Random
   Forest, Gradient Boosting** — and **Random Forest is selected** for the best
   balance of performance and interpretability.
5. Churn is **imbalanced (~18%)**, so accuracy alone is not trusted; **recall on the
   churn class is prioritized** and the **decision threshold is tuned** to a business
   tradeoff.
6. Random Forest **feature importances** confirm the headline insight: **behavioral
   signals (engagement decline, falling purchase frequency, usage inconsistency,
   recency) are the strongest churn drivers**, not contractual fields.

## Data model

Three raw tables (`sql/create_tables.sql`, loaded into `churn.db`):

| Table | Grain | Key columns |
| --- | --- | --- |
| `subscriptions` | one row per customer | `customer_id` (PK), `signup_date`, `plan_type`, `contract_type`, `monthly_fee`, `payment_method`, `region`, `is_active`, `cancel_date` |
| `products` | product/feature catalog | `product_id` (PK), `product_name`, `product_category`, `unit_price` |
| `transactions` | one row per event (time series) | `transaction_id` (PK), `customer_id` (FK), `product_id` (FK), `event_date`, `event_type` (login/feature_use/purchase/support_ticket), `amount`, `session_minutes` |

**Target & cutoff.** `churn = 1` if the subscription was cancelled within the
observation window. Each customer gets a `cutoff_date` = `cancel_date` (churned) or
the observation-window end (active); features use only events **strictly before** the
cutoff, so the cancellation event can never leak into the features.

## Behavioral feature families (`src/features.py`)

All features are per customer, computed only from pre-cutoff data. Trends compare the
last 30 days vs the prior 30 days, so a negative trend means decline.

- **Engagement** — `total_logins`, `logins_last_30d`, `logins_prev_30d`,
  `engagement_trend`, `avg_session_minutes`, `session_minutes_trend`, `active_days`,
  `days_since_last_activity`
- **Purchase patterns** — `total_purchases`, `purchases_last_30d`,
  `purchases_prev_30d`, `purchase_frequency`, `purchase_frequency_trend`,
  `avg_order_value`, `total_spend`, `spend_trend`, `distinct_product_categories`
- **Consistency / volatility** — `activity_gap_std`,
  `coefficient_of_variation_sessions`, `active_weeks_ratio`
- **Retention / tenure** — `tenure_months`, `recency_days`, `support_tickets_count`,
  `late_or_missed_payments`
- **Contractual / demographic (supporting)** — `plan_type`, `contract_type`,
  `monthly_fee`, `payment_method`, `region`

## Preprocessing & imputation (`src/preprocess.py`)

A single reusable scikit-learn `ColumnTransformer`, **fit on the training split only**
and reused for evaluation/scoring (saved with joblib). Imputation strategy:

| Column group | Imputation | Encoding/scaling |
| --- | --- | --- |
| Behavioral numerics (missing = no activity) | constant `0` | `StandardScaler` |
| Usage/transaction numerics (missing-at-random) | `median` | `StandardScaler` |
| `plan_type`, `contract_type` | `most_frequent` | one-hot |
| `payment_method`, `region` (missingness may be informative) | explicit `"unknown"` | one-hot |

A reproducible, stratified 80/20 train/test split is created with `RANDOM_STATE = 42`.

## Model comparison & selection (`src/train.py`)

Stratified 5-fold cross-validation on the training split. **Imbalance handling:**
`class_weight="balanced"` for Logistic Regression and Random Forest;
`RandomOverSampler` (train-folds only, inside an `imblearn` Pipeline) for Gradient
Boosting since it has no `class_weight`.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.881 | 0.629 | 0.849 | 0.722 | 0.929 |
| **Random Forest (selected)** | **0.932** | **0.914** | 0.692 | 0.787 | 0.933 |
| Gradient Boosting | 0.926 | 0.787 | 0.813 | 0.798 | 0.931 |

Selection is computed in code: rank by mean CV ROC-AUC; when models tie within a
tolerance, prefer Random Forest for the best performance/interpretability balance (its
feature importances drive the churn-driver analysis). Artifacts: `final_model.joblib`,
`model_card.json`, `outputs/metrics/model_comparison.csv`.

## Evaluation & threshold tuning (`src/evaluate.py`)

Held-out test set, **ROC-AUC = 0.924**. The threshold is swept and the operating point
maximizes churn recall subject to a precision floor of 0.60 (the business tradeoff:
missing an at-risk customer costs more than over-flagging). **Chosen threshold = 0.19.**

| | Default (0.50) | **Chosen (0.19)** |
| --- | --- | --- |
| Accuracy | 0.920 | **0.872** |
| Churn recall | 0.615 | **0.835** |
| Churn precision | 0.918 | 0.608 |
| Churn F1 | 0.737 | 0.704 |

Confusion matrix at 0.19: `tn=360, fp=49, fn=15, tp=76`. Outputs: `evaluation.json`,
`threshold_sweep.csv`, `confusion_matrix.csv`, and the persisted `threshold.json`.

## Churn drivers (`src/feature_importance.py`)

RF impurity importances (aggregated from one-hot columns back to the 29 source
features) plus permutation importance. **Behavioral signals dominate**: behavioral
families hold **0.958** of total importance vs **0.042** for contractual fields, and no
contractual feature appears in the top 5. Top drivers: `session_minutes_trend`,
`purchases_last_30d` (#2), `coefficient_of_variation_sessions`, `logins_last_30d`,
`engagement_trend` — falling purchase frequency now ranks as a top behavioral driver
alongside engagement decline. Outputs: `feature_importance.csv`,
`feature_importance.png`, `churn_drivers_insights.md`. The script **flags loudly** if
behavioral features ever fail to rank at the top.

## Validation & monitoring (`src/validate.py`, `src/monitor.py`)

- **Validation** (gate before training): schema/column presence, row-count and
  duplicate-key checks, per-column null-rate thresholds, and range/sanity checks
  (no negative fees/counts, `churn ∈ {0,1}`, bounded ratios, dates ordered). Critical
  failures raise; warnings are recorded. Report: `validation_report.csv`.
- **Monitoring** (production simulation): captures a baseline distribution (feature
  means/stds, churn rate, prediction rate) on first run, then flags **drift** on later
  runs via **PSI** (>0.25 significant, 0.10–0.25 moderate) and a mean-shift z-score.
  Reports: `monitor_report.csv`, `monitor_summary.csv` (baseline in
  `monitor_baseline.json`; delete or run `--reset` to re-capture).

## Scoring & risk buckets (`src/score.py`)

Scores every customer with the saved model and tuned threshold, producing
`outputs/predictions/customer_scored.csv` with `customer_id`, `churn_probability`,
`predicted_churn`, `risk_bucket` (and `actual_churn` for auditing). Risk buckets:
**Low** (<0.30), **Medium** (0.30–0.60), **High** (0.60–0.80), **Very High** (0.80+).
The buckets are well-calibrated (mean actual churn rises from ~1% in Low to ~99% in
Very High).

## Tech stack

- Python 3.10+
- SQL via **SQLite** (`churn.db`), Postgres-compatible where reasonable
- pandas, numpy, scikit-learn, matplotlib, joblib, imbalanced-learn

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run order

Each script is runnable standalone and as a step in this documented order. Run from
the project root so the `src` package imports resolve.

| Step | Command | Role |
| ---- | ------- | ---- |
| 1 | `python -m src.generate_data` | Generate synthetic data → `data/raw/*.csv` |
| 2 | `python -m src.ingest` | Load raw CSVs into `churn.db` (DDL from `sql/create_tables.sql`) |
| 3 | `python -m src.join_clean` | Clean + join; churn label from the labeling cutoff → `data/processed/` |
| 4 | `python -m src.features` | Behavioral feature engineering → `modeling_table.csv` |
| 5 | `python -m src.validate` | Data-quality validation gate (raises on critical failures) |
| 6 | `python -m src.preprocess` | Fit reusable preprocessing on train; save split + transformer |
| 7 | `python -m src.train` | Train + compare 3 models, select Random Forest |
| 8 | `python -m src.evaluate` | Metrics, confusion matrix, threshold sweep + tuning |
| 9 | `python -m src.feature_importance` | Ranked importances + chart + insights |
| 10 | `python -m src.score` | Score customers + risk buckets → `customer_scored.csv` |
| 11 | `python -m src.monitor` | Baseline capture + drift checks |

> Note: `train` will auto-run `preprocess` if its artifacts are missing, but after you
> regenerate data you should re-run `preprocess` (step 6) before `train` so the split
> isn't stale.

## Repository structure

```
subscription-churn-system/
├── data/
│   ├── raw/                  # generated synthetic CSVs
│   └── processed/            # cleaned/joined layer + modeling_table.csv
├── sql/
│   ├── create_tables.sql
│   └── features.sql
├── src/
│   ├── config.py             # shared constants (RANDOM_STATE, paths)
│   ├── generate_data.py      # synthetic data generation
│   ├── ingest.py             # load CSVs -> churn.db
│   ├── join_clean.py         # clean + join; churn label + cutoff
│   ├── features.py           # behavioral feature engineering
│   ├── preprocess.py         # reusable sklearn Pipeline (imputation/encoding/scaling)
│   ├── train.py              # train + compare 3 models, select RF
│   ├── evaluate.py           # metrics, confusion matrix, threshold sweep
│   ├── feature_importance.py # ranked importances + chart + insights
│   ├── validate.py           # data-quality validation checks
│   ├── monitor.py            # baseline capture + drift checks
│   └── score.py              # scoring + risk buckets
├── outputs/
│   ├── models/               # final_model, preprocessor, split, threshold, model_card
│   ├── metrics/              # comparison, evaluation, sweep, importances, reports
│   └── predictions/          # customer_scored.csv
├── notebooks/
│   └── 01_eda.ipynb
├── churn.db
├── requirements.txt
└── README.md
``