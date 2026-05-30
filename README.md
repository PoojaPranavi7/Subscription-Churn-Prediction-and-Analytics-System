# Subscription Churn Prediction & Analytics System

An end-to-end subscription churn prediction and analytics system in **Python + SQL**.
It turns raw subscription, product, and transaction data into **behavioral signals**
that predict which customers are at risk of churning — early enough for the business
to act.

> The full project specification lives in
> [`subscription_context_document.md`](subscription_context_document.md), which is the
> single source of truth for the data model, features, and modeling choices.

## Project goal

- Clean and **join** relational `subscriptions`, `products`, and `transactions` data
  into a single customer-level modeling table.
- Engineer **behavioral features** centered on engagement, purchase patterns,
  usage consistency, and retention/recency signals.
- Train and compare **three models** (Logistic Regression, Random Forest,
  Gradient Boosting) and **select Random Forest** for the best balance of
  performance and interpretability.
- Handle **class imbalance** (~15–20% churn) and report metrics beyond accuracy
  (precision, recall, F1, ROC-AUC), prioritizing **recall on the churn class**.
- Surface the **top churn drivers** via feature importance, with behavioral signals
  expected to dominate.
- Add **validation checks** and **lightweight monitoring** to mimic a production ML
  workflow, and produce an analytics-ready **scored customer file** with risk buckets.

Target: a final model reaching **~85% accuracy** with strong recall on churn.

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

## Intended run order

Each script is runnable standalone and as a step in this documented order:

| Step | Script | Role |
| ---- | ------ | ---- |
| 1 | `src/generate_data.py` | Kaggle seed + synthetic supplementation → `data/raw/*.csv` |
| 2 | `src/ingest.py` | Load raw CSVs into `churn.db` (schema from `sql/create_tables.sql`) |
| 3 | `src/join_clean.py` | Clean + join the three tables; define churn label from a cutoff |
| 4 | `src/features.py` | Behavioral feature engineering → customer-level modeling table |
| 5 | `src/validate.py` | Data-quality validation checks (gate before training) |
| 6 | `src/train.py` | Train + compare 3 models, select Random Forest |
| 7 | `src/evaluate.py` | Metrics, confusion matrix, threshold sweep + tuning |
| 8 | `src/feature_importance.py` | Ranked importances + chart + plain-language insights |
| 9 | `src/score.py` | Score customers + assign risk buckets → `customer_scored.csv` |
| 10 | `src/monitor.py` | Baseline capture + drift checks (production-style monitoring) |

Example:

```bash
python -m src.generate_data
python -m src.ingest
python -m src.join_clean
python -m src.features
python -m src.validate
python -m src.train
python -m src.evaluate
python -m src.feature_importance
python -m src.score
python -m src.monitor
```

## Repository structure

```
subscription-churn-system/
├── data/
│   ├── raw/                  # kaggle seed + generated synthetic CSVs
│   └── processed/            # joined/modeling table
├── sql/
│   ├── create_tables.sql
│   └── features.sql
├── src/
│   ├── config.py             # shared constants (RANDOM_STATE, paths)
│   ├── generate_data.py      # kaggle seed + synthetic supplementation
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

## Status

Phase 0 — repository scaffolding. Scripts are stubs with docstrings describing their
role; logic is implemented in later phases.
