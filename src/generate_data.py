"""Generate the raw dataset: Kaggle-style seed + synthetic supplementation.

Role in the pipeline (step 1):
    Produces the raw relational CSVs under ``data/raw/`` that everything else is
    built on. Starts from a public-style Kaggle subscription seed and supplements
    it with synthetic transactional and engagement history so the data is rich
    enough to support behavioral feature engineering.

Outputs:
    - data/raw/subscriptions.csv  (one row per customer subscription)
    - data/raw/products.csv       (product/feature catalog)
    - data/raw/transactions.csv   (time-series activity/purchase events)

Notes:
    The synthetic generation is tuned so behavioral signals (engagement decline,
    falling purchase frequency, usage inconsistency, recency) carry the churn
    signal rather than contractual fields. Uses ``RANDOM_STATE`` for determinism.

To be implemented in a later phase.
"""
