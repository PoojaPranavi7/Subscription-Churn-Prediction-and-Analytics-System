"""Clean and join the three raw tables into a single base table.

Role in the pipeline (step 3):
    Cleans ``subscriptions``, ``products``, and ``transactions`` (type coercion,
    date parsing, dedup, sanity fixes) and joins them into one customer-level base
    table that the feature engineering step aggregates over. Defines the churn
    label from a documented labeling cutoff date so features can be computed from
    data strictly before the label window (no leakage).

Inputs:
    - churn.db (raw tables)

Outputs:
    - data/processed/ base/joined table (and/or a cleaned table back in churn.db).

To be implemented in a later phase.
"""
