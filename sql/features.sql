-- features.sql
-- SQL-side aggregation of transaction history into customer-level behavioral
-- features (engagement, purchase patterns, consistency, retention). Complements
-- src/features.py; all aggregations must use only data before the label cutoff to
-- avoid leakage. See Section 4 of the context document for the full feature list.

-- TODO (later phase): queries / views that aggregate transactions per customer into
-- engagement, purchase-pattern, consistency, and retention features, joined back to
-- the subscriptions/products attributes to form the modeling table.
