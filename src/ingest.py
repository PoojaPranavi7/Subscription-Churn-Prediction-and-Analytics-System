"""Load raw CSVs into the SQLite database.

Role in the pipeline (step 2):
    Reads the raw CSVs from ``data/raw/`` and loads them into ``churn.db`` as the
    ``subscriptions``, ``products``, and ``transactions`` tables, using the schema
    defined in ``sql/create_tables.sql``. This gives every downstream step a single
    relational source to query.

Inputs:
    - data/raw/subscriptions.csv
    - data/raw/products.csv
    - data/raw/transactions.csv
    - sql/create_tables.sql

Outputs:
    - churn.db with the three raw tables populated.

To be implemented in a later phase.
"""
