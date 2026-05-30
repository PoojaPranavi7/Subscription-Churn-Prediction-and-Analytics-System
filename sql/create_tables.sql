-- create_tables.sql
-- Defines the raw relational schema loaded into churn.db by src/ingest.py.
-- Three source tables (stored unchanged): subscriptions, products, transactions.
-- See Section 3 of the context document for the authoritative column list.
--
-- Dialect notes:
--   * Written for SQLite but kept Postgres-compatible where reasonable.
--   * Dates are stored as ISO-8601 TEXT ('YYYY-MM-DD'). SQLite has no native DATE
--     type; Postgres will happily accept these strings if the column types are
--     swapped to DATE. Monetary values use REAL (NUMERIC in Postgres).
--   * Drop-if-exists order respects FK dependencies (transactions first).

DROP TABLE IF EXISTS transactions;
DROP TABLE IF EXISTS products;
DROP TABLE IF EXISTS subscriptions;

-- One row per customer subscription.
CREATE TABLE subscriptions (
    customer_id     INTEGER     PRIMARY KEY,
    signup_date     TEXT        NOT NULL,
    plan_type       TEXT        NOT NULL,
    contract_type   TEXT        NOT NULL,
    monthly_fee     REAL        NOT NULL,
    payment_method  TEXT,
    region          TEXT,
    is_active       INTEGER     NOT NULL,
    cancel_date     TEXT,                       -- nullable: NULL while still active
    CONSTRAINT chk_is_active CHECK (is_active IN (0, 1)),
    CONSTRAINT chk_monthly_fee CHECK (monthly_fee >= 0)
);

-- Catalog of products/features the customer can use or buy.
CREATE TABLE products (
    product_id        INTEGER   PRIMARY KEY,
    product_name      TEXT      NOT NULL,
    product_category  TEXT,
    unit_price        REAL,
    CONSTRAINT chk_unit_price CHECK (unit_price IS NULL OR unit_price >= 0)
);

-- One row per customer activity/purchase event (time series).
CREATE TABLE transactions (
    transaction_id  INTEGER   PRIMARY KEY,
    customer_id     INTEGER   NOT NULL,
    product_id      INTEGER,                    -- nullable: e.g. login / support_ticket
    event_date      TEXT      NOT NULL,
    event_type      TEXT      NOT NULL,
    amount          REAL,                       -- nullable for non-purchase events
    session_minutes REAL,                       -- nullable for non-session events
    CONSTRAINT fk_tx_customer
        FOREIGN KEY (customer_id) REFERENCES subscriptions (customer_id),
    CONSTRAINT fk_tx_product
        FOREIGN KEY (product_id) REFERENCES products (product_id),
    CONSTRAINT chk_event_type
        CHECK (event_type IN ('login', 'feature_use', 'purchase', 'support_ticket'))
);

-- Indexes to speed up the customer-level aggregations in features.sql / features.py.
CREATE INDEX idx_tx_customer ON transactions (customer_id);
CREATE INDEX idx_tx_customer_date ON transactions (customer_id, event_date);
CREATE INDEX idx_tx_event_type ON transactions (event_type);
