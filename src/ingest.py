"""Load raw CSVs into the SQLite database.

Role in the pipeline (step 2):
    Creates ``churn.db`` (SQLite), runs the DDL in ``sql/create_tables.sql``, and
    loads the three raw CSVs from ``data/raw/`` into the ``subscriptions``,
    ``products``, and ``transactions`` tables. This gives every downstream step a
    single relational source to query.

Inputs:
    - data/raw/subscriptions.csv
    - data/raw/products.csv
    - data/raw/transactions.csv
    - sql/create_tables.sql

Outputs:
    - churn.db with the three raw tables populated (rebuilt from scratch each run).

Load order respects foreign keys: subscriptions and products are loaded before
transactions. Empty CSV cells (nullable columns) are loaded as SQL NULL. After
loading, basic verification reports row counts per table and checks them against
the source CSVs.

Run standalone:
    python -m src.ingest
"""

from __future__ import annotations

import sqlite3

import pandas as pd

try:
    from src.config import DB_PATH, RAW_DATA_DIR, SQL_DIR
except ImportError:  # allow running as a plain script (python src/ingest.py)
    from config import DB_PATH, RAW_DATA_DIR, SQL_DIR


# Table name -> source CSV. Order matters: parents before children (FK targets).
TABLE_SOURCES = [
    ("subscriptions", "subscriptions.csv"),
    ("products", "products.csv"),
    ("transactions", "transactions.csv"),
]


def _run_ddl(conn: sqlite3.Connection) -> None:
    """Execute the schema definition from sql/create_tables.sql."""
    ddl_path = SQL_DIR / "create_tables.sql"
    ddl = ddl_path.read_text()
    conn.executescript(ddl)
    conn.commit()


def _load_csv(conn: sqlite3.Connection, table: str, csv_name: str) -> int:
    """Load one raw CSV into ``table`` and return the number of source rows.

    Empty cells are read as NaN by pandas and written as SQL NULL, which is the
    desired behavior for nullable columns (cancel_date, product_id, amount,
    session_minutes).
    """
    csv_path = RAW_DATA_DIR / csv_name
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Missing raw CSV: {csv_path}. Run `python -m src.generate_data` first."
        )

    df = pd.read_csv(csv_path)
    df.to_sql(table, conn, if_exists="append", index=False, chunksize=10_000)
    return len(df)


def _verify_counts(conn: sqlite3.Connection, expected: dict[str, int]) -> bool:
    """Compare row counts in the DB against the source CSV counts."""
    print("-" * 60)
    print("Load verification (row counts):")
    all_ok = True
    for table, exp in expected.items():
        actual = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        ok = actual == exp
        all_ok = all_ok and ok
        flag = "OK " if ok else "MISMATCH"
        print(f"  [{flag}] {table:<14}: db={actual:>9,}  csv={exp:>9,}")
    return all_ok


def main() -> None:
    # Rebuild from scratch so the database is fully reproducible on every run.
    if DB_PATH.exists():
        DB_PATH.unlink()

    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("PRAGMA foreign_keys = ON;")
        _run_ddl(conn)

        print("=" * 60)
        print(f"Ingesting raw CSVs into {DB_PATH.name}")
        print("=" * 60)

        expected: dict[str, int] = {}
        for table, csv_name in TABLE_SOURCES:
            n = _load_csv(conn, table, csv_name)
            expected[table] = n
            print(f"loaded {table:<14}: {n:>9,} rows from {csv_name}")

        conn.commit()
        all_ok = _verify_counts(conn, expected)
        print("=" * 60)
        print("Ingestion complete." if all_ok else "Ingestion finished WITH MISMATCHES.")
        print("=" * 60)

        if not all_ok:
            raise RuntimeError("Row-count verification failed; see report above.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
