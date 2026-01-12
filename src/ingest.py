import pandas as pd
import sqlite3

df = pd.read_csv("data/raw/telco_churn.csv")

# Fix TotalCharges issue
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

conn = sqlite3.connect("churn.db")
df.to_sql("customers_raw", conn, if_exists="replace", index=False)
conn.close()

print("Ingestion complete. Rows loaded:", len(df))
