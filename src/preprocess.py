from __future__ import annotations

import os
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from joblib import dump
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

@dataclass(frozen=True)
class Config:
    db_path: str = "churn.db"
    source_table: str = "customer_features"
    target_col: str = "churn"
    id_col: str = "customerID"
    test_size: float = 0.2
    random_state: int = 42

    out_dir: str = os.path.join("outputs", "processed")
    model_dir: str = os.path.join("outputs", "models")

CFG = Config()

def ensure_dirs() -> None:
    os.makedirs(CFG.out_dir, exist_ok=True)
    os.makedirs(CFG.model_dir, exist_ok=True)


def load_from_sqlite() -> pd.DataFrame:
    conn = sqlite3.connect(CFG.db_path)
    df = pd.read_sql(f"SELECT * FROM {CFG.source_table}", conn)
    conn.close()
    return df

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "TotalCharges" in df.columns and "tenure" in df.columns:
        df["charges_per_tenure"] = df["TotalCharges"] / (df["tenure"].fillna(0) + 1)

    if "MonthlyCharges" in df.columns:
        df["log_monthly_charges"] = np.log1p(df["MonthlyCharges"])

    if "TotalCharges" in df.columns:
        df["log_total_charges"] = np.log1p(df["TotalCharges"])

    return df


def build_preprocess_pipeline(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),  # works with sparse output too
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    feature_names: list[str] = []

    # Numeric features
    num_features = preprocessor.transformers_[0][2]
    feature_names.extend(list(num_features))

    # Categorical features
    cat_transformer = preprocessor.transformers_[1][1]
    ohe: OneHotEncoder = cat_transformer.named_steps["onehot"]
    cat_features = preprocessor.transformers_[1][2]
    ohe_names = ohe.get_feature_names_out(cat_features)
    feature_names.extend(list(ohe_names))

    return feature_names


def save_sparse_matrix(path: str, X) -> None:
    if sparse.issparse(X):
        sparse.save_npz(path, X.tocsr())
    else:
        np.save(path.replace(".npz", ".npy"), X)

def main() -> None:
    ensure_dirs()

    df = load_from_sqlite()

    required = {CFG.target_col, CFG.id_col}
    missing_required = required - set(df.columns)
    if missing_required:
        raise ValueError(f"Missing required columns in {CFG.source_table}: {missing_required}")

    df = add_features(df)

    y = df[CFG.target_col].astype(int)
    X = df.drop(columns=[CFG.target_col])

    if CFG.id_col in X.columns:
        ids = X[CFG.id_col].copy()
        X = X.drop(columns=[CFG.id_col])
    else:
        ids = None

    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]

    # Train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=CFG.test_size,
        random_state=CFG.random_state,
        stratify=y,
    )

    # Build preprocessing pipeline
    preprocessor = build_preprocess_pipeline(numeric_cols, categorical_cols)

    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    feature_names = get_feature_names(preprocessor)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    save_sparse_matrix(os.path.join(CFG.out_dir, "X_train.npz"), X_train_t)
    save_sparse_matrix(os.path.join(CFG.out_dir, "X_test.npz"), X_test_t)

    pd.DataFrame({"y": y_train}).to_csv(os.path.join(CFG.out_dir, "y_train.csv"), index=False)
    pd.DataFrame({"y": y_test}).to_csv(os.path.join(CFG.out_dir, "y_test.csv"), index=False)
    pd.DataFrame({"feature_name": feature_names}).to_csv(os.path.join(CFG.out_dir, "feature_names.csv"), index=False)

    dump(preprocessor, os.path.join(CFG.model_dir, "preprocess_pipeline.joblib"))

    # Save metadata for reproducibility
    meta = {
        "source_table": CFG.source_table,
        "rows_total": int(df.shape[0]),
        "train_rows": int(X_train.shape[0]),
        "test_rows": int(X_test.shape[0]),
        "churn_rate_total": float(y.mean()),
        "churn_rate_train": float(y_train.mean()),
        "churn_rate_test": float(y_test.mean()),
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "n_features_after_transform": int(X_train_t.shape[1]),
        "test_size": CFG.test_size,
        "random_state": CFG.random_state,
        "run_utc": ts,
    }
    with open(os.path.join(CFG.out_dir, "preprocess_run_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("Preprocessing complete")
    print(f"Rows: total={df.shape[0]} train={X_train.shape[0]} test={X_test.shape[0]}")
    print(f"Churn rate: total={y.mean():.4f} train={y_train.mean():.4f} test={y_test.mean():.4f}")
    print(f"Features after transform: {X_train_t.shape[1]}")
    print(f"Saved to: {CFG.out_dir}")
    print(f"Pipeline saved to: {CFG.model_dir}/preprocess_pipeline.joblib")

if __name__ == "__main__":
    main()
