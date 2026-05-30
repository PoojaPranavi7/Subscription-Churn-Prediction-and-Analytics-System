"""Reusable preprocessing pipeline (imputation / encoding / scaling).

Role in the pipeline (step 5):
    Builds a single, reusable scikit-learn ``ColumnTransformer`` that is fit on the
    training split only (no leakage) and reused unchanged for evaluation and
    scoring. It implements the Section 5 imputation strategy exactly, plus one-hot
    encoding for categoricals and standard scaling for numerics.

Imputation strategy (Section 5), encoded as four column groups:
    1. Behavioral "no activity" numerics  -> SimpleImputer(constant, fill_value=0)
       Missing genuinely means the customer did nothing (e.g. no logins -> 0), so 0
       is the correct value, not the median.
    2. Random-missing / usage & transaction numerics -> SimpleImputer(median)
       Values that are missing-at-random or undefined for sparse customers
       (activity_gap_std, coefficient_of_variation_sessions) get the robust median.
    3. Structural categoricals (plan_type, contract_type) -> most_frequent
       Missingness here is not expected to be informative.
    4. Possibly-informative categoricals (payment_method, region) -> explicit
       "unknown" category, because a missing value may itself carry signal.

Encoding / scaling:
    - Numerics are standardized (StandardScaler) so the Logistic Regression baseline
      is well-conditioned; harmless for the tree models.
    - Categoricals are one-hot encoded with handle_unknown="ignore" so unseen
      categories at scoring time don't break the transform.

Determinism & artifacts:
    A single reproducible, stratified train/test split is produced with
    ``RANDOM_STATE``. The fitted preprocessor and the split are saved with joblib so
    every downstream phase reuses the exact same data and transforms.

Exposed API (importable by train.py / evaluate.py / score.py):
    - load_modeling_table()      -> DataFrame
    - get_feature_columns()      -> dict of the four column groups
    - build_preprocessor()       -> unfitted ColumnTransformer
    - make_split(df)             -> dict with X/y/customer_id for train & test
    - prepare_and_save()         -> fit on train, persist artifacts, return bundle
    - load_artifacts()           -> reload the saved split + fitted preprocessor

Outputs (outputs/models/):
    - preprocessor.joblib        fitted ColumnTransformer (fit on train only)
    - split.joblib               dict of raw train/test splits + customer ids

Run standalone:
    python -m src.preprocess
"""

from __future__ import annotations

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from src.config import MODELS_DIR, PROCESSED_DATA_DIR, RANDOM_STATE
except ImportError:  # allow running as a plain script (python src/preprocess.py)
    from config import MODELS_DIR, PROCESSED_DATA_DIR, RANDOM_STATE


TARGET = "churn"
ID_COL = "customer_id"

# --- Column groups (Section 5 imputation strategy) ---------------------------
# Behavioral features where a missing value means "no activity" -> fill with 0.
NUMERIC_ZERO_FEATURES = [
    "total_logins", "logins_last_30d", "logins_prev_30d", "engagement_trend",
    "avg_session_minutes", "session_minutes_trend", "active_days",
    "total_purchases", "purchases_last_30d", "purchases_prev_30d",
    "purchase_frequency", "purchase_frequency_trend", "avg_order_value",
    "total_spend", "spend_trend", "distinct_product_categories",
    "support_tickets_count", "active_weeks_ratio", "late_or_missed_payments",
]

# Usage/transaction numerics that are missing-at-random or undefined for sparse
# customers -> median imputation.
NUMERIC_MEDIAN_FEATURES = [
    "days_since_last_activity", "activity_gap_std",
    "coefficient_of_variation_sessions", "tenure_months", "recency_days",
    "monthly_fee",
]

# Categoricals whose missingness is not expected to be informative -> most_frequent.
CATEGORICAL_MOST_FREQUENT = ["plan_type", "contract_type"]

# Categoricals whose missingness may itself be informative -> explicit "unknown".
CATEGORICAL_UNKNOWN = ["payment_method", "region"]

ARTIFACT_PREPROCESSOR = "preprocessor.joblib"
ARTIFACT_SPLIT = "split.joblib"


def get_feature_columns() -> dict[str, list[str]]:
    """Return the four preprocessing column groups."""
    return {
        "numeric_zero": list(NUMERIC_ZERO_FEATURES),
        "numeric_median": list(NUMERIC_MEDIAN_FEATURES),
        "categorical_most_frequent": list(CATEGORICAL_MOST_FREQUENT),
        "categorical_unknown": list(CATEGORICAL_UNKNOWN),
    }


def all_feature_columns() -> list[str]:
    """Ordered list of every model input column (excludes id and target)."""
    return (
        NUMERIC_ZERO_FEATURES
        + NUMERIC_MEDIAN_FEATURES
        + CATEGORICAL_MOST_FREQUENT
        + CATEGORICAL_UNKNOWN
    )


def load_modeling_table(path=None) -> pd.DataFrame:
    """Load the customer-level modeling table produced by src/features.py."""
    path = path or (PROCESSED_DATA_DIR / "modeling_table.csv")
    if not path.exists():
        raise FileNotFoundError(
            f"Missing modeling table: {path}. Run `python -m src.features` first."
        )
    return pd.read_csv(path)


def build_preprocessor() -> ColumnTransformer:
    """Construct the (unfitted) reusable preprocessing ColumnTransformer.

    Returned unfitted so it can be embedded inside model pipelines and fit strictly
    on training folds/splits, guaranteeing no leakage.
    """
    numeric_zero_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value=0)),
        ("scale", StandardScaler()),
    ])
    numeric_median_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    cat_most_frequent_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    cat_unknown_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer(
        transformers=[
            ("num_zero", numeric_zero_pipe, NUMERIC_ZERO_FEATURES),
            ("num_median", numeric_median_pipe, NUMERIC_MEDIAN_FEATURES),
            ("cat_mf", cat_most_frequent_pipe, CATEGORICAL_MOST_FREQUENT),
            ("cat_unknown", cat_unknown_pipe, CATEGORICAL_UNKNOWN),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_split(df: pd.DataFrame, test_size: float = 0.2) -> dict:
    """Reproducible, stratified train/test split keyed by the churn target.

    Returns raw (untransformed) feature frames plus aligned customer ids so the
    preprocessor can be fit on train only and downstream scoring can rejoin ids.
    """
    features = all_feature_columns()
    X = df[features]
    y = df[TARGET].astype(int)
    ids = df[ID_COL]

    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X, y, ids,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    return {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "id_train": id_train, "id_test": id_test,
    }


def prepare_and_save(test_size: float = 0.2, save: bool = True) -> dict:
    """Load data, split, fit the preprocessor on train only, and persist artifacts.

    Returns a bundle with the split, the fitted preprocessor, and the output feature
    names. This is the main entry point other scripts can import.
    """
    df = load_modeling_table()
    split = make_split(df, test_size=test_size)

    preprocessor = build_preprocessor()
    preprocessor.fit(split["X_train"])  # fit on TRAIN ONLY -> no leakage
    feature_names_out = list(preprocessor.get_feature_names_out())

    bundle = {**split, "preprocessor": preprocessor, "feature_names_out": feature_names_out}

    if save:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(preprocessor, MODELS_DIR / ARTIFACT_PREPROCESSOR)
        joblib.dump(
            {k: split[k] for k in ("X_train", "X_test", "y_train", "y_test", "id_train", "id_test")},
            MODELS_DIR / ARTIFACT_SPLIT,
        )

    return bundle


def load_artifacts() -> dict:
    """Reload the saved split and fitted preprocessor for downstream phases."""
    pre_path = MODELS_DIR / ARTIFACT_PREPROCESSOR
    split_path = MODELS_DIR / ARTIFACT_SPLIT
    if not pre_path.exists() or not split_path.exists():
        raise FileNotFoundError(
            "Preprocessing artifacts not found. Run `python -m src.preprocess` first."
        )
    preprocessor = joblib.load(pre_path)
    split = joblib.load(split_path)
    return {**split, "preprocessor": preprocessor,
            "feature_names_out": list(preprocessor.get_feature_names_out())}


def _print_summary(bundle: dict) -> None:
    y_train, y_test = bundle["y_train"], bundle["y_test"]
    n_out = len(bundle["feature_names_out"])
    print("=" * 60)
    print("Preprocessing pipeline summary")
    print("=" * 60)
    print(f"input features        : {len(all_feature_columns())}")
    print(f"  numeric (zero-fill) : {len(NUMERIC_ZERO_FEATURES)}")
    print(f"  numeric (median)    : {len(NUMERIC_MEDIAN_FEATURES)}")
    print(f"  categorical (mf)    : {len(CATEGORICAL_MOST_FREQUENT)}")
    print(f"  categorical (unk)   : {len(CATEGORICAL_UNKNOWN)}")
    print(f"output features (OHE) : {n_out}")
    print("-" * 60)
    print(f"train rows            : {len(y_train):,}  (churn {y_train.mean():.1%})")
    print(f"test rows             : {len(y_test):,}  (churn {y_test.mean():.1%})")
    print("-" * 60)
    print(f"saved: {MODELS_DIR / ARTIFACT_PREPROCESSOR}")
    print(f"saved: {MODELS_DIR / ARTIFACT_SPLIT}")
    print("=" * 60)


def main() -> None:
    bundle = prepare_and_save(save=True)
    _print_summary(bundle)


if __name__ == "__main__":
    main()
