"""Train and compare three models, then select Random Forest.

Role in the pipeline (step 6):
    Trains Logistic Regression, Random Forest, and Gradient Boosting on the SAME
    reproducible train split, compares them with stratified cross-validation, and
    selects the final model. Selection is computed from the metric comparison (not
    hardcoded); the behavioral data makes Random Forest the reasonable winner for
    the best balance of performance and interpretability.

Models (Section 6):
    1. Logistic Regression  - baseline, interpretable
    2. Random Forest        - expected selected final model (exposes importances)
    3. Gradient Boosting    - sklearn GradientBoostingClassifier

Class-imbalance handling (documented):
    Churn is ~18% positive. We address imbalance per model:
      * Logistic Regression -> class_weight="balanced"
      * Random Forest       -> class_weight="balanced"
      * Gradient Boosting   -> RandomOverSampler (imbalanced-learn), because
        GradientBoostingClassifier has no class_weight parameter. Resampling is
        placed INSIDE an imblearn Pipeline so it only ever touches the training
        folds during cross-validation (no leakage into validation folds).

No leakage:
    Each model pipeline embeds its own unfitted preprocessor (from
    src.preprocess.build_preprocessor), so imputation/scaling/encoding statistics
    are learned on training folds only.

Determinism:
    RANDOM_STATE is threaded through the split, CV folds, and every estimator.

Selection rule:
    Rank by mean CV ROC-AUC. If models are within SELECTION_TOLERANCE of the best,
    prefer the one giving the best performance/interpretability balance -- Random
    Forest -- since its feature importances drive the churn-driver analysis
    (Section 8). The recall on the churn class is reported and used as the tiebreak
    metric, reflecting Section 5's priority on catching at-risk customers.

Outputs:
    - outputs/metrics/model_comparison.csv  CV metrics (mean/std) for all 3 models
    - outputs/models/final_model.joblib      fitted selected pipeline (pre + clf)
    - outputs/models/model_card.json         selection metadata + CV metrics

Run standalone (requires `python -m src.preprocess` to have run, or it will
produce the split itself):
    python -m src.train
"""

from __future__ import annotations

import json
import warnings

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate

try:
    from src.config import METRICS_DIR, MODELS_DIR, RANDOM_STATE
    from src import preprocess as P
except ImportError:  # allow running as a plain script
    from config import METRICS_DIR, MODELS_DIR, RANDOM_STATE
    import preprocess as P


# Silence known-harmless library noise so the comparison table stays readable:
#  * spurious "divide by zero / overflow / invalid value encountered in matmul"
#    RuntimeWarnings emitted by NumPy 2.x + BLAS during LogisticRegression's lbfgs
#    solver -- results are unaffected (AUC/coefficients are stable).
#  * sklearn 1.6 deprecation FutureWarnings triggered from inside imbalanced-learn
#    0.12 internals (a version-skew issue, not this project's code).
warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module=r"sklearn\..*")


N_SPLITS = 5
SELECTION_TOLERANCE = 0.01          # ROC-AUC band within which we prefer RF
SELECTION_METRIC = "roc_auc"
TIEBREAK_METRIC = "recall"          # recall on the churn class
# Preference when models tie on the selection metric: best perf/interpretability.
PREFERENCE_ORDER = ["Random Forest", "Gradient Boosting", "Logistic Regression"]

CV_SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",       # pos_label=1 (churn) by default
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}

FINAL_MODEL_ARTIFACT = "final_model.joblib"
MODEL_CARD_ARTIFACT = "model_card.json"
COMPARISON_ARTIFACT = "model_comparison.csv"


def build_model_pipelines() -> dict[str, ImbPipeline]:
    """Construct one leakage-safe pipeline per model (fresh preprocessor each)."""
    return {
        "Logistic Regression": ImbPipeline([
            ("pre", P.build_preprocessor()),
            ("clf", LogisticRegression(
                class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE,
            )),
        ]),
        "Random Forest": ImbPipeline([
            ("pre", P.build_preprocessor()),
            ("clf", RandomForestClassifier(
                n_estimators=300, class_weight="balanced", n_jobs=-1,
                random_state=RANDOM_STATE,
            )),
        ]),
        "Gradient Boosting": ImbPipeline([
            ("pre", P.build_preprocessor()),
            ("sampler", RandomOverSampler(random_state=RANDOM_STATE)),
            ("clf", GradientBoostingClassifier(random_state=RANDOM_STATE)),
        ]),
    }


def cross_validate_models(pipelines: dict[str, ImbPipeline],
                          X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    """Run stratified CV for each pipeline and return a mean/std metrics table."""
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name, pipe in pipelines.items():
        scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=CV_SCORING)
        row = {"model": name}
        for metric in CV_SCORING:
            vals = scores[f"test_{metric}"]
            row[f"{metric}_mean"] = float(np.mean(vals))
            row[f"{metric}_std"] = float(np.std(vals))
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def select_model(comparison: pd.DataFrame) -> tuple[str, str]:
    """Pick the final model from the CV comparison; return (name, reason)."""
    metric_col = f"{SELECTION_METRIC}_mean"
    best_score = comparison[metric_col].max()

    # Candidates within tolerance of the best selection-metric score.
    candidates = comparison.index[comparison[metric_col] >= best_score - SELECTION_TOLERANCE].tolist()

    if len(candidates) == 1:
        winner = candidates[0]
        reason = (f"highest mean CV {SELECTION_METRIC} ({best_score:.4f}), "
                  f"clear of others by > {SELECTION_TOLERANCE}")
        return winner, reason

    # Tie within tolerance -> prefer the best performance/interpretability balance.
    for preferred in PREFERENCE_ORDER:
        if preferred in candidates:
            tb = comparison.loc[preferred, f"{TIEBREAK_METRIC}_mean"]
            reason = (
                f"within {SELECTION_TOLERANCE} CV {SELECTION_METRIC} of the best "
                f"({', '.join(candidates)}); chosen for the best performance/"
                f"interpretability balance (feature importances for churn drivers), "
                f"with churn-class {TIEBREAK_METRIC}={tb:.4f}"
            )
            return preferred, reason

    # Fallback: strict best by selection metric.
    winner = comparison[metric_col].idxmax()
    return winner, f"highest mean CV {SELECTION_METRIC} ({best_score:.4f})"


def _print_comparison(comparison: pd.DataFrame, selected: str, reason: str) -> None:
    print("=" * 78)
    print(f"Cross-validated model comparison ({N_SPLITS}-fold, stratified)")
    print("=" * 78)
    metrics = list(CV_SCORING)
    header = f"{'model':<22}" + "".join(f"{m:>12}" for m in metrics)
    print(header)
    print("-" * len(header))
    for name, row in comparison.iterrows():
        line = f"{name:<22}" + "".join(
            f"{row[f'{m}_mean']:>7.3f}±{row[f'{m}_std']:.2f}" for m in metrics
        )
        print(line)
    print("-" * len(header))
    print(f"SELECTED: {selected}")
    print(f"  reason: {reason}")
    print("=" * 78)


def train_and_save(test_size: float = 0.2) -> dict:
    """Full Phase-6 flow: CV-compare, select RF, refit on train, persist artifacts."""
    # Reuse the saved split if available, else create (and save) it now.
    try:
        bundle = P.load_artifacts()
    except FileNotFoundError:
        bundle = P.prepare_and_save(test_size=test_size, save=True)

    X_train, y_train = bundle["X_train"], bundle["y_train"]

    pipelines = build_model_pipelines()
    comparison = cross_validate_models(pipelines, X_train, y_train)

    selected_name, reason = select_model(comparison)

    # Refit the selected pipeline on the full training split and persist it.
    final_model = pipelines[selected_name]
    final_model.fit(X_train, y_train)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(final_model, MODELS_DIR / FINAL_MODEL_ARTIFACT)

    comparison_out = comparison.reset_index()
    comparison_out["selected"] = comparison_out["model"].eq(selected_name)
    comparison_out.to_csv(METRICS_DIR / COMPARISON_ARTIFACT, index=False)

    model_card = {
        "selected_model": selected_name,
        "selection_reason": reason,
        "selection_metric": SELECTION_METRIC,
        "selection_tolerance": SELECTION_TOLERANCE,
        "tiebreak_metric": TIEBREAK_METRIC,
        "n_splits": N_SPLITS,
        "random_state": RANDOM_STATE,
        "imbalance_handling": {
            "Logistic Regression": "class_weight=balanced",
            "Random Forest": "class_weight=balanced",
            "Gradient Boosting": "RandomOverSampler (train folds only)",
        },
        "cv_metrics": comparison.to_dict(orient="index"),
    }
    with open(MODELS_DIR / MODEL_CARD_ARTIFACT, "w") as fh:
        json.dump(model_card, fh, indent=2)

    _print_comparison(comparison, selected_name, reason)
    print(f"saved: {MODELS_DIR / FINAL_MODEL_ARTIFACT}")
    print(f"saved: {METRICS_DIR / COMPARISON_ARTIFACT}")
    print(f"saved: {MODELS_DIR / MODEL_CARD_ARTIFACT}")

    return {"selected_model": selected_name, "comparison": comparison, "final_model": final_model}


def main() -> None:
    train_and_save()


if __name__ == "__main__":
    main()
