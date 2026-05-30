"""Evaluate the selected model and tune the decision threshold.

Role in the pipeline (step 7):
    Loads the selected Random Forest pipeline (src/train.py) and the held-out test
    split (src/preprocess.py), computes the full Section 7 evaluation suite, sweeps
    decision thresholds, and selects an operating threshold that maximizes recall on
    the churn class subject to a reasonable precision floor (the business tradeoff:
    missing an at-risk customer costs more than over-flagging). All metrics, the
    sweep table, and the chosen threshold are persisted for downstream scoring and
    auditing.

Reported metrics (Section 7 -- never accuracy alone):
    - Accuracy (target ~85%)
    - Precision / Recall / F1 per class, with churn-class (label 1) recall highlighted
    - ROC-AUC (threshold-independent)
    - Confusion matrix
    - Threshold sweep table (precision/recall/F1/accuracy across thresholds)

Threshold selection rule:
    Among thresholds whose churn-class precision >= PRECISION_FLOOR, pick the one
    with the highest churn recall (ties -> higher precision, then the more
    conservative/higher threshold). If no threshold clears the floor, fall back to
    the threshold that maximizes F1 and flag it.

Outputs:
    - outputs/metrics/threshold_sweep.csv   full sweep grid + selection flags
    - outputs/metrics/confusion_matrix.csv  confusion matrix at the chosen threshold
    - outputs/metrics/evaluation.json       all metrics (chosen + default threshold)
    - outputs/models/threshold.json         the persisted chosen threshold (for scoring)

Run standalone (requires `python -m src.train` to have produced the model):
    python -m src.evaluate
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

try:
    from src.config import METRICS_DIR, MODELS_DIR
    from src import preprocess as P
    from src import train as T
except ImportError:  # allow running as a plain script
    from config import METRICS_DIR, MODELS_DIR
    import preprocess as P
    import train as T


# Business tradeoff: the minimum acceptable churn-class precision. We push recall as
# high as possible while keeping at least this share of flagged customers correct.
PRECISION_FLOOR = 0.60

# Fine grid for selection; the full grid is also saved for auditing.
THRESHOLD_GRID = np.round(np.arange(0.05, 0.96, 0.01), 2)
DEFAULT_THRESHOLD = 0.5

SWEEP_ARTIFACT = "threshold_sweep.csv"
CONFUSION_ARTIFACT = "confusion_matrix.csv"
EVALUATION_ARTIFACT = "evaluation.json"
THRESHOLD_ARTIFACT = "threshold.json"


def load_model_and_test() -> tuple:
    """Load the fitted final pipeline, test features/labels, and the model name."""
    model_path = MODELS_DIR / T.FINAL_MODEL_ARTIFACT
    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing model: {model_path}. Run `python -m src.train` first."
        )
    model = joblib.load(model_path)
    split = joblib.load(MODELS_DIR / P.ARTIFACT_SPLIT)

    model_name = "Selected model"
    card_path = MODELS_DIR / T.MODEL_CARD_ARTIFACT
    if card_path.exists():
        model_name = json.loads(card_path.read_text()).get("selected_model", model_name)

    return model, split["X_test"], split["y_test"].astype(int), model_name


def threshold_sweep(y_true: np.ndarray, proba: np.ndarray,
                    grid: np.ndarray = THRESHOLD_GRID) -> pd.DataFrame:
    """Compute churn-class precision/recall/F1, accuracy, and flag rate per threshold."""
    rows = []
    for thr in grid:
        pred = (proba >= thr).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, pred, labels=[1], average=None, zero_division=0
        )
        rows.append({
            "threshold": float(thr),
            "precision": float(p[0]),
            "recall": float(r[0]),
            "f1": float(f1[0]),
            "accuracy": float(accuracy_score(y_true, pred)),
            "predicted_churn_rate": float(pred.mean()),
        })
    return pd.DataFrame(rows)


def select_threshold(sweep: pd.DataFrame, precision_floor: float = PRECISION_FLOOR) -> tuple[float, bool]:
    """Pick the operating threshold; return (threshold, used_fallback)."""
    eligible = sweep[sweep["precision"] >= precision_floor]
    if not eligible.empty:
        # Highest recall, then higher precision, then the more conservative threshold.
        best = eligible.sort_values(
            ["recall", "precision", "threshold"], ascending=[False, False, False]
        ).iloc[0]
        return float(best["threshold"]), False

    # Fallback: maximize F1 when the precision floor cannot be met anywhere.
    best = sweep.sort_values(["f1", "threshold"], ascending=[False, False]).iloc[0]
    return float(best["threshold"]), True


def evaluate_at_threshold(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict:
    """Full per-class metrics + confusion matrix at a given threshold."""
    pred = (proba >= threshold).astype(int)
    p, r, f1, support = precision_recall_fscore_support(
        y_true, pred, labels=[0, 1], zero_division=0
    )
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "per_class": {
            "no_churn": {"precision": float(p[0]), "recall": float(r[0]),
                          "f1": float(f1[0]), "support": int(support[0])},
            "churn": {"precision": float(p[1]), "recall": float(r[1]),
                       "f1": float(f1[1]), "support": int(support[1])},
        },
        # churn-class recall highlighted as the headline metric.
        "churn_recall": float(r[1]),
        "churn_precision": float(p[1]),
        "churn_f1": float(f1[1]),
        "confusion_matrix": {
            "labels": ["no_churn(0)", "churn(1)"],
            "matrix": cm.tolist(),
            "tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
        },
    }


def _print_report(model_name: str, roc_auc: float, chosen: float, fallback: bool,
                  at_chosen: dict, at_default: dict) -> None:
    print("=" * 64)
    print(f"Evaluation — {model_name} (held-out test set)")
    print("=" * 64)
    print(f"ROC-AUC (threshold-independent): {roc_auc:.4f}")
    print(f"Precision floor (business)     : {PRECISION_FLOOR:.2f}")
    print(f"Chosen threshold               : {chosen:.2f}"
          + ("  [FALLBACK: floor unmet, max-F1]" if fallback else ""))
    print("-" * 64)
    for label, res in [("default (0.50)", at_default), (f"chosen ({chosen:.2f})", at_chosen)]:
        print(f"@ {label}:")
        print(f"    accuracy            : {res['accuracy']:.4f}"
              + ("   <- target ~0.85" if "chosen" in label else ""))
        c = res["per_class"]["churn"]; nc = res["per_class"]["no_churn"]
        print(f"    churn      P/R/F1   : {c['precision']:.3f} / "
              f"{c['recall']:.3f} / {c['f1']:.3f}   <- recall highlighted")
        print(f"    no_churn   P/R/F1   : {nc['precision']:.3f} / "
              f"{nc['recall']:.3f} / {nc['f1']:.3f}")
        cm = res["confusion_matrix"]
        print(f"    confusion [tn fp / fn tp]: [{cm['tn']} {cm['fp']} / {cm['fn']} {cm['tp']}]")
    print("=" * 64)


def run_evaluation() -> dict:
    """Full Phase-7 flow: score test set, sweep + select threshold, persist outputs."""
    model, X_test, y_test, model_name = load_model_and_test()
    y_true = y_test.to_numpy()
    proba = model.predict_proba(X_test)[:, 1]

    roc_auc = float(roc_auc_score(y_true, proba))

    sweep = threshold_sweep(y_true, proba)
    chosen_threshold, fallback = select_threshold(sweep)

    at_chosen = evaluate_at_threshold(y_true, proba, chosen_threshold)
    at_default = evaluate_at_threshold(y_true, proba, DEFAULT_THRESHOLD)

    # --- persist ---
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    sweep_out = sweep.copy()
    sweep_out["meets_precision_floor"] = sweep_out["precision"] >= PRECISION_FLOOR
    sweep_out["selected"] = np.isclose(sweep_out["threshold"], chosen_threshold)
    sweep_out.to_csv(METRICS_DIR / SWEEP_ARTIFACT, index=False)

    cm = at_chosen["confusion_matrix"]["matrix"]
    pd.DataFrame(cm, index=["actual_no_churn", "actual_churn"],
                 columns=["pred_no_churn", "pred_churn"]).to_csv(METRICS_DIR / CONFUSION_ARTIFACT)

    evaluation = {
        "model": model_name,
        "roc_auc": roc_auc,
        "precision_floor": PRECISION_FLOOR,
        "chosen_threshold": chosen_threshold,
        "threshold_selection_fallback": fallback,
        "default_threshold": DEFAULT_THRESHOLD,
        "metrics_at_chosen_threshold": at_chosen,
        "metrics_at_default_threshold": at_default,
    }
    with open(METRICS_DIR / EVALUATION_ARTIFACT, "w") as fh:
        json.dump(evaluation, fh, indent=2)

    # Persist the chosen threshold on its own for the scoring step (Phase 9).
    with open(MODELS_DIR / THRESHOLD_ARTIFACT, "w") as fh:
        json.dump({"threshold": chosen_threshold, "precision_floor": PRECISION_FLOOR,
                   "fallback": fallback}, fh, indent=2)

    _print_report(model_name, roc_auc, chosen_threshold, fallback, at_chosen, at_default)
    print(f"saved: {METRICS_DIR / SWEEP_ARTIFACT}")
    print(f"saved: {METRICS_DIR / CONFUSION_ARTIFACT}")
    print(f"saved: {METRICS_DIR / EVALUATION_ARTIFACT}")
    print(f"saved: {MODELS_DIR / THRESHOLD_ARTIFACT}")

    return evaluation


def main() -> None:
    run_evaluation()


if __name__ == "__main__":
    main()
