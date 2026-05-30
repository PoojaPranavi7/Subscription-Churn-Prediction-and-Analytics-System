"""Lightweight production-style monitoring with drift detection.

Role in the pipeline (step 10):
    Simulates a production monitoring workflow over the modeling table + scored
    predictions. On the FIRST run it captures a baseline distribution (per-feature
    means/stds + binning, the actual churn rate, and the model's prediction rate)
    and saves it. On every SUBSEQUENT run it compares the current distribution
    against that baseline and flags drift using a simple PSI + z-score check.

Drift rules:
    - PSI (Population Stability Index) per numeric feature:
        < 0.10            -> no drift
        0.10 - 0.25       -> moderate drift (warning)
        > 0.25            -> significant drift (flag)
    - z-score of the mean shift |(cur_mean - base_mean) / base_std| > 3 also flags.
    - churn_rate / prediction_rate: flagged if they move by more than
      RATE_DRIFT_ABS (absolute) from baseline.

Outputs (Section 9):
    - outputs/metrics/monitor_report.csv    one row per monitored metric
    - outputs/metrics/monitor_summary.csv   one-row run summary
    - outputs/metrics/monitor_baseline.json  persisted baseline (delete to re-capture)

Run standalone (requires `python -m src.features`; predictions also need
`python -m src.train` + `python -m src.evaluate`):
    python -m src.monitor            # establish baseline (first run) or compare
    python -m src.monitor --reset    # force re-capture of the baseline
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

try:
    from src.config import METRICS_DIR, MODELS_DIR, PROCESSED_DATA_DIR
    from src import preprocess as P
    from src import train as T
    from src import evaluate as E
except ImportError:  # allow running as a plain script
    from config import METRICS_DIR, MODELS_DIR, PROCESSED_DATA_DIR
    import preprocess as P
    import train as T
    import evaluate as E


BASELINE_ARTIFACT = "monitor_baseline.json"
REPORT_ARTIFACT = "monitor_report.csv"
SUMMARY_ARTIFACT = "monitor_summary.csv"

N_BINS = 10
PSI_MODERATE = 0.10
PSI_SIGNIFICANT = 0.25
Z_FLAG = 3.0
RATE_DRIFT_ABS = 0.05       # 5 percentage-point shift in churn/prediction rate
EPS = 1e-6

# Numeric features whose distribution we track.
MONITORED_FEATURES = P.NUMERIC_ZERO_FEATURES + P.NUMERIC_MEDIAN_FEATURES


def _load_current() -> pd.DataFrame:
    path = PROCESSED_DATA_DIR / "modeling_table.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing modeling table: {path}. Run `python -m src.features` first."
        )
    return pd.read_csv(path)


def _prediction_rate(df: pd.DataFrame) -> float | None:
    """Share of customers the model flags as churn at the tuned threshold."""
    model_path = MODELS_DIR / T.FINAL_MODEL_ARTIFACT
    thr_path = MODELS_DIR / E.THRESHOLD_ARTIFACT
    if not model_path.exists() or not thr_path.exists():
        return None
    model = joblib.load(model_path)
    threshold = float(json.loads(thr_path.read_text())["threshold"])
    proba = model.predict_proba(df[P.all_feature_columns()])[:, 1]
    return float((proba >= threshold).mean())


def _psi(base_props: np.ndarray, cur_props: np.ndarray) -> float:
    base = np.clip(base_props, EPS, None)
    cur = np.clip(cur_props, EPS, None)
    return float(np.sum((cur - base) * np.log(cur / base)))


def capture_baseline(df: pd.DataFrame) -> dict:
    """Build the baseline distribution snapshot from the current data."""
    features = {}
    for col in MONITORED_FEATURES:
        if col not in df.columns:
            continue
        x = df[col].to_numpy(dtype=float)
        edges = np.unique(np.quantile(x, np.linspace(0, 1, N_BINS + 1)))
        if edges.size < 2:  # constant feature -> degenerate single bin
            edges = np.array([x.min() - 0.5, x.max() + 0.5])
        counts, _ = np.histogram(x, bins=edges)
        props = counts / max(counts.sum(), 1)
        features[col] = {
            "mean": float(np.mean(x)), "std": float(np.std(x)),
            "bin_edges": edges.tolist(), "bin_props": props.tolist(),
        }

    return {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_rows": int(len(df)),
        "churn_rate": float(df["churn"].mean()) if "churn" in df.columns else None,
        "prediction_rate": _prediction_rate(df),
        "features": features,
    }


def _feature_rows(baseline: dict, df: pd.DataFrame) -> list[dict]:
    rows = []
    for col, base in baseline["features"].items():
        if col not in df.columns:
            continue
        x = df[col].to_numpy(dtype=float)
        edges = np.array(base["bin_edges"])
        counts, _ = np.histogram(x, bins=edges)
        cur_props = counts / max(counts.sum(), 1)
        psi = _psi(np.array(base["bin_props"]), cur_props)

        cur_mean, cur_std = float(np.mean(x)), float(np.std(x))
        base_std = base["std"] if base["std"] > 0 else 1.0
        z = (cur_mean - base["mean"]) / base_std

        if psi > PSI_SIGNIFICANT or abs(z) > Z_FLAG:
            level = "significant"
        elif psi > PSI_MODERATE:
            level = "moderate"
        else:
            level = "none"

        rows.append({
            "metric": col, "type": "feature",
            "baseline_mean": base["mean"], "current_mean": cur_mean,
            "baseline_std": base["std"], "current_std": cur_std,
            "psi": psi, "z_score": z,
            "drift_level": level, "drift_flag": level != "none",
        })
    return rows


def _rate_row(name: str, base_val, cur_val) -> dict:
    if base_val is None or cur_val is None:
        return {"metric": name, "type": "rate", "baseline_mean": base_val,
                "current_mean": cur_val, "baseline_std": None, "current_std": None,
                "psi": None, "z_score": None, "drift_level": "unknown", "drift_flag": False}
    diff = abs(cur_val - base_val)
    level = "significant" if diff > RATE_DRIFT_ABS else "none"
    return {"metric": name, "type": "rate", "baseline_mean": base_val,
            "current_mean": cur_val, "baseline_std": None, "current_std": None,
            "psi": None, "z_score": float(cur_val - base_val),
            "drift_level": level, "drift_flag": level != "none"}


def run_monitoring(reset: bool = False) -> dict:
    """Establish the baseline (first run / --reset) or compare current vs baseline."""
    df = _load_current()
    baseline_path = METRICS_DIR / BASELINE_ARTIFACT
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    first_run = reset or not baseline_path.exists()

    if first_run:
        baseline = capture_baseline(df)
        baseline_path.write_text(json.dumps(baseline, indent=2))
    else:
        baseline = json.loads(baseline_path.read_text())

    # Current snapshot for the rate metrics.
    cur_churn = float(df["churn"].mean()) if "churn" in df.columns else None
    cur_pred = _prediction_rate(df)

    rows = _feature_rows(baseline, df)
    rows.append(_rate_row("churn_rate", baseline.get("churn_rate"), cur_churn))
    rows.append(_rate_row("prediction_rate", baseline.get("prediction_rate"), cur_pred))
    report = pd.DataFrame(rows)

    n_moderate = int((report["drift_level"] == "moderate").sum())
    n_significant = int((report["drift_level"] == "significant").sum())
    run_type = "baseline_established" if first_run else "comparison"
    status = "BASELINE" if first_run else ("DRIFT" if n_significant > 0 else
                                           ("WATCH" if n_moderate > 0 else "OK"))

    summary = pd.DataFrame([{
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_type": run_type,
        "status": status,
        "baseline_created_at": baseline["created_at"],
        "n_metrics": len(report),
        "n_drift_moderate": n_moderate,
        "n_drift_significant": n_significant,
        "baseline_rows": baseline["n_rows"],
        "current_rows": len(df),
        "churn_rate_baseline": baseline.get("churn_rate"),
        "churn_rate_current": cur_churn,
        "prediction_rate_baseline": baseline.get("prediction_rate"),
        "prediction_rate_current": cur_pred,
    }])

    report.to_csv(METRICS_DIR / REPORT_ARTIFACT, index=False)
    summary.to_csv(METRICS_DIR / SUMMARY_ARTIFACT, index=False)

    # --- report ---
    print("=" * 70)
    print(f"Monitoring — {run_type} (status: {status})")
    print("=" * 70)
    if first_run:
        print("Baseline captured from current data; future runs compare against it.")
    else:
        flagged = report[report["drift_flag"]]
        if flagged.empty:
            print("No drift detected (all PSI < 0.10 and |z| < 3).")
        else:
            print("Drifting metrics:")
            for _, r in flagged.iterrows():
                psi = "n/a" if pd.isna(r["psi"]) else f"{r['psi']:.3f}"
                print(f"  [{r['drift_level']:>11}] {r['metric']:<32} "
                      f"PSI={psi}  z={r['z_score']:.2f}  "
                      f"(base={r['baseline_mean']:.3f} -> cur={r['current_mean']:.3f})")
    print("-" * 70)
    print(f"churn_rate     : {baseline.get('churn_rate')} -> {cur_churn}")
    print(f"prediction_rate: {baseline.get('prediction_rate')} -> {cur_pred}")
    print(f"moderate={n_moderate} significant={n_significant}")
    print(f"saved: {METRICS_DIR / REPORT_ARTIFACT}")
    print(f"saved: {METRICS_DIR / SUMMARY_ARTIFACT}")
    print("=" * 70)

    return {"report": report, "summary": summary, "status": status}


def main() -> None:
    run_monitoring(reset="--reset" in sys.argv)


if __name__ == "__main__":
    main()
