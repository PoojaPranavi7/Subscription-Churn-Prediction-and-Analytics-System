"""Rank feature importances and surface churn drivers.

Role in the pipeline (step 8):
    Explains the selected Random Forest by ranking which features drive churn.
    Combines two complementary views:
      * Impurity-based importance (RandomForest.feature_importances_), aggregated
        from the one-hot-encoded columns back to the 29 original features so the
        ranking is read at the human feature level.
      * Permutation importance on the held-out test set (model-agnostic; permutes
        each raw input column through the full pipeline), if feasible.

    It then writes a ranked CSV, a bar chart of the top drivers, and a short
    plain-language insights file, and verifies the project's headline claim that
    behavioral signals -- engagement decline, purchase-frequency change, usage
    inconsistency, recency -- rank at the top rather than contractual fields. If
    they do not, it flags this loudly.

Outputs (outputs/metrics/):
    - feature_importance.csv        ranked table (impurity + permutation, family)
    - feature_importance.png        bar chart of the top drivers
    - churn_drivers_insights.md     generated plain-language summary

Run standalone (requires `python -m src.train`):
    python -m src.feature_importance
"""

from __future__ import annotations

import json

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # headless / file-only rendering
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402

try:
    from src.config import METRICS_DIR, MODELS_DIR, RANDOM_STATE
    from src import preprocess as P
    from src import train as T
except ImportError:  # allow running as a plain script
    from config import METRICS_DIR, MODELS_DIR, RANDOM_STATE
    import preprocess as P
    import train as T


# --- Feature family taxonomy -------------------------------------------------
# Behavioral families (the project's headline drivers) vs the supporting
# contractual/demographic family.
FAMILY_OF = {
    # engagement
    "total_logins": "engagement", "logins_last_30d": "engagement",
    "logins_prev_30d": "engagement", "engagement_trend": "engagement",
    "avg_session_minutes": "engagement", "session_minutes_trend": "engagement",
    "active_days": "engagement", "days_since_last_activity": "engagement",
    # purchase patterns
    "total_purchases": "purchase", "purchases_last_30d": "purchase",
    "purchases_prev_30d": "purchase", "purchase_frequency": "purchase",
    "purchase_frequency_trend": "purchase", "avg_order_value": "purchase",
    "total_spend": "purchase", "spend_trend": "purchase",
    "distinct_product_categories": "purchase",
    # consistency / volatility
    "activity_gap_std": "consistency", "coefficient_of_variation_sessions": "consistency",
    "active_weeks_ratio": "consistency",
    # retention / tenure
    "tenure_months": "retention", "recency_days": "retention",
    "support_tickets_count": "retention", "late_or_missed_payments": "retention",
    # contractual / demographic (supporting, not headline)
    "plan_type": "contractual", "contract_type": "contractual",
    "monthly_fee": "contractual", "payment_method": "contractual", "region": "contractual",
}
BEHAVIORAL_FAMILIES = {"engagement", "purchase", "consistency", "retention"}

# Headline behavioral signals -> representative features (best-ranked member is
# reported in the insights file).
SIGNAL_FEATURES = {
    "engagement decline": [
        "engagement_trend", "session_minutes_trend", "logins_last_30d",
        "logins_prev_30d", "avg_session_minutes", "total_logins", "active_days",
    ],
    "purchase-frequency change": [
        "purchase_frequency_trend", "purchases_last_30d", "purchases_prev_30d",
        "purchase_frequency", "spend_trend", "total_spend", "avg_order_value",
        "total_purchases", "distinct_product_categories",
    ],
    "usage inconsistency": [
        "activity_gap_std", "coefficient_of_variation_sessions", "active_weeks_ratio",
    ],
    "recency": ["recency_days", "days_since_last_activity"],
}

TOP_N_CHART = 15
CSV_ARTIFACT = "feature_importance.csv"
CHART_ARTIFACT = "feature_importance.png"
INSIGHTS_ARTIFACT = "churn_drivers_insights.md"

FAMILY_COLORS = {
    "engagement": "#1f77b4", "purchase": "#2ca02c", "consistency": "#9467bd",
    "retention": "#17becf", "contractual": "#d62728",
}


def _source_feature(transformed_name: str) -> str:
    """Map a (possibly one-hot) transformed column back to its original feature."""
    if transformed_name in FAMILY_OF:
        return transformed_name
    for cat in P.CATEGORICAL_MOST_FREQUENT + P.CATEGORICAL_UNKNOWN:
        if transformed_name == cat or transformed_name.startswith(cat + "_"):
            return cat
    return transformed_name


def load_model_and_test() -> tuple:
    """Load the fitted RF pipeline plus the held-out test split."""
    model_path = MODELS_DIR / T.FINAL_MODEL_ARTIFACT
    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing model: {model_path}. Run `python -m src.train` first."
        )
    model = joblib.load(model_path)
    split = joblib.load(MODELS_DIR / P.ARTIFACT_SPLIT)
    return model, split["X_test"], split["y_test"].astype(int)


def aggregated_impurity_importance(model) -> pd.Series:
    """RF impurity importances aggregated from OHE columns to original features."""
    pre = model.named_steps["pre"]
    clf = model.named_steps["clf"]
    transformed_names = pre.get_feature_names_out()
    importances = clf.feature_importances_

    agg: dict[str, float] = {}
    for name, imp in zip(transformed_names, importances):
        src = _source_feature(name)
        agg[src] = agg.get(src, 0.0) + float(imp)
    return pd.Series(agg, name="rf_importance")


def compute_permutation_importance(model, X_test, y_test) -> pd.DataFrame | None:
    """Permutation importance over the raw input columns (model-agnostic).

    Returns a frame indexed by original feature, or None if it cannot be computed.
    """
    try:
        # n_jobs=1: single-process keeps this deterministic and avoids spawning
        # workers (only 29 features x 10 repeats x ~500 rows -- fast anyway).
        result = permutation_importance(
            model, X_test, y_test,
            scoring="roc_auc", n_repeats=10, random_state=RANDOM_STATE, n_jobs=1,
        )
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[warn] permutation importance skipped: {exc}")
        return None
    return pd.DataFrame(
        {
            "perm_importance_mean": result.importances_mean,
            "perm_importance_std": result.importances_std,
        },
        index=list(X_test.columns),
    )


def build_ranking(model, X_test, y_test) -> pd.DataFrame:
    """Assemble the ranked importance table (impurity + permutation + family)."""
    rf_imp = aggregated_impurity_importance(model)
    ranking = rf_imp.to_frame()
    ranking["family"] = ranking.index.map(FAMILY_OF)

    perm = compute_permutation_importance(model, X_test, y_test)
    if perm is not None:
        ranking = ranking.join(perm, how="left")

    ranking = ranking.sort_values("rf_importance", ascending=False)
    ranking.insert(0, "rank", range(1, len(ranking) + 1))
    ranking.index.name = "feature"
    return ranking


def verify_behavioral_dominance(ranking: pd.DataFrame, top_k: int = 5) -> dict:
    """Check the headline claim: behavioral features rank at the top."""
    top = ranking.head(top_k)
    top1_family = ranking.iloc[0]["family"]
    contractual_in_top = top.index[top["family"].eq("contractual")].tolist()

    behavioral_total = ranking.loc[ranking["family"].isin(BEHAVIORAL_FAMILIES), "rf_importance"].sum()
    contractual_total = ranking.loc[ranking["family"].eq("contractual"), "rf_importance"].sum()

    passed = (top1_family in BEHAVIORAL_FAMILIES) and (len(contractual_in_top) == 0)
    return {
        "passed": bool(passed),
        "top_feature": ranking.index[0],
        "top_feature_family": top1_family,
        "contractual_in_top_k": contractual_in_top,
        "top_k": top_k,
        "behavioral_importance_total": float(behavioral_total),
        "contractual_importance_total": float(contractual_total),
    }


def save_chart(ranking: pd.DataFrame, path) -> None:
    """Horizontal bar chart of the top drivers, colored by feature family."""
    top = ranking.head(TOP_N_CHART).iloc[::-1]  # smallest at bottom for barh
    colors = [FAMILY_COLORS.get(f, "#7f7f7f") for f in top["family"]]

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top.index, top["rf_importance"], color=colors)
    ax.set_xlabel("Random Forest importance (impurity, aggregated)")
    ax.set_title(f"Top {TOP_N_CHART} churn drivers")

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in FAMILY_COLORS.values()]
    ax.legend(handles, FAMILY_COLORS.keys(), title="family", loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _top_signal_features(ranking: pd.DataFrame) -> dict[str, tuple[str, int]]:
    """For each headline signal, the best-ranked representative feature."""
    out = {}
    rank_of = ranking["rank"].to_dict()
    for signal, feats in SIGNAL_FEATURES.items():
        present = [(f, rank_of[f]) for f in feats if f in rank_of]
        if present:
            best = min(present, key=lambda x: x[1])
            out[signal] = best
    return out


def write_insights(ranking: pd.DataFrame, verdict: dict, path) -> None:
    """Generate the plain-language churn-driver insights file."""
    top5 = ranking.head(5)
    signal_tops = _top_signal_features(ranking)

    lines = ["# Churn drivers — key insights", ""]

    if verdict["passed"]:
        lines.append(
            "**Behavioral signals are the strongest churn drivers.** Engagement "
            "decline, falling purchase frequency, inconsistent usage, and recency "
            "dominate the model's importance ranking — contractual/demographic "
            "fields are secondary, exactly as intended."
        )
    else:
        lines.append(
            "> **FLAG:** behavioral signals did NOT clearly dominate the importance "
            "ranking — see the verification section below."
        )
    lines += ["", "## Top 5 drivers", ""]
    for _, row in top5.iterrows():
        perm = ("" if "perm_importance_mean" not in row or pd.isna(row.get("perm_importance_mean"))
                else f", permutation Δ={row['perm_importance_mean']:.3f}")
        lines.append(
            f"{int(row['rank'])}. **{row.name}** ({row['family']}) — "
            f"importance {row['rf_importance']:.3f}{perm}"
        )

    lines += ["", "## By headline behavioral signal", ""]
    for signal, (feat, rnk) in signal_tops.items():
        lines.append(f"- **{signal.capitalize()}** → top feature `{feat}` (rank #{rnk}).")

    lines += ["", "## Verification", ""]
    lines.append(
        f"- Top feature: `{verdict['top_feature']}` (family: {verdict['top_feature_family']})."
    )
    lines.append(
        f"- Contractual features in top {verdict['top_k']}: "
        f"{verdict['contractual_in_top_k'] or 'none'}."
    )
    lines.append(
        f"- Behavioral importance total: {verdict['behavioral_importance_total']:.3f} "
        f"vs contractual: {verdict['contractual_importance_total']:.3f}."
    )
    lines.append(f"- Result: {'PASS — behavioral features rank at the top.' if verdict['passed'] else 'FLAGGED — see above.'}")
    lines.append("")

    path.write_text("\n".join(lines))


def run() -> dict:
    """Full Phase-8 flow: rank, verify, and persist CSV + chart + insights."""
    model, X_test, y_test = load_model_and_test()
    ranking = build_ranking(model, X_test, y_test)
    verdict = verify_behavioral_dominance(ranking)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(METRICS_DIR / CSV_ARTIFACT)
    save_chart(ranking, METRICS_DIR / CHART_ARTIFACT)
    write_insights(ranking, verdict, METRICS_DIR / INSIGHTS_ARTIFACT)

    # --- report ---
    print("=" * 64)
    print("Feature importance & churn drivers")
    print("=" * 64)
    cols = [c for c in ["rank", "rf_importance", "perm_importance_mean", "family"] if c in ranking.columns]
    print(ranking.head(10)[cols].to_string())
    print("-" * 64)
    if verdict["passed"]:
        print("VERDICT: PASS — behavioral signals rank at the top.")
    else:
        print("VERDICT: *** FLAG *** — behavioral signals did NOT dominate:")
        print(f"  top feature = {verdict['top_feature']} ({verdict['top_feature_family']}); "
              f"contractual in top {verdict['top_k']} = {verdict['contractual_in_top_k']}")
    print(f"  behavioral total={verdict['behavioral_importance_total']:.3f} | "
          f"contractual total={verdict['contractual_importance_total']:.3f}")
    print("=" * 64)
    for art in (CSV_ARTIFACT, CHART_ARTIFACT, INSIGHTS_ARTIFACT):
        print(f"saved: {METRICS_DIR / art}")

    return {"ranking": ranking, "verdict": verdict}


def main() -> None:
    run()


if __name__ == "__main__":
    main()
