"""Evaluate the selected model and tune the decision threshold.

Role in the pipeline (step 7):
    Computes the full evaluation suite for the selected model and sweeps decision
    thresholds to pick one that maximizes recall on the churn class subject to a
    reasonable precision floor (business tradeoff). Persists metrics so they are
    reproducible and auditable.

Reported metrics (see Section 7 — never accuracy alone):
    - Accuracy (target ~85%)
    - Precision / Recall / F1 per class, with churn-class recall highlighted
    - ROC-AUC
    - Confusion matrix
    - Threshold sweep table (precision/recall/F1 across thresholds)

Outputs:
    - outputs/metrics/ metrics + threshold sweep (CSV/JSON)
    - the chosen decision threshold saved for scoring.

To be implemented in a later phase.
"""
