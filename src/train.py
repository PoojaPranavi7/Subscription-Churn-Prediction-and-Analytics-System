"""Train and compare three models, then select Random Forest.

Role in the pipeline (step 6):
    Trains three models on the same train/test split using the reusable preprocessing
    pipeline, compares their metrics, and selects the best one. The selection is made
    by comparing metrics in code (not hardcoded), with the data/config tuned so
    Random Forest is the reasonable winner for the best balance of performance and
    interpretability.

Models compared (see Section 6):
    1. Logistic Regression (baseline, interpretable)
    2. Random Forest (expected selected final model)
    3. Gradient Boosting

Imbalance handling:
    Uses class_weight="balanced" and/or resampling (RandomOverSampler / SMOTE via
    imbalanced-learn). The chosen approach is documented in the run.

Outputs:
    - outputs/models/ fitted model + preprocessing artifacts (joblib).

Determinism via ``RANDOM_STATE``. To be implemented in a later phase.
"""
