"""Reusable preprocessing pipeline (imputation / encoding / scaling).

Role in the pipeline:
    Builds a single, fitted scikit-learn ``Pipeline`` / ``ColumnTransformer`` that is
    reused across training, evaluation, and scoring. Keeping preprocessing reusable
    (and fit on train only) prevents data leakage and guarantees consistent
    transforms at scoring time. The fitted transformer is persisted with joblib.

Imputation strategy (see Section 5 of the context document):
    - Numerical (usage counts, transaction values): median imputation.
    - Behavioral features where missing means no activity: impute 0.
    - Categorical: most frequent, or an explicit "unknown" category when missingness
      may be informative.

To be implemented in a later phase.
"""
