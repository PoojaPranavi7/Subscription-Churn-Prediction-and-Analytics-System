"""Rank feature importances and surface churn drivers.

Role in the pipeline (step 8):
    Extracts Random Forest feature importances (and optionally permutation
    importance), saves a ranked table and a bar chart of the top drivers, and writes
    a short plain-language insights file. The headline finding must be that
    behavioral signals (engagement decline, falling purchase frequency, usage
    inconsistency, recency) rank at the top — not contractual fields.

Outputs:
    - outputs/metrics/ ranked importance table (CSV)
    - outputs/metrics/ top-drivers bar chart (matplotlib)
    - a generated insights text/markdown file describing the top drivers.

To be implemented in a later phase.
"""
