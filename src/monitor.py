"""Lightweight production-style monitoring with drift detection.

Role in the pipeline (step 10):
    Simulates a production monitoring workflow. On the first run it captures a
    baseline feature distribution (means/stds, churn rate, prediction rate). On each
    subsequent run it compares the current distribution against the baseline and flags
    drift using a simple z-score / PSI-style check.

Outputs (see Section 9):
    - outputs/metrics/monitor_report.csv
    - outputs/metrics/monitor_summary.csv

To be implemented in a later phase.
"""
