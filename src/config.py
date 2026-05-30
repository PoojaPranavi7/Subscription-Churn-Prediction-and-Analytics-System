"""Shared configuration constants for the subscription churn system.

Centralizes paths, the deterministic random seed, and other settings that must
be consistent across every step of the pipeline (data generation, training,
evaluation, scoring, and monitoring). Import from here instead of redefining
values locally so the whole project stays reproducible.
"""

from pathlib import Path

# Determinism: reused everywhere a random seed is needed (numpy, sklearn,
# resampling, train/test splits) so runs are reproducible.
RANDOM_STATE = 42

# --- Project paths -----------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

SQL_DIR = PROJECT_ROOT / "sql"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
METRICS_DIR = OUTPUTS_DIR / "metrics"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"

# SQLite database file that holds the raw relational tables.
DB_PATH = PROJECT_ROOT / "churn.db"
