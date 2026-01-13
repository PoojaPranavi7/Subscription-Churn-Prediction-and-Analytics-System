# Subscription Churn Prediction & Analytics System

## Overview
A production-style machine learning system that predicts customer churn, explains key churn drivers, and generates analytics-ready outputs for downstream BI tools.
This project follows an end-to-end ML lifecycle: data ingestion → validation → feature engineering → model training → evaluation → monitoring → scoring.

## Project Objective
The goal of this project is to:
- Predict customer churn using supervised machine learning
- Achieve strong predictive performance (~85% accuracy / strong ROC-AUC)
- Identify key behavioral and contractual churn drivers
- Simulate a real-world production ML workflow
- Export clean, analytics-ready datasets for business intelligence tools
This project is designed to mirror how churn models are built and operationalized in real companies.

## Key Features
- SQL-backed ingestion & feature table creation
- Automated preprocessing (missing values, encoding, scaling, log features)
- Model training + evaluation (Accuracy/F1/ROC-AUC)
- Feature importance to explain churn drivers
- Monitoring to detect data/pipeline drift across runs
- Tableau-ready outputs (risk scoring + churn insights dashboards)

## Dataset
- Source: Telco Customer Churn dataset
- Granularity: One row per customer
- Target Variable: Churn (Yes / No)
- Key Features:
  - Tenure
  - Contract type
  - Monthly charges
  - Total charges
  - Internet service
  - Payment method
  - Demographics (senior citizen, dependents, partner)
Raw data is stored unchanged and all transformations are reproducible.

## Tech Stack
Languages & Core Tools
- Python 3.10+
- SQL (SQLite / Postgres compatible)
- Bash (pipeline execution)

Machine Learning
- scikit-learn
- XGBoost
- joblib
- numpy
- pandas

Data Validation & Monitoring
- Custom validation logic
- Schema & expectation checks
- Baseline vs run monitoring reports

Visualization / BI (downstream only)
- Tableau 

## Exploratory Data Analysis (EDA)
Notebook: notebooks/01_eda.ipynb

EDA focused on:
- Churn rate by tenure band
- Churn vs contract type
- Distribution of monthly & total charges
- Correlation analysis
- Missing value inspection

Key findings:
- Month-to-month contracts have the highest churn
- New customers (0–6 months) churn significantly more
- Higher monthly charges increase churn probability
- Fiber optic internet customers churn more frequently

EDA insights are summarized in outputs/metrics/eda_insights.md.

## Data Preprocessing
Implemented in src/preprocess.py using scikit-learn Pipelines.

Preprocessing steps:
- Missing value imputation
- One-hot encoding for categorical features
- Log transformation for skewed monetary features
- Feature scaling
- Train/test split with reproducibility

Artifacts saved:
- preprocess_pipeline.joblib
- feature_names.csv
- X_train.npy, X_test.npy
- y_train.csv, y_test.csv

## Model Training
Implemented in src/train.py.

Models trained:
- Logistic Regression (baseline)
- Random Forest 
- XGBoost

Training features:
- Hyperparameter tuning
- Cross-validation
- Class imbalance handling
- Deterministic runs via random seeds

Final model saved as:
outputs/models/best_model.joblib

Metadata stored in:
outputs/models/best_model_meta.json

## Model Evaluation
Implemented in src/evaluate.py.

Metrics calculated:
- Accuracy
- Precision / Recall
- F1-Score
- ROC-AUC
- Confusion matrix
- Threshold sensitivity analysis

Outputs:
- metrics.json
- model_metrics.csv
- roc_points.csv
- threshold_sweep.csv
- confusion_matrix.csv

Target performance achieved:
- Accuracy ≈ 85%
- Strong ROC-AUC

## Feature Importance & Explainability
Implemented in src/feature_importance.py.

Generated:
- Global feature importance ranking
- Visual bar chart for top drivers

Top churn drivers:
- Contract type (Month-to-month)
- Monthly charges
- Tenure
- Internet service type
- Payment method

Artifacts:
- feature_importance.csv
- feature_importance.png

## Monitoring & Drift Baseline
Implemented in:
- src/monitor.py
- src/monitor_summary.py

Simulates production ML monitoring:
- Baseline distribution capture
- Feature drift checks
- Prediction distribution tracking

Outputs:
- monitor_baseline.json
- monitor_report.csv
- monitor_summary.csv

## Scoring Pipeline
Generates customer-level predictions:
- Predicted churn probability
- Binary churn label
- Risk bucket assignment:
  - Low (< 0.30)
  - Medium (0.30–0.60)
  - High (0.60–0.80)
  - Very High (0.80+)

Final scoring output:
outputs/predictions/customer_scored.csv

This file is analytics-ready and used for downstream reporting.

## SQL Layer (Optional Analytics Modeling)
Located in sql/:
- create_tables.sql
- features.sql
Provides example feature tables and schemas for loading predictions into a relational database.

## How to Run 
pip install -r requirements.txt

python src/ingest.py
python src/preprocess.py
python src/train.py
python src/evaluate.py
python src/feature_importance.py
python src/monitor.py
python src/score.py

## Key Skills Demonstrated
- End-to-end ML system design
- Feature engineering with pipelines
- Model evaluation & explainability
- Production-style validation & monitoring
- Versioned artifacts & reproducibility
- Analytics-ready data exports

## Next Phase (Not Covered in This README)
- Tableau dashboarding
- Executive KPI views
- Risk bucket selectors
- Customer-level drilldowns
(Documented separately)