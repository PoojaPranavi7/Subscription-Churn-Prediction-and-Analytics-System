"""Behavioral feature engineering (core of the project).

Role in the pipeline (step 4):
    Aggregates the cleaned/joined data into a single customer-level modeling table.
    All features are computed per customer using only data before the label cutoff.

Feature families (see Section 4 of the context document):
    - Engagement: total_logins, logins_last_30d/prev_30d, engagement_trend,
      avg_session_minutes, session_minutes_trend, active_days, days_since_last_activity.
    - Purchase patterns: total_purchases, purchases_last_30d/prev_30d,
      purchase_frequency, purchase_frequency_trend, avg_order_value, total_spend,
      spend_trend, distinct_product_categories.
    - Consistency / volatility: activity_gap_std, coefficient_of_variation_sessions,
      active_weeks_ratio.
    - Retention / tenure: tenure_months, recency_days, support_tickets_count,
      late_or_missed_payments.
    - Contractual / demographic (supporting): plan_type, contract_type,
      monthly_fee, payment_method, region.

Outputs:
    - data/processed/ modeling table (one row per customer) including the churn label.

To be implemented in a later phase.
"""
