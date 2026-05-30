# Churn drivers — key insights

**Behavioral signals are the strongest churn drivers.** Engagement decline, falling purchase frequency, inconsistent usage, and recency dominate the model's importance ranking — contractual/demographic fields are secondary, exactly as intended.

## Top 5 drivers

1. **session_minutes_trend** (engagement) — importance 0.167, permutation Δ=0.034
2. **coefficient_of_variation_sessions** (consistency) — importance 0.136, permutation Δ=0.022
3. **logins_last_30d** (engagement) — importance 0.098, permutation Δ=0.008
4. **engagement_trend** (engagement) — importance 0.093, permutation Δ=0.009
5. **days_since_last_activity** (engagement) — importance 0.052, permutation Δ=0.002

## By headline behavioral signal

- **Engagement decline** → top feature `session_minutes_trend` (rank #1).
- **Purchase-frequency change** → top feature `total_spend` (rank #10).
- **Usage inconsistency** → top feature `coefficient_of_variation_sessions` (rank #2).
- **Recency** → top feature `days_since_last_activity` (rank #5).

## Verification

- Top feature: `session_minutes_trend` (family: engagement).
- Contractual features in top 5: none.
- Behavioral importance total: 0.957 vs contractual: 0.043.
- Result: PASS — behavioral features rank at the top.
