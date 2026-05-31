# Churn drivers — key insights

**Behavioral signals are the strongest churn drivers.** Engagement decline, falling purchase frequency, inconsistent usage, and recency dominate the model's importance ranking — contractual/demographic fields are secondary, exactly as intended.

## Top 5 drivers

1. **session_minutes_trend** (engagement) — importance 0.120, permutation Δ=0.008
2. **purchases_last_30d** (purchase) — importance 0.111, permutation Δ=0.020
3. **coefficient_of_variation_sessions** (consistency) — importance 0.109, permutation Δ=0.012
4. **logins_last_30d** (engagement) — importance 0.088, permutation Δ=0.005
5. **engagement_trend** (engagement) — importance 0.066, permutation Δ=0.003

## By headline behavioral signal

- **Engagement decline** → top feature `session_minutes_trend` (rank #1).
- **Purchase-frequency change** → top feature `purchases_last_30d` (rank #2).
- **Usage inconsistency** → top feature `coefficient_of_variation_sessions` (rank #3).
- **Recency** → top feature `days_since_last_activity` (rank #6).

## Verification

- Top feature: `session_minutes_trend` (family: engagement).
- Contractual features in top 5: none.
- Behavioral importance total: 0.958 vs contractual: 0.042.
- Result: PASS — behavioral features rank at the top.
