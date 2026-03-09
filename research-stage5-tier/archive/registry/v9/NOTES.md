# V9: Binding Frequency Feature

## Summary

v9 adds `binding_freq_6` — a 6-month historical binding frequency — as the 14th feature.
This produces **extraordinary dev-set improvements** that warrant careful scrutiny before trusting.

## Configuration

- **Method**: LightGBM regression, tiered labels (0/1/2/3)
- **Features**: 14 (12 base FEATURES_V1B + v7_formula_score + binding_freq_6)
- **Training**: 8 months, 0 validation
- **Eval**: 36 months (2020-06 to 2023-05)

## Feature Definition

```
binding_freq_6(constraint_id, month M) =
    count(months in M-6..M-1 where constraint had realized_sp > 0) / 6
```

Source: realized DA cache (`data/realized_da/{YYYY-MM}.parquet`).
Each training month gets its OWN temporal window (no future data used).

## Dev Results (36 months)

| Metric | v0 (formula) | v8c (prev best) | v9 | v9 vs v0 | v9 vs v8c |
|--------|-------------|-----------------|-----|----------|-----------|
| VC@20 | 0.3442 | 0.3702 | **0.4475** | +30% | +21% |
| VC@100 | 0.6296 | — | **0.7445** | +18% | — |
| Recall@20 | 0.2181 | — | **0.3514** | +61% | — |
| NDCG | 0.4802 | — | **0.6067** | +26% | — |
| Spearman | 0.1970 | 0.2011 | **0.3276** | +66% | +63% |

v9c ensemble (alpha=0.30): VC@20=0.4689, VC@100=0.6977, NDCG=0.6157, Spearman=0.2597

## Feature Importance (average gain)

| Feature | Gain | % of Total |
|---------|------|-----------|
| binding_freq_6 | 694.4 | 73.6% |
| v7_formula_score | 140.4 | 14.9% |
| da_rank_value | 57.5 | 6.1% |
| prob_exceed_110 | 15.0 | 1.6% |
| all others combined | 36.9 | 3.9% |

binding_freq_6 dominates by 5x over the next feature. The model is essentially a binding_freq classifier with minor adjustments from other features.

---

## SELF-AUDIT

### 1. Is this leakage?

**Strict temporal check: NO LEAKAGE.**

- For eval month M, binding_freq uses M-6..M-1. The label is realized_sp for month M.
- The eval month is NEVER in the lookback window. Verified programmatically.
- For training months, each month T's binding_freq uses T-6..T-1. The label for month T is T's own realized_sp. No overlap.

**Cross-month dependency in training data:**

Training months T₁ < T₂ can share data: T₂'s binding_freq lookback may include T₁, whose realized_sp is also a training label. Example: for eval=2021-06, train month 2020-12's binding_freq lookback includes 2020-10, which is also a training month with its own label.

This is standard time-series feature engineering (using lagged targets as features). It is NOT target leakage — the feature for each row only uses data from BEFORE that row's time point.

### 2. Why is the signal so strong?

**Binding persistence.** Constraints that bound recently tend to bind again because:
- Grid topology changes slowly (same transmission lines, same capacity limits)
- Congestion patterns are seasonal/structural (same load pockets, same generation mix)
- The constraint universe is relatively stable month-to-month

**Evidence from audit diagnostics:**
- 23.7% of test rows have binding_freq > 0 (ever bound in prior 6 months)
- Of those: 32.7% actually bind (6.5x the base rate of 5.0% for never-bound constraints)
- Spearman(binding_freq, realized_sp) = 0.40 — very high for a single feature

### 3. Concerns and risks

**A. The model is doing something simple.** With 73.6% of total feature importance, binding_freq_6 reduces the model to approximately: "predict binding if constraint has bound recently, predict non-binding if it hasn't." The other 13 features contribute marginally. This is powerful but fragile.

**B. Cannot predict NEW binding constraints.** 5.0% of constraints with binding_freq=0 DO actually bind. These are constraints that start binding for the first time (or after a >6 month gap). The model has no signal for these cases beyond the original features — which performed at v8-level (VC@20 ≈ 0.37).

**C. Dev-set autocorrelation may inflate results.** The 36-month rolling eval creates heavy autocorrelation: adjacent eval months share 5/6 of their binding_freq lookback. This means a constraint that binds persistently across 2020-2023 gets consistently high binding_freq AND consistently binds, making the feature look better than it would on a structurally different period.

**D. Holdout will be the real test.** The 2024-2025 holdout is 12+ months separated from the dev eval period. If binding patterns shift (new transmission projects, load growth, generation retirement), binding_freq will degrade. Expect significant regression from dev numbers.

**E. binding_freq is NOT the same as da_rank_value.** Correlation between them is only -0.30 (Spearman). da_rank_value is a 60-month lookback of shadow price magnitude (from V6.2B historical data). binding_freq is a 6-month lookback of binary binding occurrence (from realized DA). They capture complementary aspects of binding persistence.

### 4. Production viability

**YES — feature is producible.** Realized DA shadow prices are published by MISO daily. By the time V6.2B signal runs (~5th of month), all prior months' DA data is available. We can compute binding_freq_6 from cached realized DA parquets.

### 5. Verdict

**Legitimate signal, not leakage, but almost certainly overstated on dev.**

The improvement is real — binding persistence is a genuine physical phenomenon. But:
- The 30% VC@20 gain over formula likely includes substantial autocorrelation inflation
- Holdout validation is MANDATORY before updating champion or production plans
- If holdout shows <15% improvement over formula, the dev numbers were inflated
- If holdout shows >15%, binding_freq is a genuine breakthrough feature

**Recommended next steps:**
1. Run holdout (2024-2025) for v9 and v9c
2. Compare holdout degradation ratio (dev gain / holdout gain) with v6b's ratio
3. If validated, consider adding binding_freq as a production feature
4. Investigate binding_freq lookback windows (3, 12, 24 months) to find optimal
