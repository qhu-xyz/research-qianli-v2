# V10e: Multi-Window Binding Frequency + DA Rank

## Summary

v10e uses 9 features: 5 binding frequency windows (1/3/6/12/15 months), v7 formula score,
prob_exceed_110, constraint_limit, and da_rank_value. It beats v9 (14f) on ALL key metrics
while using fewer features.

## Configuration

- **Method**: LightGBM regression, tiered labels (0/1/2/3)
- **Features**: 9 (5 bf windows + v7_formula + prob_exceed_110 + constraint_limit + da_rank_value)
- **Training**: 8 months rolling, 0 validation
- **Eval**: 36 months (2020-06 to 2023-05)

## Features

| Feature | Monotone | Description |
|---------|----------|-------------|
| binding_freq_1 | +1 | Fraction of prior 1 month where constraint was binding |
| binding_freq_3 | +1 | Fraction of prior 3 months |
| binding_freq_6 | +1 | Fraction of prior 6 months |
| binding_freq_12 | +1 | Fraction of prior 12 months |
| binding_freq_15 | +1 | Fraction of prior 15 months (seasonal) |
| v7_formula_score | -1 | 0.85*da_rank + 0.15*density_ori_rank |
| prob_exceed_110 | +1 | Spice6 probability flow exceeds 110% |
| constraint_limit | 0 | MW thermal limit |
| da_rank_value | -1 | Historical DA shadow price percentile rank |

All bf features normalize by actual months available: `count / len(prior)`.
For bf_15, if only 12 months exist, it divides by 12.

## Dev Results (36 months)

| Metric | v9 (14f) | v10e (9f) | Delta |
|--------|----------|-----------|-------|
| VC@20 | 0.4475 | 0.4536 | +1.4% |
| VC@50 | 0.5994 | 0.6543 | +9.2% |
| VC@100 | 0.7445 | 0.7838 | +5.3% |
| Recall@20 | 0.3514 | 0.3597 | +2.4% |
| NDCG | 0.6067 | 0.6231 | +2.7% |
| Spearman | 0.3276 | 0.3482 | +6.3% |

v10e beats v9 on all 6 key metrics. Only loses on VC@200 (0.8593 vs 0.8629, deep tail).

## Key Design Decisions

1. **bf_1 (last month)**: Sharp recency signal. 43.5% importance in v10g variant.
   Captures "did this constraint bind LAST month" — the strongest single predictor.

2. **bf_15 (seasonal)**: Captures annual/seasonal binding patterns beyond the 12-month
   horizon. Many constraints bind seasonally (summer peaks, winter gas constraints).

3. **da_rank_value added back**: Was 6.1% importance in v9 but got pruned in v10.
   Adding it back to v10c (creating v10e) consistently improves VC@20 and Recall@20.
   Provides complementary signal for constraints that haven't bound recently but have
   high historical DA shadow prices.

4. **Pruned 5 features from v9**: mean_branch_max, ori_mean, mix_mean,
   density_mix_rank_value, density_ori_rank_value were all <1% importance.
   Removing them reduces noise and slightly improves generalization.
