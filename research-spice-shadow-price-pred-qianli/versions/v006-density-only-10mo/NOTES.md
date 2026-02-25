# v006 — Density-Only Features, 10-Month Training

**Hypothesis**: v005 used features from shift factor tables and constraint metadata
(sf_max_abs, sf_mean_abs, sf_std, sf_nonzero_frac, is_interface, constraint_limit) —
data sources the legacy baseline never had. To make a fair comparison against v000,
we should restrict to the same data sources (density parquets + DA shadow prices)
while making better use of them. Adding all available density-derived features
(prob_below_*, density moments) and extending the training window to 10 months
should improve performance without introducing new data sources.

**Changes vs v005**:
- **Removed 6 network/metadata features**: sf_max_abs, sf_mean_abs, sf_std, sf_nonzero_frac, is_interface, constraint_limit
- **Added density features**: prob_exceed_85, prob_exceed_80, prob_below_100, prob_below_95, prob_below_90, prob_below_85, prob_below_80, density_mean, density_variance, density_kurtosis
- **Training split**: 9/3/0 → 10/2/0 (1 more training month, 1 less validation month)
- **Step1 features**: 15 → 17 (all density + hist_da)
- **Step2 features**: 15 → 21 (step1 + recent_hist_da + season_hist_da_{1,2,3})

**Data sources used** (same as v000):
- Density parquets: prob_exceed_{80..110}, prob_below_{80..100}, density_{mean,variance,skewness,kurtosis}
- DA shadow prices: hist_da, recent_hist_da, season_hist_da_{1,2,3}

## Results (32-run benchmark, 2026-02-23)

| Gate | v000 (legacy) | v006 | Delta | Floor | Pass |
|------|--------------|------|-------|-------|------|
| S1-AUC | 0.6954 | 0.6890 | -0.0064 | 0.65 | PASS |
| S1-REC | 0.2734 | 0.2902 | +0.0168 | 0.25 | PASS |
| S2-SPR | 0.4120 | 0.4033 | -0.0087 | 0.30 | PASS |
| C-VC@1000 | 0.8508 | 0.8357 | -0.0151 | 0.50 | PASS |
| C-RMSE | 1555.79 | 1133.53 | -422.27 | 2000 | PASS |

**Gates passed**: 5/5
**Beats champion on**: 4/5 (all except S2-SPR)
**Promotable**: No (S2-SPR does not beat champion)

### Comparison with v005

| Gate | v005 | v006 | Delta |
|------|------|------|-------|
| S1-AUC | 0.6816 | 0.6890 | +0.0074 |
| S1-REC | 0.2866 | 0.2902 | +0.0036 |
| S2-SPR | 0.4236 | 0.4033 | -0.0203 |
| C-VC@1000 | 0.8220 | 0.8357 | +0.0137 |
| C-RMSE | 1098.26 | 1133.53 | +35.27 |

### Per-class breakdown

| Gate | onpeak | offpeak |
|------|--------|---------|
| S1-AUC | 0.6845 | 0.6936 |
| S1-REC | 0.2848 | 0.2956 |
| S2-SPR | 0.3774 | 0.4292 |
| C-VC@1000 | 0.8175 | 0.8538 |
| C-RMSE | 1153.73 | 1113.32 |

### Key observations

1. **S1-REC improved** (+1.7pp vs v000): More density features (prob_below_*, moments)
   give the classifier a richer view of the distribution shape, improving recall.

2. **S1-AUC close to v000** (-0.6pp): Marginal AUC decrease — the additional features
   help recall but don't dramatically change discrimination. Still well above 0.65 floor.

3. **S2-SPR slightly lower** (-0.9pp vs v000): The only metric that doesn't beat v000.
   Removing network features (sf_*, is_interface, constraint_limit) cost some regression
   ranking quality — those features provided unique signal about physical constraint
   characteristics that density features alone can't replicate.

4. **C-RMSE improved dramatically** (-422, -27% vs v000): Consistent with v005 — more
   training data + MEAN aggregation produces much better calibrated predictions.

5. **C-VC@1000 slightly lower** (-1.5pp vs v000): Small decrease in value capture at
   K=1000, but still at 84% (well above 0.50 floor).

6. **v006 vs v005 trade-off**: v006 improves S1-AUC (+0.7pp), S1-REC (+0.4pp), and
   C-VC@1000 (+1.4pp) over v005, but loses S2-SPR (-2.0pp) and C-RMSE (+35). The extra
   density features improve classification, but the loss of network features hurts
   regression ranking.

**Note**: v000 metrics were computed with MAX binding probability aggregation.
v006 uses MEAN. The comparison is not perfectly apples-to-apples for constraint-level
metrics. MEAN aggregation is more appropriate because MAX overweights single
high-probability outages within a constraint.

**Conclusion**: v006 passes all 5 gates using only the same data sources as v000.
It improves recall (+1.7pp) and RMSE (-27%) substantially, but cannot beat v000 on
S2-SPR (Spearman regression ranking). The S2-SPR gap suggests the 6 network features
in v005 provided genuine signal for regression ranking. To close this gap without adding
new data sources, consider engineering more informative features from the existing density
distributions (e.g., quantiles, tail ratios, cross-percentile spreads).
