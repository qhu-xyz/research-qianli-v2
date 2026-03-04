# Experiment Log

## Iter 1 — feat-eng-3-20260304-102111

**Version**: v0009 (39 features, duplicate of v0008)
**Champion**: v0007 (34 features)
**Hypothesis**: Add 5 distributional/raw features (density_skewness, density_kurtosis, density_cv, season_hist_da_3, prob_below_85)
**Batch constraint**: Feature engineering only, no HP changes

### Screening (2 months: 2022-06 weak, 2022-12 strong)

| Metric | Champion | Hyp A (37) | Hyp B (39) |
|--------|----------|------------|------------|
| Mean EV-VC@100 | 0.103 | 0.072 | **0.095** |
| Mean EV-NDCG | 0.710 | 0.689 | **0.706** |

Winner: Hypothesis B (+32% EV-VC@100 vs A). Spearman safety: within ±0.003 on both months.

### Full 12-Month Results (v0009 vs v0007)

| Metric | v0007 | v0009 | Delta | % |
|--------|-------|-------|-------|---|
| EV-VC@100 | 0.0699 | 0.0762 | +0.0063 | **+9.0%** |
| EV-VC@500 | 0.2294 | 0.2329 | +0.0035 | **+1.5%** |
| EV-NDCG | 0.7513 | 0.7548 | +0.0035 | **+0.5%** |
| Spearman | 0.3932 | 0.3910 | -0.0022 | -0.6% |
| C-RMSE | 2916.4 | 2827.4 | -89.0 | **-3.1%** |
| C-MAE | 1151.5 | 1136.7 | -14.8 | **-1.3%** |

### Gate Status: ALL PASS (3 layers)
- EV-VC@100: L1 P, L2 P (0 tail fails), L3 P (bot2 -0.0006, within tolerance)
- EV-VC@500: L1 P, L2 P (0 tail fails), L3 P (bot2 +0.0056)
- EV-NDCG: L1 P, L2 P (0 tail fails), L3 P (bot2 -0.0056, margin +0.0145)
- Spearman: L1 P, L2 P (0 tail fails), L3 P (bot2 +0.0036)

### Decision: PROMOTE v0009
- All Group A gates pass all 3 layers
- EV-VC@100 +9% is a material improvement on the primary business metric
- Spearman -0.6% is within noise and tail actually improved

### Process Note
v0009 is byte-identical to v0008 (registered in an earlier partial run). Duplicate version counter increment — no new empirical information vs v0008.
