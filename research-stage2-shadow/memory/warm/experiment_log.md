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

---

## Iter 1 — feat-eng-3-20260304-121042

**Version**: v0011 (34 features — pruned 5 dead features from v0009's 39)
**Champion**: v0009 (39 nominal / 34 effective features)
**Hypothesis tested**: Prune 5 always-zero features (hist_physical_interaction, overload_exceedance_product, band_severity, sf_exceed_interaction, hist_seasonal_band)
**Batch constraint**: Feature engineering / selection only

### Screening (2 months: 2022-09 weak, 2021-09 strong)

| Metric | Month | v0009 | Hyp A (34, prune) | Hyp B (35, prune+flow_dir) |
|--------|-------|-------|-------------------|---------------------------|
| EV-VC@100 | 2022-09 | 0.030 | 0.028 (-6.1%) | 0.028 (-7.2%) |
| EV-VC@100 | 2021-09 | 0.200 | 0.235 (+17.6%) | 0.220 (+10.0%) |
| Spearman | 2022-09 | 0.330 | 0.328 (-0.4%) | 0.332 (+0.6%) |
| Spearman | 2021-09 | 0.408 | 0.416 (+1.9%) | 0.409 (+0.2%) |

Winner: Hypothesis A (mean EV-VC@100=0.1316 vs B=0.1238, +6.3%). No Spearman veto.

### Full 12-Month Results (v0011 vs v0009)

| Metric | v0009 | v0011 | Delta | % |
|--------|-------|-------|-------|---|
| EV-VC@100 | 0.0762 | 0.0801 | +0.0039 | **+5.2%** |
| EV-VC@500 | 0.2329 | 0.2270 | -0.0059 | **-2.5%** |
| EV-NDCG | 0.7548 | 0.7499 | -0.0048 | -0.6% |
| Spearman | 0.3910 | 0.3925 | +0.0015 | +0.4% |
| C-RMSE | 2827.4 | 2866.6 | +39.2 | +1.4% |
| C-MAE | 1136.7 | 1142.5 | +5.8 | +0.5% |

### Gate Status: ALL PASS (3 layers) — but tight margins

| Gate | L1 (Mean) | L2 (Tail, max 1 fail) | L3 (bot2 >= champ-0.02) | Overall |
|------|-----------|----------------------|-------------------------|---------|
| EV-VC@100 | P (+20.6%) | P (0 fails) | P (0.0111 vs -0.0135 thresh) | **P** |
| EV-VC@500 | P (+4.2%) | P (1 fail, AT LIMIT) | P (0.0541 vs 0.0518, margin +0.0023) | **P** |
| EV-NDCG | P (+5.1%) | P (0 fails) | P (0.6403 vs 0.6246) | **P** |
| Spearman | P (+5.1%) | P (0 fails) | P (0.2678 vs 0.2505) | **P** |

### Key Observations
1. **Precision-vs-breadth tradeoff**: EV-VC@100 +5.2% but EV-VC@500 -2.5%. Pruning dead features sharpens top-100 at the expense of top-500 coverage.
2. **EV-VC@500 tail at limit**: 2022-09 (0.0527) is the sole tail failure, barely below tail_floor (0.0536). One more bad month would flip L2.
3. **EV-VC@500 L3 margin razor-thin**: bot2 dropped from 0.0718 to 0.0541, margin only +0.0023 above threshold.
4. **EV-VC@100 gains concentrated**: Driven by 2021-05 (+183%), 2021-11 (+168%), 2021-09 (+17.6%) — but 5/12 months degraded.
5. **flow_direction added no signal over pruning alone** — Hyp B lost screen cleanly.

### Decision: PROMOTE v0011
- All Group A gates pass all 3 layers
- EV-VC@100 +5.2% is material on primary business metric
- Spearman +0.4% is a slight bonus
- EV-VC@500 degradation is a genuine tradeoff, acknowledged, not hidden
- Code change is clean (dead feature removal, no risk)
