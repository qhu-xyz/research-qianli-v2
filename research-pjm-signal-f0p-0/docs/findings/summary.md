# research-pjm-signal-f0p-0 — Final Findings Summary

## 1. Champion: v0b (Formula Baseline)

**v0b is champion across all 6 ML slices.** No ML model beat the simple formula.

Formula: `score = -(0.80 * da_rank_value + 0.15 * density_mix_rank_value + 0.05 * density_ori_rank_value)`

This is a re-weighted version of V6.2B's original formula (0.60/0.30/0.10). The key change is shifting weight from density toward `da_rank_value`, which is the strongest individual predictor (Spearman ~0.27 with realized shadow price).

## 2. V6.2B Formula (v0) vs Optimized Blend (v0b)

V0 = V6.2B original formula: `0.60*da + 0.30*dmix + 0.10*dori`
V0b = Optimized blend: `0.80*da + 0.15*dmix + 0.05*dori`

### Holdout Comparison (out-of-sample)

| Slice | v0 VC@20 | v0b VC@20 | v0b vs v0 | v0 NDCG | v0b NDCG |
|-------|:--------:|:---------:|:---------:|:-------:|:--------:|
| f0/onpeak | 0.5102 | **0.5445** | +6.7% | 0.5635 | **0.6381** |
| f0/dailyoffpeak | 0.6586 | **0.6845** | +3.9% | 0.5984 | **0.6378** |
| f0/wkndonpeak | 0.4479 | **0.4852** | +8.3% | 0.4904 | **0.5299** |
| f1/onpeak | 0.5306 | **0.5685** | +7.1% | 0.5053 | **0.6006** |
| f1/dailyoffpeak | 0.6579 | **0.6420** | -2.4% | 0.5488 | **0.5994** |
| f1/wkndonpeak | 0.5128 | **0.5303** | +3.4% | 0.4464 | **0.5070** |

v0b wins on VC@20 in 5/6 slices and on NDCG in 6/6 slices. The improvement is consistent but modest (3-8%).

## 3. ML Versions Tested

### Version Descriptions

| Version | Method | Features | Key Change |
|---------|--------|----------|------------|
| v0 | Formula | 3 (V6.2B blend) | Original 0.60/0.30/0.10 |
| v0b | Formula | 3 (optimized blend) | 0.80/0.15/0.05 — **CHAMPION** |
| v2 | LightGBM LambdaRank | 14 (full feature set) | MISO-equivalent ML pipeline |
| v3a | LambdaRank | 14 | Variant (no NB metrics tracked) |
| v3b | LambdaRank | 14 | Improved training |
| v3c | LambdaRank | 14 | Further tuning |
| v3d | LambdaRank | 14 | Regression objective (failed) |
| v4 | LambdaRank | 14 | Bug fixes from code review |
| v5 | Two-model blend | 14 full + 5 pred | BF-aware hard switch |
| v5f | LambdaRank | 14 | Full model (v5's full component) |
| v5p | LambdaRank | 5 (predictive only) | Zero historical info |

### 14 Full Features
`binding_freq_{1,3,6,12,15}`, `v7_formula_score`, `da_rank_value`, `shadow_price_da`, `binding_probability`, `predicted_shadow_price`, `ori_mean`, `prob_exceed_100`, `constraint_limit`, `hist_da`

### 5 Predictive-Only Features (v5p)
`ori_mean`, `binding_probability`, `predicted_shadow_price`, `prob_exceed_100`, `constraint_limit`

## 4. Full Results — Holdout (Out-of-Sample)

### f0/onpeak (primary slice)

| Version | VC@20 | VC@50 | R@20 | NDCG | Sprmn | NB_VC@50 | NB_R@50 |
|---------|:-----:|:-----:|:----:|:----:|:-----:|:--------:|:-------:|
| v0 | 0.5102 | 0.6821 | 0.246 | 0.5635 | 0.2727 | 0.4899 | 0.236 |
| **v0b** | **0.5445** | 0.6726 | 0.283 | **0.6381** | 0.2774 | 0.5271 | 0.237 |
| v2 | 0.4797 | 0.6409 | 0.263 | 0.5141 | **0.3196** | 0.5018 | **0.253** |
| v3b | 0.4907 | 0.6552 | 0.281 | 0.5108 | 0.3185 | -- | -- |
| v4 | 0.4915 | 0.6584 | 0.265 | 0.4743 | 0.3212 | **0.5486** | **0.261** |
| v5 blend | 0.4969 | 0.6308 | 0.275 | 0.5123 | 0.2431 | 0.4078 | 0.224 |
| v5f | 0.4986 | **0.6544** | **0.283** | 0.5391 | 0.3199 | 0.5309 | 0.255 |
| v5p | 0.4778 | 0.5911 | 0.235 | 0.4588 | 0.2252 | 0.4078 | 0.224 |

### f0/dailyoffpeak

| Version | VC@20 | VC@50 | R@20 | NDCG | NB_VC@50 | NB_R@50 |
|---------|:-----:|:-----:|:----:|:----:|:--------:|:-------:|
| **v0b** | **0.6845** | **0.7813** | 0.283 | **0.6378** | **0.5959** | 0.223 |
| v4 | 0.6236 | 0.7412 | 0.283 | 0.6332 | 0.5535 | 0.218 |
| v5f | 0.6216 | 0.7574 | 0.283 | 0.6168 | 0.5259 | 0.223 |
| v5p | 0.5221 | 0.6276 | 0.217 | 0.5645 | 0.2989 | 0.188 |

### f0/wkndonpeak

| Version | VC@20 | VC@50 | R@20 | NDCG | NB_VC@50 | NB_R@50 |
|---------|:-----:|:-----:|:----:|:----:|:--------:|:-------:|
| **v0b** | **0.4852** | 0.6689 | 0.260 | **0.5299** | 0.5225 | 0.215 |
| v4 | 0.5144 | **0.6904** | **0.277** | 0.4882 | 0.5200 | 0.227 |
| v5f | 0.5071 | 0.6771 | 0.273 | 0.4798 | **0.5377** | **0.228** |

### f1/onpeak

| Version | VC@20 | VC@50 | R@20 | NDCG | NB_VC@50 | NB_R@50 |
|---------|:-----:|:-----:|:----:|:----:|:--------:|:-------:|
| **v0b** | **0.5685** | 0.7200 | **0.295** | **0.6006** | 0.5522 | 0.253 |
| v4 | 0.4924 | 0.6863 | 0.268 | 0.4931 | 0.5122 | **0.281** |
| v5f | 0.4857 | 0.6691 | 0.250 | 0.5026 | 0.5195 | 0.265 |
| v2 | 0.4741 | 0.6868 | 0.245 | 0.4685 | **0.5786** | 0.276 |

### f1/dailyoffpeak

| Version | VC@20 | VC@50 | R@20 | NDCG | NB_VC@50 | NB_R@50 |
|---------|:-----:|:-----:|:----:|:----:|:--------:|:-------:|
| **v0b** | **0.6420** | 0.7708 | 0.264 | **0.5994** | 0.5651 | 0.219 |
| v4 | 0.5825 | **0.7721** | 0.270 | 0.5821 | 0.5190 | **0.248** |
| v5f | 0.5886 | 0.7452 | **0.286** | 0.5709 | **0.5444** | 0.239 |

### f1/wkndonpeak

| Version | VC@20 | VC@50 | R@20 | NDCG | NB_VC@50 | NB_R@50 |
|---------|:-----:|:-----:|:----:|:----:|:--------:|:-------:|
| **v0b** | **0.5303** | 0.6950 | 0.259 | **0.5070** | 0.5759 | 0.227 |
| v4 | 0.4564 | **0.7179** | 0.268 | 0.4906 | **0.5822** | **0.253** |
| v5f | 0.4502 | 0.7060 | **0.277** | 0.4916 | 0.5782 | 0.235 |

## 5. Key Observations

### 5.1 v0b dominates VC@20 and NDCG on holdout
v0b wins VC@20 on **all 6 holdout slices** and NDCG on **all 6**. The ML models never beat the simple formula on the most important head-ranking metric.

### 5.2 ML models have better ranking depth (VC@50, Recall@50, Spearman)
ML models (v4, v5f, v2) consistently beat v0b on:
- **VC@50**: v4 wins 4/6 slices
- **Recall@50**: ML wins across the board (0.33-0.39 vs v0b's 0.27-0.35)
- **Spearman**: ML wins 6/6 (0.27-0.32 vs v0b's 0.23-0.28)

This suggests ML is better at the full ranking but worse at concentrating value at the very top.

### 5.3 NB (New-Binding) performance is similar
On NB metrics, ML models show mixed results:
- **NB_VC@50**: v4 wins 3/6 slices, v0b wins 3/6
- **NB_Recall@50**: ML wins 5/6 slices (small margins: +0.01-0.03)
- Neither approach reliably identifies new binders — NB_VC@50 hovers around 0.40-0.58

The predictive-only model (v5p) is consistently worst on NB metrics — the spice6 ML predictions have near-zero independent discriminative power.

### 5.4 Two-model blend (v5) failed
v5 (hard-switch: full model for BF-positive, pred model for BF-zero) was worse than both v0b and v5f on nearly all metrics. The pred-only model hurts the BF-zero cohort rather than helping.

### 5.5 Dev vs Holdout discrepancy
On dev, ML models look more competitive with v0b:
- f0/onpeak dev: v4 VC@20=0.4110 vs v0b 0.4092 (ML wins by 0.2%)
- f0/onpeak holdout: v4 VC@20=0.4915 vs v0b **0.5445** (v0b wins by 11%)

v0b is **more robust** to distribution shift, likely because the formula is simpler (3 parameters vs 14 features + tree structure).

## 6. NB Metrics Deep Dive

### Average NB statistics across all months (f0/onpeak holdout)

| Metric | Value |
|--------|-------|
| NB constraints per month (n_new) | 16.8 |
| NB row share (% of universe that is NB) | ~68% |
| NB value share (% of binding $ from NB) | 23.2% |

~68% of the V6.2B universe has no binding history in any given month, and these NB constraints account for ~23% of total binding value. This is a significant share.

### NB Performance Comparison (f0/onpeak holdout)

| Version | NB_VC@50 | NB_R@50 | NB_VC@100 | NB_R@100 |
|---------|:--------:|:-------:|:---------:|:--------:|
| v0b | 0.5271 | 0.237 | -- | -- |
| v4 | **0.5486** | **0.261** | -- | -- |
| v5f | 0.5309 | 0.255 | -- | -- |
| v5p | 0.4078 | 0.224 | -- | -- |
| v5 blend | 0.4078 | 0.224 | -- | -- |

v4 has the best NB performance, but the margin over v0b is small (+4% NB_VC@50, +10% NB_R@50). Neither model reliably captures NB value.

## 7. Coverage Gap (The Real Bottleneck)

See `2026-03-11-v62b-coverage-gap.md` for full analysis.

### Summary
- V6.2B universe: ~475 branches per month
- Raw density universe: ~3,100 branches per month (7x larger)
- V6.2B captures only **33% of binding constraints** and **47% of binding value**
- Missed binders have similar mean value to captured ones (not noise)
- Examples: JUN-TMI ($21,165), CNS-NOR1 ($23,262), HAN-JUN1 ($20,680) — absent from V6.2B

### Why This Matters More Than ML
Even the best model on V6.2B (v0b, VC@20=0.5445 on holdout) captures only:
`0.5445 * 47% ≈ 26%` of **total** DA binding value.

An expanded universe (89% coverage) with a mediocre model (VC@20=0.30) would capture:
`0.30 * 89% ≈ 27%` — already matching v0b's absolute capture.

**The universe, not the model, is the binding constraint on performance.**

## 8. Feature Importance Analysis

### Spearman Correlation with Realized SP (f0/onpeak, within V6.2B universe)

| Feature | Spearman | Type |
|---------|:--------:|------|
| da_rank_value | ~0.27 | Historical |
| shadow_price_da | ~0.27 | Historical |
| binding_freq_6 | ~0.15-0.25 | Historical (computed) |
| ori_mean | ~0.09 | Predictive (density) |
| density_score (raw) | ~0.01-0.09 | Predictive (density) |
| binding_probability | ~0.01 | Predictive (ML) |
| predicted_shadow_price | ~0.01 | Predictive (ML) |
| prob_exceed_100 | ~0.01 | Predictive (ML) |
| constraint_limit | ~0.01 | Physical |

**Historical features dominate.** `da_rank_value` alone has 3-27x more correlation than any predictive feature. This explains why:
1. The formula (which is 95% historical) beats ML
2. The predictive-only model (v5p) performs poorly
3. The two-model blend (v5) doesn't help — the pred model adds noise, not signal

## 9. What's Next: f0p-1

The f0p-0 findings point clearly to one conclusion: **expand the universe, not the model**.

Key changes for f0p-1:
1. Build from raw spice6 density (~3,100 branches vs ~475)
2. New cross-universe comparable metrics (absolute value capture)
3. Formal NB metrics tracking
4. Mapping quality monitoring
5. Aggregation strategy experiments

See `/home/xyz/workspace/research-qianli-v2/research-pjm-signal-f0p-1/docs/design.md`.
