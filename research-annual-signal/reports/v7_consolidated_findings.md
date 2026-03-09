# Annual Signal: Consolidated Findings (v7 Rebase)

**Date**: 2026-03-09
**Scope**: ML rebase on v0b stronger baseline + blending strategies

## Executive Summary

The original ML narrative overstated the value of machine learning for annual FTR constraint ranking. After discovering that the V6.1 formula's density components (30% density_mix + 10% density_ori) actively hurt performance, the properly-tuned formula baseline (v0b: pure da_rank_value) captures most of the gains previously attributed to ML. The best blended approach adds only +3.9% VC@20 over v0b.

## Results Overview

### All Versions Comparison (mean over 12 groups, 2022-2024)

| Version | Type | VC@20 | VC@100 | Recall@20 | Recall@50 | NDCG | Spearman |
|---------|------|-------|--------|-----------|-----------|------|----------|
| v0 | Formula (0.60/0.30/0.10) | 0.2329 | 0.6573 | 0.2167 | 0.3817 | 0.5889 | 0.3392 |
| **v0b** | **Formula (pure DA rank)** | **0.2997** | **0.6879** | **0.3208** | **0.4600** | **0.6028** | **0.3678** |
| v1 | ML Set A (6f), rank | 0.2934 | 0.6854 | 0.2708 | 0.4383 | 0.6071 | 0.3642 |
| v5 | ML Set AF (7f), tiered | 0.3075 | 0.6792 | 0.3208 | 0.4367 | 0.6098 | 0.3695 |
| v7a | ML Set A (6f), tiered | 0.3024 | 0.6708 | 0.3125 | 0.4400 | 0.6043 | 0.3612 |
| v7b | ML Lean (4f), tiered | 0.3072 | 0.6690 | 0.3208 | 0.4433 | 0.6012 | 0.3643 |
| v7c | ML Lean+da_rank (5f), tiered | 0.3025 | 0.6928 | 0.3208 | 0.4583 | 0.6026 | 0.3704 |
| v7d | ML Set A+da_rank (7f), tiered | 0.3033 | 0.6797 | 0.3125 | 0.4600 | 0.5997 | 0.3697 |
| **blend_v7d_a70** | **Score blend (70% ML)** | **0.3113** | **0.6935** | **0.3417** | **0.4533** | **0.5987** | **0.3715** |
| blend_v7d_rank | Rank blend | 0.3033 | 0.6958 | 0.3292 | 0.4617 | 0.6109 | 0.3657 |
| blend_v1_a50 | Score blend v1 (50%) | 0.3029 | 0.6921 | 0.3167 | 0.4567 | 0.6128 | 0.3645 |
| blend_v7d_rrf | RRF | 0.3020 | 0.6934 | 0.3292 | 0.4633 | 0.6105 | 0.3653 |

### Delta vs v0b (properly-tuned formula baseline)

| Version | VC@20 delta | Recall@20 delta | NDCG delta |
|---------|-------------|-----------------|------------|
| v1 (ML, rank) | -2.1% | -15.6% | +0.7% |
| v5 (ML, tiered) | +2.6% | 0.0% | +1.2% |
| v7b (lean 4f) | +2.5% | 0.0% | -0.3% |
| v7d (full+da_rank) | +1.2% | -2.6% | -0.5% |
| **blend_v7d_a70** | **+3.9%** | **+6.5%** | **-0.7%** |

## Key Findings

### 1. Density Features Actively Hurt
Grid search over formula weights found optimal = pure da_rank_value (alpha=1.0). The 30% density_mix + 10% density_ori in V6.1's formula add noise that degrades ranking quality by -28.7% on VC@20.

### 2. ML Marginal Value is Small
Against the proper v0b baseline:
- Best standalone ML (v5): +2.6% VC@20
- Best blend (score_blend_v7d_a70): +3.9% VC@20, +6.5% Recall@20
- Original claim (v1 vs v0): +26% was misleading — most of that gain came from the formula being suboptimal, not from ML learning new patterns

### 3. Feature Count Doesn't Matter for ML
- v7b (4 features, lean) matches v5 (7 features) on VC@20 (0.3072 vs 0.3075)
- Adding da_rank_value as an explicit feature doesn't help (v7c, v7d ≈ v7a, v7b)
- The LightGBM model essentially learns to weight shadow_price_da heavily — same thing as v0b

### 4. Blending Provides Modest Improvement
- Score blend (70% ML, 30% formula) is the best strategy: +3.9% VC@20
- Rank blend and RRF don't improve over either component alone
- ML-heavy blends (a70) work better than formula-heavy (a30) — ML adds some edge at the top-k

### 5. Gate Compliance Remains Challenging
- No version passes all v0b-calibrated gates
- Root cause: Recall@100 tail regression — sharpening top-k necessarily trades off worst-case breadth
- This is a fundamental tradeoff, not a model deficiency

## Recommendation

**For production**: Use v0b (pure da_rank_value) as the primary signal. It's simpler, requires no ML training, and captures 96% of the best possible ranking quality.

**If pursuing ML**: The best option is score_blend_v7d_a70 (+3.9% VC@20), but the complexity vs. value tradeoff is marginal. This should be validated on holdout before committing.

**Action item for V6.1 signal**: The density components should be removed or downweighted in the annual formula. They may still be useful for non-annual period types (monthly showed positive density contribution).

## Appendix: Version Descriptions

| Version | Features | Labels | Notes |
|---------|----------|--------|-------|
| v0 | Formula: 0.60*da_rank + 0.30*density_mix + 0.10*density_ori | - | Original V6.1 |
| v0b | Formula: pure da_rank_value | - | Best formula (grid search) |
| v1 | shadow_price_da, mean_branch_max, ori/mix_mean, density_mix/ori_rank | rank | First ML |
| v5 | v1 features + rank_ori | tiered | Best ML (original) |
| v7a | Same as v1 (Set A, 6f) | tiered | Tiered labels, no formula feature |
| v7b | shadow_price_da, mean_branch_max, ori_mean, mix_mean (4f) | tiered | Lean — drops density |
| v7c | v7b + da_rank_value (5f) | tiered | Lean + clean formula signal |
| v7d | v7a + da_rank_value (7f) | tiered | Full + clean formula signal |
