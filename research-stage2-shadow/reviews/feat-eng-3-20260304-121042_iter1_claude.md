# Review — feat-eng-3-20260304-121042 iter1 (v0011)

**Reviewer**: Claude
**Date**: 2026-03-04
**Champion**: v0009 (39 features, 34 effective)
**Candidate**: v0011 (34 features — prune 5 dead features)

## Summary

v0011 removes 5 always-zero features (`hist_physical_interaction`, `overload_exceedance_product`, `band_severity`, `sf_exceed_interaction`, `hist_seasonal_band`) from the regressor, reducing the feature set from 39 nominal to 34 actual features. This is a clean, low-risk housekeeping change that was recommended by both reviewers in the previous batch. The screening correctly selected Hypothesis A (prune only) over Hypothesis B (prune + `flow_direction`) based on higher mean EV-VC@100.

The 12-month results show a mixed profile: **EV-VC@100 improved +5.2%** (the primary business metric), but **EV-VC@500 degraded -2.5%** and **EV-NDCG degraded -0.6%**. All Group A gates pass all three layers. The improvement pattern — sharper top-100 at the expense of broader top-500 — is consistent with the hypothesis that removing dead features improves `colsample_bytree` sampling efficiency, letting XGBoost focus on real signal for the highest-conviction predictions. However, the EV-VC@500 degradation and near-miss on its tail safety layer (1 month below tail_floor, exactly at the allowed limit) warrants attention.

## Gate-by-Gate Analysis

### Group A (Hard Gates)

| Gate | v0009 Mean | v0011 Mean | Delta | Delta% | L1 | L2 (tail fails) | L3 (bot2) | Overall |
|------|-----------|-----------|-------|--------|----|----|------|---------|
| EV-VC@100 | 0.0762 | 0.0801 | +0.0039 | **+5.2%** | P (floor=0.0664) | P (0 fails) | P (0.0111 vs champ 0.0065) | **P** |
| EV-VC@500 | 0.2329 | 0.2270 | -0.0059 | **-2.5%** | P (floor=0.2179, margin +4.2%) | P (1 fail, limit=1) | P (0.0541 vs champ 0.0718, delta=-0.0177) | **P** |
| EV-NDCG | 0.7548 | 0.7499 | -0.0048 | -0.6% | P (floor=0.7137, margin +5.1%) | P (0 fails) | P (0.6403 vs champ 0.6446, delta=-0.0043) | **P** |
| Spearman | 0.3910 | 0.3925 | +0.0015 | +0.4% | P (floor=0.3736, margin +5.1%) | P (0 fails) | P (0.2678 vs champ 0.2705, delta=-0.0027) | **P** |

### Group B (Monitor)

| Gate | v0009 Mean | v0011 Mean | Delta | Delta% | L1 | L2 | L3 | Overall |
|------|-----------|-----------|-------|--------|----|----|------|---------|
| C-RMSE | 2827.4 | 2866.6 | +39.2 | +1.4% | P | P (0 fails) | P (5292 vs champ 5337) | **P** |
| C-MAE | 1136.7 | 1142.5 | +5.8 | +0.5% | P | P (0 fails) | P (2086 vs champ 2105) | **P** |
| EV-VC@1000 | 0.3152 | 0.3040 | -0.0112 | -3.6% | P (barely, floor=0.3014) | P (0 fails) | P | **P** |
| R-REC@500 | 0.0356 | 0.0347 | -0.0009 | -2.5% | P (floor=0.0337) | P (1 fail) | P | **P** |

### Key Concerns

1. **EV-VC@500 L2 at limit**: 2022-09 (0.0527) is below tail_floor (0.0536). This is the sole tail failure, exactly at the `tail_max_failures=1` threshold. One more bad month would flip L2 to FAIL. The champion had 0 tail failures on this gate.

2. **EV-VC@500 L3 degradation**: bot2_mean dropped from 0.0718 to 0.0541, a delta of -0.0177. This passes L3 because `noise_tolerance=0.02`, but the margin is only 0.0023. This is a genuine regression in worst-case EV-VC@500 performance.

3. **EV-VC@1000 barely passing L1**: Mean 0.3040 vs floor 0.3014, margin only +0.9%. This is the tightest mean-quality margin across all gates.

4. **R-REC@500 gained a tail failure**: 1 month below tail_floor (at limit). Champion had 0.

## Seasonal/Monthly Analysis

### Month-by-Month Comparison (v0011 vs v0009, Group A)

| Month | EV-VC@100 | EV-VC@500 | EV-NDCG | Spearman |
|-------|-----------|-----------|---------|----------|
| 2020-09 | +29.1% | +6.4% | +3.4% | +0.2% |
| 2020-11 | -5.2% | +2.5% | -4.1% | +1.7% |
| 2021-01 | -21.4% | +10.6% | +1.1% | +0.0% |
| 2021-03 | -25.6% | -2.8% | -0.7% | +0.2% |
| 2021-05 | +183% | -36.9% | -1.2% | +0.2% |
| 2021-07 | +9.8% | -3.3% | -1.3% | -0.7% |
| 2021-09 | **+17.6%** | -0.9% | -0.5% | +1.9% |
| 2021-11 | +168% | -21.7% | +0.7% | -2.2% |
| 2022-03 | -2.9% | +4.7% | -1.1% | +1.2% |
| 2022-06 | +14.2% | -18.3% | -0.1% | +0.2% |
| 2022-09 | -6.1% | -14.6% | -1.1% | -0.4% |
| 2022-12 | -9.0% | -3.1% | -2.4% | +0.7% |

### Patterns

**EV-VC@100 improvement is concentrated**: The +5.2% mean gain is driven by 3 months with large improvements: 2021-05 (+183%, from 0.0012 to 0.0034 — tiny absolute values), 2021-11 (+168%, from 0.0118 to 0.0317), and 2021-09 (+17.6%, from 0.2000 to 0.2352). The remaining 9 months are mixed, with 5 months degrading.

**EV-VC@500 degradation is broad**: 7 of 12 months degrade. The worst are 2021-05 (-36.9%), 2021-11 (-21.7%), 2022-06 (-18.3%), and 2022-09 (-14.6%). This is a systematic pattern, not noise in 1-2 months.

**Weakest months (structurally persistent)**:
- **2022-06** and **2022-09** remain weak across all versions. These appear to be structurally difficult months (2022 summer/fall), likely reflecting a regime change or market structural shift.
- **2021-05** is consistently the worst EV-VC month across all versions, though v0011 slightly improved its EV-VC@100 (still near zero: 0.0034).
- **2021-11** has the worst Spearman (0.2616 in v0011 vs 0.2674 in v0009), and worsened further with pruning.

## Code Review

### Correctness: PASS

The code change is clean and correct:

1. **`_DEAD_FEATURES` set**: Properly defines the 5 features to remove. Using a set for O(1) lookup is appropriate.

2. **List comprehension filtering**: `_V1_CLF_FOR_REGRESSOR` and `_V1_CLF_MONO_FOR_REGRESSOR` correctly use `zip()` to keep features and monotone constraints synchronized when filtering.

3. **No classifier modification**: `_V1_CLF_FEATURES`, `_V1_CLF_MONOTONE`, and `ClassifierConfig` are untouched. Correct — the classifier is frozen.

4. **Feature count**: 29 V1 classifier features - 5 dead = 24, plus 10 additional = 34. The math is verified in the updated tests.

5. **Test updates**: Both `test_config.py` and `test_data_loader.py` feature count assertions correctly updated from 39 to 34.

### Minor Observation

The comment at line 92 says "Regressor features: ALL available (34)" — this is correct now but will need updating if features are added in future iterations. Consider using `len(_ALL_REGRESSOR_FEATURES)` in the comment to avoid staleness, or just remove the count from the comment.

## Regression Quality

| Metric | v0009 | v0011 | Delta |
|--------|-------|-------|-------|
| C-RMSE (mean) | 2827.4 | 2866.6 | +1.4% (worse) |
| C-MAE (mean) | 1136.7 | 1142.5 | +0.5% (worse) |
| Spearman (mean) | 0.3910 | 0.3925 | +0.4% (better) |

Marginal calibration degradation (C-RMSE/C-MAE slightly worse) but rank-order correlation (Spearman) slightly improved. Since the business objective prioritizes ranking quality over calibration, this is an acceptable tradeoff. The C-RMSE/C-MAE changes are well within Group B thresholds.

Notably, C-RMSE bot2_mean actually **improved** (5292 vs 5337 champion), meaning the worst-month regression accuracy got slightly better even as average accuracy got slightly worse. This suggests the pruning reduced overfitting on easy months while maintaining floor quality.

## EV Ranking Quality

The core tension in this version is **EV-VC@100 vs EV-VC@500**:

- **EV-VC@100 +5.2%**: Better concentration of value in the top-100 predictions. This directly benefits capital allocation for the highest-conviction positions.
- **EV-VC@500 -2.5%**: Broader portfolio coverage degraded. If the trading strategy extends beyond top-100, this matters.

The interpretation: pruning dead features sharpens the regressor's discrimination at the very top of the distribution but slightly weakens it in the 101-500 range. The `colsample_bytree=0.8` sampling efficiency improvement (from 31/39 to 27/34 useful features per tree) appears to help the regressor differentiate between "very high value" and "high value" constraints, but at the cost of some value redistribution from the 100-500 tier.

**For a business that concentrates capital in top-100 positions, EV-VC@100 +5.2% is the more impactful number.** The EV-VC@500 -2.5% is a tradeoff, not a free loss.

## Regressor Feature Importance

This iteration's change was feature pruning, not addition. The 5 removed features were confirmed zero-filled (never populated by the data loader). The pruning is unambiguously correct — these features contributed no signal and only consumed `colsample_bytree` sampling slots.

**Remaining feature set (34)**: 24 from classifier (filtered) + 10 additional. The feature set is now clean with no known dead features. Future iterations should consider:
- Feature importance analysis to identify low-signal features among the remaining 34
- Interaction features (e.g., `hist_da * prob_exceed_100`)
- Whether the additional regressor features (density_skewness, density_kurtosis, density_cv, season_hist_da_3, prob_below_85 added in prior batch) are pulling their weight

## Unified vs Gated Mode

Not tested in this iteration (batch constraint was feature-only). The gated mode remains default (`unified_regressor=False`). Given that EV-VC@100 is improving with gated mode, there is no urgent reason to switch. However, unified mode could potentially help EV-VC@500 by training on a broader sample — worth exploring in a future HP iteration.

## Statistical Rigor

With 12 eval months:
- **EV-VC@100**: 7 months improved, 5 degraded. Not a slam dunk, but the improvements are larger in magnitude.
- **EV-VC@500**: 5 months improved, 7 degraded. Systematic degradation — not noise.
- **EV-NDCG**: 2 months improved, 10 degraded. Nearly all months slightly worse, suggesting a broad (but small) ranking quality decrease.
- **Spearman**: 8 months improved, 4 degraded. Most consistent positive signal.

The improvement is **not** consistent across metrics. EV-VC@100 and Spearman improve; EV-VC@500 and EV-NDCG degrade. This is a genuine tradeoff, not an unambiguous win. The magnitude of EV-VC@100 improvement (+5.2%) outweighs the EV-VC@500 degradation (-2.5%) for the stated business objective (top-of-ranking quality), but this should be acknowledged as a tradeoff, not presented as purely positive.

## Gate Calibration

### Current Assessment

| Gate | v0011 Mean | Floor | Margin | Assessment |
|------|-----------|-------|--------|------------|
| EV-VC@100 | 0.0801 | 0.0664 | +20.6% | Loose — 20% headroom |
| EV-VC@500 | 0.2270 | 0.2179 | +4.2% | Tight but passing |
| EV-NDCG | 0.7499 | 0.7137 | +5.1% | Reasonable |
| Spearman | 0.3925 | 0.3736 | +5.1% | Was binding, now slightly looser |

### Suggestions

1. **EV-VC@500 tail_floor may be too tight**: The tail_floor of 0.0536 was set at 0.90x of v0007's worst month. With v0011 showing a tail failure at 0.0527, this gate is at its exact limit. If the business truly prioritizes top-100 over top-500, consider loosening the tail_floor slightly (e.g., 0.85x instead of 0.90x) to avoid blocking versions that trade EV-VC@500 tail for EV-VC@100 gains.

2. **L3 noise_tolerance scale-awareness**: Still an open issue from last batch. For EV-VC@100 (bot2 ~0.011), the tolerance of 0.02 is larger than the bot2_mean itself, making L3 meaningless. For EV-VC@500 (bot2 ~0.054), 0.02 is ~37% of the value, which is very generous. Recommend switching to `max(0.02, 0.15 * champ_bot2_mean)` as proposed previously.

3. **EV-VC@1000 floor at 0.3014 with v0011 at 0.3040**: Only 0.9% margin. If this gate were Group A, v0011 would be at risk. As Group B (monitor), it signals that broader value capture is degrading. Worth watching.

## Recommendations for Next Iteration

### Priority 1: Hyperparameter Tuning (Now Unblocked)
The batch constraint (FE-only) is lifted. The HPs were tuned for 29 effective features (v0007). With 34 effective features (v0011), consider:
- **`colsample_bytree`**: Currently 0.8 → ~27 features per tree. With 34 clean features, try 0.7 (24 features/tree) for more regularization, or 0.85 (29 features/tree) to match the old effective count.
- **`n_estimators`**: 400 may be underfitting with cleaner features. Try 500-600.
- **`min_child_weight`**: Currently 25. Try 15-20 to allow finer splits now that feature noise is reduced.
- **`learning_rate`**: Currently 0.05. If increasing n_estimators, reduce to 0.03-0.04.

### Priority 2: Address EV-VC@500 Degradation
The -2.5% EV-VC@500 is a real tradeoff. If the next HP iteration doesn't recover EV-VC@500 alongside EV-VC@100, investigate:
- Whether higher tree count helps the 100-500 tier
- Whether value weighting (`value_weighted=True`) improves calibration for mid-tier constraints

### Priority 3: Feature Importance Audit
With the feature set now clean, run feature importance analysis to identify the bottom 3-5 contributors. Candidates for future pruning if they add noise without signal.

### Not Recommended
- **Unified mode**: Not yet — gated mode is producing improving EV-VC@100 results. Explore after HP tuning.
- **New features**: The feature set has grown from 24 to 34 across the last two batches. Let the HP tuning settle before adding more.

## Verdict

**v0011 is a valid candidate for promotion.** All Group A gates pass all three layers. The EV-VC@100 +5.2% improvement is material and aligns with the business objective. The EV-VC@500 -2.5% degradation is a genuine tradeoff but acceptable given the stated priority of top-of-ranking quality.

**Risk factors to acknowledge**:
- EV-VC@500 L2 at exact limit (1 tail failure, max allowed is 1)
- EV-VC@500 L3 margin is thin (0.0023 remaining tolerance)
- EV-VC@1000 L1 margin is only 0.9%
- The improvement is concentrated in a few months, not broadly distributed

The next iteration should focus on HP tuning to see if both EV-VC@100 and EV-VC@500 can be improved simultaneously.
