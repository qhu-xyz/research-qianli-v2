# Claude Review — feat-eng-2-20260303-092848 iter1 (v0007)

## Summary

v0007 is a **breakthrough result** — the most significant improvement in 7 real-data experiments. Adding 6 shift factor and constraint metadata features (13→19 total) produced AUC +0.0137 (12W/0L, p≈0.0002) and AP +0.0455 (11W/1L, p≈0.006), shattering the AUC ceiling at ~0.836 that held across the previous 6 experiments. The AP bottom-2-mean trend, which had been monotonically worsening for 6 consecutive experiments (from -0.0017 to -0.0094 vs v0), was dramatically reversed to +0.0363. BRIER also improved unexpectedly (-0.0108), breaking the 6-experiment narrowing trend.

The one material concern is **NDCG**: the mean is neutral (0.7333 vs 0.7333) but the distribution shifted adversely (5W/7L), and the bottom-2-mean regressed by -0.0154, leaving only **0.0046 margin** to Layer 3 failure. This is the closest any Group A metric has come to failing across all experiments. Despite this, v0007 passes all three layers on all four Group A gates and is clearly **promotion-worthy**.

## Gate-by-Gate Analysis (Group A)

### Three-Layer Detail

| Gate | v0007 Mean | v0 Mean | Δ Mean | Floor | L1 Headroom | v0007 Bot2 | v0 Bot2 | Δ Bot2 | L3 Margin | All Pass? |
|------|-----------|---------|--------|-------|-------------|-----------|---------|--------|-----------|-----------|
| **S1-AUC** | 0.8485 | 0.8348 | **+0.0137** | 0.7848 | +0.064 | 0.8188 | 0.8105 | +0.0083 | +0.028 | **YES** |
| **S1-AP** | 0.4391 | 0.3936 | **+0.0455** | 0.3436 | +0.096 | 0.3685 | 0.3322 | **+0.0363** | +0.056 | **YES** |
| **S1-VCAP@100** | 0.0247 | 0.0149 | **+0.0098** | -0.0351 | +0.060 | 0.0094 | 0.0014 | +0.0080 | +0.028 | **YES** |
| **S1-NDCG** | 0.7333 | 0.7333 | +0.0000 | 0.6833 | +0.050 | 0.6562 | 0.6716 | **-0.0154** | **0.005** | **YES** (barely) |

### Per-Month Win/Loss Analysis

| Gate | W/L/T | p (sign test) | Consistency | Notes |
|------|-------|--------------|-------------|-------|
| S1-AUC | **12W/0L** | 0.0002 | Unanimous | First 12/12 in all experiments. Min Δ = +0.003 (2022-12) |
| S1-AP | **11W/1L** | 0.006 | Near-unanimous | Only loss: 2021-12 (-0.010) |
| S1-VCAP@100 | 9W/3L | 0.073 | Strong | Losses in 2021-04, 2022-03, 2022-06 |
| S1-NDCG | 5W/7L | 0.387 | **Weak** | Broadly distributed regression, not outlier-driven |

### Per-Month Detail (AUC)

| Month | v0007 | v0 | Δ |
|-------|-------|-----|------|
| 2020-09 | 0.854 | 0.843 | +0.011 |
| 2020-11 | 0.841 | 0.830 | +0.011 |
| 2021-01 | 0.864 | 0.856 | +0.009 |
| 2021-04 | 0.844 | 0.835 | +0.009 |
| 2021-06 | 0.846 | 0.825 | **+0.021** |
| 2021-08 | 0.872 | 0.853 | **+0.019** |
| 2021-10 | 0.865 | 0.851 | +0.015 |
| 2021-12 | 0.826 | 0.812 | +0.014 |
| 2022-03 | 0.854 | 0.845 | +0.010 |
| 2022-06 | 0.851 | 0.826 | **+0.026** |
| 2022-09 | 0.853 | 0.833 | **+0.019** |
| 2022-12 | 0.812 | 0.809 | +0.003 |

The improvement is remarkably **broad**: the smallest delta (+0.003 in 2022-12) is still positive. The largest gains are in summer months (2021-06, 2021-08, 2022-06) and the previously "structurally broken" 2022-09 (+0.019). This is the first time ANY intervention has improved 2022-09 AUC meaningfully.

### NDCG Concern — Detailed Decomposition

NDCG is the only Group A metric that didn't improve. The per-month breakdown reveals a redistribution rather than uniform degradation:

| Month | v0007 | v0 | Δ | Direction |
|-------|-------|-----|------|-----------|
| 2020-09 | 0.790 | 0.758 | +0.032 | WIN |
| 2020-11 | 0.746 | 0.729 | +0.017 | WIN |
| 2021-01 | 0.743 | 0.764 | -0.021 | LOSS |
| 2021-04 | **0.648** | 0.660 | -0.012 | LOSS |
| 2021-06 | 0.806 | 0.767 | +0.039 | WIN |
| 2021-08 | 0.700 | 0.683 | +0.017 | WIN |
| 2021-10 | 0.809 | 0.773 | +0.036 | WIN |
| 2021-12 | 0.761 | 0.787 | -0.026 | LOSS |
| 2022-03 | **0.664** | 0.695 | **-0.031** | LOSS |
| 2022-06 | 0.747 | 0.763 | -0.017 | LOSS |
| 2022-09 | 0.692 | 0.709 | -0.017 | LOSS |
| 2022-12 | 0.693 | 0.712 | -0.018 | LOSS |

The bottom-2 months shifted: v0 had 2021-04 (0.660) and 2021-08 (0.683); v0007 has 2021-04 (0.648) and 2022-03 (0.664). The 2022-03 regression (-0.031) is the single largest NDCG drop. Late-2022 months (2022-03, 2022-06, 2022-09, 2022-12) ALL regressed, suggesting the new features may add noise to position-weighted ranking in the distribution-shift period even while improving broad discrimination (AUC).

**Layer 3 margin**: bot2 delta = -0.0154 against tolerance 0.02, margin = **0.0046**. This is concerning. If v0007 becomes champion and a future version regresses NDCG bot2 by even 0.005, it would fail Layer 3.

### Group B Monitor Gates

| Gate | v0007 Mean | v0 Mean | Δ | Floor | Headroom | Pass? | Concern? |
|------|-----------|---------|------|-------|----------|-------|----------|
| S1-BRIER | **0.1395** | 0.1503 | **-0.0108** | 0.1703 | +0.031 | YES | NO — REVERSED the 6-experiment narrowing trend! |
| S1-VCAP@500 | 0.0920 | 0.0908 | +0.0012 | 0.0408 | +0.051 | YES | No |
| S1-VCAP@1000 | 0.1401 | 0.1591 | -0.0190 | 0.1091 | +0.031 | YES | Moderate — 12% relative decline |
| S1-REC | 0.4318 | 0.4192 | +0.0126 | 0.1000 | +0.332 | YES | No |
| **S1-CAP@100** | **0.7342** | 0.7825 | **-0.0483** | 0.7325 | **0.002** | YES | **CRITICAL** — 0.002 headroom! |
| **S1-CAP@500** | **0.7280** | 0.7740 | **-0.0460** | 0.7240 | **0.004** | YES | **HIGH** — 0.004 headroom! |

**CAP@100 and CAP@500 are dangerously close to their Group B floors.** Both dropped ~0.05 from v0, suggesting that while the model ranks better broadly (AUC) and for the positive class (AP), the top-K capture rate at threshold-dependent metrics has degraded. This is consistent with v0007 using a higher threshold (mean 0.851 vs v0's 0.834), which reduces the number of predicted positives and can hurt CAP metrics.

## Code Review

### Quality: GOOD — No bugs found

The code changes are clean, minimal, and well-structured:

1. **`ml/config.py`**: 6 features correctly added with appropriate monotone constraints. `sf_max_abs` and `sf_mean_abs` at monotone=1 (higher → more binding) is physically sound — higher shift factors indicate greater sensitivity to generation changes, which correlates with binding risk. The 4 unconstrained features (monotone=0) are reasonable given uncertain monotonic relationships.

2. **`ml/data_loader.py`**:
   - Diagnostic print after real-data load is useful for debugging.
   - `_load_smoke()` synthetic data generation is appropriate: abs(randn) for SF features, 30% binary for is_interface, log-transformed uniform for constraint_limit.
   - Minor note: The `sf_meta_features` set and the hardcoded feature lists in the smoke generator are redundant with config — if features are pruned later, the smoke generator will still produce orphan columns. Harmless but worth cleaning up if v0007 is promoted.

3. **`ml/features.py`**: Source feature verification is correctly placed before the general missing-column check, providing a more informative error message. The set intersection logic (`source_needed = source_features & set(cols)`) correctly handles the case where not all source features are in the config.

4. **Tests**: Updated correctly. `conftest.py` now dynamically reads from `FeatureConfig()` — good practice that prevents future breakage.

### Potential Issue: Feature Verification Order

In `features.py`, the source feature check (line 49-57) runs after interaction feature computation (line 38-47). If a source feature name collided with an interaction feature name (not the case now, but a maintenance hazard), the interaction computation could mask the missing source column. This is a minor structural concern, not a current bug.

## Statistical Rigor

**This is the most statistically robust result in the entire experiment history:**

- AUC 12W/0L: Binomial sign test p = 2^(-12) ≈ 0.0002 (two-sided p ≈ 0.0005). The probability of 12 consecutive wins by chance is effectively zero.
- AP 11W/1L: p ≈ 0.006. Highly significant.
- The AUC improvement is NOT outlier-driven: the standard deviation of per-month deltas is 0.007, and no single month contributes disproportionately. Excluding the best month (2022-06, +0.026), the mean AUC delta is still +0.012 — still the largest ever.
- The AP improvement is also broad: excluding the best month (2021-06, +0.086), the mean AP delta is still +0.042.

**Contrast with prior experiments**: The previous best W/L was 10W/2L for VCAP@100 in v0006. v0007's AUC 12W/0L and AP 11W/1L represent a qualitative improvement in signal strength.

## Feature Importance Analysis

The 6 new features contribute only **4.66% combined gain**, yet produce a +0.0137 AUC improvement. This apparent paradox has an important explanation: feature importance (gain) measures how much a feature contributes to reducing training loss, NOT how much it improves generalization. The shift factor features may help break ties or provide discrimination signal in regions where existing features are ambiguous, even though they contribute little to aggregate training loss.

None of the new features cracked the top 10 by importance. This suggests they are acting as **auxiliary discriminators** — useful at the margins but not primary drivers. The implication: future pruning should be careful not to remove these features based on low importance alone, as their generalization value exceeds their training-loss contribution.

## Key Findings

### What Worked
1. **Network topology features broke the ceiling**: AUC jumped from [0.832-0.836] range to 0.849 — a +0.012 step change above the prior ceiling.
2. **2022-09 improved**: AUC 0.853 (+0.019) and AP 0.347 (+0.032). Five prior interventions failed to move this month; shift factors succeeded. The constraint's network position (shift factors) helps discriminate when flow-based features are ambiguous in low-binding-rate periods.
3. **AP bot2 trend reversed**: From monotonic decline (-0.0017 → -0.0094 over 6 experiments) to +0.0363. The new features help the model maintain ranking quality even in the weakest months.
4. **BRIER improved**: -0.0108, reversing the 6-experiment narrowing trend. Topology features appear to improve probability calibration, not just discrimination.

### What Didn't Work
1. **NDCG did not benefit** (5W/7L): The new features improve broad discrimination (AUC) and positive-class ranking (AP) but hurt position-weighted ranking (NDCG), especially in late-2022 months. This may indicate that shift factor features help separate binders from non-binders but add noise to the relative ordering of binders vs each other.
2. **CAP@100/500 degraded**: Top-K capture rate dropped ~0.05. The model is predicting with higher confidence (threshold up from 0.834 to 0.851) but its absolute top-K predictions are less calibrated in aggregate capture.
3. **VCAP@1000 degraded** (-0.019): Broader value capture worsened, consistent with the CAP regression.

## Recommendations for Next Iteration

### If v0007 Is Promoted (Recommended)

1. **NDCG-targeted investigation**: The 5W/7L NDCG result and -0.0154 bot2 regression are the primary risk. Investigate whether selective monotone constraint relaxation on the new features (e.g., `sf_std` and `sf_nonzero_frac` with monotone=0 currently could be tested at +1 or -1) would improve NDCG without regressing AUC/AP.

2. **Monotone constraint sensitivity analysis**: v0007 has 4 unconstrained features (monotone=0) out of 19. v0006 showed that full monotone enforcement sharpened NDCG at the cost of AP. The reverse experiment — testing whether enforcing monotone=1 on `constraint_limit` (larger constraints may bind more) or `sf_nonzero_frac` (more connected → more likely to bind) — could recover NDCG.

3. **Distribution shape features**: With the AUC ceiling broken, re-introducing density shape features (density_skewness, density_kurtosis, density_cv — pruned in v0006) on top of v0007's 19-feature base could provide additional marginal signal. Target: 22 features.

4. **HP tuning on v0007**: The v0003-HP experiment showed HP tuning didn't help at the 14-feature level. But with 19 features providing more signal, tree depth or regularization adjustments may now have room to work. Specifically, try `max_depth=5` or `n_estimators=300` to exploit the richer feature space.

### Gate Calibration Suggestions

1. **NDCG Layer 3 tolerance**: v0007's NDCG bot2 at -0.0154 vs tolerance 0.02 (margin 0.0046) is dangerously thin. If v0007 becomes champion, future versions will be measured against v0007's bot2 of 0.6562. Two options:
   - **Keep tolerance at 0.02**: Allows NDCG bot2 down to 0.6362, which gives reasonable room.
   - **Tighten to 0.015**: Would have failed v0007 (margin would be -0.004). NOT recommended.
   - **Recommendation**: Keep 0.02 for now but **set champion to v0007** to activate Layer 3 for future iterations.

2. **CAP@100/500 floors**: v0007 mean CAP@100 = 0.7342 (headroom 0.002 from floor 0.7325) and CAP@500 = 0.728 (headroom 0.004 from floor 0.724). These are Group B so non-blocking, but the tight headroom means minor future regressions could trigger monitoring alerts. Consider relaxing these floors by 0.02 if v0007 is promoted, since the model profile has shifted from threshold-dependent capture (CAP) to ranking quality (AUC/AP).

3. **VCAP@100 floor**: Still at -0.035 (non-binding). With v0007 at 0.0247 mean, tightening to 0.0 would be informative without being restrictive. Reiterate prior recommendation.

4. **BRIER headroom**: Recovered to +0.031 (was 0.016 for v0006). No longer a concern. The shift factor features improved calibration — a welcome surprise.

5. **Activate Layer 3**: Set champion to v0007. This enables non-regression checks against the new performance bar. With v0007's strong bot2 numbers (AUC 0.819, AP 0.369, VCAP@100 0.009), future versions have healthy margins on AUC and AP but tight on NDCG (0.656).

## Promotion Assessment

| Criterion | Threshold | v0007 | Pass? |
|-----------|-----------|-------|-------|
| AUC > 0.840 | 0.840 | **0.849** | **YES** |
| AP > 0.400 | 0.400 | **0.439** | **YES** |
| 8+/12 AUC wins | 8 | **12** | **YES** |
| All Group A gates pass | 3 layers × 4 gates | 12/12 pass | **YES** |

**Verdict: PROMOTE v0007.** This is a clear promotion-worthy result — the first version to satisfy all three success criteria simultaneously, with the strongest statistical evidence in the experiment history. The NDCG Layer 3 margin (0.005) is the only cautionary note, and it passes within tolerance.
