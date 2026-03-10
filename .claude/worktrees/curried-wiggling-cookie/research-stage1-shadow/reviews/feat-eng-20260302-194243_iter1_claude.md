# Claude Review — feat-eng-20260302-194243, Iteration 1

## Summary

v0003 tests whether expanding the training window from 10 to 14 months (H5) can break the AUC ceiling at ~0.835, targeting the persistent late-2022 weakness observed in two prior iterations (H3 HP tuning: AUC 0W/11L; H4 interaction features: AUC 5W/6L/1T). The interaction features from v0002 were correctly reverted to isolate the training window effect. The result is a **small, directionally positive** improvement: mean AUC +0.0013 (7W/4L/1T), mean AP +0.0012 (8W/4L), mean NDCG +0.0019 (7W/4L/1T), VCAP@100 +0.0034 (9W/3L). All gate layers pass. However, effect sizes are well under 0.1σ for AUC, AP, and NDCG, and the targeted month 2022-09 actually regressed on AP (-0.0091). This is the strongest result of the three real-data iterations so far, but it does not represent a decisive breakthrough.

The code changes are clean and include a welcome bug fix (benchmark.py train_months/val_months plumbing), a proper interaction-feature guard in features.py, and correctly updated tests. One structural concern: benchmark.py now hardcodes `train_months=14` in its function signatures alongside PipelineConfig's default, creating dual-default fragility.

## Gate-by-Gate Analysis

### Group A (Hard Gates) — Per-Month Detail

| Month | AUC v0 | AUC v3 | Δ | AP v0 | AP v3 | Δ | NDCG v0 | NDCG v3 | Δ | VCAP@100 v0 | VCAP@100 v3 | Δ |
|-------|--------|--------|---|-------|-------|---|---------|---------|---|-------------|-------------|---|
| 2020-09 | 0.8434 | 0.8461 | **+0.0027** | 0.3866 | 0.3976 | **+0.0110** | 0.7579 | 0.7608 | +0.0029 | 0.0051 | 0.0102 | +0.0051 |
| 2020-11 | 0.8300 | 0.8332 | **+0.0032** | 0.4330 | 0.4341 | +0.0011 | 0.7290 | 0.7324 | +0.0034 | 0.0066 | 0.0024 | -0.0042 |
| 2021-01 | 0.8555 | 0.8528 | -0.0027 | 0.4442 | 0.4425 | -0.0017 | 0.7643 | 0.7869 | **+0.0226** | 0.0189 | 0.0433 | **+0.0244** |
| 2021-04 | 0.8353 | 0.8337 | -0.0016 | 0.4198 | 0.4243 | +0.0045 | 0.6601 | 0.6619 | +0.0018 | 0.0092 | 0.0124 | +0.0032 |
| 2021-06 | 0.8246 | 0.8262 | +0.0016 | 0.3494 | 0.3495 | +0.0001 | 0.7671 | 0.7671 | 0.0000 | 0.0024 | 0.0080 | +0.0056 |
| 2021-08 | 0.8532 | 0.8543 | +0.0011 | 0.3959 | 0.3901 | -0.0058 | 0.6830 | 0.6695 | **-0.0135** | 0.0121 | 0.0081 | -0.0040 |
| 2021-10 | 0.8507 | 0.8506 | -0.0001 | 0.4439 | 0.4337 | **-0.0102** | 0.7730 | 0.7699 | -0.0031 | 0.0434 | 0.0271 | -0.0163 |
| 2021-12 | 0.8123 | 0.8138 | +0.0015 | 0.4253 | 0.4334 | **+0.0081** | 0.7872 | 0.8030 | **+0.0158** | 0.0181 | 0.0245 | +0.0064 |
| 2022-03 | 0.8446 | 0.8450 | +0.0004 | 0.3625 | 0.3643 | +0.0018 | 0.6946 | 0.7024 | +0.0078 | 0.0290 | 0.0464 | **+0.0174** |
| 2022-06 | 0.8258 | 0.8249 | -0.0009 | 0.3850 | 0.3857 | +0.0007 | 0.7634 | 0.7577 | -0.0057 | 0.0225 | 0.0234 | +0.0009 |
| 2022-09 | 0.8334 | 0.8334 | 0.0000 | 0.3150 | 0.3059 | **-0.0091** | 0.7085 | 0.7094 | +0.0009 | 0.0109 | 0.0126 | +0.0017 |
| 2022-12 | 0.8088 | 0.8186 | **+0.0098** | 0.3623 | 0.3765 | **+0.0142** | 0.7116 | 0.7019 | **-0.0097** | 0.0005 | 0.0008 | +0.0003 |

### Three-Layer Gate Summary

| Gate | Mean v0 | Mean v3 | Δ Mean | L1 | Bot2 v0 | Bot2 v3 | Δ Bot2 | L3 (tol=0.02) | Tail Fails | L2 | Overall |
|------|---------|---------|--------|-----|---------|---------|--------|---------------|------------|-----|---------|
| S1-AUC | 0.8348 | 0.8361 | +0.0013 | P | 0.8105 | 0.8162 | **+0.0057** | P | 0 | P | **P** |
| S1-AP | 0.3936 | 0.3948 | +0.0012 | P | 0.3322 | 0.3277 | -0.0045 | P (within 0.02) | 0 | P | **P** |
| S1-VCAP@100 | 0.0149 | 0.0183 | +0.0034 | P | 0.0014 | 0.0016 | +0.0002 | P | 0 | P | **P** |
| S1-NDCG | 0.7333 | 0.7352 | +0.0019 | P | 0.6716 | 0.6657 | -0.0059 | P (within 0.02) | 0 | P | **P** |
| S1-BRIER | 0.1503 | 0.1514 | +0.0011 | P | 0.1584 | 0.1599 | +0.0015 | P | 0 | P | **P** |

**Layer 3 note**: Champion is null, so L3 defaults to pass. Had v0 been champion, AP bottom-2 (-0.0045) and NDCG bottom-2 (-0.0059) would still pass the 0.02 tolerance. No layer is close to flipping.

### Seasonal Analysis

**Months that improved most** (AUC): 2022-12 (+0.0098), 2020-11 (+0.0032), 2020-09 (+0.0027).

**Months that degraded most** (AUC): 2021-01 (-0.0027), 2021-04 (-0.0016), 2022-06 (-0.0009).

**Pattern**: No clear seasonal pattern. 2022-12 (winter, the weakest month) improved substantially, supporting the hypothesis that more training context helps. But 2021-01 (also winter) degraded, suggesting the effect is not purely seasonal — it may be specific to the 2022 distribution shift rather than a general winter pattern.

**2022-09 (target month)**: AUC unchanged (0.8334), AP regressed (-0.0091). The longer window adds data from 2021-05 onward instead of 2021-09 onward, but the additional spring/summer 2021 data didn't help this month. The 2022-09 weakness likely stems from a feature-level distributional shift rather than insufficient training diversity.

**NDCG anomaly at 2021-08 and 2022-12**: These months show opposing AUC vs NDCG behavior. 2022-12 has AUC +0.0098 but NDCG -0.0097; 2021-08 has AUC +0.0011 but NDCG -0.0135. The model discriminates better overall but ranks the positive cases worse within their stratum. This could indicate the longer window dilutes the fine-grained ordering signal for certain months.

### Group B Highlights

| Gate | v0 Mean | v3 Mean | Δ | Concern |
|------|---------|---------|---|---------|
| S1-BRIER | 0.1503 | 0.1514 | +0.0011 | Slight regression. Headroom: 0.0189 (was 0.0200). Still safe but tightening. |
| S1-REC | 0.4192 | 0.4130 | -0.0062 | Model more conservative (pred_binding_rate 0.0754→0.0733). Consistent with larger training set raising thresholds. |
| S1-VCAP@500 | 0.0908 | 0.0845 | -0.0063 | Broader value capture degraded. Top-100 improved at expense of top-500. |
| S1-VCAP@1000 | 0.1591 | 0.1610 | +0.0019 | Marginal improvement at wider capture. |
| S1-CAP@100 | 0.7825 | 0.7708 | -0.0117 | Headroom to floor (0.7325) now only 0.0383. Not critical but notable. |
| S1-CAP@500 | 0.7740 | 0.7633 | -0.0107 | Headroom to floor (0.7240) now 0.0393. Similar trajectory as CAP@100. |

**Trend**: VCAP@100 improved while VCAP@500 and CAP metrics degraded. The longer training window improves the very top of the ranking at some cost to the broader ranking. This is consistent with the business objective (precision at the top matters most) but worth monitoring.

## Code Review Findings

### 1. Dual-default fragility in benchmark.py (MEDIUM)

`_eval_single_month()` and `run_benchmark()` both hardcode `train_months: int = 14` in their function signatures. `PipelineConfig` also has `train_months: int = 14`. If a future experiment changes `PipelineConfig.train_months` back to 10 (or any other value), the benchmark function signatures must be manually updated in lockstep, otherwise calling `run_benchmark()` without explicit `train_months` would use the stale function-level default (14) rather than the new `PipelineConfig` default.

**Suggested fix**: Use `PipelineConfig().train_months` as the default, or use `None` sentinel with fallback to PipelineConfig:
```python
def run_benchmark(..., train_months: int | None = None, ...):
    if train_months is None:
        train_months = PipelineConfig().train_months
```

### 2. Schema guard subtlety in features.py (LOW)

The guard `missing = set(cols) - set(df.columns) - interaction_cols` unconditionally subtracts all `interaction_cols` from missing. This means if an interaction column is listed in `cols` but the `if interaction_cols & set(cols)` branch doesn't execute (impossible in current code, but fragile), the missing check would silently ignore it. Additionally, if only a subset of interaction features is requested, all three are computed — minor inefficiency but not a bug.

### 3. Revert of interaction features vs D18 decision (NOTE)

Decision log D18 explicitly said "Keep interaction features in iter2." The direction_iter1.md overrides this by reverting to 14 features to "isolate the training window effect cleanly." The direction's rationale is methodologically correct — you cannot attribute results to window expansion if you simultaneously change the feature set. However, this means we still lack data on the combined effect (14-month window + interaction features). This should be considered for a future iteration.

### 4. All other changes are clean

- Docstring updates correctly remove hardcoded counts.
- Test updates correctly match new config (14 features, train_months=14).
- Monotone constraint string correctly matches 14 features.
- `eval_config` in metrics.json correctly records `train_months: 14`.

### Carried Issues (not introduced by this iteration)

- **Threshold leakage** (HIGH, since v0001): Train/val split shares the same threshold optimization window. Deferred to HUMAN_SYNC.
- **Threshold > vs >=** (MEDIUM, since smoke-v7): PR curve inclusive, apply_threshold exclusive. Deferred to HUMAN_SYNC.
- **Layer 3 disabled** (MEDIUM): Champion is null, so bottom-2 non-regression is not checked. Acceptable until first promotion.

## Statistical Rigor

With 12 eval months and AUC std=0.015:
- **AUC δ = +0.0013**: This is 0.087σ — far below 1σ significance. A paired sign test on 7W/4L/1T gives p ≈ 0.27 (one-sided binomial with n=11, k=7). Not statistically significant.
- **AP δ = +0.0012**: 0.029σ. 8W/4L gives p ≈ 0.19. Not significant.
- **NDCG δ = +0.0019**: 0.046σ. 7W/4L/1T gives p ≈ 0.27. Not significant.
- **VCAP@100 δ = +0.0034**: 0.27σ. 9W/3L gives p ≈ 0.07. Closest to significance but still above 0.05.

**Assessment**: The improvements are directionally consistent but not statistically significant at conventional levels. The 7/12 and 8/12 win rates are better than the 5/12 from v0002 and 0/12 from v0003-HP, suggesting real (if small) signal. The VCAP@100 result (9W/3L) is the most compelling individual metric.

The improvement pattern is also not concentrated in 1-2 outlier months — the largest single-month AUC gain is 2022-12 (+0.0098) while other improving months show modest gains (+0.0004 to +0.0032). This is consistent with a small but distributed effect rather than an artifact of one outlier.

## Promotion Assessment

**Recommendation: Do not promote.** The improvement is real but marginal. Mean AUC rose from 0.8348 to 0.8361 — a +0.15% relative improvement. The bottom-2 is mixed (AUC improved, AP and NDCG regressed). The targeted month 2022-09 didn't improve. This is not sufficient to justify changing the production baseline.

However, v0003 should serve as the starting point for future iterations. The 14-month window is strictly better than 10 months on the primary metric (AUC) and doesn't hurt any gate.

## Recommendations for Next Iteration

### 1. Combine 14-month window with interaction features (HIGH priority)

The direction reverted interaction features to isolate the window effect. Now that the window effect is characterized, the logical next step is to combine both: 14-month window + 3 interaction features. This tests whether the two effects are additive (v0002 had NDCG 8W/4L on its own, v0003 has NDCG 7W/4L/1T — potentially complementary).

### 2. Investigate 2022-09 specifically (MEDIUM priority)

2022-09 is the persistently weakest AP month across all three iterations (0.3150 baseline, 0.3143 v0002, 0.3059 v0003). Three independent levers failed to improve it. This strongly suggests a distributional shift in the underlying data that these features cannot capture. Worth analyzing: (a) binding rate in 2022-09 (0.0663, lowest in the dataset), (b) whether the feature distributions shifted relative to other months, (c) whether the constraint mix changed in this period.

### 3. Consider asymmetric training windows (LOW priority)

Instead of uniform 14-month lookback, weight recent months more heavily through sample weighting or use a 14-month window but undersample months >10 months back. This might preserve the seasonal diversity benefit while reducing stale data dilution.

### 4. Explore new feature categories (MEDIUM priority)

Three iterations have shown the 14 base features have an AUC ceiling at ~0.835. The interaction features and training window each provided ~+0.001 AUC. Breaking significantly past 0.835 likely requires fundamentally new information — e.g., features derived from network topology, temporal autocorrelation of binding, or constraint-level historical patterns beyond hist_da and hist_da_trend.

## Gate Calibration Assessment

No changes recommended.

- **Group A floors remain appropriate**: v0003 passes all layers with ~0.05 headroom on mean. Gates are not blocking valid candidates — there simply haven't been clear improvements yet.
- **Layer 3 tolerance (0.02)**: Still effectively disabled (champion=null). When champion is eventually set, the 0.02 tolerance appears reasonable — observed bottom-2 deltas range from -0.006 to +0.006, well within tolerance.
- **BRIER headroom narrowing**: v0003 BRIER=0.1514 vs floor=0.1703, headroom=0.0189. Was 0.0200 for v0. Worth watching but not critical.
- **S1-CAP@100 headroom narrowing**: v0003 mean=0.7708 vs floor=0.7325, headroom=0.0383. Was 0.0500 for v0. CAP metrics have very high variance (std=0.25), so this headroom is thin relative to variability. Not actionable yet but flag if it degrades further.
- **Metric-specific noise_tolerance**: Still premature (3 real-data iterations, all within narrow delta range). Revisit after 5+ iterations with more varied results.
