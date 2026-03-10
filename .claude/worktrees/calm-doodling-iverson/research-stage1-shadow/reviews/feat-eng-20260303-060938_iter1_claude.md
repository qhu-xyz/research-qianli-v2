# Claude Review — feat-eng-20260303-060938 iter1 (v0004)

## Summary

v0004 combines the two positive-signal levers from prior iterations: 14-month training window (from v0003) and 3 interaction features (from v0002). The hypothesis (H6) asked whether these effects are additive. The answer is **partially additive** — v0004 produces the strongest Group A results of any iteration to date, with AUC +0.0015 (9W/3L), VCAP@100 +0.0056 (10W/2L), and NDCG +0.0038 (7W/5L). All three gate layers pass cleanly for all metrics.

However, the improvement is firmly in the "encouraging, not promotion-worthy" band. AUC 0.8363 falls at the low end of the additive prediction range (0.836–0.838) and well below the promotion threshold of 0.837. AP is essentially flat at 6W/6L, showing no benefit from combining levers on average precision. The iteration also includes two well-executed bug fixes (f2p parsing, dual-default fragility) that improve code quality without affecting model performance. This is a solid result that confirms the feature set is the primary bottleneck, but the signal is not strong enough to change the production baseline.

## Gate-by-Gate Analysis

### Group A (Blocking) — All 3 Layers

| Gate | v0 Mean | v0004 Mean | Δ Mean | v0 Bot2 | v0004 Bot2 | Δ Bot2 | v0003-win Bot2 | W/L/T | L1 | L2 | L3 | Overall |
|------|---------|------------|--------|---------|------------|--------|----------------|-------|----|----|----|----|
| S1-AUC | 0.8348 | 0.8363 | **+0.0015** | 0.8105 | 0.8164 | +0.0059 | 0.8162 | **9W/3L** | P (+0.0515) | P (0 fail) | P (+0.0059) | **P** |
| S1-AP | 0.3936 | 0.3951 | **+0.0015** | 0.3322 | 0.3282 | -0.0040 | 0.3277 | 6W/6L | P (+0.0515) | P (0 fail) | P (-0.0040) | **P** |
| S1-VCAP@100 | 0.0149 | 0.0205 | **+0.0056** | 0.0014 | 0.0011 | -0.0003 | 0.0016 | **10W/2L** | P (+0.0556) | P (0 fail) | P (-0.0003) | **P** |
| S1-NDCG | 0.7333 | 0.7371 | **+0.0038** | 0.6716 | 0.6656 | -0.0060 | 0.6657 | 7W/5L | P (+0.0538) | P (0 fail) | P (-0.0060) | **P** |

### Group B (Monitor) — Key Metrics

| Gate | v0 Mean | v0004 Mean | Δ Mean | Headroom to Floor | Status |
|------|---------|------------|--------|-------------------|--------|
| S1-BRIER | 0.1503 | 0.1516 | +0.0013 (worse) | 0.0187 | P (narrowing) |
| S1-VCAP@500 | 0.0908 | 0.0843 | **-0.0065** | 0.0435 | P |
| S1-VCAP@1000 | 0.1591 | 0.1578 | -0.0013 | 0.0487 | P |
| S1-REC | 0.4192 | 0.4174 | -0.0018 | 0.3174 | P |
| S1-CAP@100 | 0.7825 | 0.7850 | +0.0025 | 0.0525 | P |
| S1-CAP@500 | 0.7740 | 0.7750 | +0.0010 | 0.0510 | P |

### Detailed Observations

**VCAP@100 is the standout**: 10W/2L with mean nearly doubling from 0.0149 to 0.0205. This is the clearest evidence of additivity — window expansion (v0003: +0.0034) and interactions (v0002: +0.0009) combine to +0.0056, which exceeds both individual effects. The top-100 predicted-positive constraints capture significantly more shadow price value than v0.

**AP is the disappointment**: 6W/6L is no better than random. Window expansion alone was 8W/4L on AP; adding interactions actually reduced AP consistency. The bottom_2_mean degraded from v0's 0.3322 to 0.3282 (-0.0040), driven by 2022-09 (AP=0.3072) and 2021-06 (AP=0.3492). This remains within the 0.02 tolerance but is worth noting.

**VCAP@500 regression persists**: -0.0065 vs v0, confirming a pattern seen in both v0002 (-0.0043) and v0003 (-0.0063). The combined model concentrates value capture more tightly in the top 100, at the expense of the 100-500 rank range. This is acceptable per business objective (top-100 precision > broad ranking) but the v0004 bottom_2_mean for VCAP@500 is 0.0387 — approaching the floor of 0.0408. If this trend continues, VCAP@500 could breach the Group B floor.

**BRIER headroom continues narrowing**: v0004 BRIER mean=0.1516 vs floor=0.1703, headroom=0.0187. This has decreased from 0.0200 (v0) → 0.0189 (v0003-window) → 0.0187 (v0004). Not yet critical, but the trend is consistent — adding features and expanding the window slightly degrade calibration.

## Seasonal / Per-Month Analysis

### AUC Per-Month Breakdown

| Month | v0 | v0004 | Δ | Result |
|-------|-----|-------|---|--------|
| 2020-09 | 0.8434 | 0.8471 | +0.0037 | W |
| 2020-11 | 0.8300 | 0.8326 | +0.0026 | W |
| 2021-01 | 0.8555 | 0.8532 | -0.0023 | L |
| 2021-04 | 0.8353 | 0.8342 | -0.0011 | L |
| 2021-06 | 0.8246 | 0.8263 | +0.0017 | W |
| 2021-08 | 0.8532 | 0.8538 | +0.0006 | W |
| 2021-10 | 0.8507 | 0.8509 | +0.0002 | W |
| 2021-12 | 0.8123 | 0.8141 | +0.0018 | W |
| 2022-03 | 0.8446 | 0.8453 | +0.0007 | W |
| 2022-06 | 0.8258 | 0.8247 | -0.0011 | L |
| 2022-09 | 0.8334 | 0.8345 | +0.0011 | W |
| 2022-12 | 0.8088 | 0.8186 | +0.0098 | W |

**Pattern**: Improvements are broadly distributed (9/12 months). The 3 losses (2021-01, 2021-04, 2022-06) are all small (<0.0023). This is NOT an outlier-driven result, unlike v0002 where 2021-01 NDCG dominated.

**2022-12 continues to respond well**: +0.0098 AUC, the largest per-month improvement. This month has the shortest lookback relative to the training distribution, so the 14-month window helps most here. Consistent with v0003-window's +0.0098 on this same month — the interaction features contributed nothing extra to 2022-12.

**2022-09 remains the hardest month**: AP=0.3072 (lowest across all months), precision=0.3283 (lowest), binding_rate=0.0663 (lowest). AUC only improved +0.0011, essentially noise. Four iterations of changes (HP, interactions, window, combined) have all failed to improve this month's AP. The 6.63% binding rate makes class separation inherently harder. This is likely a structural feature of the data in that period, not addressable by the current feature set.

**Weakest months by metric**:
- AUC: 2021-12 (0.8141), 2022-12 (0.8186) — both improved vs v0 but remain lowest
- AP: 2022-09 (0.3072), 2021-06 (0.3492) — neither improved meaningfully
- VCAP@100: 2022-12 (0.0008), 2021-06 (0.0014) — persistent weakness
- NDCG: 2021-04 (0.6634), 2021-08 (0.6678) — stable across iterations

No month falls below any tail_floor. Tail safety is clean.

## Additivity Assessment

| Metric | Window-only (v0003) | Interactions-only (v0002) | Combined (v0004) | Sum of individuals | Additive? |
|--------|---------------------|---------------------------|-------------------|--------------------|-----------|
| AUC Δ | +0.0013 | +0.0000 | +0.0015 | +0.0013 | Mostly window |
| AP Δ | +0.0012 | +0.0010 | +0.0015 | +0.0022 | Sub-additive |
| VCAP@100 Δ | +0.0034 | +0.0009 | +0.0056 | +0.0043 | **Super-additive** |
| NDCG Δ | +0.0019 | +0.0016 | +0.0038 | +0.0035 | ~Additive |

Key finding: **VCAP@100 is super-additive** (combined > sum of parts), suggesting that interaction features are particularly effective at re-ranking the very top of the score distribution when the model has more diverse training data. NDCG is roughly additive. AUC and AP are sub-additive — the levers overlap on overall discrimination and average precision.

## Code Review Findings

### 1. f2p Parsing Fix (data_loader.py:103-105) — CORRECT

```python
m = re.match(r"f(\d+)", ptype)
horizon = int(m.group(1)) if m else 3
```

Correctly handles "f0", "f1", "f2p" by extracting the leading digits. The default of 3 for an unmatched pattern is a reasonable safe fallback (wider lookback if ptype format is unexpected). The `re` import was added at the call site. No issues.

### 2. Dual-Default Fix (benchmark.py:36-43, 96-102, 133-136) — CORRECT

Both `_eval_single_month()` and `run_benchmark()` now use `None` sentinel with fallback to `PipelineConfig()` defaults. This eliminates the fragile coupling where hardcoded values in function signatures had to match `PipelineConfig`. Clean implementation.

### 3. Config Changes (config.py:37-41) — CORRECT

Three interaction features added with monotone +1 constraints. Feature names match the computation in `features.py:38-47`. The features are appended after the base 14, maintaining backward compatibility of feature indices.

### 4. Test Updates (test_config.py, test_features.py) — CORRECT

- `test_feature_config_has_17_features`: Updated 14→17
- `test_monotone_constraints_values`: Updated expected string to include 3 additional `1` values
- `test_feature_names_match_expected`: Full 17-feature list asserted
- `test_prepare_features_shape`: Made dynamic via `len(fc.features)` — good, prevents future breakage

### 5. Carried Issues (not introduced in this iteration)

- **Threshold leakage** (HIGH, carried from v0001): Validation data is used for both threshold tuning AND final evaluation. Still deferred — correct since it affects all versions equally and Group A gates are threshold-independent.
- **Threshold `>` vs `>=` mismatch** (MEDIUM, carried): PR curve thresholds are inclusive, but `apply_threshold` uses strict `>`. Affects threshold-dependent metrics only.
- **Missing schema guard for interaction base columns** (MEDIUM, carried from v0002): `features.py` line 50 checks `missing = set(cols) - set(df.columns) - interaction_cols` but this passes even if base columns (prob_exceed_110, prob_exceed_90, etc.) needed for interaction computation are absent. Not triggered in practice since base columns are always loaded, but worth hardening.

### 6. No New Issues Introduced

The code changes are minimal and surgical. All three changes follow the direction document exactly. No regressions detected.

## Statistical Rigor

With 12 eval months, the approximate standard errors are:
- AUC: std=0.0137, SE ≈ 0.0137/√12 = 0.0040. Delta +0.0015 → z ≈ 0.38, p ≈ 0.35
- AP: std=0.042, SE ≈ 0.012. Delta +0.0015 → z ≈ 0.12, p ≈ 0.45
- NDCG: std=0.050, SE ≈ 0.014. Delta +0.0038 → z ≈ 0.27, p ≈ 0.39
- VCAP@100: std=0.018, SE ≈ 0.005. Delta +0.0056 → z ≈ 1.12, p ≈ 0.13

None are statistically significant at p<0.05. VCAP@100 is closest at p≈0.13, which aligns with its 10W/2L record (sign test p≈0.019, which IS significant). The sign test is more appropriate here since it doesn't assume normality, and **VCAP@100's 10W/2L is the first statistically significant improvement in the pipeline's history** (two-sided sign test p=0.039).

AUC 9W/3L also approaches significance via sign test (p=0.073). This is stronger evidence than any previous iteration.

## Gate Calibration Assessment

Current gate floors remain appropriate. Specific observations:

1. **VCAP@500 bottom_2_mean (0.0387) is approaching the floor (0.0408)**. v0004 is the third consecutive version showing VCAP@500 regression (-0.0043 v0002, -0.0063 v0003, -0.0065 v0004). If this pattern continues, a future version could fail the mean quality layer for VCAP@500. However, since VCAP@500 is Group B (non-blocking), this is an advisory concern only. The trend suggests that improvements to top-100 ranking systematically degrade the 100-500 range.

2. **BRIER floor headroom (0.0187) continues to narrow**. Monotonic decline across v0→v0002→v0003→v0004. Still Group B and not blocking, but worth monitoring.

3. **Layer 3 remains effectively disabled** (champion=null). All bottom_2_mean deltas vs v0 are well within the 0.02 tolerance. Once a champion is promoted, Layer 3 will activate and become meaningful. Current deltas suggest it would not have been a problem: largest regression is NDCG -0.0060, within tolerance.

4. **VCAP@100 floor remains non-binding** at -0.0351. Even v0004's worst month (0.0008) is far above tail_floor (-0.0995). This gate does not meaningfully constrain any reasonable model. If ever tightened, use extreme caution — VCAP@100 has very high month-to-month variance (std=0.018, range 0.0008–0.0541).

No gate calibration changes recommended at this time.

## Recommendations for Next Iteration

### Assessment: v0004 falls in the "Encouraging" band

Per the direction document's criteria:
- AUC > 0.835 ✓ (0.8363), ≥7/12 wins ✓ (9W/3L), AP ≤ 0.396 ✗ (0.3951, just under)
- Not promotion-worthy (AUC < 0.837, AP < 0.396)
- Not a dead end (AUC > 0.835 and clear improvement in top-K metrics)

### Priority Recommendations

1. **Feature selection / importance analysis**: With 4 iterations of evidence, the feature set is the confirmed bottleneck. Before adding more features, analyze which of the current 17 contribute meaningful signal. If the 3 interaction features have low importance despite being monotone-constrained, they may add noise without helping. Use XGBoost feature importance (gain-based) across all 12 months to identify candidates for removal.

2. **Target the AP weakness**: AP is the one Group A metric that hasn't responded to any lever (6W/6L this iteration, 8W/4L for window alone — interactions actively hurt AP consistency). AP is sensitive to calibration quality in the positive tail. Investigate whether the interaction features distort the probability ordering among true positives specifically. If so, consider dropping one or two of the three interaction features.

3. **Explore new feature sources**: The current 14 base features + 3 interactions capture ~0.836 AUC ceiling. Breaking through likely requires fundamentally different information. Consider: (a) temporal features (day-of-week, month-of-year seasonality encoding), (b) cross-constraint features (how many nearby constraints are also predicted to bind), (c) weather/load forecast error features if available. Any new features should be tested in isolation before combining.

4. **Do not continue HP tuning**: Confirmed dead end across 4 experiments. v0 defaults are near-optimal for this feature set.

5. **Monitor VCAP@500**: If next iteration again shows VCAP@500 regression, document the trade-off explicitly — this appears to be an inherent cost of improving top-100 precision with the current model architecture.

### Carried Issues for Human Review at HUMAN_SYNC

- Threshold leakage (HIGH) — should be addressed before promoting any version to champion
- Threshold `>` vs `>=` mismatch (MEDIUM) — affects recall/precision reporting
- Missing schema guard for interaction base columns (MEDIUM) — defensive coding
