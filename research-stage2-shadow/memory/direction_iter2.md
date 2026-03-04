# Direction — Iteration 2 (feat-eng-3-20260304-102111)

## Batch Constraint
**Feature cleanup + value-weighted training experiment.** HPs remain at v0009 defaults EXCEPT `value_weighted` may change.
All regressor HPs must remain: n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=1.0, min_child_weight=25.

## Analysis

**Champion v0009** (39 nominal / 34 effective features, mcw=25, reg_lambda=1.0):
- Mean: EV-VC@100=0.0762, EV-VC@500=0.2329, EV-NDCG=0.7548, Spearman=0.3910
- Weakest months: 2021-05 (EV-VC@100=0.0012), 2021-11 (EV-VC@100=0.0118, Spearman=0.267), 2022-06 (EV-NDCG=0.607, Spearman=0.274)
- Strongest months: 2021-09 (EV-VC@100=0.200), 2020-11 (EV-VC@100=0.149), 2022-12 (EV-VC@100=0.174)
- **5 zero-filled features**: hist_physical_interaction, overload_exceedance_product, band_severity, sf_exceed_interaction, hist_seasonal_band — data loader doesn't provide these, always zero
- **Improvement concentration**: v0009's +9% EV-VC@100 gain is driven by 3 months; 5/12 months degraded vs v0007

**Key opportunity**: Both reviewers recommend removing zero-filled features (reduces noise, saves tree splits). Additionally, `value_weighted=True` is untested and directly targets the business objective — weighting training by shadow price magnitude should improve the model's ability to rank high-value constraints.

**Rationale**: Feature cleanup is a necessary hygiene step. Value weighting is a training mode change that aligns loss function with business objective (maximize value capture at top of ranking). Since EV-VC@100 improvement was concentrated in few months, value weighting may spread the benefit more uniformly by emphasizing high-$ constraints.

---

## Hypothesis A (control): Feature cleanup only (39 → 34)

**What**: Remove the 5 zero-filled features. Keep all HPs and `value_weighted=False`.

**Why**: Establishes the isolated effect of removing noise features. Expected to be neutral or slightly positive — XGBoost should already be mostly ignoring constant columns, but removing them guarantees no wasted splits and provides a clean baseline.

**Feature list** (34 features):
```json
{"regressor": {"features": ["prob_exceed_110", "prob_exceed_105", "prob_exceed_100", "prob_exceed_95", "prob_exceed_90", "prob_below_100", "prob_below_95", "prob_below_90", "expected_overload", "hist_da", "hist_da_trend", "sf_max_abs", "sf_mean_abs", "sf_std", "sf_nonzero_frac", "is_interface", "constraint_limit", "density_mean", "density_variance", "density_entropy", "tail_concentration", "prob_band_95_100", "prob_band_100_105", "hist_da_max_season", "prob_exceed_85", "prob_exceed_80", "recent_hist_da", "season_hist_da_1", "season_hist_da_2", "density_skewness", "density_kurtosis", "density_cv", "season_hist_da_3", "prob_below_85"], "monotone_constraints": [1, 1, 1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, -1], "value_weighted": false}}
```

---

## Hypothesis B (experiment): Feature cleanup + value weighting (39 → 34, value_weighted=True)

**What**: Same 34-feature cleanup as Hypothesis A, but with `value_weighted=True`.

**Why**: Value weighting emphasizes high-shadow-price rows during training. Since the business objective is ranking by expected value (EV = P(binding) × predicted_$), the regressor should be most accurate on the constraints that matter most — the high-$ ones. This directly targets EV-VC@100 improvement.

**Feature list** (34 features, same as A):
```json
{"regressor": {"features": ["prob_exceed_110", "prob_exceed_105", "prob_exceed_100", "prob_exceed_95", "prob_exceed_90", "prob_below_100", "prob_below_95", "prob_below_90", "expected_overload", "hist_da", "hist_da_trend", "sf_max_abs", "sf_mean_abs", "sf_std", "sf_nonzero_frac", "is_interface", "constraint_limit", "density_mean", "density_variance", "density_entropy", "tail_concentration", "prob_band_95_100", "prob_band_100_105", "hist_da_max_season", "prob_exceed_85", "prob_exceed_80", "recent_hist_da", "season_hist_da_1", "season_hist_da_2", "density_skewness", "density_kurtosis", "density_cv", "season_hist_da_3", "prob_below_85"], "monotone_constraints": [1, 1, 1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, -1], "value_weighted": true}}
```

---

## Screen Months

| Month | Role | Rationale |
|-------|------|-----------|
| **2021-11** | Weak | Worst Spearman (0.267), degraded EV-VC@100 from v0007→v0009 (0.039→0.012). Value weighting may help here if high-$ constraints are being misranked. |
| **2021-09** | Strong | Best EV-VC@100 (0.200), largest single improvement from v0007→v0009. Must confirm cleanup doesn't regress and value weighting doesn't redistribute away from this month. |

---

## Winner Criteria

1. **Primary**: Higher mean EV-VC@100 across the 2 screen months
2. **Safety check**: Spearman must not drop > 0.03 on either screen month vs champion v0009
3. **Tiebreaker**: Higher mean EV-NDCG across screen months
4. **If both degrade vs champion**: Pick the one with smallest EV-VC@100 degradation. If both are substantially worse (> 10% mean EV-VC@100 drop), STOP and escalate.
5. **If A and B are within noise** (<2% mean EV-VC@100 difference): Prefer A (simpler, no value weighting) — avoids adding training mode complexity without clear benefit.

---

## Code Changes for Winner

1. **For both hypotheses**: The worker must create a new config.json with the 34-feature list (removing the 5 zero-filled features) and corresponding monotone_constraints (length 34).

2. **For Hypothesis B winner**: Additionally set `"value_weighted": true` in the regressor config.

3. **Verification**: After the full 12-month benchmark, confirm:
   - config.json has exactly 34 features and 34 monotone_constraints
   - The 5 zero-filled features are NOT in the feature list
   - `value_weighted` matches the winning hypothesis (false for A, true for B)
   - Classifier config is UNCHANGED from v0009 (14 features, frozen)
   - All other regressor HPs match v0009 exactly

4. **IMPORTANT**: Do NOT modify `ml/config.py` or any ML code. The config.json override is sufficient.

---

## Expected Impact

| Gate | Hyp A (cleanup) | Hyp B (cleanup + value wt) |
|------|-----------------|---------------------------|
| EV-VC@100 | ±0-2% (neutral) | +3-8% (value weighting targets this) |
| EV-VC@500 | ±0-1% | +1-5% |
| EV-NDCG | ±0-0.5% | +0.5-2% |
| Spearman | ±0% | ±1-2% (could go either way) |
| C-RMSE | ±0-1% | +2-5% worse on low-$ constraints (acceptable) |

**Key uncertainty for B**: Value weighting could hurt Spearman if it makes the model less accurate on low-$ constraints that still contribute to rank correlation. Watch for Spearman degradation > 1%.

---

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Feature cleanup changes results unexpectedly | Very low | XGBoost should already ignore constant features; removing them is cleanup only |
| Value weighting hurts Spearman significantly | Medium | Screen months include 2021-11 (worst Spearman) for diagnostic. Safety check at ±0.03. |
| Value weighting causes C-RMSE/C-MAE regression | Medium | These are Group B (monitor), not blocking. Acceptable if EV ranking improves. |
| Worker uses wrong feature list (includes zero-filled features) | Low | Explicit verification step in instructions. Feature count 34 is hard constraint. |
| Duplicate version (like v0009=v0008) | Medium | Worker should check if proposed config matches any existing version before running full benchmark. |
