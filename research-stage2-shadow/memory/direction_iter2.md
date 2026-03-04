# Direction — Iteration 2 (feat-eng-3-20260304-102111)

## Batch Constraint
**Feature engineering / selection ONLY.** No hyperparameter changes. No training mode changes.
All regressor HPs must remain at v0009 defaults: n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=1.0, min_child_weight=25.
`unified_regressor=false`, `value_weighted=false` — UNCHANGED.

## Analysis

**Champion v0009** (39 nominal / 34 effective features, mcw=25, reg_lambda=1.0):
- Mean: EV-VC@100=0.0762, EV-VC@500=0.2329, EV-NDCG=0.7548, Spearman=0.3910
- Weakest months: 2021-05 (EV-VC@100=0.0012, classifier-limited), 2021-11 (EV-VC@100=0.0118, Spearman=0.267), 2022-06 (EV-NDCG=0.607, Spearman=0.274)
- Strongest months: 2021-09 (EV-VC@100=0.200), 2022-12 (EV-VC@100=0.174), 2020-11 (EV-VC@100=0.149, Spearman=0.519)
- **5 zero-filled features**: hist_physical_interaction, overload_exceedance_product, band_severity, sf_exceed_interaction, hist_seasonal_band — computed by `features.py:compute_interaction_features()` but always produce zeros
- **Gate margins**: EV-VC@100 +14.8%, EV-VC@500 +6.9%, EV-NDCG +5.8%, **Spearman +4.7% (binding constraint)**

**Key findings from codebase investigation**:
1. Zero-filled features are interaction features computed in `ml/features.py` that produce zeros because underlying column combinations don't exist. They waste ~13% of tree split budget.
2. Remaining unused data loader columns: `flow_direction` (±1, genuinely new signal), `prob_below_105/110/80` (mathematically redundant — they're `1 - prob_exceed_*`, which XGBoost can learn from the exceed versions).
3. `flow_direction` differentiates import vs export constraints. Import and export constraints have different shadow price distributions — direction could help the regressor estimate magnitude more accurately.

**Rationale**: Pruning zero-filled features is a consensus recommendation from both reviewers. Adding `flow_direction` is the only available axis for genuinely new signal from the data loader within the FE-only constraint. Testing pruning alone vs pruning+flow_direction tells us whether the new feature carries signal.

---

## Hypothesis A (control): Prune zero-filled features only (39 → 34)

**What**: Remove the 5 zero-filled interaction features. No new features added.

**Why**: Establishes the isolated effect of removing noise features. Frees 13% of tree split budget. XGBoost should already mostly ignore constant columns, but removing them guarantees no wasted splits. Both reviewers recommend unanimously. This is the conservative, hygienic bet.

Hypothesis A overrides:
```json
{"regressor": {"features": ["prob_exceed_110", "prob_exceed_105", "prob_exceed_100", "prob_exceed_95", "prob_exceed_90", "prob_below_100", "prob_below_95", "prob_below_90", "expected_overload", "hist_da", "hist_da_trend", "sf_max_abs", "sf_mean_abs", "sf_std", "sf_nonzero_frac", "is_interface", "constraint_limit", "density_mean", "density_variance", "density_entropy", "tail_concentration", "prob_band_95_100", "prob_band_100_105", "hist_da_max_season", "prob_exceed_85", "prob_exceed_80", "recent_hist_da", "season_hist_da_1", "season_hist_da_2", "density_skewness", "density_kurtosis", "density_cv", "season_hist_da_3", "prob_below_85"], "monotone_constraints": [1, 1, 1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, -1]}}
```

---

## Hypothesis B (experiment): Prune zero-filled + add flow_direction (39 → 35)

**What**: Remove 5 zero-filled features AND add `flow_direction` (±1 integer from data loader). Net: 34 effective → 35 features.

**Why**: `flow_direction` is the only remaining data loader column that provides genuinely independent information (not a linear transform of existing features). Import vs export constraints have structurally different shadow price distributions — the direction indicator should help the regressor differentiate them. If B beats A, flow_direction carries real signal; if A is better or tied, the feature is noise for this task.

**Monotone constraint for flow_direction**: 0 (unconstrained) — the relationship between flow direction and shadow price magnitude is not monotonic.

Hypothesis B overrides:
```json
{"regressor": {"features": ["prob_exceed_110", "prob_exceed_105", "prob_exceed_100", "prob_exceed_95", "prob_exceed_90", "prob_below_100", "prob_below_95", "prob_below_90", "expected_overload", "hist_da", "hist_da_trend", "sf_max_abs", "sf_mean_abs", "sf_std", "sf_nonzero_frac", "is_interface", "constraint_limit", "density_mean", "density_variance", "density_entropy", "tail_concentration", "prob_band_95_100", "prob_band_100_105", "hist_da_max_season", "prob_exceed_85", "prob_exceed_80", "recent_hist_da", "season_hist_da_1", "season_hist_da_2", "density_skewness", "density_kurtosis", "density_cv", "season_hist_da_3", "prob_below_85", "flow_direction"], "monotone_constraints": [1, 1, 1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, -1, 0]}}
```

---

## Screen Months

| Month | Role | v0009 Metrics | Rationale |
|-------|------|---------------|-----------|
| **2021-11** | Weak | EV-VC@100=0.0118, Spearman=0.267 (worst), C-RMSE=4776 | Worst Spearman month — since Spearman is the binding gate (4.7% margin), changes must not degrade here. Also tests if pruning/flow_direction helps on high-error months. |
| **2020-11** | Strong | EV-VC@100=0.149, Spearman=0.519 (best), EV-NDCG=0.803 | Best Spearman month with strong EV metrics. "Do no harm" diagnostic — any regression here signals damage to the model's best behavior. |

**Note**: Iter 1 used 2022-06/2022-12. Using different months provides broader diagnostic coverage across the eval set.

---

## Winner Criteria

1. **Primary**: Higher mean EV-VC@100 across the 2 screen months wins
2. **Safety check**: Spearman must not drop > 0.02 on either screen month vs champion v0009 (tighter threshold than iter 1 because Spearman margin is only 4.7%)
3. **Tiebreaker**: Higher mean EV-NDCG across screen months
4. **If both degrade vs champion**: Pick the one with smallest EV-VC@100 degradation. If both are substantially worse (> 10% mean EV-VC@100 drop), STOP and escalate.
5. **If A and B are within noise** (<2% mean EV-VC@100 difference): Prefer A (parsimony — fewer features, simpler model)

---

## Code Changes for Winner

1. **For Hypothesis A winner**: No code changes to `ml/features.py` needed. The worker creates a new version config.json with the 34-feature list, removing the 5 zero-filled features and their monotone_constraints entries.

2. **For Hypothesis B winner**: The worker must:
   - Verify `flow_direction` is already available as a raw column from the data loader (it should be — `data_loader.py` computes it as ±1)
   - Add `"flow_direction"` to `_ADDITIONAL_FEATURES` in `ml/config.py` (if needed for the pipeline to include it)
   - Ensure `flow_direction` is included in the feature preparation step in `ml/features.py` (may already be passthrough from data loader)
   - Create config.json with 35 features and 35 monotone_constraints

3. **For both winners**: Remove the 5 zero-filled features from `_REGRESSOR_FEATURES` in `ml/config.py`:
   - Delete: `hist_physical_interaction`, `overload_exceedance_product`, `band_severity`, `sf_exceed_interaction`, `hist_seasonal_band`
   - Also remove the `compute_interaction_features()` function from `ml/features.py` if it only produces these 5 features
   - Update `_REGRESSOR_MONOTONE` to match the new feature list length

4. **Verification** (CRITICAL): After the full 12-month benchmark, confirm:
   - config.json has exactly 34 features (A) or 35 features (B) and matching monotone_constraints length
   - The 5 zero-filled features are NOT in the feature list
   - `value_weighted=false` and `unified_regressor=false` (batch constraint)
   - All regressor HPs match v0009 exactly (n_estimators=400, max_depth=5, lr=0.05, subsample=0.8, colsample=0.8, reg_alpha=1.0, reg_lambda=1.0, mcw=25)
   - Classifier config is UNCHANGED from v0009 (14 features, frozen)

---

## Expected Impact

| Gate | Hyp A (prune only) | Hyp B (prune + flow_direction) |
|------|--------------------|---------------------------------|
| EV-VC@100 | ±0-2% (neutral to slight positive) | +1-4% (if direction differentiates constraint pricing) |
| EV-VC@500 | ±0-1% | +0-3% |
| EV-NDCG | ±0-0.5% | +0-1% |
| Spearman | ±0% (cleanup should be transparent) | ±0-1% (direction adds information, not noise) |
| C-RMSE | ±0-1% | -1-3% (direction may reduce residuals on directional constraints) |

**Realistic expectations**: Pruning zero-filled features is primarily hygienic — XGBoost already mostly ignores constant columns, so the effect should be small. The real test is whether `flow_direction` provides meaningful signal. Given that import/export constraints have different economics, there's a reasonable prior that direction matters for magnitude estimation.

---

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Feature cleanup changes results unexpectedly | Very low | XGBoost ignores constant features; removing them is transparent cleanup |
| flow_direction doesn't exist in processed data | Low | Confirmed in data_loader.py:362 as ±1 parameter; used as raw column |
| flow_direction adds noise, degrading weak months | Low | Unconstrained monotone gives XGBoost freedom to ignore it. Screen on 2021-11 catches regression. |
| Spearman regression from feature changes | Low | Tighter safety check (0.02 threshold) protects the binding gate |
| Worker uses wrong feature list | Low | Explicit feature counts (34 or 35) and verification checklist |
| flow_direction is correlated with is_interface | Medium | Both capture constraint structure. If highly correlated, B adds no value over A — that's fine, we pick A. |
