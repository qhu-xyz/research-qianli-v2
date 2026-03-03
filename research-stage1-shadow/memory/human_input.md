# Human Input — Feature Engineering Batch 2 (3 iterations)

## Overall Research Direction

**Priority A: Aggressive feature engineering.** There are 15+ features available in the source data loader that we haven't tried. The current ~0.836 AUC ceiling is a feature ceiling, not a model ceiling. We need to add MANY new features, not tweak existing ones.

**Priority B: Minor param tweaks.** train_months=14 is confirmed optimal and is the HARD MAXIMUM. Do not go above 14.

Read `memory/research_direction.md` for the full catalog of unused features and engineering opportunities.

## What We've Learned (6 real-data experiments, 2 batches)

1. **HP tuning is a dead end.** v0 defaults are near-optimal. (3 experiments)
2. **Interaction features alone don't break through.** AUC +0.0000 with 3 interactions.
3. **Training window 14 months is optimal.** AUC +0.0013 vs 10-month. 18-month was worse.
4. **train_months=14 is the ABSOLUTE MAXIMUM.** Human constraint — do NOT try 15, 16, 18, or higher.
5. **Window + interactions are partially additive.** v0004 (AUC 0.8363, VCAP@100 0.0205) is best version. VCAP@100 10W/2L (p=0.039) — first statistically significant improvement.
6. **Feature pruning results pending** (v0006, current batch iter 3). Even if successful, the gains are marginal.
7. **The model is feature-starved, not HP-starved.** The source data loader produces shift factor features, constraint metadata, distribution statistics, probability band features, and temporal features — NONE of which we currently use.

## Strategy for This Batch: Add Many New Features

### Iter 1: Add shift factor + constraint metadata features (H9)
- **Add 4 shift factor features**: sf_max_abs, sf_mean_abs, sf_std, sf_nonzero_frac
- **Add 2 constraint metadata features**: is_interface, constraint_limit
- These are entirely new signal categories (network topology + structural) independent from the density curve
- Requires updating `ml/features.py` to pass these through from source data (they're already computed by the source loader)
- Requires updating `ml/config.py` FeatureConfig with monotone constraints: sf_max_abs→1, sf_mean_abs→1, sf_std→0, sf_nonzero_frac→0, is_interface→0, constraint_limit→0
- Keep train_months=14, v0 HP defaults
- Expected impact: LARGE if network topology is discriminative (and it should be — where a constraint sits in the network matters)

### Iter 2: Add distribution shape + probability band features (H10)
- **Add 3 distribution features**: density_mean, density_variance, density_entropy
- **Add 3 probability band features**: tail_concentration, prob_band_95_100, prob_band_100_105
- These are already computed by the source loader, just need to be wired through
- Monotone: density_mean→1, density_variance→0, density_entropy→0, tail_concentration→1, prob_band_95_100→1, prob_band_100_105→1
- Can be COMBINED with iter 1 features (cumulative) OR tested independently depending on iter 1 results

### Iter 3: Add derived interaction features + hist_da_max_season (H11)
- **Add hist_da_max_season** (peak seasonal signal)
- **Create new interactions**: sf_max_abs × prob_exceed_100, hist_da × sf_max_abs
- **Create ratio features**: prob_exceed_110 / prob_exceed_90, expected_overload / prob_exceed_100
- These require code changes in features.py to compute new columns

### Alternative: If iter 1 shows big gains from SF features
- Iter 2: Combine ALL available features at once (kitchen sink approach ~25 features)
- Iter 3: Feature selection via importance to trim back to most valuable ~15-18 features
- This is the "add everything, then prune" strategy

## Technical Notes for Worker

The source data loader (`research-spice-shadow-price-pred-qianli/src/shadow_price_prediction/data_loader.py`) already computes these features in `load_data_for_outage()` → `calculate_direction_features()`. They flow through `load_training_data()` into the DataFrame. The shadow pipeline's `ml/data_loader.py` calls the source loader and converts to polars — the new columns should already be present in the DataFrame. The worker just needs to:
1. Verify the columns exist in the loaded DataFrame (add a print of available columns)
2. Add them to `FeatureConfig.step1_features` with correct monotone constraints
3. Handle them in `features.py` (they don't need computation like interaction features — they should already be in the DataFrame)
4. Update tests

## Business Constraint (unchanged)
**Precision over recall.** Do NOT lower threshold, do NOT increase beta above 1.0, do NOT optimize for recall.

## What NOT to Do
- No HP tuning (proven dead end — 5 experiments confirm)
- No threshold_beta changes (keep 0.7)
- No architecture changes (XGBoost + monotone constraints is correct)
- **No train_months above 14** (HARD MAX — human constraint)
- Don't spend more than 1 iteration on something that shows zero signal

## Success Criteria
- **Promotion-worthy**: AUC > 0.840 AND at least 8/12 months winning AND AP > 0.400
- **Encouraging**: AUC > 0.837 with 7+/12 wins (continue refining)
- **Marginal**: AUC 0.835-0.837 → the new features helped but not enough yet
- **Dead end signal**: AUC ≤ 0.835 or fewer than 6/12 wins → pivot approach

## Gate Notes
- Don't change gate floors — we need more data points before calibrating
- Layer 3 (non-regression) is effectively disabled since champion=null — this is fine for now
- Promote the first version that meets promotion criteria above
