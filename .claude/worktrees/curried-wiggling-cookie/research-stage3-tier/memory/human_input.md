# Per-Batch Human Input — Feature Engineering & Selection ONLY

## Batch Constraint (MANDATORY)

**Feature engineering and feature selection ONLY.** No hyperparameter changes. No class weight changes. No tier bin/midpoint changes. Only the feature list, monotone_constraints, and feature computation code (`features.py`) may change.

The 12-month evaluation configuration is FROZEN — do not change train_months, val_months, eval_months, or class_type.

## Current State

- Champion: v0 (34 features, early stopping on val)
- Architecture: single XGBoost multi:softprob, 5 tiers, val-based early stopping
- All hyperparams FROZEN at defaults

## Stage-2 FE Learnings (from feat-eng-3 batch)

These findings from the stage-2 regression pipeline inform what works:

1. **Adding density_skewness/kurtosis/cv, season_hist_da_3, prob_below_85** was the biggest win (+9% EV-VC@100)
2. **Pruning 5 dead features** improved sampling efficiency (+5.2% EV-VC@100)
3. **flow_direction** had no effect (likely constant or not meaningful for shadow prices)
4. **value_weighted** had no effect

## Candidate New Features to Explore

### Interaction Features (not currently used)
These can be computed from existing raw columns:
- `overload_x_hist = expected_overload * hist_da` — historical binding × overload signal
- `prob110_x_hist = prob_exceed_110 * recent_hist_da` — flow exceedance × recent price
- `log1p_hist_da = log1p(hist_da)` — compress long-tailed price history
- `prob_range_high = prob_exceed_100 - prob_exceed_110` — probability mass in 100-110% range
- `tail_x_hist = tail_concentration * hist_da` — density tail × price signal
- `sf_x_overload = sf_max_abs * expected_overload` — shift factor × overload
- `density_range = density_entropy * density_cv` — distribution spread indicator

### Selection Opportunities
- Check feature importance from v0 — any features with near-zero importance should be candidates for removal
- Check `registry/v0/feature_importance.json` for ranked features

## Priority Order

1. **Add 2-3 high-signal interaction features** (pick from candidates above based on domain reasoning)
2. **Prune low-importance features** (check v0 feature importance, drop bottom 3-5 if near zero)
3. **Test new density-derived features** if interactions don't help

## Worker: Allowed File Modifications

- `ml/config.py` — ONLY `_ALL_TIER_FEATURES` list and `_ALL_TIER_MONOTONE` list
- `ml/features.py` — `compute_interaction_features()` function for new derived features
- `ml/tests/` — update tests for feature count changes
- `registry/${VERSION_ID}/` — version artifacts

## Worker: FORBIDDEN Changes

- Any field in TierConfig except `features` and `monotone_constraints`
- `ml/train.py` — no training logic changes
- `ml/pipeline.py` — no pipeline structure changes
- `ml/evaluate.py` — no metric changes
- `registry/gates.json` — no gate changes
