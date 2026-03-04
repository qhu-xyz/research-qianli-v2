# Feature Engineering Focus — Parallel Batch

## Constraint
This batch is EXCLUSIVELY about feature engineering and feature selection for the **regressor**.
- Do NOT change hyperparameters (n_estimators, learning_rate, max_depth, subsample, etc.)
- Do NOT change the classifier (it is frozen)
- Do NOT change pipeline architecture or evaluation harness
- ONLY modify: regressor feature list, monotone_constraints, and interaction feature computation in ml/features.py

## Starting point
v0007 champion: 34 regressor features (see registry/v0007/config.json).

## Iteration protocol
Each iteration:
1. **Research**: Analyze feature importance from the current champion. Identify dead/low-value features and potential new features or interaction terms from available data columns in MisoDataLoader.
2. **Generate 2 hypotheses** using `--overrides` JSON. Examples:
   - H1: Drop the 5 lowest-importance features to reduce noise
   - H2: Add 3 new interaction terms (e.g., prob_exceed_100 * density_entropy)
3. **Screen both** on 2 months: 2022-06 (weak) and 2021-09 (strong)
4. **Implement winner** in code, run full 12-month benchmark

## Reporting
Report on **target month only** — val set is NOT used for reporting metrics.

## Feature ideas to explore
- Feature pruning: which of the 34 features contribute least? Remove noise.
- New interaction terms: ratios or products of exceedance/density/shift-factor features
- Temporal features: any untapped lagged or seasonal features in the data loader
- Monotone constraint audit: are current constraints correct per domain knowledge?
- Consider: exceed_severity_ratio = prob_exceed_110 / (prob_exceed_90 + 1e-6)
