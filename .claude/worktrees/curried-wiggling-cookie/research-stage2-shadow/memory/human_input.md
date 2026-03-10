# Per-Batch Human Input — Feature Engineering Only

## Batch Constraint: FEATURE ENGINEERING / SELECTION ONLY

**This batch is restricted to feature engineering and feature selection/deselection.**
- NO hyperparameter changes (n_estimators, max_depth, learning_rate, subsample, colsample_bytree, reg_alpha, reg_lambda, min_child_weight must stay at v0007 defaults)
- NO training mode changes (unified_regressor, value_weighted stay as-is)
- ONLY changes allowed: features list, monotone_constraints list, and new feature computation in ml/features.py

## Target Phase: REGRESSOR (most impactful)

The classifier is frozen. All FE/selection targets the regressor phase.

## What to Explore

### 1. Unused raw columns (available from MisoDataLoader but not in regressor)

The data loader provides 46 columns. The regressor uses 34 features but MISSES these raw columns:

| Column | Description | Potential |
|--------|-------------|-----------|
| `density_skewness` | Skewness of flow distribution | In classifier but NOT regressor — high-value omission |
| `density_kurtosis` | Kurtosis of flow distribution | In classifier but NOT regressor — high-value omission |
| `density_cv` | Coefficient of variation | In classifier but NOT regressor — measures flow variability |
| `prob_below_105` | P(flow < 105% limit) | Complements prob_below_100 |
| `prob_below_110` | P(flow < 110% limit) | Complements prob_below_100 |
| `prob_below_80` | P(flow < 80% limit) | Deep below threshold |
| `prob_below_85` | P(flow < 85% limit) | Below threshold |
| `season_hist_da_3` | 3rd seasonal DA component | Complements season_hist_da_1/2 |
| `flow_direction` | Direction of flow (int) | Could differentiate constraint behavior |

Note: `density_skewness`, `density_kurtosis`, `density_cv` are used by the v0 classifier (14 features) but were NEVER added to the regressor. This is likely an oversight — the regressor should benefit from these distributional shape features.

### 2. Feature engineering opportunities (derived from existing columns)

- **Ratio features**: expected_overload / constraint_limit, sf_max_abs / constraint_limit
- **Spread/range features**: prob_exceed_100 - prob_below_100 (binding probability margin)
- **Interaction features**: hist_da * tail_concentration, expected_overload * density_mean
- **Nonlinear transforms**: log1p(hist_da), sqrt(expected_overload)
- **Cross-band features**: prob_exceed_110 - prob_exceed_105 (incremental exceedance)

### 3. Feature deselection

- L1=1.0 had negligible effect (learning #6), suggesting all 34 features carry signal
- However, feature importance analysis could identify low-value features for removal
- Removing noise features could improve generalization on weak months

## Strategy: Choose Most Impactful

The orchestrator should prioritize hypotheses by expected impact:

**HIGH impact** (likely): Adding the 3 missing distributional features (density_skewness, density_kurtosis, density_cv) — these are proven useful in the classifier and were simply never included in the regressor.

**MEDIUM impact** (worth testing): Engineered ratio/interaction features that capture relationships between raw signals.

**LOWER impact** (per learning #6): Pure deselection — unlikely to help since L1 showed all features carry signal.

## Worker: Allowed File Modifications

- `ml/features.py` — add new feature computation functions
- `ml/config.py` — modify `_ADDITIONAL_FEATURES`, `_ADDITIONAL_MONOTONE`, `_REGRESSOR_FEATURES`, `_REGRESSOR_MONOTONE` (NOT classifier features)
- `ml/tests/` — update tests for new features
- `registry/${VERSION_ID}/` — version artifacts
