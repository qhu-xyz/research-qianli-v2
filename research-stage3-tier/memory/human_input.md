# Per-Batch Human Input — Tier Classification v0 → v1

## Batch Goal: Improve Tier-Recall@1 and Tier-VC@100

The v0 baseline reveals two critical weaknesses:
1. **Tier-Recall@1 = 0.098** (catastrophically low — missing 90% of strongly binding constraints)
2. **Tier-VC@100 = 0.075** (top-100 ranking quality very poor)

## Allowed Changes

### Hyperparameter Tuning
- Class weights (especially tier 1 weight — currently only 5)
- min_child_weight (currently 25 — may be too conservative for rare tiers)
- n_estimators, max_depth, learning_rate
- reg_alpha, reg_lambda, subsample, colsample_bytree

### Feature Engineering
- New derived features from existing 34
- Feature selection/deselection

### Tier Configuration
- Consider reducing to 4 classes (tier 4 always has 0 samples)
- Adjust tier bin edges if well-motivated
- Adjust tier midpoints for EV scoring

## Priority Hypotheses

### 1. Aggressive Class Weight Rebalancing (HIGH priority)
Current: {0:10, 1:5, 2:2, 3:1, 4:0.5}
Try: {0:15, 1:15, 2:3, 3:1, 4:0.5} — dramatically increase tier 1 weight

### 2. Lower min_child_weight (HIGH priority)
Current: 25 → Try: 5-10
Rationale: 25 is very conservative — prevents fine-grained splits that could help detect rare tier 0/1 patterns

### 3. Drop Tier 4 / 4-Class Model (MEDIUM priority)
Tier 4 has 0 samples in all months. Model wastes capacity on an impossible class.
Change bins to [0, 100, 1000, 3000, inf] with labels [3, 2, 1, 0] and num_class=4

## Worker: Allowed File Modifications
- `ml/config.py` — TierConfig parameters
- `ml/features.py` — feature computation
- `ml/train.py` — training logic (if needed for class reduction)
- `ml/tests/` — update tests for changes
- `registry/${VERSION_ID}/` — version artifacts
