# Direction — Iter 3 (feat-eng-3-20260303-104101)

## Hypothesis H12: More Trees + Slower Learning Rate (Final Optimization)

**Core idea**: v0009 confirmed 29 features with 17.13% interaction importance — the feature space is well-explored and saturated. Both reviewers independently recommend against adding more features. The model currently uses n_estimators=200 with learning_rate=0.1 — these are the original v0 defaults, unchanged through 9 experiments. With 29 features and strong interaction signal, the model may be under-treed: 200 trees at learning_rate=0.1 produces an effective model capacity that may not fully exploit the richer feature space. Increasing to 300 trees with learning_rate=0.07 maintains the total gradient magnitude (~21 vs ~20) while allowing the model to make finer-grained splits — a standard final-stage optimization.

**Secondary target**: Improve BRIER further (currently 0.1376, best ever) — slower learning typically improves calibration. Also monitor whether NDCG bot2 improves (currently 0.6648, tightest L3 constraint).

## Specific Changes

### 1. `ml/config.py` — HyperparamConfig.n_estimators

Change from 200 to 300:
```python
n_estimators: int = 300
```

### 2. `ml/config.py` — HyperparamConfig.learning_rate

Change from 0.1 to 0.07:
```python
learning_rate: float = 0.07
```

### 3. `ml/tests/test_config.py` — Update HP assertions

Update any assertions that check n_estimators (200→300) and learning_rate (0.1→0.07).

### What NOT to change

- **features**: Keep all 29 features exactly as-is. Do NOT add, remove, or reorder.
- **colsample_bytree**: Keep at 0.9 (validated by v0009)
- **threshold_beta**: Keep at 0.7
- **train_months**: Keep at 14 (HARD MAX)
- **Other HPs**: Keep subsample=0.8, max_depth=4, reg_alpha=0.1, reg_lambda=1.0, min_child_weight=10
- **gates.json and evaluate.py**: NEVER modify
- **Feature computations in features.py**: No changes
- **data_loader.py**: No changes

### Worker verification checklist

1. **After config change**: Run smoke tests (`SMOKE_TEST=true python -m pytest ml/tests/ -v`)
2. **After real data run**: Verify 29 features in model (unchanged from v0009)
3. **Critical checks**:
   - BRIER should not degrade (currently 0.1376). If BRIER worsens > 0.001, the extra trees may be overfitting.
   - AUC should maintain or improve (currently 0.8495).
   - NDCG bot2 ≥ 0.6448 (L3 floor with v0009 as champion). Target: maintain or improve.
   - All other Group A L3 floors: AUC ≥ 0.7989, AP ≥ 0.3512, VCAP@100 ≥ -0.0111.
4. **Training time**: Expect ~50% longer training (300 vs 200 trees). This is acceptable for the final iteration.

## Expected Impact

| Metric | v0009 (champion) | Expected direction | Reasoning |
|--------|------------------|-------------------|-----------|
| S1-AUC | 0.8495 | Maintain or slight improve | More trees → finer discrimination at decision boundaries |
| S1-AP | 0.4445 | Maintain or slight improve | Same reasoning; AP is ranking-based |
| S1-NDCG | 0.7359 | Maintain or slight improve | More trees help with finer relative ordering |
| S1-VCAP@100 | 0.0266 | Maintain | Top-100 quality is more feature-dependent than tree-count dependent |
| S1-BRIER | 0.1376 | **Improve** | Slower learning rate is the strongest known lever for calibration |

**Key expectation**: This is a conservative change. The most likely outcome is modest improvement across all metrics (0.001-0.003) with no single metric showing large regression. If results are flat, that confirms the model has reached its capacity ceiling with 29 features and the pipeline is ready for HUMAN_SYNC.

## Risk Assessment

1. **Overfitting**: LOW — BRIER has been improving for 3 consecutive versions. 300 trees with lr=0.07 and colsample_bytree=0.9 is conservative. The model also has max_depth=4 and min_child_weight=10 as structural regularizers. If BRIER degrades > 0.001, this would be the signal.
2. **AUC regression**: LOW — more trees with lower LR rarely hurts discrimination. The effective gradient magnitude is similar (300 × 0.07 = 21 vs 200 × 0.1 = 20).
3. **Training time**: EXPECTED — ~50% increase. Not a concern for a single evaluation.
4. **Null result**: MEDIUM — if the model is already at capacity, more trees won't help. This is acceptable for the final iteration.

## Layer 3 Non-Regression Floors (v0009 as champion)

| Metric | Champion bot2 | L3 floor (bot2 - 0.02) |
|--------|--------------|------------------------|
| S1-AUC | 0.8189 | **0.7989** |
| S1-AP | 0.3712 | **0.3512** |
| S1-VCAP@100 | 0.0089 | **-0.0111** |
| S1-NDCG | 0.6648 | **0.6448** (tightest) |

## Why This Direction and Not Others

- **NOT more features**: Both reviewers independently recommend against more features. 29 features with 17.13% from interactions is well-balanced. Feature engineering is saturated — each marginal feature adds less unique discrimination signal.
- **NOT colsample_bytree 1.0**: 0.9 was validated by v0009 with no overfitting. Moving to 1.0 removes feature subsampling regularization entirely — unnecessary risk.
- **NOT max_depth increase**: max_depth=4 with 29 features is well-calibrated. Deeper trees risk overfitting on 270K-row training sets. HP tuning (v0003-HP) showed depth changes don't help.
- **NOT subsample change**: subsample=0.8 is standard. No evidence it's suboptimal.
- **NOT regularization changes**: reg_alpha=0.1 and reg_lambda=1.0 are reasonable defaults. Without evidence of overfitting, no reason to change.

## Batch Status

- **Iter 1**: H10 — 7 new features. **PROMOTED as v0008.**
- **Iter 2**: H11 — 3 interactions + colsample_bytree=0.9. **PROMOTED as v0009.**
- **Iter 3** (this): H12 — n_estimators 200→300, learning_rate 0.1→0.07. **Final optimization pass.**

This is the last iteration of batch feat-eng-3-20260303-104101. After this iteration, the orchestrator will produce an executive summary and prepare for HUMAN_SYNC.
