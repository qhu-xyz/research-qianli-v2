# Direction — Iteration 1 (feat-eng-20260303-060938, v0003)

## Hypothesis: H6 — Combined 14-Month Window + Interaction Features (Additivity Test)

**Core question**: Are the two positive-signal levers (window expansion, interaction features) additive?

Three real-data experiments have isolated individual levers:
- **HP tuning (v0003-HP)**: AUC -0.0025 (0W/11L) — REFUTED. Model not complexity-limited.
- **Interaction features (v0002)**: AUC +0.0000 (5W/6L/1T) — NOT SUPPORTED. But NDCG 8W/4L, AP 7W/5L show marginal ranking signal.
- **Window expansion (v0003)**: AUC +0.0013 (7W/4L/1T) — INCONCLUSIVE. Best lever so far. VCAP@100 9W/3L (p≈0.07).

If effects are additive, we expect AUC ~0.836–0.838, NDCG ~+0.0035, VCAP@100 ~+0.0043. Even partial additivity would be the strongest result yet.

## Specific Changes

### 1. Add 3 interaction features to FeatureConfig (MAIN CHANGE)

**File**: `ml/config.py` → `FeatureConfig.step1_features`

Add these 3 tuples after the existing 14 base features:
```python
# --- Interaction features ---
("exceed_severity_ratio", 1),      # prob_exceed_110 / (prob_exceed_90 + 1e-6)
("hist_physical_interaction", 1),   # hist_da * prob_exceed_100
("overload_exceedance_product", 1), # expected_overload * prob_exceed_105
```

Total features: 14 → 17. All three are monotone +1 (higher = more likely to bind).

The computation logic already exists in `ml/features.py:prepare_features()` (lines 38–47) — it dynamically computes interaction columns when they appear in `config.features`. No changes needed in `features.py`.

### 2. Keep train_months=14 (NO CHANGE)

`ml/config.py` → `PipelineConfig.train_months` is already 14 from the previous v0003. Verify it is still 14 before running — do NOT change it.

### 3. Keep v0 HP defaults (NO CHANGE)

`ml/config.py` → `HyperparamConfig` should remain:
- n_estimators=200, max_depth=4, learning_rate=0.1
- subsample=0.8, colsample_bytree=0.8
- reg_alpha=0.1, reg_lambda=1.0, min_child_weight=10

### 4. Fix f2p parsing crash (BUG FIX — HIGH)

**File**: `ml/benchmark.py` (or wherever `int(ptype[1:])` is used for cascade stage parsing)

**Problem**: `int(ptype[1:])` crashes for ptype="f2p" → `int("2p")` raises ValueError. This blocks cascade stage-3 evaluation.

**Fix**: Replace `int(ptype[1:])` with a robust parser. Options:
- Use a mapping dict: `{"f0": 0, "f1": 1, "f2p": 2}`
- Or use regex: `int(re.match(r'f(\d+)', ptype).group(1))`
- Test with all 3 cascade ptypes: f0, f1, f2p

### 5. Fix dual-default fragility in benchmark.py (BUG FIX — MEDIUM)

**File**: `ml/benchmark.py`

**Problem**: `_eval_single_month()` and `run_benchmark()` have `train_months=14` hardcoded in function signatures alongside `PipelineConfig.train_months=14`. If one changes, the other must change in lockstep — fragile.

**Fix**: Use `None` sentinel with fallback:
```python
def _eval_single_month(
    ...,
    train_months: int | None = None,
    val_months: int | None = None,
    ...
):
    # Resolve from PipelineConfig if not provided
    if train_months is None:
        train_months = PipelineConfig().train_months
    if val_months is None:
        val_months = PipelineConfig().val_months
```

Apply the same pattern to `run_benchmark()`.

### 6. Update tests for 17 features

**File**: `ml/tests/` — update any test fixtures that hardcode 14-wide feature arrays to 17-wide (or make them dynamic from FeatureConfig).

Codex flagged (LOW) that synthetic fixtures are 17-wide while production config was 14 — this was from a previous iteration. Now that we're going back to 17 features, ensure fixtures match.

## Execution Order

1. Make config change (add 3 interaction features)
2. Fix f2p parsing crash
3. Fix dual-default fragility
4. Run tests: `python -m pytest ml/tests/ -v`
5. Run benchmark pipeline (full 12-month eval)
6. Run validate + compare against v0
7. Commit, then write handoff

## Expected Impact

| Metric | v0 Baseline | Expected (if additive) | Expected (if not additive) |
|--------|-------------|----------------------|---------------------------|
| S1-AUC | 0.8348 | 0.836–0.838 | ~0.836 |
| S1-AP | 0.3936 | 0.396–0.400 | ~0.395 |
| S1-VCAP@100 | 0.0149 | 0.019–0.023 | ~0.018 |
| S1-NDCG | 0.7333 | 0.737–0.740 | ~0.736 |
| AUC W/L | — | ≥8/12 (additive) | 6-7/12 (not additive) |

**Success criteria** (from human input):
- **Promotion-worthy**: AUC > 0.837 AND ≥8/12 wins AND AP > 0.396
- **Encouraging**: AUC > 0.835, 7+/12 wins → continue refining in iter 2
- **Dead end**: AUC ≤ 0.835 or <6/12 wins → feature set has hard ceiling, pivot in iter 2

## Risk Assessment

1. **Non-additivity (MEDIUM)**: Window expansion and interactions may overlap in signal — both primarily help early months (2020–2021H1). If so, combined effect ≈ max(individual effects) rather than sum. Mitigation: the experiment still provides valuable information about which features carry independent signal.

2. **Broader ranking degradation (LOW-MEDIUM)**: Both v0002 (interactions) and v0003 (window) showed VCAP@500 and CAP@100/500 regression. Combined may amplify this. Mitigation: acceptable per business objective (top-100 precision > broad ranking), but monitor closely.

3. **BRIER regression (LOW)**: v0003 showed BRIER +0.0011 (slightly worse calibration). Combined with interactions may push BRIER closer to floor (headroom only 0.019). Mitigation: BRIER is Group B (non-blocking), and AUC/ranking improvements are more business-relevant.

4. **2022-09 remains stuck (HIGH likelihood, LOW impact)**: Three independent levers all failed to improve 2022-09 (lowest binding rate 6.63%, AP consistently worst). This iteration will not fix it. Mitigation: document the result; if iter 1 confirms the ceiling, iter 2 can try feature selection to explicitly address this month.

5. **Bug fix scope creep (LOW)**: The f2p and dual-default fixes are surgical. Risk of unintended side effects is minimal since both are in evaluation/benchmarking code, not in the training pipeline itself.

## What NOT To Do

- Do NOT change hyperparameters (proven dead end — 3 experiments confirm)
- Do NOT change threshold_beta (keep 0.7)
- Do NOT change val_months (keep 2)
- Do NOT modify gates.json or evaluate.py
- Do NOT add more than the 3 specified interaction features this iteration
- Do NOT touch registry/v0/ or any other registry/v*/ except registry/v0003/
