# Changes Summary — v0003 (Iteration 1, Batch hp-tune-20260302-134412)

## Hypothesis

**H3: Hyperparameter tuning improves ranking quality over untuned v0 defaults.**

## Changes Made

### File: `ml/config.py` → `HyperparamConfig`

| Param | v0 | v0003 | Rationale |
|---|---|---|---|
| `n_estimators` | 200 | 400 | More boosting rounds to compensate for halved learning rate |
| `max_depth` | 4 | 6 | Deeper trees for better feature interactions at 270K rows/month |
| `learning_rate` | 0.1 | 0.05 | Slower learning for better generalization |
| `min_child_weight` | 10 | 5 | Finer leaf splits to capture rarer binding patterns |

All other hyperparameters unchanged (subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, random_state=42).

### File: `ml/tests/test_config.py`

Updated `test_hyperparam_defaults` and `test_hyperparam_to_dict` to reflect new default values.

### No changes to:
- Features (14 features, monotone constraints unchanged)
- Pipeline config (threshold_beta=0.7, train_months=10, val_months=2)
- `ml/evaluate.py` or `registry/gates.json` (HUMAN-WRITE-ONLY)

## Results — 12-Month Benchmark (f0, onpeak)

### Group A (blocking gates) — v0003 vs v0

| Gate | v0 Mean | v0003 Mean | Delta | v0 Bot2 | v0003 Bot2 | Delta | Pass |
|------|---------|------------|-------|---------|------------|-------|------|
| S1-AUC | 0.8348 | 0.8323 | -0.0025 | 0.8105 | 0.8089 | -0.0016 | YES |
| S1-AP | 0.3936 | 0.3921 | -0.0015 | 0.3322 | 0.3299 | -0.0023 | YES |
| S1-VCAP@100 | 0.0149 | 0.0164 | +0.0015 | 0.0014 | 0.0007 | -0.0007 | YES |
| S1-NDCG | 0.7333 | 0.7323 | -0.0010 | 0.6716 | 0.6675 | -0.0041 | YES |

### Group B (monitor gates)

| Gate | v0 Mean | v0003 Mean | Delta | Pass |
|------|---------|------------|-------|------|
| S1-BRIER | 0.1503 | 0.1462 | -0.0041 (improved) | YES |
| S1-REC | 0.4192 | 0.4220 | +0.0028 | YES |
| S1-CAP@100 | 0.7825 | 0.7833 | +0.0008 | YES |
| S1-CAP@500 | 0.7740 | 0.7712 | -0.0028 | YES |

### Overall: All gates PASS (Group A: YES, Group B: YES)

## Assessment

The HP tuning had **near-zero net effect** on ranking quality (Group A metrics). All changes are within noise tolerance (±0.005 on AUC/AP/NDCG). The one positive signal is BRIER improving by 0.004 (better calibration), but BRIER is Group B (non-blocking).

**Per-month patterns:**
- v0003 improved on some months (e.g., 2021-06 AUC +0.004, 2021-12 AP +0.006) but regressed on others
- Weakest months (2022-09, 2022-12) remain weak in both versions — the distribution shift in late-2022 is not addressed by these HP changes
- Bottom-2 metrics are marginally worse across the board (within noise tolerance)

**Interpretation:** The v0 defaults (max_depth=4, n_estimators=200, lr=0.1) were already reasonably well-suited. The standard "deeper trees + slower learning" pattern did not produce the expected 0.005–0.015 improvement on Group A metrics. The model may be limited more by feature informativeness than by tree complexity at these data scales.
