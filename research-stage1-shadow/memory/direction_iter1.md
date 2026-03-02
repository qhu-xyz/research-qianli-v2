# Direction — Iteration 1 (Batch: hp-tune-20260302-132826)

## Hypothesis (H3: Hyperparameter Tuning for Ranking Quality)

Increasing tree depth and boosting rounds while reducing learning rate will improve ranking quality (AUC, AP, NDCG) by allowing the model to capture more complex feature interactions in the 270K-row dataset. The v0 baseline used conservative defaults (max_depth=4, n_estimators=200, learning_rate=0.1) that were never tuned for this data. With adequate regularization already in place (subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0), deeper trees at a slower learning rate should generalize well.

## Specific Changes

### File: `ml/config.py` — `HyperparamConfig` class

Change exactly these 4 default values in the `HyperparamConfig` dataclass:

```python
# BEFORE (v0 defaults):
n_estimators: int = 200
max_depth: int = 4
learning_rate: float = 0.1
min_child_weight: int = 10

# AFTER (v0001):
n_estimators: int = 400
max_depth: int = 6
learning_rate: float = 0.05
min_child_weight: int = 5
```

**Do NOT change any other parameters.** Keep:
- `subsample: float = 0.8` (unchanged)
- `colsample_bytree: float = 0.8` (unchanged)
- `reg_alpha: float = 0.1` (unchanged)
- `reg_lambda: float = 1.0` (unchanged)
- `random_state: int = 42` (unchanged)

### File: `ml/config.py` — `PipelineConfig` class

**No changes.** Keep:
- `threshold_beta: float = 0.7` (precision-favoring, business requirement)
- `threshold_scaling_factor: float = 1.0`
- `train_months: int = 10`
- `val_months: int = 2`

### File: `ml/config.py` — `FeatureConfig` class

**No changes.** Keep all 14 features and their monotone constraints exactly as-is.

## What to Run

1. Apply the hyperparameter changes above
2. Run `python -m pytest ml/tests/ -v` to verify tests pass
3. Run the full pipeline: train + evaluate across all 12 primary eval months (f0, onpeak)
4. Run `validate` to confirm metrics.json schema compliance
5. Run `compare` against v0 baseline to check gates
6. Write `changes_summary.md` in `registry/${VERSION_ID}/` documenting:
   - Exact hyperparameter changes
   - Per-month metrics comparison table (all 12 months)
   - Gate pass/fail status for all 3 layers
   - Any anomalies in specific months (especially late-2022)

## Expected Impact

| Metric | v0 Mean | Expected Direction | Reasoning |
|--------|---------|-------------------|-----------|
| S1-AUC | 0.835 | +0.005 to +0.015 | Deeper trees + more rounds improve discrimination |
| S1-AP | 0.394 | +0.010 to +0.030 | AP is more sensitive to improved top-ranking |
| S1-NDCG | 0.733 | +0.005 to +0.015 | Better overall ranking quality |
| S1-VCAP@100 | 0.015 | +0.005 to +0.020 | Better top-of-list separation of high-value constraints |
| S1-BRIER | 0.150 | +/- 0.005 | May shift slightly; monitor but HP changes rarely hurt calibration |
| Precision | 0.442 | Stable or +slight | Better ranking → better precision at same threshold |

**Bottom line**: The primary target is lifting AUC, AP, and NDCG means. Even modest improvements (+0.005 AUC, +0.01 AP) demonstrate the hypothesis and establish a new champion candidate.

## Gate Analysis — What to Watch

### Group A (blocking) — all should pass easily
- **S1-AUC**: Floor=0.785, v0 mean=0.835. Need to stay above 0.785 mean, no month below 0.709. Very comfortable.
- **S1-AP**: Floor=0.344, v0 mean=0.394. Need to stay above 0.344 mean. Watch 2022-09 (v0 worst: 0.315).
- **S1-VCAP@100**: Floor=-0.035 (negative). Effectively non-binding.
- **S1-NDCG**: Floor=0.683, v0 mean=0.733. Watch 2021-04 (v0 worst: 0.660).

### Group B (monitor) — track but don't block
- **S1-BRIER**: Tightest gate. Floor=0.170, v0 mean=0.150. Only 0.02 headroom. If BRIER mean exceeds 0.165, flag as a concern for reviewers.
- **S1-REC**: Very loose floor (0.10). Will pass easily.
- **S1-CAP@100/500**: High variance — may shift month-to-month. Not blocking.

### Layer 3 — Non-Regression Check
Champion is v0 (since no promotion yet). New model's bottom_2_mean for each Group A metric must be >= v0's bottom_2_mean - 0.02:
- S1-AUC bottom_2: v0=0.811 → must be >= 0.791
- S1-AP bottom_2: v0=0.332 → must be >= 0.312
- S1-NDCG bottom_2: v0=0.672 → must be >= 0.652
- S1-VCAP@100 bottom_2: v0=0.001 → must be >= -0.019

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Overfitting (deeper trees memorize noise) | Low-Medium | AUC/AP degrade on test months | Learning rate halved (0.05), regularization intact. If test months degrade, revert max_depth to 5. |
| BRIER degradation | Low | Monitor gate approaches ceiling | HP changes alone rarely hurt calibration. If mean BRIER > 0.165, flag in changes_summary. |
| Compute timeout | Low | Worker times out during training | 400 trees × 12 months ≈ 2x v0 compute. Should fit in 600s timeout with margin. |
| Late-2022 months get worse | Medium | Bottom-2 regression on AP | These months (2022-09, 2022-12) were already v0's weakest. If they degrade, the improvement in other months may compensate. Document per-month deltas. |

## Rationale for This First Move

Hyperparameter tuning is the lowest-risk, highest-signal first experiment for a new real-data batch:

1. **No code architecture changes** — only 4 numeric constants change
2. **Well-understood mechanism** — more trees + slower learning + deeper splits is a standard XGBoost improvement path
3. **Directly addresses business objective** — ranking improvements (AUC, AP, NDCG) improve precision at any threshold
4. **Establishes improvement baseline** — if HP tuning lifts metrics, we know the v0 defaults were suboptimal and future iterations can refine further (e.g., feature engineering, training window adjustments)
5. **Human directive** — this is explicitly what the human requested for the first real-data batch
