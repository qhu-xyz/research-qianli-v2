# Direction — Iteration 1 (Batch hp-tune-20260302-134412)

## Hypothesis

**H3: Hyperparameter tuning improves ranking quality over untuned v0 defaults.**

The v0 baseline uses XGBoost defaults (max_depth=4, n_estimators=200, lr=0.1, min_child_weight=10) that were never optimized on real data (~270K rows/month, 14 features, 7.5% binding rate). Deeper trees with slower learning should better capture feature interactions and improve discrimination without overfitting, given the dataset size and existing regularization.

## Specific Changes

### File: `ml/config.py` → `HyperparamConfig`

Change exactly these 4 defaults:

```python
# BEFORE (v0 defaults)
n_estimators: int = 200
max_depth: int = 4
learning_rate: float = 0.1
min_child_weight: int = 10

# AFTER (iter1)
n_estimators: int = 400
max_depth: int = 6
learning_rate: float = 0.05
min_child_weight: int = 5
```

**Do NOT change any other parameter.** Specifically, keep:
- `subsample: float = 0.8` (unchanged)
- `colsample_bytree: float = 0.8` (unchanged)
- `reg_alpha: float = 0.1` (unchanged)
- `reg_lambda: float = 1.0` (unchanged)
- `random_state: int = 42` (unchanged)

### File: `ml/config.py` → `PipelineConfig`

**No changes.** Keep:
- `threshold_beta: float = 0.7` (unchanged — precision-favoring, business requirement)
- `threshold_scaling_factor: float = 1.0` (unchanged)
- `train_months: int = 10` (unchanged)
- `val_months: int = 2` (unchanged)

### File: `ml/config.py` → `FeatureConfig`

**No changes.** Keep all 14 features and monotone constraints as-is.

## Rationale for Each Change

| Param | v0 | New | Why |
|---|---|---|---|
| `max_depth` | 4 | 6 | At 270K rows, depth 4 likely underfits. Depth 6 allows 3-way feature interactions (e.g., prob_exceed_110 × hist_da × density_skewness) without extreme complexity. |
| `n_estimators` | 200 | 400 | More boosting rounds compensate for the halved learning rate. Combined effect: slower, more careful learning. |
| `learning_rate` | 0.1 | 0.05 | Halved rate + doubled trees is a standard XGBoost pattern for better generalization. Each tree contributes less, reducing overfitting risk from deeper trees. |
| `min_child_weight` | 10 | 5 | Allows finer leaf splits. At 270K rows with ~20K positives per month, min_child_weight=5 is still conservative. Enables the model to capture rarer binding patterns. |

## Expected Impact

### Group A (blocking) — target improvements:
| Gate | v0 Mean | Expected | Reasoning |
|------|---------|----------|-----------|
| S1-AUC | 0.835 | 0.840–0.850 | Better discrimination from deeper trees + slower learning |
| S1-AP | 0.394 | 0.405–0.425 | AP benefits most from better ranking of rare positives (7.5% binding rate) |
| S1-NDCG | 0.733 | 0.740–0.750 | Better ordering of predicted probabilities |
| S1-VCAP@100 | 0.015 | 0.015–0.030 | May improve if model better separates high-value constraints, but highly variable (std=0.012) |

### Group B (monitor) — expected effects:
| Gate | v0 Mean | Expected Direction |
|------|---------|-------------------|
| S1-BRIER | 0.150 | Stable or slight increase. Floor=0.170, headroom=0.02. Flag if >0.165. |
| S1-REC | 0.419 | Stable — threshold is tuned per-month, ranking changes may shift it slightly |
| S1-CAP@100 | 0.783 | Likely stable — high variance (std=0.25) makes this hard to shift systematically |

### Tail risk:
- **Bottom-2 AUC** (v0: 0.811 from months 2022-12, 2022-09) should improve — deeper trees help with harder periods
- **Bottom-2 AP** (v0: 0.332 from months 2022-09, 2022-12) is the main target — these late-2022 months may have distribution shift that deeper trees can handle better
- **Bottom-2 NDCG** (v0: 0.672 from months 2021-04, 2021-06) should improve modestly

## Risk Assessment

### Low risk:
- **Overfitting**: Mitigated by lower learning rate (0.05), existing regularization (subsample=0.8, colsample=0.8, L1=0.1, L2=1.0), and moderate depth (6, not extreme). 270K rows per month provides ample data.
- **Gate regression**: All Group A gates have +0.05 headroom from mean to floor. Even a slight regression on some months won't break gates. Layer 3 (bottom_2 non-regression) requires bottom_2_mean >= champion_bottom_2_mean - 0.02, and v0 is the baseline.

### Medium risk:
- **BRIER degradation**: Only 0.02 headroom, but BRIER is Group B (non-blocking). Deeper trees may slightly hurt calibration. If BRIER mean exceeds 0.165, note it for reviewers.
- **Compute time**: 400 trees × depth 6 ≈ 2-3x more compute than v0 (200 × depth 4). Should fit within worker timeout (3000s), but monitor.

### No risk:
- **Precision degradation**: These are ranking-quality changes. The threshold (beta=0.7) is tuned per-month from validation data and is independent of tree hyperparameters. Precision at the tuned threshold may actually improve if the model's probability estimates become more accurate.

## Worker Checklist

1. Read `VERSION_ID` from `${PROJECT_DIR}/state.json` (not worktree copy)
2. Modify `ml/config.py` → `HyperparamConfig` (4 param changes only)
3. Run `python -m ml.pipeline run --version-id ${VERSION_ID}` for all 12 eval months
4. Run `python -m ml.pipeline validate --version-id ${VERSION_ID}`
5. Run `python -m ml.pipeline compare --version-id ${VERSION_ID} --baseline v0`
6. Verify all Group A gates pass all 3 layers
7. Check BRIER is below 0.165 (note if between 0.165–0.170)
8. Commit changes, write `changes_summary.md` and `comparison.md`
9. Write handoff JSON with status and results
