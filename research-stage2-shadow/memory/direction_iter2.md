# Direction — Iteration 2

**Batch**: smoke-test-20260303-223300
**Champion**: v0 (baseline) — unchanged, iter 1 worker failed
**Iteration**: 2 (recovery from iter 1 failure)

## Context

Iteration 1 planned value-weighted regressor training but the worker **phantom-completed** — it wrote a handoff JSON claiming "done" but produced no artifacts (no registry/v0001/, no metrics, no code changes, no commits). The hypothesis was never tested.

This iteration retries the core hypothesis with a **simplified scope**: only one change (value_weighted=True), keeping all hyperparams at v0 defaults. This isolates the effect of value weighting and reduces implementation complexity.

## Hypothesis

**H1 (retry): Value-weighted regressor training improves EV-based ranking quality by teaching the model to prioritize accuracy on high-value constraints.**

Same rationale as iter 1: the v0 regressor treats a $0.50 and $50 constraint equally during training. The business objective (EV-VC@100, EV-VC@500) rewards accurate ranking of high-value constraints. By weighting samples proportional to their log1p target, the model allocates more capacity to high-value predictions.

## Specific Changes

### 1. Edit `ml/pipeline.py` — Wire up value-weighted sample weights

**Location**: Phase 4, between line 209 (end of the if/else block that sets `X_reg_fit`/`y_reg_fit`) and line 211 (the `train_regressor` call).

**Replace this exact line** (line 211):
```python
        reg_model = train_regressor(X_reg_fit, y_reg_fit, config.regressor)
```

**With this block:**
```python
        # Compute value-weighted sample weights if configured
        sample_weight = None
        if config.regressor.value_weighted:
            sample_weight = y_reg_fit.copy()
            sample_weight = np.maximum(sample_weight, 0.1)  # floor to avoid zero weights
            sample_weight = sample_weight / sample_weight.mean()  # normalize mean=1
            print(f"[phase 4] value_weighted: w_min={sample_weight.min():.3f} "
                  f"w_max={sample_weight.max():.3f} w_mean={sample_weight.mean():.3f}")

        reg_model = train_regressor(X_reg_fit, y_reg_fit, config.regressor,
                                    sample_weight=sample_weight)
```

**Why this works**: `y_reg_fit` is the log1p-transformed target. Using it as the weight upweights high-value constraints. The 0.1 floor prevents zero weights for near-zero shadow prices. Normalization to mean=1 preserves the effective learning rate.

### 2. Set config — ONLY change value_weighted

The worker should use these `RegressorConfig` values:

| Parameter | v0 | v2 (this iter) | Changed? |
|---|---|---|---|
| `value_weighted` | False | **True** | YES — the only change |
| `n_estimators` | 400 | 400 | No |
| `learning_rate` | 0.05 | 0.05 | No |
| `max_depth` | 5 | 5 | No |
| `subsample` | 0.8 | 0.8 | No |
| `colsample_bytree` | 0.8 | 0.8 | No |
| `reg_alpha` | 0.1 | 0.1 | No |
| `reg_lambda` | 1.0 | 1.0 | No |
| `min_child_weight` | 10 | 10 | No |
| `unified_regressor` | False | False | No |

**Rationale for keeping all hyperparams the same**: We need a clean signal. If we change 5 things at once and performance moves, we won't know which change caused it. Isolate the independent variable.

### 3. No feature changes

Keep all 34 regressor features unchanged.

## Verification Checklist (CRITICAL)

Before writing the handoff JSON, the worker MUST verify:

1. **Code change exists**: `grep -n "value_weighted" ml/pipeline.py` should show the new sample_weight block
2. **Config is correct**: The version's config.json should have `"value_weighted": true` and all other regressor params matching v0
3. **Pipeline ran**: `registry/{VERSION_ID}/metrics.json` must exist with per_month and aggregate sections
4. **Changes committed**: `git log --oneline -1` should show the worker's commit
5. **Changes summary written**: `registry/{VERSION_ID}/changes_summary.md` must exist

**If any verification fails, write `"status": "failed"` in the handoff JSON with the specific error. Do NOT write `"status": "done"` unless ALL verification checks pass.**

## Expected Impact

| Metric | v0 Mean | Expected Direction | Confidence |
|---|---|---|---|
| EV-VC@100 | 0.069 | **+3-10%** | Medium |
| EV-VC@500 | 0.216 | **+2-5%** | Medium |
| EV-NDCG | 0.747 | **+1-3%** | Medium |
| Spearman | 0.393 | **±2%** | Low |
| C-RMSE | 3133 | **+0-10% (worse)** | Low-Medium |
| C-MAE | 1158 | **+0-5% (worse)** | Low-Medium |

**Key gate risk**: Spearman. Without the extra regularization from iter 1's plan (higher reg_lambda, min_child_weight), value weighting could pull Spearman down if the model over-specializes. However, with v0's hyperparams unchanged, the risk is moderate — the model capacity is the same, only the loss landscape shifts.

## Risk Assessment

1. **Spearman regression** (medium risk): Value weighting may reduce global rank correlation. v0 Spearman floor is 0.393 (= v0 mean). If Spearman drops below this, value weighting is confirmed harmful at this weight scale. Signal for iter 3: lighter weights or pair with stronger regularization.

2. **Tail month 2021-05** (low risk): Already catastrophic in v0 (EV-VC@100=0.0001). Value weighting won't make it worse — it was a structural issue.

3. **C-RMSE/C-MAE degradation** (low concern): These are Group B (monitor). Mild degradation is acceptable if EV ranking quality improves.

## Success Criteria

- **Pass**: Any Group A metric mean improves without another Group A metric failing Layer 1
- **Strong pass**: EV-VC@100 mean improves by ≥5% relative (from 0.069 to ≥0.072)
- **Fail signal**: Spearman drops below floor (0.393) — value weighting too aggressive at this scale
- **Diagnostic value (even if fail)**: We learn whether sample weighting helps or hurts, informing iter 3 direction
