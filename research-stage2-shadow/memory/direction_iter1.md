# Direction — Iteration 1

**Batch**: smoke-test-20260303-223300
**Champion**: v0 (baseline)
**Iteration**: 1

## Analysis of Current State

### v0 Baseline Strengths
- EV-NDCG mean 0.747 is solid — ranking quality is reasonable
- Best months (2022-03, 2022-12) show EV-VC@100 > 0.12, proving the pipeline can capture value when conditions align
- Spearman mean 0.393 shows moderate rank correlation

### v0 Baseline Weaknesses
- **Extreme EV-VC@100 variance**: min 0.000114 (2021-05) vs max 0.194 (2022-12) — 1700x spread
- **Two catastrophic months** for value capture: 2021-05 (EV-VC@100=0.0001) and 2022-06 (EV-VC@100=0.014)
- **C-RMSE high-variance tail**: 2022-06 has C-RMSE=5918, nearly 4x the best month (1481)
- **Spearman tail**: 2021-11 (0.265) and 2022-06 (0.273) both weak
- **Value-weighted training not wired up**: the `value_weighted` flag exists in config but pipeline Phase 4 doesn't compute or pass sample weights

### Gate Analysis (to promote beyond v0)
All gates were calibrated from v0's own performance. To promote, a new version must:
- **Layer 1 (mean)**: match or exceed v0 means (which ARE the floors)
- **Layer 2 (tail)**: at most 1 month below v0's worst month per metric
- **Layer 3 (non-regression)**: bottom_2_mean within 0.02 of v0's bottom_2_mean

The primary challenge: **improve the mean while not breaking tail months**.

## Hypothesis

**Value-weighted regressor training improves EV-based ranking quality by teaching the model to prioritize accuracy on high-value constraints.**

### Rationale
The v0 regressor treats a $0.50 constraint and a $50 constraint equally during training. But the business objective (EV-VC@100, EV-VC@500) rewards ranking the $50 constraint correctly far more than the $0.50 one. By weighting training samples proportional to their shadow price magnitude (in log-space to match the log1p target transform), the model will:

1. Allocate more model capacity to accurately predict high-value constraints
2. Reduce the frequency of catastrophic mis-rankings where a high-value constraint is predicted low
3. Improve value capture in the top-100 and top-500 positions

Paired with a moderate increase in regularization and model capacity to stabilize learning under the new weighting scheme.

## Specific Changes

### 1. Wire up value-weighted training in `ml/pipeline.py` (Phase 4)

After computing `X_reg_fit` and `y_reg_fit` (lines ~191-209), add sample weight computation:

```python
# After X_reg_fit and y_reg_fit are determined, compute sample weights
sample_weight = None
if config.regressor.value_weighted:
    # Use the log1p target as weight — matches the loss scale,
    # upweights high-value constraints proportionally
    sample_weight = y_reg_fit.copy()
    # Floor at 0.1 to avoid zero weights for near-zero shadow prices
    sample_weight = np.maximum(sample_weight, 0.1)
    # Normalize so mean weight = 1.0 to preserve effective learning rate
    sample_weight = sample_weight / sample_weight.mean()
    print(f"[phase 4] value_weighted: w_min={sample_weight.min():.3f} "
          f"w_max={sample_weight.max():.3f} w_mean={sample_weight.mean():.3f}")

reg_model = train_regressor(X_reg_fit, y_reg_fit, config.regressor,
                            sample_weight=sample_weight)
```

This replaces the existing `reg_model = train_regressor(X_reg_fit, y_reg_fit, config.regressor)` call on line 211.

### 2. Set regressor hyperparameters in the config override

The worker should use these `RegressorConfig` values:

| Parameter | v0 | v1 (this iter) | Rationale |
|---|---|---|---|
| `value_weighted` | False | **True** | Core hypothesis |
| `n_estimators` | 400 | **600** | More capacity for weighted loss landscape |
| `learning_rate` | 0.05 | **0.03** | Lower rate for stability with sample weights |
| `reg_lambda` | 1.0 | **3.0** | Stronger L2 to prevent overfitting to high-weight samples |
| `min_child_weight` | 10 | **15** | Slightly more conservative splits |
| `max_depth` | 5 | 5 | Keep unchanged |
| `subsample` | 0.8 | 0.8 | Keep unchanged |
| `colsample_bytree` | 0.8 | 0.8 | Keep unchanged |
| `reg_alpha` | 0.1 | 0.1 | Keep unchanged |
| `unified_regressor` | False | False | Keep gated mode |

### 3. No feature changes

Keep all 34 regressor features. Feature engineering changes are deferred to iteration 2+ based on results.

## Expected Impact

| Metric | v0 Mean | Expected Direction | Confidence |
|---|---|---|---|
| EV-VC@100 | 0.069 | **+5-15%** | Medium — high-value constraints get better predictions |
| EV-VC@500 | 0.216 | **+3-8%** | Medium — broader set, less sensitive to individual predictions |
| EV-NDCG | 0.747 | **+1-3%** | Medium — ranking quality should improve at the top |
| Spearman | 0.393 | **±2%** | Low — global rank correlation may not change much since weighting favors the tail |
| C-RMSE | 3133 | **+5-15% (worse)** | Medium — lower accuracy on low-value samples expected |
| C-MAE | 1158 | **+3-10% (worse)** | Medium — same reason |

**Key gate risk**: Spearman. Value weighting improves top-of-ranking accuracy at the expense of global rank correlation. If the model over-specializes on high-value constraints, Spearman could drop below the floor (0.393). The reg_lambda increase mitigates this.

## Risk Assessment

1. **Spearman regression** (medium risk): Value weighting could harm global rank correlation if the model ignores low-value constraints entirely. Mitigation: moderate weighting (log-scale, not linear-scale) and increased regularization.

2. **Tail month instability** (low risk): 2021-05 (EV-VC@100=0.0001) was already catastrophic in v0. Value weighting is unlikely to make it worse — it was a structural issue (low binding count or different market conditions), not a training issue.

3. **Training instability** (low risk): The lower learning rate (0.03) and higher reg_lambda (3.0) should prevent the model from oscillating under the weighted loss. The log-scale weights are bounded (max ~10x the floor), not unbounded.

4. **Overfitting to high-weight samples** (medium risk): With 600 trees and log-scale weights, the model could overfit to the small number of very-high-value constraints. The min_child_weight increase from 10→15 and reg_lambda 1.0→3.0 hedge against this.

## Success Criteria

- **Pass**: EV-VC@100 mean improves by ≥3% relative (from 0.069 to ≥0.071) AND no Group A gate Layer 1 fails
- **Strong pass**: EV-VC@100 mean improves by ≥10% AND EV-VC@500 improves by ≥5%
- **Fail signal**: Spearman drops below floor (0.393) — indicates value weighting is too aggressive, need lighter weights or more regularization in iter 2
