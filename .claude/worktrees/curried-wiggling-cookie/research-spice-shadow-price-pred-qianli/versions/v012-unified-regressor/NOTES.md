# v012-unified-regressor: Notes

> **Status**: RUNNING (13/32 parquets at time of writing)
> **Date**: 2026-02-25
> **Hypothesis**: Training the regressor on ALL samples (not just binding) with target=log1p(max(0,shadow_price)) eliminates the binary gate bottleneck that has been the root cause of poor ranking metrics across all previous versions.

---

## Motivation

### The Binary Gate Bottleneck

In all versions v000–v011, Stage 2 (regression) only runs for samples where the binary classifier predicted `binding=1`. This creates a hard structural limit:

- Only ~5% of constraints get non-zero shadow price predictions (those passing the binary gate)
- v000 (threshold ~0.5): covers 46% of binding constraints with non-zero predictions
- v008/v011 (threshold ~0.9): covers only 42% of binding constraints with non-zero predictions
- The remaining 54–58% of truly binding constraints ALWAYS get shadow_price=0, regardless of model quality

This bottleneck corrupts ranking metrics (R-REC@500, C-VC@K, C-NDCG) because constraints with predicted_shadow_price=0 automatically rank last. Better model quality cannot compensate if the binary gate misses too many binding constraints.

### What Changed

**Training**: Regressors (branch + default) now train on ALL samples with target = `log1p(max(0, label))`:
- Non-binding samples: target = 0
- Binding samples: target = log1p(shadow_price) > 0

**Prediction**: No binary gate. The regressor is called on ALL samples. XGBoost learns both the binding/non-binding boundary AND the severity in one model.

**Sample count**: Default regressor trains on ~1.35M samples (vs ~67K binding-only previously).

---

## Config

All settings identical to v008 **except** `unified_regressor=True`.

```python
PredictionConfig(
    market_round=1,
    training=TrainingConfig(
        train_months=10,
        val_months=2,
        unified_regressor=True,  # <-- only change
    ),
    threshold=ThresholdConfig(threshold_beta=0.7),
    # features, models: same as v008
)
```

---

## Smoke Test (2020-07/onpeak/f0)

```
Default regressor: trained on 1,349,036 samples (unified mode)
Branch regressors: 796 trained

Stage 1 (Classifier) — unchanged:
  AUC-ROC=0.7123  AUC-PR=0.1977
  Prec=0.3652  Recall=0.3419  F1=0.3531

Stage 2 (Regressor on 3,051 TPs):
  MAE=$1,281.81  RMSE=$2,197.40  Spearman=0.4234
  mean_actual=$1,301.80  mean_pred=$468.62  bias=-$833.18  ← LARGE NEGATIVE BIAS

Combined (all 142,660):
  RMSE=$451.20  MAE=$54.05  ← excellent

Constraint ranking (NDCG=0.403):
  K=100: VC=34.1%
  K=500: VC=70.0%
  K=1000: VC=82.7%
```

**Key concern**: The regressor severely underpredicts shadow prices for binding constraints (bias = -$833). The 95%+ zero-target training distribution pulls XGBoost toward near-zero predictions. Despite this bias, absolute ranking quality (VC@K) looks comparable to v008.

---

## Full Benchmark Results

> **To be filled in after benchmark completes and scoring runs.**

```
Gate values (32-run average):
  S1-AUC:    TBD
  S1-AP:     TBD
  R-REC@500: TBD
  C-VC@100:  TBD
  C-VC@500:  TBD
  C-VC@1000: TBD
  C-NDCG:    TBD
  C-RMSE:    TBD

Per-segment:
  onpeak/f0:  TBD
  onpeak/f1:  TBD
  offpeak/f0: TBD
  offpeak/f1: TBD

Promotable: TBD
```

---

## Expected Outcome

**If unified regressor helps ranking** (best case):
- R-REC@500 and C-VC@K improve vs v008 because more binding constraints get non-zero predictions
- C-RMSE may stay similar (bias hurts absolute prediction, but RMSE@0 is already 0)
- S1-AUC unchanged (classifier is the same)

**If unified regressor fails ranking** (bias dominates):
- R-REC@500 and C-VC@K similar to or worse than v008
- C-RMSE better (unified model assigns tiny non-zero values everywhere, reducing RMSE)
- This would indicate the class imbalance problem is too severe for standard MSE loss

**Recommended next step if v012 fails on ranking**:
- v013: Try Tweedie loss (`objective='reg:tweedie'`, `tweedie_variance_power=1.5`) in the unified regressor. Tweedie is the statistically correct loss for zero-inflated positive outcomes and should reduce the negative bias.
