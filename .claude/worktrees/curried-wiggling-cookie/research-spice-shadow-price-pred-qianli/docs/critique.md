# Critique: Shadow Price Prediction Pipeline

> **Last updated**: 2026-02-25
> **Applies to**: `src/shadow_price_prediction/` codebase

---

## Bugs

### BUG-1: `reg_weights` used instead of `clf_weights` in classifier training

**File**: `models.py:913`
**Severity**: Medium (silent now, breaks if ensemble weights are configured)

```python
# CURRENT (wrong):
"weight_overrides": self.config.models.reg_weights.get(group_name),

# SHOULD BE:
"weight_overrides": self.config.models.clf_weights.get(group_name),
```

In `train_classifiers()`, when building the param dict for `_train_single_branch_classifier`, the code passes **regressor weight overrides** instead of **classifier weight overrides**. Since both dicts default to empty `{}`, and the ensemble is a single model, this is currently harmless. But if `clf_weights` or `reg_weights` are ever configured differently, the classifier's threshold optimization will use wrong weights.

**Fix**: One-line change, `reg_weights` → `clf_weights`.

**Status**: FIXED (2026-02-19)

---

### BUG-2: Default scaler overwritten by regressor training

**File**: `models.py:818` (classifier) and `models.py:1006` (regressor)
**Severity**: Low (data is the same, but conceptually fragile)

Both `train_classifiers` and `train_regressors` write to `self.scalers_default[group_name]`. The regressor training runs second and **overwrites** the classifier's scaler. At prediction time, `get_classifier_ensemble()` returns `self.scalers_default[group]` — which is now the regressor's scaler, not the classifier's.

In practice, both scalers are fit on the same `group_data[feature_cols]`, so values are identical. But this is fragile: if the data subsets ever diverge (e.g., filtering), predictions will silently use the wrong scaler.

**Fix**: Split into `scalers_default_clf`/`scalers_default_reg` and `scalers_branch_clf`/`scalers_branch_reg`. Updated all references in `models.py`, `prediction.py`, `pipeline.py`.

**Status**: FIXED (2026-02-19)

---

### BUG-3: Threshold optimized on in-sample training data (see also BUG-4 NaN variant)

**File**: `models.py:874-884` (default) and `models.py:677-685` (branch)
**Severity**: High (causes overfitting)

The threshold optimization calls `predict_ensemble()` on the **same training data** used to fit the model, then finds the threshold maximizing F-beta on those predictions. This overfits the threshold — the model's in-sample predictions are more confident than out-of-sample, so the threshold will be biased low (predicting too many as binding).

**Fix**: Implemented 6/3/3 train/val/test split. Threshold optimization now uses the 3-month validation set instead of in-sample predictions. For branch models, falls back to 0.5 if val data is insufficient (<10 samples or single class). Changes in `config.py` (TrainingConfig), `models.py` (val_data parameter threading), `pipeline.py` (data splitting).

**Status**: FIXED (2026-02-19)

---

### BUG-4: NaN poisoning in `find_optimal_threshold()` argmax

**File**: `models.py` — `find_optimal_threshold()`
**Severity**: High (silently returns degenerate threshold)

When computing F-beta across the precision-recall curve:

```python
fbeta = (1 + beta**2) * p * r / (beta**2 * p + r)
optimal_idx = np.argmax(fbeta)
```

At high thresholds where the model predicts nothing, `p + r = 0` produces `NaN`. `np.argmax` on a NaN-containing array has undefined behavior — in practice it returns the index of the first `NaN`, corresponding to the highest threshold on the curve (~0.93).

With `beta=2.0` (recall-favoring), the optimizer should find ~0.51. Instead it returned 0.930, suppressing recall (0.38 vs 0.74 after fix). See `document/learnings.md` Section 1 for full impact table.

**Fix**: One line — `fbeta = np.nan_to_num(fbeta, nan=0.0)` before `np.argmax`.

**Status**: FIXED (2026-02-18)

---

### BUG-5: `spearmanr` returns array instead of scalar

**File**: `models.py` — `select_features()` → `_get_subset()`
**Severity**: Medium (crashes feature selection for certain features)

`scipy.stats.spearmanr(x_feat, y)` can return an array or matrix instead of a scalar in some edge cases (constant features, NaN-heavy inputs, or newer scipy versions). The code then calls `np.isnan(corr)` which fails with `ValueError: The truth value of an array with more than one element is ambiguous`.

**Fix**: Cast result to scalar: `corr = float(np.asarray(corr).flat[0])`.

**Status**: FIXED (2026-02-18)

---

## Improvements

### IMP-1: Add `hist_da` to Step 1 (classification) features

**Priority**: HIGH
**Impact**: Expected significant improvement in recall

Historical DA shadow price is the single strongest predictor of future binding. A constraint that has been binding historically is far more likely to bind again. Currently, `hist_da` is available in training data but **commented out** in `FeatureConfig.step1_features`.

**Result**: Uncommented in config.py and smoke test. In Run 2, feature selection dropped `hist_da` (8/9 selected). In Run 3, after BUG-3 fix (validation-based threshold optimization changed the training data subset), `hist_da` was **KEPT** (AUC=0.835-0.851, Spearman=0.292-0.304 on the 6-month fit split). Instead, `density_skewness` was dropped (AUC<0.5, Spearman<0 — wrong-sign relationship with constraint=+1).

**Status**: DONE (2026-02-18) — feature is now retained by selection.

---

### IMP-2: Add `hist_da` and `recent_hist_da` to Step 2 (regression) features

**Priority**: HIGH
**Impact**: Expected significant improvement in shadow price magnitude prediction

The regressor currently uses the **same 8 features as the classifier** — no historical price information. Predicting shadow price magnitude without knowing the historical shadow price is like predicting stock returns without knowing the stock price.

**Result**: Both features added to config.py and smoke test. **Spearman rank correlation improved +3-4%** on holdout TPs (onpeak 0.462→0.476, offpeak 0.521→0.540). MAE/RMSE largely unchanged (regression clamping had larger effect on RMSE).

**Status**: DONE (2026-02-18)

---

### IMP-3: Out-of-fold threshold optimization

**Priority**: HIGH
**Impact**: Prevents threshold overfitting, more honest F-beta

See BUG-3 above. Replaced in-sample prediction with validation-set prediction via 6/3/3 temporal split. Branch models with <10 val samples or single class fall back to threshold 0.5.

**Status**: DONE (2026-02-19, via BUG-3 fix)

---

### IMP-4: Enable ensemble diversity

**Priority**: HIGH
**Impact**: More robust predictions, reduced model risk

The `EnsembleConfig` supports multiple models with weighted averaging, but currently only XGBoost is active — ElasticNet and LogisticRegression are commented out. A single-model "ensemble" provides no diversity.

Even a simple 70/30 XGBoost + LogisticRegression blend should improve robustness, especially for the classifier where tree models and linear models have uncorrelated errors.

**Suggested change**: Uncomment the LogisticRegression/ElasticNet model specs in `config.py:163-265`. Start with equal weights.

**Status**: OPEN

---

### IMP-5: Vectorize anomaly detection loop

**Priority**: MEDIUM
**Impact**: Performance (currently O(n) Python loop per sample)

In `prediction.py:204`, the anomaly detection runs a **per-sample Python loop**:
```python
for i, idx in enumerate(branch_indices):
    is_anomaly, confidence, reason = self.anomaly_detector.detect_flow_anomaly(...)
```

This should be vectorized to process all samples for a branch at once. The detection logic (compare feature values against P99 + k*IQR thresholds) is purely arithmetic and can be done with numpy broadcasting.

**Status**: OPEN

---

### IMP-6: Include near-threshold constraints in signal

**Priority**: MEDIUM
**Impact**: Better coverage of borderline constraints

`SignalGenerator` filters to `predicted_shadow_price > 0` only. Constraints with high binding probability but `predicted_shadow_price == 0` (because the regressor had no model or the branch wasn't predicted as binding) are **silently dropped**. These should appear in lower tiers rather than being excluded entirely.

**Suggested fix**: Include constraints where `binding_probability > some_threshold` even if shadow price is 0, assigning them to tier 3 or 4.

**Status**: OPEN

---

### IMP-7: Add seasonal features to Step 2

**Priority**: MEDIUM
**Impact**: Better long-horizon predictions

`season_hist_da_{1,2,3}` are computed by the data loader but unused. Seasonal patterns matter for congestion — summer and winter have different constraint profiles. These are particularly valuable for quarterly (q2/q3/q4) predictions where horizon is 3+ months.

**Result**: Added `season_hist_da_1`, `season_hist_da_2`, `season_hist_da_3` to `step2_features` in `config.py` and smoke test. Added defensive handling for missing seasonal columns (filled with 0 if absent). Holdout Spearman improved +2% on both classes (onpeak 0.476→0.484, offpeak 0.540→0.552).

**Status**: DONE (2026-02-18)

---

### IMP-8: Re-enable metrics calculation

**Priority**: MEDIUM
**Impact**: Faster iteration loop

Both `prediction.py:632` and `pipeline.py:605` hardcode `metrics = {}`. The `evaluate_signal_v2` backtesting catches this externally, but having in-pipeline metrics (precision, recall, F1 per tier) would significantly speed up development iteration. Don't need to wait for signal save + external evaluation.

**Fix**: Added `evaluate_split()` to `evaluation.py` and integrated it into `pipeline.py`. Val and holdout splits are now evaluated after training with classification metrics (P/R/F1/F-beta/AUC-ROC/AUC-PR), regression metrics on TPs (MAE/RMSE/Spearman), and top-K ranking metrics (Precision@K/Lift@K/Value Capture@K/NDCG). Metrics are aggregated across auction months in `run()` and returned in the `metrics` slot. Note: `prediction.py` still returns empty metrics for per-market-month prediction — only the post-training evaluation on val/holdout is implemented.

**Status**: DONE (2026-02-18)

---

### IMP-9: Separate default scalers for clf vs reg

**Priority**: LOW
**Impact**: Code correctness, prevents future bugs

See BUG-2. Even though current behavior is correct by accident, the architecture should make it impossible for the regressor to overwrite the classifier's scaler.

**Status**: DONE (2026-02-19, via BUG-2 fix)

---

### IMP-10: Add LightGBM as ensemble member

**Priority**: LOW
**Impact**: Model diversity, potentially better calibrated probabilities

LightGBM uses histogram-based splitting and handles categorical features differently from XGBoost. Adding it as a second tree model (alongside XGBoost) provides diversity with similar computational cost. Would require adding `lightgbm` to `extra_modules` in Ray init.

**Status**: OPEN

---

### IMP-11: Hyperparameter tuning

**Priority**: LOW
**Impact**: Unknown — current params may be near-optimal or far from it

All model parameters are hand-set. No Optuna/cross-val hyperparameter search has been run. The current `max_depth=4, learning_rate=0.1, n_estimators=200` are reasonable defaults but may not be optimal for this specific problem.

**Approach**: Walk-forward cross-validation with Optuna, optimizing F-beta for classifier and RMSE for regressor. Use GroupKFold splitting by auction_month to prevent temporal leakage.

**Status**: OPEN

---

### IMP-12: PJM constraint path — hardcoded `class_type=onpeak`

**Priority**: BLOCKED (on upstream investigation)

`iso_configs.py:101` hardcodes `class_type=onpeak` in the PJM constraint path template, preventing generation of `dailyoffpeak` and `wkndonpeak` signals. See `blockers.md` Blocker #2.

**Status**: BLOCKED

---

### IMP-13: Clamp regression output before `expm1()`

**Priority**: HIGH
**Impact**: Prevents absurd shadow price predictions (offpeak holdout RMSE = $90,911)

The regression target is `log1p(label)`. At prediction time, `expm1(raw_pred)` can produce extreme values if `raw_pred` is large. Offpeak holdout RMSE of $90,911 is inflated by these outliers.

**Result**: Applied `np.clip(raw_pred, None, 12)` at all 3 `expm1` call sites in `prediction.py`. Offpeak holdout RMSE dropped **87%** ($90,911 → $12,034). Onpeak RMSE unchanged (no extreme outliers there).

**Status**: DONE (2026-02-18)

---

### IMP-14: Add constraint-level top-K metrics

**Priority**: HIGH
**Impact**: More representative evaluation of signal quality

Current top-K metrics are computed at the per-outage level (~400K rows). The signal output aggregates across outage dates to the constraint level (~1,300 constraints). Value Capture@K < 2% at K=2000 (per-outage) is misleading — constraint-level top-K would be more aligned with how the signal is consumed.

**Result**: Implemented constraint-level aggregation in `evaluate_split()` in `evaluation.py`. Groups by `(branch_name, flow_direction)`, aggregates shadow prices (SUM), binding probability (MAX), and actual binding (MAX). Top-K computed with `CONSTRAINT_K_VALUES = (50, 100, 260, 520, 1000)`. Key finding: **top 1K constraints capture 74-77% of total congestion value** on holdout. Constraint NDCG=0.27-0.34 (lower than per-outage NDCG=0.60-0.62 because per-outage is inflated by easy non-binding rows).

**Status**: DONE (2026-02-18)

---

### IMP-16: Fix `density_skewness` constraint direction

**Priority**: MEDIUM
**Impact**: Better feature utilization — currently dropped by selection

`density_skewness` had `constraint=+1` in `step1_features` but the actual relationship with binding is **negative** (AUC=0.431-0.440, Spearman=-0.047 to -0.052). This means higher skewness correlates with *less* binding, not more. Feature selection correctly dropped it.

Changed constraint to `0` (unconstrained) in both step1 and step2 feature lists, letting the model decide the direction. This was applied as part of the v005-9mo-training experiment.

**Status**: FIXED (2026-02-23, v005-9mo-training)

---

### IMP-17: Add feature selection verbose diagnostics

**Priority**: LOW
**Impact**: Better debugging of feature selection decisions

Added `verbose` parameter to `select_features()` in `models.py`. When `verbose=True`, prints per-feature AUC, Spearman, and KEEP/DROP decisions. Propagated from `train_classifiers()`.

**Status**: DONE (2026-02-18)

---

### IMP-15: Investigate val top-K anomaly

**Priority**: MEDIUM
**Impact**: Understand why highest-probability predictions are wrong

Val top-K shows Lift < 1.7x (barely above random) while holdout Lift is 3.4-5.6x. The model's top predictions on validation data are mostly wrong — likely overconfident default-model probabilities for never-binding branches. Understanding this could improve probability calibration.

**Status**: OPEN

---

### BUG-6: Threshold artifact file race condition (f0/f1 workers overwrite each other)

**File**: `scripts/_experiment_worker.py` → `_save_worker_artifacts()`
**Severity**: Low (artifacts only — predictions use correct in-memory thresholds)

The threshold JSON file is named `thresholds_{auction_month}_{class_type}.json` without a `period_type` suffix. When f0 and f1 workers run concurrently for the same auction_month and class_type, whichever finishes last overwrites the other's threshold artifact. This makes threshold artifacts unreliable for post-hoc analysis (e.g., v008's f0 threshold appeared as 0.5 in the file, but the actual in-memory optimized threshold was ~0.885).

**Fix**: Add `period_type` to the threshold artifact filename in `naming.py` → `worker_threshold_path()`.

**Status**: OPEN

---

### BUG-7: S2-SPR gate silently excludes NaN months (inflates metric)

**File**: `scripts/run_experiment.py` → `_compute_gates()`
**Severity**: Medium (metric inflation for sparse data months)

`df["spearman_tp"].mean()` silently drops NaN values. Months where Spearman cannot be computed (e.g., no true positives, constant predictions) are excluded from the mean, inflating the S2-SPR gate value. Should either fill NaN with 0 (conservative) or track NaN count separately.

**Status**: OPEN

---

### BUG-8: threshold_override ignored in group=None early-return path

**File**: `models.py` → `get_classifier_ensemble()`
**Severity**: Medium (threshold_override experiments returned wrong results for ungrouped data)

When `group is None`, the early return hardcoded `threshold=0.5`, ignoring `ThresholdConfig.threshold_override`. Fixed by checking `self.config.threshold.threshold_override` in that path.

**Status**: FIXED (2026-02-24)

---

### IMP-19: Per-period gate enforcement

**Priority**: HIGH
**Impact**: Prevents weak period types from hiding behind strong ones in aggregate metrics

Previously, gates were checked at the class_type level (onpeak/offpeak average). A version could pass gates despite f1 having terrible recall if f0 compensated. Now gates are checked per (class_type, period_type) segment — each of onpeak/f0, onpeak/f1, offpeak/f0, offpeak/f1 must independently pass all floor gates.

Changes: `registry.py` (GateCheck.failed_segments, check_gates per-segment iteration, gate_values_by_period method on VersionEntry), `run_experiment.py` (_compute_gates helper, gate_values_by_period computation and storage).

**Status**: DONE (2026-02-24)

---

### IMP-20: threshold_override for experiment flexibility

**Priority**: MEDIUM
**Impact**: Enables controlled threshold experiments without modifying core pipeline

Added `ThresholdConfig.threshold_override: float | None` that bypasses threshold optimization when set. Applied in `get_classifier_ensemble()` and `_experiment_worker.py` `apply_overrides()`. Enables experiments like "what if we used 0.5 threshold everywhere?" without code changes.

**Status**: DONE (2026-02-24)

---

### IMP-11: Hyperparameter tuning (partial)

**Priority**: LOW → MEDIUM (partially addressed)
**Impact**: v007/v008 experiments validated several hyperparameter directions

v007 tested aggressive changes (lr=0.05, depth=5, subsample/colsample=0.8, L1/L2 reg) — hurt classifier AUC. v008 split the difference: conservative classifier (v000 defaults + regularization), aggressive regressor. This improved RMSE by $447 vs v000.

Remaining: no formal Optuna/cross-val search has been run. The v008 hyperparameters are informed by v007/v008 A/B testing but not systematically optimized.

**Status**: OPEN (partially addressed by v007/v008)

---

### IMP-21: Threshold tuning is a dead end

**Priority**: LOW (resolved — no further action needed)
**Impact**: Saves future effort by documenting that threshold is not a useful optimization lever

Tested three threshold strategies against v008's natural mixed-threshold behavior:
- v009: threshold_override=0.7 — FAILS S1-REC gate (onpeak/f1 = 0.228 < 0.25). Raising branch fallbacks from 0.50 to 0.70 kills recall.
- v010-thr050: threshold_override=0.5 — All gates pass, but nearly IDENTICAL to v008. f1 metrics unchanged (v008 already used 0.50). f0 metrics differ by ~2pp (noise).
- v008 (mixed): f0 default ~0.91, f1 default 0.50, all branches 0.50 — best overall.

Root cause: hundreds of branch models all fall back to threshold=0.50 (insufficient per-branch validation data). They dominate aggregate metrics. The f0 default model's optimized threshold (~0.91) affects only the minority of predictions from constraints without branch models.

**Conclusion**: Future improvements must come from model quality (features, architecture, training data), not threshold selection. The optimizer's natural output works well.

**Status**: CLOSED (2026-02-25)

---

## Change Log

| Date | Item | Change |
|------|------|--------|
| 2026-02-18 | Initial | Created critique from full pipeline analysis |
| 2026-02-19 | BUG-1 | FIXED: `reg_weights` → `clf_weights` in `models.py` |
| 2026-02-19 | BUG-2 | FIXED: Split scaler dicts into `_clf`/`_reg` variants across `models.py`, `prediction.py`, `pipeline.py` |
| 2026-02-19 | BUG-3 | FIXED: Threshold optimization now uses 3-month validation set instead of in-sample predictions |
| 2026-02-19 | IMP-3 | DONE: Implemented via BUG-3 fix (6/3/3 temporal split) |
| 2026-02-19 | IMP-9 | DONE: Implemented via BUG-2 fix (separate clf/reg scalers) |
| 2026-02-18 | BUG-4 | FIXED: `nan_to_num` before `argmax` in `find_optimal_threshold()` |
| 2026-02-18 | IMP-8 | DONE: `evaluate_split()` added to `evaluation.py`, integrated into `pipeline.py` with top-K ranking metrics |
| 2026-02-18 | IMP-1 | DONE: `hist_da` added to step1_features — dropped by feature selection |
| 2026-02-18 | IMP-2 | DONE: `hist_da` + `recent_hist_da` added to step2_features — Spearman +3-4% |
| 2026-02-18 | IMP-13 | DONE: Clamped regression output — offpeak RMSE -87% ($90K→$12K) |
| 2026-02-18 | BUG-5 | FIXED: `spearmanr` returns array in some cases — `np.isnan(corr)` fails. Added `float(np.asarray(corr).flat[0])` |
| 2026-02-18 | IMP-14 | DONE: Constraint-level top-K metrics — top 1K captures 74-77% value |
| 2026-02-18 | IMP-7 | DONE: Seasonal features added to Step 2 — Spearman +2% |
| 2026-02-18 | IMP-1 | Updated: `hist_da` now KEPT by selection (was dropped in Run 2 only) |
| 2026-02-18 | IMP-16 | OPEN: `density_skewness` has wrong-sign constraint (+1 but negative relationship) |
| 2026-02-18 | IMP-17 | DONE: Feature selection verbose diagnostics added |
| 2026-02-18 | IMP-15 | OPEN: Investigate val top-K anomaly |
| 2026-02-19 | IMP-18 | REMOVED: Reverted registry.py (replaced by legacy baseline benchmark approach) |
| 2026-02-20 | IMP-18 | DONE: Model promotion registry implemented (`registry.py`) with versioning, gating, checksum verification |
| 2026-02-21 | IMP-18 | Updated: Registry enriched with 7 artifacts, per-class gate enforcement, noise tolerance, version hashes, 3 audit scripts, naming.py. Dead code removed (StackingModel, HyperparameterTuner, tuning_utils.py, run_f0_smoke_test.py) |
| 2026-02-23 | IMP-16 | FIXED: `density_skewness` constraint changed from +1 to 0 (unconstrained) in both step1 and step2 |
| 2026-02-23 | RESTRUCTURE | Repo restructure: `document/`→`docs/`, `notebook/`→`scripts/`+`notebooks/`, `registry/`→`versions/`. All path references updated. |
| 2026-02-23 | PIPELINE | Pipeline fixes: train_months 6→9, test_months 3→0, evaluation MAX→MEAN, gate floors relaxed. v005 benchmark: 32/32 OK, all 5 gates passed. |
| 2026-02-24 | BUG-8 | FIXED: threshold_override ignored in group=None early-return in `get_classifier_ensemble()` |
| 2026-02-24 | BUG-6 | OPEN: Threshold artifact race condition — f0/f1 workers overwrite same file |
| 2026-02-24 | BUG-7 | OPEN: S2-SPR gate silently drops NaN months |
| 2026-02-24 | IMP-19 | DONE: Per-period gate enforcement — each (class_type, period_type) segment checked independently |
| 2026-02-24 | IMP-20 | DONE: threshold_override feature for controlled experiments |
| 2026-02-24 | IMP-11 | Updated: Partially addressed by v007/v008 hyperparameter experiments |
| 2026-02-25 | IMP-21 | CLOSED: Threshold tuning dead end. v009 (0.7) fails gate, v010-thr050 (0.5) identical to v008. Branch fallbacks dominate. |
| 2026-02-25 | IMP-22 | DONE: Gate restructuring — PROMOTION_GATES (8, threshold-independent) + MONITORING_GATES (7, informational). Eliminates apples-to-oranges comparisons across threshold configs. |
| 2026-02-25 | IMP-23 | DONE: EV scoring (expected_value_scoring=True) + regressor value weighting (value_weighted_reg=True). v011: all gates pass, C-RMSE -16% vs v008. But promotion ranking metrics unchanged — regressor coverage bottleneck identified. |
| 2026-02-25 | IMP-24 | IN PROGRESS: Unified regressor (unified_regressor=True) — v012 running. Trains on all samples, no binary gate at prediction. Addresses root cause: regressor coverage bottleneck. |
| 2026-02-25 | CONCERN-1 | OPEN: Unified regressor shows severe negative bias in smoke test ($468 predicted vs $1302 actual on TPs). 95%+ zero training data pulls XGBoost toward near-zero predictions. Consider Tweedie loss (`reg:tweedie`) as next experiment if v012 ranking metrics don't improve. |
| 2026-02-25 | CONCERN-2 | OPEN: Even "threshold-independent" promotion gates (R-REC@500, C-VC@K) are indirectly affected by threshold — the binary gate controls which constraints get non-zero shadow price predictions, affecting top-K ranking. Only AUC/AP are truly threshold-independent. This is a fundamental property of the two-stage architecture. |
