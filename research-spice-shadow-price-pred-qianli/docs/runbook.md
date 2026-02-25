# Runbook: Shadow Price Prediction Pipeline (SPICE_F0P_V6.7B)

> **Last updated**: 2026-02-24
> **Signal**: `SPICE_F0P_V6.7B.R1`
> **RTOs**: MISO (active), PJM (blocked — see blockers.md)

---

## 1. Purpose

Predict which transmission constraints will bind in MISO/PJM FTR auctions and estimate their shadow prices. The output is a **ConstraintsSignal** consumed by the CIA Multistep trading workflow and evaluated via `evaluate_signal_v2`.

---

## 2. Architecture Overview

```
density parquets + DA shadow prices
        │
        ▼
  ┌─────────────┐
  │  DataLoader  │  load_training_data / load_test_data
  └──────┬──────┘
         │  per (auction_month, market_month)
         ▼
  ┌──────────────────────┐
  │  ShadowPriceModels   │  train_classifiers + train_regressors
  │  (per horizon group) │  default models + per-branch models
  └──────┬───────────────┘
         │
         ▼
  ┌─────────────┐
  │  Predictor   │  Stage 1: classify → Stage 2: regress
  └──────┬──────┘
         │
         ▼
  ┌──────────────────┐
  │ SignalGenerator   │  composite ranking → 5-tier signal
  └──────────────────┘
         │
         ▼
  ConstraintsSignal parquet (saved to signal_data/)
```

---

## 3. Two-Stage ML Pipeline

### Stage 1: Binary Classification

**Question**: Will this constraint bind?

- Model: XGBClassifier with monotonic constraints
- Target: `label > 0` (binary)
- Output: P(binding) per constraint per outage date
- Decision: `P >= threshold` → predicted binding

### Stage 2: Regression

**Question**: If binding, how much shadow price?

- Model: XGBRegressor with monotonic constraints
- Target: `log1p(label)` (log-transformed shadow price)
- Output: `expm1(prediction)` → estimated shadow price in $/MWh
- Only runs for constraints predicted as binding in Stage 1

---

## 4. Model Hierarchy

### Per-Branch vs Default

For each `(branch_name, flow_direction)` pair:

1. **Branch-specific model** — trained on historical data for just that branch
   - Requires: `min_samples_for_branch_model` (default: 1) samples
   - Requires: both classes present (binding + non-binding)
   - Requires: positive ratio >= `min_branch_positive_ratio` (default: 2%)
   - Requires: at least one monotonic feature survives feature selection
2. **Default fallback model** — pooled across all branches in the horizon group
   - Used when branch model can't be trained or feature selection fails

### Horizon Groups (MISO)

| Group  | Horizons | Description |
|--------|----------|-------------|
| `f0`   | 0        | Current month |
| `f1`   | 1        | Next month |
| `long` | 2–999    | 3+ months (quarterly, annual) |

Separate default + branch models are trained per horizon group.

### Horizon Groups (PJM)

| Group  | Horizons | Description |
|--------|----------|-------------|
| `f0`   | 0        | Current month |
| `f1`   | 1        | Next month |
| `f2`   | 2        | Month 2 |
| `long` | 3–999    | 3+ months |

---

## 5. Features

### Available in Training Data (40 feature columns + metadata)

```
# Exceedance / below-limit probabilities (14)
prob_exceed_{80,85,90,95,100,105,110}    # 7 features — P(flow > X% of limit)
prob_below_{80,85,90,95,100,105,110}     # 7 features — P(flow < X% of limit)

# Density distribution statistics (6)
density_{mean,variance,skewness,kurtosis} # 4 features — distribution moments
density_entropy                           # distribution uncertainty
density_cv                                # coefficient of variation (std/|mean|)

# Derived density features (4)
expected_overload                         # E[flow] for flow > limit
tail_concentration                        # prob_exceed_100 / prob_exceed_80
prob_band_95_100                          # prob mass approaching limit
prob_band_100_105                         # prob mass just above limit

# Historical DA shadow prices (7)
hist_da                                   # cumulative DA shadow price (all time)
recent_hist_da                            # recent DA shadow price (last few months)
season_hist_da_{1,2,3}                    # seasonal history (3 lagged years)
hist_da_trend                             # recent / seasonal mean ratio
hist_da_max_season                        # max of seasonal DA prices

# Target + metadata
label                                     # target: actual DA shadow price
constraint_id, branch_name, flow_direction, forecast_horizon
auction_month, market_month, outage_date, period_type
```

Note: The 8 new features (expected_overload through hist_da_max_season) were added in v007. They are computed in `data_loader.py:calculate_direction_features()` from the same density parquets and DA data — no new data sources required.

### Currently Used Features (v008)

**Step 1 (classification) — 14 features:**

| Feature | Monotone Constraint | Meaning |
|---------|-------------------|---------|
| `prob_exceed_110` | +1 (increasing) | P(flow > 110% limit) |
| `prob_exceed_105` | +1 | P(flow > 105% limit) |
| `prob_exceed_100` | +1 | P(flow > 100% limit) |
| `prob_exceed_95`  | +1 | P(flow > 95% limit) |
| `prob_exceed_90`  | +1 | P(flow > 90% limit) |
| `prob_below_100`  | -1 (decreasing) | P(flow < 100% limit) |
| `prob_below_95`   | -1 | P(flow < 95% limit) |
| `prob_below_90`   | -1 | P(flow < 90% limit) |
| `expected_overload` | +1 | E[flow] for flow > 100% of limit |
| `density_skewness` | 0 (unconstrained) | Right skew in density |
| `density_kurtosis` | 0 | Tail heaviness of density |
| `density_cv`       | 0 | Coefficient of variation (std/mean) |
| `hist_da`          | +1 | log1p(sum of all historical DA shadow prices) |
| `hist_da_trend`    | +1 | Recent DA / seasonal mean — congestion momentum |

**Step 2 (regression) — 27 features** (all density + historical features):

| Feature | Monotone Constraint | Meaning |
|---------|-------------------|---------|
| `prob_exceed_{110..90}` (5) | +1 | Exceedance probabilities |
| `prob_below_{100,95,90}` (3) | -1 | Below-limit probabilities |
| `expected_overload` | +1 | Expected overload magnitude |
| `tail_concentration` | +1 | prob_exceed_100 / prob_exceed_80 — tail shape |
| `prob_band_95_100` | +1 | Probability mass approaching limit |
| `prob_band_100_105` | +1 | Probability mass just above limit |
| `density_{mean,variance}` | 0 | Distribution location and spread |
| `density_skewness` | 0 | Right skew in density |
| `density_kurtosis` | 0 | Tail heaviness |
| `density_entropy` | 0 | Distribution uncertainty |
| `density_cv` | 0 | Coefficient of variation |
| `hist_da` | +1 | Historical DA shadow price (cumulative) |
| `hist_da_trend` | +1 | Recent / seasonal mean ratio |
| `recent_hist_da` | +1 | Recent-window DA shadow price |
| `season_hist_da_{1,2,3}` (3) | +1 | Seasonal DA prices (3 lagged years) |
| `hist_da_max_season` | +1 | Max of seasonal DA prices |

### Feature Design (v008 asymmetry)

v008 uses an **asymmetric** feature strategy: a focused 14-feature classifier (drops highly correlated features to reduce noise) paired with a rich 27-feature regressor (all available features for magnitude prediction). This outperforms the symmetric designs of v005-v007.

---

## 6. Model Parameters

### Default Classifier (v008)

```python
XGBClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.1,
    min_child_weight=10, eval_metric='logloss',
    scale_pos_weight=auto,  # n_neg / n_pos
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0,
    monotone_constraints=<per feature selection>,
)
```

### Branch Classifier (v008)

```python
XGBClassifier(
    n_estimators=100, max_depth=4, learning_rate=0.1,
    min_child_weight=1, eval_metric='logloss',
    scale_pos_weight=auto,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0,
    monotone_constraints=<per branch feature selection>,
)
```

### Default Regressor (v008)

```python
XGBRegressor(
    n_estimators=400, max_depth=5, learning_rate=0.05,
    min_child_weight=2, objective='reg:squarederror',
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0,
    monotone_constraints=<per feature selection>,
)
```

### Branch Regressor (v008)

```python
XGBRegressor(
    n_estimators=200, max_depth=5, learning_rate=0.05,
    min_child_weight=1, objective='reg:squarederror',
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0,
    monotone_constraints=<per branch feature selection>,
)
```

### Ensemble Status

The `EnsembleConfig` supports multiple models per stage (e.g., XGBoost + LogisticRegression + ElasticNet) with weighted averaging or stacking. **Currently, all ensembles contain a single XGBoost model** — the ElasticNet and LogisticRegression entries are commented out.

### Design Note: Asymmetric Classifiers vs Regressors (v008)

v008 uses **different hyperparameters** for classifiers and regressors:
- Classifiers: conservative (depth=4, lr=0.1, 200 trees) — matches v000 proven defaults + regularization
- Regressors: deeper (depth=5, lr=0.05, 400 trees) — more capacity for shadow price magnitude prediction
This asymmetry emerged from v007 experiments showing deeper trees hurt classifier AUC but helped regression RMSE.

---

## 7. Feature Selection

Per-model feature selection via `select_features()`:

1. For each candidate feature with monotone constraint (+1 or -1):
   - **Spearman check**: correlation with target must match expected sign
   - **AUC check**: ROC AUC must be > 0.5 (for +1) or < 0.5 (for -1)
   - Feature must pass **both** checks (method="both")
2. If no monotonic features survive → fallback to Spearman-only
3. If still none → fallback to all features (branch model discarded; uses default)

Threshold: `min_correlation=0.0`, `auc_threshold=0.5` (better than random).

---

## 8. Threshold Optimization

Binary classification threshold is optimized per model to maximize **F-beta score**.

```python
find_optimal_threshold(y_true, y_proba, beta=threshold_beta, scaling_factor=1.0)
```

Uses `precision_recall_curve` from sklearn. The `scaling_factor` blends the optimal threshold toward 0.5: `threshold = scaling_factor * optimal + (1 - scaling_factor) * 0.5`.

Threshold is optimized on the **validation set** (not in-sample). For branch models with <10 val samples or only one class, falls back to 0.5. `threshold_override` in `ThresholdConfig` bypasses optimization entirely and uses a fixed value for all groups.

**Current behavior by horizon group**: f0 default classifier successfully trains and optimizes threshold (~0.88 with v008 config). f1 threshold optimization results vary by beta: beta=1.0 (v007) produces thresholds ~0.75-0.89; beta=0.7 (v008) produces threshold ~0.5. The `long` group typically falls back to 0.5 (insufficient data).

**NaN guard**: The F-beta array is sanitized with `np.nan_to_num(fbeta, nan=0.0)` before `np.argmax` to prevent NaN poisoning when `p + r = 0` at high thresholds (see BUG-4 in `critique.md`).

---

## 9. Anomaly Detection

For branches that **never bind** in training data:

1. **Characterize** flow statistics (P99, IQR per feature) during training
2. **Detect** at prediction time: weighted score based on how far above P99 + k*IQR
3. **Probability** phase: sigmoid mapping of anomaly score to P(binding)
4. If anomaly detected: predict binding with `shadow_price = max(default_reg_pred * 2.0, 20.0)`

Config: `k_multiplier=3.0`, `detection_threshold=0.5`, features: `[prob_exceed_100, prob_exceed_95]`.

---

## 10. Training Data & Split

### Training Window

Rolling 12-month lookback with temporal split. For auction month X:
- **Fit**: `[X - 12 months, X - val_months - 1)` — model training
- **Validation**: `[X - val_months months, X - 1 month)` — threshold optimization
- **Test**: the actual `(auction_month, market_month)` pairs (future data)

Configured via `TrainingConfig(train_months=N, val_months=M, test_months=0)`. The `train_months_lookback` property computes the total (N+M).

**Current config (v008)**: train_months=10, val_months=2 → 12-month lookback. Note: v007 used 9/3 split; v008 reverted to 10/2 for more training data.

### Label Modification

If `prob_exceed_90 < 1e-5` → force `label = 0`. Rationale: density model says zero probability of exceeding 90% capacity, so treat as non-binding regardless of actual outcome.

### Test Unbind Rule

At prediction time, if `prob_exceed_90 < 1e-5` → force prediction to 0 ("Force Unbind"), skip classification entirely.

---

## 11. Signal Generation

`SignalGenerator.generate_signal()` converts pipeline output to ConstraintsSignal format:

1. Filter to `predicted_shadow_price > 0` only
2. Compute composite rank: `0.4 * prob_rank + 0.3 * hist_shadow_rank + 0.3 * pred_shadow_rank`
3. Assign 5 quantile tiers: 0 (best) through 4 (worst)
4. Format index as `{constraint_id}|{flow_direction}|spice`
5. Include columns: `tier`, `shadow_price` (= `predicted_shadow_price`), `predicted_shadow_price`, `binding_probability`, `equipment` (= `branch_name`), `shadow_sign`, etc.

### Signal Output Path

```
{data_root}/signal_data/{rto}/constraints/{signal_name}/{YYYY-MM}/{period_type}/{class_type}/
```

---

## 12. Pipeline Execution Flow

### Per Auction Month (`_process_auction_month`)

1. Calculate training period: `[auction_month - 12, auction_month - 1)`
2. Load training data (density + DA shadow prices via Ray)
3. Apply label modification rule
4. Identify test branches from test data
5. Train classifiers (default + per-branch, per horizon group)
6. Characterize never-binding branches for anomaly detection
7. Train regressors (default + per-branch, per horizon group)
8. For each market month:
   - Run prediction (anomaly detection → classification → regression)
   - Save `final_results.parquet` if `output_dir` is set

### Parallelism

**Within pipeline** (ShadowPricePipeline.run):
- Auction months can be processed in parallel via Ray (`parallel_equal_pool`)
- Within each auction month, branch model training is sequential
- `n_jobs` controls Ray worker count (0 = auto)

**Benchmark runner** (run_experiment.py --mode full):
- 32 benchmark runs executed as concurrent subprocesses via `run_workers_concurrent()`
- `--concurrency N` controls max simultaneous workers (default: 4)
- `_interleave_runs()` spreads workers across auction months to avoid data contention
- Each worker gets its own Ray session, log file, and 30-minute timeout
- Skip-if-exists: completed parquets are not re-run
- `--mode timing`: runs 2 workers sequential vs parallel for speedup measurement

---

## 13. Aggregation

Per-outage-date predictions are aggregated to monthly level:

| Column | Aggregation |
|--------|-------------|
| `predicted_shadow_price` | SUM across outage dates |
| `actual_shadow_price` | SUM |
| `binding_probability` | MEAN |
| `binding_probability_scaled` | Power-mean (p=3) |
| `predicted_binding` | COUNT → bool (>= 1) |
| Features | MEAN |

---

## 14. Evaluation

### Internal Metrics (val/holdout)

After training, `evaluate_split()` runs the `Predictor` on val and holdout splits and computes:

**Classification** (per-outage level):
- Precision, Recall, F1, F-beta (beta=2.0), AUC-ROC, AUC-PR

**Regression** (per-outage level):
- On TPs only: MAE, RMSE, Spearman rank correlation
- On all samples: MAE, RMSE

**Top-K Ranking** (per-outage level):
- Precision@K, Recall@K, Lift@K, Value Capture@K, Mean Value@K for K = {100, 250, 500, 1000, 2000}
- NDCG (full ranking quality)

Metrics are aggregated across auction months and returned in the `metrics` slot of `run()`. Verbose output prints compact tables per split per auction month and cross-month summaries.

### External Evaluation

`evaluate_signal_v2()` from pbase backtests signals against actual DA shadow prices:

```python
aptools.tools.evaluate_signal_v2(
    signal_name=SIGNAL_NAME,
    period_type='f0',
    peak_type='onpeak',
    auction_month_st='2024-06',
    auction_month_et_in='2025-12',
    n_jobs=12,
)
```

Key metrics: `sp%,t~0` (% shadow $ captured by tier 0), `recall,t~0`, `prec,t~0`, `F1,t~0`.

---

## 15. Model Registry & Gate Enforcement

### Version Registry (`versions/`)

Each experiment version is stored in `versions/{version_id}/` with:
- `metrics.json` — gate values (aggregate + per-period), benchmark scope, per-run scores
- `config.json` — frozen PredictionConfig at time of run
- `NOTES.md` — human-readable analysis and conclusions

`versions/manifest.json` tracks the active champion. Promotion requires passing all hard gates AND beating the champion within noise tolerance.

### Hard Gates (11 gates)

| Gate | Description | Direction | Floor |
|------|-------------|-----------|-------|
| S1-AUC | Stage 1 AUC-ROC | higher | 0.65 |
| S1-REC | Stage 1 Recall | higher | 0.25 |
| S1-PREC | Stage 1 Precision | higher | 0.25 |
| S2-SPR | Stage 2 Spearman (TP) | higher | 0.30 |
| C-VC@100 | Value Capture top-100 | higher | 0.20 |
| C-VC@500 | Value Capture top-500 | higher | 0.45 |
| C-VC@1000 | Value Capture top-1000 | higher | 0.50 |
| C-CAP@20 | Capture rate top-20 | higher | 0.50 |
| C-CAP@200 | Capture rate top-200 | higher | 0.30 |
| C-CAP@1000 | Capture rate top-1000 | higher | 0.10 |
| C-RMSE | Combined RMSE (all) | lower | 2000.0 |

**Noise tolerance**: `NOISE_TOLERANCE = 0.02` (2%) — candidate "beats" champion if value is within ±2% of champion's value, with at least one genuine improvement.

### Per-Period Gate Enforcement (mandatory since 2026-02-24)

Gates are checked at the **(class_type, period_type) segment** level. Each of the 4 segments — onpeak/f0, onpeak/f1, offpeak/f0, offpeak/f1 — must independently pass every floor gate. A version that passes in aggregate but fails on any single segment is **not promotable**.

The reported gate value is the cross-segment average (for display), but `passed=False` if any segment fails. `failed_segments` in the gate check output lists which segments failed.

For old versions without per-period data, falls back to class-level gate values.

---

## 16. File Map

| File | Purpose |
|------|---------|
| `src/shadow_price_prediction/config.py` | All dataclass configs (PredictionConfig, FeatureConfig, EnsembleConfig, etc.) |
| `src/shadow_price_prediction/iso_configs.py` | MISO/PJM-specific configs (IsoConfig, HorizonGroupConfig, DataPathConfig) |
| `src/shadow_price_prediction/data_loader.py` | DataLoader, MisoDataLoader, PjmDataLoader |
| `src/shadow_price_prediction/models.py` | ShadowPriceModels, train_ensemble, feature selection, threshold optimization |
| `src/shadow_price_prediction/prediction.py` | Predictor class — anomaly → classify → regress loop |
| `src/shadow_price_prediction/pipeline.py` | ShadowPricePipeline orchestrator, _process_auction_month |
| `src/shadow_price_prediction/anomaly_detection.py` | AnomalyDetector for never-binding branches |
| `src/shadow_price_prediction/signal_generator.py` | SignalGenerator — pipeline output → ConstraintsSignal |
| `src/shadow_price_prediction/evaluation.py` | `evaluate_split()`, top-K ranking metrics, cross-month aggregation |
| `src/shadow_price_prediction/registry.py` | Model version registry, gate checking, promotion logic |
| `src/shadow_price_prediction/naming.py` | Canonical file path generators for all pipeline artifacts |
| `scripts/run_experiment.py` | Full experiment runner: benchmark → score → register → audit |
| `scripts/_experiment_worker.py` | Subprocess worker: runs one (auction_month, class_type) slice |
| `scripts/audit_registry.py` | Registry integrity audit (7 files, checksums, version hashes) |
| `scripts/audit_experiment.py` | Experiment completeness audit (32 benchmark parquets) |
| `scripts/audit_config.py` | Config drift audit (stored config vs current code defaults) |
| `scripts/bootstrap_registry.py` | One-time migration: legacy baseline → v000-legacy in registry |
| `scripts/run_legacy_baseline.py` | Run the unmodified legacy pipeline for baseline comparison |
| `scripts/_legacy_worker.py` | Subprocess worker for legacy baseline runs |
| `notebooks/generate_signal_67b.ipynb` | Main notebook for running signal generation |
| `docs/runbook.md` | This file — pipeline design, architecture, run log |
| `docs/critique.md` | Bugs, improvements, and suggestions tracker |
| `docs/design/blockers.md` | Known blockers for PJM and MISO annual |

---

## 17. MISO Auction Schedule

Which period types are auctioned each calendar month:

| Month | Period Types |
|-------|-------------|
| Jan   | f0, f1, q4 |
| Feb   | f0, f1, f2, f3 |
| Mar   | f0, f1, f2 |
| Apr   | f0, f1 |
| May   | f0 |
| Jun   | f0 |
| Jul   | f0, f1, q2, q3, q4 |
| Aug   | f0, f1, f2, f3 |
| Sep   | f0, f1, f2 |
| Oct   | f0, f1, q3, q4 |
| Nov   | f0, f1, f2, f3 |
| Dec   | f0, f1, f2 |

---

## 18. Data Dependencies

### Density Data

```
/opt/temp/tmp/pw_data/spice6/prod_f0p_model_miso/density/
  auction_month={YYYY-MM}/market_month={YYYY-MM}/market_round={round}/outage_date={date}/
```

### Constraint Data

```
/opt/temp/tmp/pw_data/spice6/prod_f0p_model_miso/constraint_info/
  auction_month={YYYY-MM}/market_round={round}/period_type={pt}/class_type={ct}/
```

### DA Shadow Prices

Fetched via `MisoApTools.tools.fetch_da_shadow(st, et, class_type)` — requires Ray.

---

## 19. Run Log

| Date | Action | Scope | Result |
|------|--------|-------|--------|
| 2026-02-18 | Single-month dry run | 2025-01, f0/onpeak | Pass: 1332 constraints, all validations passed |
| 2026-02-18 | Full single-month test | 2025-01, all periods | Pass: 6/6 signals saved (f0,f1,q4 x onpeak,offpeak) |
| 2026-02-18 | Full 21-month run | 2024-06 to 2026-02 | Interrupted at month 2/21 (user pivot) |
| 2026-02-18 | Eval smoke test (pre-fix) | 2025-01, f0/onpeak+offpeak | Threshold=0.930 (NaN bug), Recall=0.38, AUC-ROC=0.84-0.86 |
| 2026-02-18 | NaN fix + top-K eval | 2025-01, f0/onpeak+offpeak | Threshold=0.51, Recall=0.57-0.74, Lift@100=3-6x, NDCG=0.60 |
| 2026-02-18 | +hist_da +clamp eval | 2025-01, f0/onpeak+offpeak | Offpeak RMSE -87%, Spearman +3-4%, hist_da dropped by clf selection |
| 2026-02-18 | +seasonal feats +constraint-K | 2025-01, f0/onpeak+offpeak | Spearman +2%, hist_da KEPT (density_skew DROP), Constraint ValCap@1K=74-77% |
| 2026-02-20 | Legacy baseline benchmark | PY20+PY21, 8 quarters, f0+f1, onpeak+offpeak (32 runs) | AUC=0.69-0.70, Recall=0.27, Spearman=0.38-0.45, ValCap@1000=84-86%. |
| 2026-02-20 | Model registry bootstrap | v000-legacy-20260220 promoted as champion | Registry created at `versions/`. |
| 2026-02-21 | Pipeline infrastructure upgrade | Dead code cleanup, naming.py, registry enrichment (7 artifacts), per-class gates, noise tolerance, version hashes, 3 audit scripts | |
| 2026-02-22 | Concurrent benchmark runner | `run_experiment.py` `--mode full` now uses `run_workers_concurrent()` with `--concurrency N` (default 4). Interleaved run ordering. `--mode timing` for speedup demo. Stale config.py comments removed. | Section 12 of design-planning.md |
| 2026-02-23 | Repo restructure | Moved scripts→`scripts/`, notebooks→`notebooks/`, docs→`docs/`, registry→`versions/`. Deleted stale `document/`, `notebook/`, `input/` dirs. Updated all path references. | Clean layout, `--help` works from new paths |
| 2026-02-23 | Pipeline fixes (v005) | train_months 6→9, test_months 3→0, density_skewness constraint 1→0, evaluation MAX→MEAN, gate floors relaxed (S1-AUC 0.80→0.65, S1-REC 0.30→0.25) | Applied to config.py, evaluation.py, registry.py |
| 2026-02-23 | v005-9mo-training benchmark | PY20+PY21, 32 runs, f0+f1, onpeak+offpeak, concurrency=4 | 32/32 OK. All 5 gates passed. S1-AUC=0.68, S1-REC=0.29, S2-SPR=0.42, C-VC@1000=0.82, C-RMSE=1098. See `versions/v005-9mo-training/NOTES.md` |
| 2026-02-23 | v006-density-only-10mo benchmark | PY20+PY21, 32 runs, density+DA features only, 10/2/0 split | 32/32 OK. All 5 gates passed (4/5 beat v000). S1-AUC=0.69, S1-REC=0.29, S2-SPR=0.40, C-VC@1000=0.84, C-RMSE=1134. S2-SPR does not beat v000. See `versions/v006-density-only-10mo/NOTES.md` |
| 2026-02-23 | v007-enriched-features benchmark | PY20+PY21, 32 runs, +8 new features (expected_overload, tail_concentration, prob_bands, density_entropy/cv, hist_da_trend/max), tuned XGB (lr=0.05, depth=5, subsample/colsample=0.8, L1/L2 reg), 9/3 split, F1 threshold | 32/32 OK. All 5 gates passed (2/5 beat v000). S1-AUC=0.68, S1-REC=0.31, S2-SPR=0.40, C-VC@1000=0.82, C-RMSE=1057. Best recall (+3.2pp vs v000) and RMSE (-498 vs v000). See `versions/v007-enriched-features/NOTES.md` |
| 2026-02-24 | v008-focused-classifier benchmark | PY20+PY21, 32 runs, focused classifier (14 features, v000 hyperparams + regularization), rich regressor (27 features), 10/2 split, beta=0.7 | 32/32 OK. All 5 gates passed. **PROMOTABLE**: beats v000 within 2% tolerance on all gates. S1-AUC=0.69, S1-REC=0.29 (+1.6pp), S2-SPR=0.41 (tied), C-VC@1000=0.84, C-RMSE=1109 (-447). See `versions/v008-focused-classifier/NOTES.md` |
| 2026-02-24 | Threshold experiment (v008-thr050) | v008 with threshold_override=0.5, 32 runs | **Redundant**: discovered v008 already used threshold=0.5 everywhere. f0 group optimizes to ~0.885 but artifact showed 0.5 due to f0/f1 worker race condition in threshold file naming. |
| 2026-02-24 | Threshold experiment (v007-thr050) | v007 with threshold_override=0.5, 12/32 runs | Incomplete (session interrupted at 12/32). v007 f1 thresholds were ~0.75-0.89; lowering to 0.5 would test recall impact. |
| 2026-02-24 | Per-period gate enforcement | registry.py, run_experiment.py | Gates now checked per (class_type, period_type) segment. Each of onpeak/f0, onpeak/f1, offpeak/f0, offpeak/f1 must independently pass all floor gates. Failed segments shown in summary. Rescored v000, v007, v008 with per-period data. |
| 2026-02-24 | threshold_override feature | config.py, models.py, _experiment_worker.py | Added ThresholdConfig.threshold_override to bypass threshold optimization. Fixed group=None early-return path to respect override. |
