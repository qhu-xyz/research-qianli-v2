# Business Context — Shadow Price Classification (Stage 1)

## What This Project Does

We predict whether transmission constraints in MISO (Midcontinent Independent System Operator) will **bind** — i.e., whether the shadow price will be non-zero. This is a binary classification problem: bind vs. not-bind.

**Why it matters**: Binding constraints create price differentials across nodes in the electricity grid. Predicting which constraints bind allows us to take positions in Financial Transmission Rights (FTRs), which pay the shadow price differential. Capital is limited — we can only hold a finite number of FTR positions — so we need to pick the constraints most likely to bind AND most likely to bind heavily.

## Business Objective: Precision Over Recall

**We prioritize precision.** When the model says "this constraint will bind," it needs to be right.

- Binding rate is ~7.5%. We do NOT want the model to predict 20% will bind — that wastes capital on false positives.
- Missing some binding events (lower recall) is acceptable. Over-predicting (lower precision) is not.
- The threshold is intentionally high (~0.83) and should stay precision-favoring (beta <= 1.0).
- The real value is in **ranking quality** — among predicted positives, are the highest-confidence ones actually the most valuable?

**Do NOT suggest lowering the threshold, increasing recall, or using beta > 1.0 to favor recall.** These go against the business objective.

## What the Features Represent

The 14 input features come from probabilistic power flow analysis:

| Feature | Meaning | Monotone |
|---|---|---|
| `prob_exceed_110` | P(line flow > 110% of limit) | +1 (higher = more likely to bind) |
| `prob_exceed_105` | P(line flow > 105% of limit) | +1 |
| `prob_exceed_100` | P(line flow > 100% of limit) | +1 |
| `prob_exceed_95` | P(line flow > 95% of limit) | +1 |
| `prob_exceed_90` | P(line flow > 90% of limit) | +1 |
| `prob_below_100` | P(line flow < 100% of limit) | -1 (higher = less likely to bind) |
| `prob_below_95` | P(line flow < 95% of limit) | -1 |
| `prob_below_90` | P(line flow < 90% of limit) | -1 |
| `expected_overload` | Expected MW above limit | +1 |
| `density_skewness` | Skewness of flow distribution | 0 (unconstrained) |
| `density_kurtosis` | Kurtosis of flow distribution | 0 |
| `density_cv` | Coefficient of variation | 0 |
| `hist_da` | Historical DA shadow price | +1 |
| `hist_da_trend` | Trend in DA shadow price | +1 |

Monotone constraints are enforced in XGBoost — e.g., higher prob_exceed_110 can only increase the predicted probability of binding.

## Data Structure

Each row represents: `(auction_month, constraint_name, ptype, ctype)` → binary target (bind/not-bind).

- **auction_month**: when the FTR auction occurs
- **ptype**: planning period (f0 = prompt month, f1 = next month, f2+ = further out)
- **ctype**: class type (onpeak, offpeak)
- Training uses a rolling window: 10 months train + 2 months validation, sliding forward
- Evaluation spans 12 primary months (2020-09 through 2022-12), ~270K rows each

## What the Key Metrics Mean

**Group A (hard gates, blocking) — all are ranking metrics, threshold-independent:**
- **S1-AUC**: Overall discrimination — can the model separate binders from non-binders?
- **S1-AP**: Average precision — quality of the positive class ranking (important for imbalanced data)
- **S1-VCAP@100**: Value capture at top-100 — do the top-100 predictions capture actual shadow price value?
- **S1-NDCG**: Normalized discounted cumulative gain — ranking quality weighted by position

**Group B (monitor, non-blocking) — mixed threshold-dependent and independent:**
- **S1-BRIER**: Calibration — are the predicted probabilities accurate? (lower is better)
- **S1-REC**: Recall at the optimized threshold
- **S1-CAP@100, S1-CAP@500**: Capture rate at top-K
- **S1-VCAP@500, S1-VCAP@1000**: Value capture at larger K

**Improving Group A metrics directly improves precision** at any threshold, because the model's probability scores become more accurate.

## Available Levers for Improvement

### Hyperparameters (in `ml/config.py` → `HyperparamConfig`)
| Param | v0 Default | Effect |
|---|---|---|
| `n_estimators` | 200 | More trees = better fit (diminishing returns, more compute) |
| `max_depth` | 4 | Deeper = more complex interactions (risk: overfitting) |
| `learning_rate` | 0.1 | Lower = needs more trees but generalizes better |
| `subsample` | 0.8 | Row sampling per tree (regularization) |
| `colsample_bytree` | 0.8 | Feature sampling per tree (regularization) |
| `reg_alpha` | 0.1 | L1 regularization |
| `reg_lambda` | 1.0 | L2 regularization |
| `min_child_weight` | 10 | Min samples in leaf (higher = more conservative) |

### Threshold (in `ml/config.py` → `PipelineConfig`)
- `threshold_beta`: F-beta parameter (v0=0.7, precision-favoring). **Keep <= 1.0.**
- `threshold_scaling_factor`: Multiplier on optimal threshold (v0=1.0)

### Features (in `ml/config.py` → `FeatureConfig`)
- Current: 14 features (5 exceedance, 3 below-threshold, 1 severity, 3 distribution shape, 2 historical)
- Monotone constraints enforced per feature
- New features could be derived from existing data columns

### Training window (in `ml/config.py` → `PipelineConfig`)
- `train_months`: 10 (rolling lookback for training)
- `val_months`: 2 (validation window for threshold tuning)

## v0 Baseline Summary (12 months, f0, onpeak)

| Metric | Mean | Std | Min | Bottom-2 Mean |
|---|---|---|---|---|
| S1-AUC | 0.835 | 0.015 | 0.809 | 0.811 |
| S1-AP | 0.394 | 0.041 | 0.315 | 0.332 |
| S1-VCAP@100 | 0.015 | 0.012 | 0.001 | 0.001 |
| S1-NDCG | 0.733 | 0.041 | 0.660 | 0.672 |
| S1-BRIER | 0.150 | 0.006 | 0.137 | 0.158 (worst) |
| S1-REC | 0.419 | 0.050 | 0.313 | 0.332 |
| Precision | 0.442 | 0.055 | 0.328 | — |
| Threshold | 0.834 | 0.021 | 0.791 | — |
| pred_binding_rate | 0.075 | 0.011 | 0.055 | — |

**Weakest months**: 2022-09 (AP=0.315), 2022-12 (AUC=0.809), 2022-06 (REC=0.313). Late-2022 consistently weaker — possible distribution shift.
