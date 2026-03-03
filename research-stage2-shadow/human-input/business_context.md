# Business Context — Shadow Price Regression (Stage 2)

## What This Project Does

We predict the **magnitude** of shadow prices (in dollars) for transmission constraints in MISO that are predicted to bind. This is a regression problem: given a constraint the stage 1 classifier flags as likely binding, estimate how much it will cost.

**Why it matters**: Binding constraints create price differentials across nodes in the electricity grid. Stage 1 tells us *which* constraints will bind; stage 2 tells us *how much* they will cost. The combined score (P(binding) x predicted_$) drives capital allocation for Financial Transmission Rights (FTR). The dollar prediction converts a probability into a dollar-weighted ranking that directly maps to portfolio value.

## Relationship to Stage 1

The **classifier is frozen** from stage 1 (champion v0008, 13 features, XGBoost binary classifier). Stage 2 never modifies or retrains the classifier. The two stages form a pipeline:

1. **Stage 1 (frozen)**: Binary classification — will this constraint bind? Outputs P(binding).
2. **Stage 2 (iterable)**: Regression — if it binds, what is the shadow price magnitude in dollars? Outputs predicted_$.
3. **Combined**: EV score = P(binding) x predicted_$ — used for ranking and capital allocation.

Only the regressor is subject to the agentic iteration loop. The classifier is immutable infrastructure.

## Business Objective: Maximize Value Capture

**We prioritize ranking quality.** When the model says "invest in this constraint," the predicted dollar value should be accurate enough that high-EV constraints are ranked above low-EV constraints.

- The EV-based ranking is **threshold-independent** — we rank all constraints by expected value, not by a binary cutoff
- Accuracy at the top of the ranking matters most — the top-100 and top-500 positions drive capital allocation
- Systematic over-prediction or under-prediction is acceptable if it preserves rank order (Spearman correlation)
- Catastrophic mis-ranking (predicting $0.10 for a $50 constraint) directly costs money
- Conditional accuracy (C-RMSE, C-MAE on predicted positives) provides calibration quality for position sizing

**Do NOT optimize for unconditional regression accuracy** (e.g., overall RMSE including non-binders). The regression only matters for constraints that are predicted to bind.

## What the Features Represent

The 24 regressor features consist of the 13 classifier features plus 11 additional features:

### Classifier Features (13, shared with stage 1)

| Feature | Meaning | Monotone |
|---|---|---|
| `prob_exceed_110` | P(line flow > 110% of limit) | +1 |
| `prob_exceed_105` | P(line flow > 105% of limit) | +1 |
| `prob_exceed_100` | P(line flow > 100% of limit) | +1 |
| `prob_exceed_95` | P(line flow > 95% of limit) | +1 |
| `prob_exceed_90` | P(line flow > 90% of limit) | +1 |
| `prob_below_100` | P(line flow < 100% of limit) | -1 |
| `prob_below_95` | P(line flow < 95% of limit) | -1 |
| `prob_below_90` | P(line flow < 90% of limit) | -1 |
| `expected_overload` | Expected MW above limit | +1 |
| `density_skewness` | Skewness of flow distribution | 0 (unconstrained) |
| `density_kurtosis` | Kurtosis of flow distribution | 0 |
| `hist_da` | Historical DA shadow price | +1 |
| `hist_da_trend` | Trend in DA shadow price | +1 |

### Additional Regressor Features (11, stage 2 only)

| Feature | Meaning | Monotone |
|---|---|---|
| `prob_exceed_85` | P(line flow > 85% of limit) | +1 |
| `prob_exceed_80` | P(line flow > 80% of limit) | +1 |
| `tail_concentration` | Fraction of flow density in the upper tail | +1 |
| `prob_band_95_100` | P(flow in the 95-100% band) | 0 (unconstrained) |
| `prob_band_100_105` | P(flow in the 100-105% band) | 0 |
| `density_mean` | Mean of flow distribution | 0 |
| `density_variance` | Variance of flow distribution | 0 |
| `density_entropy` | Entropy of flow distribution | 0 |
| `recent_hist_da` | Recent historical DA shadow price (shorter window) | +1 |
| `season_hist_da_1` | Seasonal historical DA component 1 | +1 |
| `season_hist_da_2` | Seasonal historical DA component 2 | +1 |

## Data Structure

Each row represents: `(auction_month, constraint_name, ptype, ctype)` with a continuous target (shadow price magnitude in dollars).

- **auction_month**: when the FTR auction occurs
- **ptype**: planning period (f0 = prompt month, f1 = next month, f2+ = further out)
- **ctype**: class type (onpeak, offpeak)
- Training uses a rolling window: 10 months train + 2 months validation, sliding forward
- Evaluation spans 12 primary months
- In **gated mode** (default), the regressor trains only on rows where the classifier predicts positive
- In **unified mode**, the regressor trains on all rows (experimental)

## What the Key Metrics Mean

**Group A (hard gates, blocking) — EV-based ranking quality:**
- **EV-VC@100**: Expected-value capture at top-100 — do the 100 highest EV-scored constraints capture actual shadow price value?
- **EV-VC@500**: Expected-value capture at top-500 — broader value capture quality
- **EV-NDCG**: Normalized discounted cumulative gain on EV scores — ranking quality weighted by position
- **Spearman**: Spearman rank correlation between EV scores and actual shadow prices — monotonic relationship quality

**Group B (monitor, non-blocking) — regression calibration and secondary ranking:**
- **C-RMSE**: Conditional RMSE on predicted positives — regression calibration (lower is better)
- **C-MAE**: Conditional MAE on predicted positives — regression calibration (lower is better)
- **EV-VC@1000**: Expected-value capture at top-1000 — even broader value capture
- **R-REC@500**: Regressor-only recall at top-500 — raw regression ranking quality without classifier

## Available Levers for Improvement

### Regressor Hyperparameters (in `ml/config.py` -> `RegressorConfig`)
| Param | v0 Default | Effect |
|---|---|---|
| `n_estimators` | 400 | More trees for regression (needs more for continuous targets) |
| `max_depth` | 5 | Deeper than classifier (regression benefits from more interactions) |
| `learning_rate` | 0.05 | Lower rate for regression stability |
| `subsample` | 0.8 | Row sampling per tree (regularization) |
| `colsample_bytree` | 0.8 | Feature sampling per tree (regularization) |
| `reg_alpha` | 0.1 | L1 regularization |
| `reg_lambda` | 1.0 | L2 regularization |
| `min_child_weight` | 10 | Min samples in leaf (higher = more conservative) |

### Feature Engineering
- Current: 24 features (13 classifier + 11 additional)
- Monotone constraints enforced per feature
- New features can be derived from existing data columns or new signal sources
- Interaction features between classifier and additional features are unexplored

### Training Mode (in `ml/config.py` -> `RegressorConfig`)
- `unified_regressor`: False (gated, default) vs True (unified) — gated trains only on classifier-positive rows
- `value_weighted`: False (default) vs True — weight training samples by shadow price magnitude

### Training Window (in `ml/config.py` -> `PipelineConfig`)
- `train_months`: 10 (rolling lookback for training)
- `val_months`: 2 (validation window)

## Gate System

### Gate Groups

| Group | Gates | Role |
|---|---|---|
| **A (hard)** | EV-VC@100, EV-VC@500, EV-NDCG, Spearman | Must pass all 3 layers to promote |
| **B (monitor)** | C-RMSE, C-MAE, EV-VC@1000, R-REC@500 | Tracked, don't block promotion |

### Three-Layer Promotion Checks (same as stage 1)

| Layer | What it checks | Formula | Purpose |
|---|---|---|---|
| **1. Mean Quality** | Average performance | `mean(metric) >= floor` | Basic quality bar |
| **2. Tail Safety** | Catastrophic month protection | `count(months below tail_floor) <= 1` | No single-month disasters |
| **3. Tail Non-Regression** | Worst-case improvement | `bottom_2_mean(new) >= bottom_2_mean(champ) - 0.02` | Worst months must not regress |

### Lower-is-Better Metrics

C-RMSE and C-MAE have inverted directions: `floor` is a ceiling, `tail_floor` is a ceiling, and `bottom_2_mean` picks the highest (worst) 2 values.
