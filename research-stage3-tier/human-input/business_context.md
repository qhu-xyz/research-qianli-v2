# Business Context — Tier Classification (Stage 3)

## What This Project Does

We predict the **shadow price tier** (0-4) for transmission constraints in MISO using a single multi-class XGBoost classifier. This replaces the two-stage pipeline (stage 1 binary classifier + stage 2 regressor) with a direct tier prediction that maps shadow prices into 5 business-meaningful categories.

**Why it matters**: Binding constraints create price differentials across nodes in the electricity grid. Tier classification directly predicts how severely a constraint will bind, enabling capital allocation for Financial Transmission Rights (FTR). The tier expected-value score (`tier_ev = sum(P(tier=t) * midpoint[t])`) produces a continuous ranking that drives portfolio construction.

## Tier Definitions

| Tier | Shadow Price Range | Meaning | Midpoint |
|------|-------------------|---------|----------|
| 0 | [3000, +inf) | Heavily binding | $4000 |
| 1 | [1000, 3000) | Strongly binding | $2000 |
| 2 | [100, 1000) | Moderately binding | $550 |
| 3 | [0, 100) | Lightly binding | $50 |
| 4 | (-inf, 0) | Not binding | $0 |

Bin edges: `[-inf, 0, 100, 1000, 3000, inf]` mapped to labels `[4, 3, 2, 1, 0]`.

## Relationship to Stage 1 and Stage 2

Stage 3 is a **standalone replacement** for the two-stage pipeline:

1. **Stage 1** (retired): Binary classification — will this constraint bind?
2. **Stage 2** (retired): Regression — if it binds, what is the shadow price magnitude?
3. **Stage 3** (this): Multi-class classification — directly predict the tier. The tier EV score replaces P(binding) x predicted_$.

There is no frozen classifier. All `TierConfig` parameters are mutable and subject to the agentic iteration loop.

## Business Objective: Maximize Value Capture via Tier Rankings

**We prioritize ranking quality.** The tier EV score (`tier_ev = P(tier=0)*4000 + P(tier=1)*2000 + P(tier=2)*550 + P(tier=3)*50 + P(tier=4)*0`) should rank high-value constraints above low-value ones for capital allocation.

- The EV-based ranking is **threshold-independent** — we rank all constraints by expected value, not by a binary cutoff
- Accuracy at the top of the ranking matters most — the top-100 and top-500 positions drive capital allocation
- **Tier-VC@100 is "the money metric"** — top-100 capital allocation quality
- Ordinal consistency (QWK) matters — predicting tier 4 for a tier 0 constraint is catastrophic
- **Missing tier 0/1 constraints is catastrophic** — these heavily/strongly binding constraints represent the highest value opportunities
- Adjacent tier errors (e.g., predicting tier 2 for a tier 1) are tolerable; distant errors (predicting tier 4 for tier 0) directly cost money

## What the Features Represent

The tier classifier uses 34 features covering flow probability, distribution shape, and historical prices:

### Flow Probability Features (11)

| Feature | Meaning | Monotone |
|---|---|---|
| `prob_exceed_110` | P(line flow > 110% of limit) | +1 |
| `prob_exceed_105` | P(line flow > 105% of limit) | +1 |
| `prob_exceed_100` | P(line flow > 100% of limit) | +1 |
| `prob_exceed_95` | P(line flow > 95% of limit) | +1 |
| `prob_exceed_90` | P(line flow > 90% of limit) | +1 |
| `prob_exceed_85` | P(line flow > 85% of limit) | +1 |
| `prob_exceed_80` | P(line flow > 80% of limit) | +1 |
| `prob_below_100` | P(line flow < 100% of limit) | -1 |
| `prob_below_95` | P(line flow < 95% of limit) | -1 |
| `prob_below_90` | P(line flow < 90% of limit) | -1 |
| `tail_concentration` | Fraction of flow density in the upper tail | +1 |

### Flow Distribution Shape Features (7)

| Feature | Meaning | Monotone |
|---|---|---|
| `prob_band_95_100` | P(flow in the 95-100% band) | 0 |
| `prob_band_100_105` | P(flow in the 100-105% band) | 0 |
| `density_mean` | Mean of flow distribution | 0 |
| `density_variance` | Variance of flow distribution | 0 |
| `density_entropy` | Entropy of flow distribution | 0 |
| `density_skewness` | Skewness of flow distribution | 0 |
| `density_kurtosis` | Kurtosis of flow distribution | 0 |

### Overload Features (1)

| Feature | Meaning | Monotone |
|---|---|---|
| `expected_overload` | Expected MW above limit | +1 |

### Historical Price Features (5)

| Feature | Meaning | Monotone |
|---|---|---|
| `hist_da` | Historical DA shadow price | +1 |
| `hist_da_trend` | Trend in DA shadow price | +1 |
| `recent_hist_da` | Recent historical DA shadow price (shorter window) | +1 |
| `season_hist_da_1` | Seasonal historical DA component 1 | +1 |
| `season_hist_da_2` | Seasonal historical DA component 2 | +1 |

### Feature-Engineered Features (10)

| Feature | Meaning | Monotone |
|---|---|---|
| `overload_x_hist` | expected_overload * hist_da | +1 |
| `prob110_x_hist` | prob_exceed_110 * hist_da | +1 |
| `prob105_x_hist` | prob_exceed_105 * hist_da | +1 |
| `prob100_x_hist` | prob_exceed_100 * hist_da | +1 |
| `log1p_hist_da` | log1p(hist_da) | +1 |
| `log1p_expected_overload` | log1p(expected_overload) | +1 |
| `prob_range_high` | prob_exceed_90 - prob_exceed_110 | 0 |
| `overload_x_recent_hist` | expected_overload * recent_hist_da | +1 |
| `prob110_x_recent_hist` | prob_exceed_110 * recent_hist_da | +1 |
| `tail_x_hist` | tail_concentration * hist_da | +1 |

## Data Structure

Each row represents: `(auction_month, constraint_name, ptype, ctype)` with a discrete target (tier label 0-4).

- **auction_month**: when the FTR auction occurs
- **ptype**: planning period (f0 = prompt month, f1 = next month, f2+ = further out)
- **ctype**: class type (onpeak, offpeak)
- Training uses a rolling window: 6 months train + 2 months validation, sliding forward
- Evaluation spans 12 primary months
- Class imbalance is handled via per-tier sample weights

## What the Key Metrics Mean

**Group A (hard gates, blocking) — EV-based ranking quality + ordinal consistency:**
- **Tier-VC@100**: Tier-based value capture at top-100 — do the 100 highest tier_ev-scored constraints capture actual shadow price value?
- **Tier-VC@500**: Tier-based value capture at top-500 — broader value capture quality
- **Tier-NDCG**: Normalized discounted cumulative gain on tier EV scores — ranking quality weighted by position
- **QWK**: Quadratic Weighted Kappa — ordinal consistency between predicted and actual tiers (penalizes distant errors quadratically)

**Group B (monitor, non-blocking) — classification quality:**
- **Macro-F1**: Macro-averaged F1 across all tiers — balanced classification quality
- **Tier-Accuracy**: Overall tier prediction accuracy
- **Adjacent-Accuracy**: Accuracy allowing +-1 tier error — tolerance for near-misses
- **Tier-Recall@0**: Recall for tier 0 (heavily binding) — critical for capturing highest-value constraints
- **Tier-Recall@1**: Recall for tier 1 (strongly binding) — critical for capturing high-value constraints

## Available Levers for Improvement

### XGBoost Hyperparameters (in `ml/config.py` -> `TierConfig`)
| Param | v0 Default | Effect |
|---|---|---|
| `n_estimators` | 400 | More trees = more capacity |
| `max_depth` | 5 | Deeper trees = more feature interactions |
| `learning_rate` | 0.05 | Lower rate = more stable but slower convergence |
| `subsample` | 0.8 | Row sampling per tree (regularization) |
| `colsample_bytree` | 0.8 | Feature sampling per tree (regularization) |
| `reg_alpha` | 1.0 | L1 regularization |
| `reg_lambda` | 1.0 | L2 regularization |
| `min_child_weight` | 25 | Min samples in leaf (higher = more conservative) |

### Class Weights (in `ml/config.py` -> `TierConfig`)
| Tier | Default Weight | Rationale |
|---|---|---|
| 0 | 10 | Heavily upweight rare, high-value tier |
| 1 | 5 | Upweight rare, high-value tier |
| 2 | 2 | Moderate upweight |
| 3 | 1 | Baseline |
| 4 | 0.5 | Downweight common, low-value tier |

### Feature Engineering
- Current: 34 features (11 flow probability + 7 distribution shape + 1 overload + 5 historical price + 10 engineered)
- Monotone constraints enforced per feature
- New features can be derived from existing data columns or new signal sources

### Tier Configuration
- `bins`: Tier bin edges — `[-inf, 0, 100, 1000, 3000, inf]` — use with caution, changes tier definitions
- `tier_midpoints`: EV score midpoints — `[4000, 2000, 550, 50, 0]` — affects ranking signal

### Training Window (in `ml/config.py` -> `PipelineConfig`)
- `train_months`: 6 (rolling lookback for training)
- `val_months`: 2 (validation window)

## Gate System

### Gate Groups

| Group | Gates | Role |
|---|---|---|
| **A (hard)** | Tier-VC@100, Tier-VC@500, Tier-NDCG, QWK | Must pass all 3 layers to promote |
| **B (monitor)** | Macro-F1, Tier-Accuracy, Adjacent-Accuracy, Tier-Recall@0, Tier-Recall@1 | Tracked, don't block promotion |

### Three-Layer Promotion Checks

| Layer | What it checks | Formula | Purpose |
|---|---|---|---|
| **1. Mean Quality** | Average performance | `mean(metric) >= floor` | Basic quality bar |
| **2. Tail Safety** | Catastrophic month protection | `count(months below tail_floor) <= 1` | No single-month disasters |
| **3. Tail Non-Regression** | Worst-case improvement | `bottom_2_mean(new) >= bottom_2_mean(champ) - 0.02` | Worst months must not regress |

### All Metrics Are Higher-Is-Better

Unlike stage 2 (which had lower-is-better C-RMSE and C-MAE), all tier metrics are higher-is-better. No direction inversions needed.
