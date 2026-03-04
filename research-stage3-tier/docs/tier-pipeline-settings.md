# Tier Classification Pipeline — Complete Settings & Knowledge Transfer

**Last updated:** 2026-03-04
**Pipeline:** `research-stage3-tier`
**Status:** v0 baseline running (with early stopping + Value-QWK)

---

## 1. What This Pipeline Does

Predicts **5 shadow-price tiers** for MISO transmission constraints using a single multi-class XGBoost classifier. This replaces the two-stage approach (stage-1 binary classifier + stage-2 regressor) with one model that directly outputs tier probabilities.

**Why tiers instead of continuous prediction?**
- The downstream trading system (pmodel/ftr24/v1) ultimately bins constraints into tiers for path pool construction and capital allocation
- A direct tier classifier aligns the ML objective with the actual decision boundary
- Probability distributions over tiers provide a richer ranking signal than point estimates

---

## 2. Tier Definitions (FROZEN)

Matches the existing SPICE tier system from `pbase/analysis/tier_threshold_generator_1.py`.

| Tier | Shadow Price Range | Meaning | Midpoint ($) | Class Weight |
|------|-------------------|---------|-------------|-------------|
| 0 | [3000, +inf) | Heavily binding | 4000 | 10 |
| 1 | [1000, 3000) | Strongly binding | 2000 | 5 |
| 2 | [100, 1000) | Moderately binding | 550 | 2 |
| 3 | [0, 100) | Lightly binding | 50 | 1 |
| 4 | (-inf, 0) | Not binding | 0 | 0.5 |

**Bins:** `[-inf, 0, 100, 1000, 3000, inf]` with labels `[4, 3, 2, 1, 0]` (reverse order: highest bin index = highest tier).

**Key fact:** Tier 4 has **0 samples** in all real training months — there are no negative shadow prices in MISO data. The model effectively learns 4 classes.

---

## 3. Model Architecture

### Single XGBoost Multi-Class Classifier

| Parameter | Value | Notes |
|-----------|-------|-------|
| `objective` | `multi:softprob` | Outputs probability distribution over 5 tiers |
| `num_class` | 5 | Tiers 0-4 |
| `n_estimators` | 400 | Max trees (early stopping usually stops at ~100-200) |
| `max_depth` | 5 | |
| `learning_rate` | 0.05 | |
| `subsample` | 0.8 | |
| `colsample_bytree` | 0.8 | |
| `reg_alpha` | 1.0 | L1 regularization |
| `reg_lambda` | 1.0 | L2 regularization |
| `min_child_weight` | 25 | Prevents small leaves for rare classes |
| `early_stopping_rounds` | 50 | Stops training when val loss plateaus |
| `eval_metric` | `mlogloss` | Multi-class log loss for early stopping |
| `tree_method` | `hist` | Histogram-based (fast) |
| `random_state` | 42 | Reproducibility |
| `monotone_constraints` | per-feature | See feature table below |

### No Frozen Components

Unlike stage 2 (which had a frozen stage-1 classifier), the entire model is the iterable component. There is no separate classifier or regressor — one model, one output.

---

## 4. Training Configuration (FROZEN)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `train_months` | 6 | Historical months for training data |
| `val_months` | 2 | Historical months for early stopping validation |
| `eval_months` | 12 | Months evaluated in benchmark (see list below) |
| `class_type` | `onpeak` | Only onpeak evaluated currently |
| `period_type` | `f0` | Only f0 (forward month) evaluated |

### Evaluation Months

```
2020-09, 2020-11, 2021-01, 2021-03, 2021-05, 2021-07,
2021-09, 2021-11, 2022-03, 2022-06, 2022-09, 2022-12
```

### How Train/Val/Eval Works

For each eval month (e.g., 2021-07):
1. **Train set**: 6 months of historical data before the eval month
2. **Val set**: 2 months between train and eval month (used for early stopping, NOT for hyperparameter tuning)
3. **Test set**: The actual target month (2021-07) — this is what we report metrics on

The val set is used as XGBoost's `eval_set` with `early_stopping_rounds=50`. When `mlogloss` on val doesn't improve for 50 rounds, training stops. This typically saves ~50-75% of trees (e.g., 96/400, 127/400, 191/400 in observed runs).

---

## 5. Feature List (34 Features)

### Current Feature Set

The features are inherited from the stage-2 regressor champion (v0012). 5 dead features were already removed. The autonomous loop is constrained to modify ONLY features and monotone constraints.

| # | Feature | Monotone | Category |
|---|---------|----------|----------|
| 1 | `prob_exceed_110` | +1 | Flow probability |
| 2 | `prob_exceed_105` | +1 | Flow probability |
| 3 | `prob_exceed_100` | +1 | Flow probability |
| 4 | `prob_exceed_95` | +1 | Flow probability |
| 5 | `prob_exceed_90` | +1 | Flow probability |
| 6 | `prob_below_100` | -1 | Flow probability |
| 7 | `prob_below_95` | -1 | Flow probability |
| 8 | `prob_below_90` | -1 | Flow probability |
| 9 | `expected_overload` | +1 | Overload |
| 10 | `hist_da` | +1 | Historical price |
| 11 | `hist_da_trend` | +1 | Historical price |
| 12 | `sf_max_abs` | +1 | Shift factor |
| 13 | `sf_mean_abs` | +1 | Shift factor |
| 14 | `sf_std` | 0 | Shift factor |
| 15 | `sf_nonzero_frac` | 0 | Shift factor |
| 16 | `is_interface` | 0 | Constraint type |
| 17 | `constraint_limit` | 0 | Constraint type |
| 18 | `density_mean` | 0 | Distribution shape |
| 19 | `density_variance` | 0 | Distribution shape |
| 20 | `density_entropy` | 0 | Distribution shape |
| 21 | `tail_concentration` | +1 | Distribution shape |
| 22 | `prob_band_95_100` | 0 | Flow probability |
| 23 | `prob_band_100_105` | 0 | Flow probability |
| 24 | `hist_da_max_season` | +1 | Historical price |
| 25 | `prob_exceed_85` | +1 | Flow probability |
| 26 | `prob_exceed_80` | +1 | Flow probability |
| 27 | `recent_hist_da` | +1 | Historical price |
| 28 | `season_hist_da_1` | +1 | Historical price |
| 29 | `season_hist_da_2` | +1 | Historical price |
| 30 | `density_skewness` | 0 | Distribution shape |
| 31 | `density_kurtosis` | 0 | Distribution shape |
| 32 | `density_cv` | 0 | Distribution shape |
| 33 | `season_hist_da_3` | +1 | Historical price |
| 34 | `prob_below_85` | -1 | Flow probability |

### Dead Features (removed in stage 2)

These 5 features were pruned in stage-2 FE optimization:
- `hist_physical_interaction`
- `overload_exceedance_product`
- `band_severity`
- `sf_exceed_interaction`
- `hist_seasonal_band`

### Interaction Features (computed in `ml/features.py`)

The 5 dead features above are still computed by `compute_interaction_features()` for backward compatibility but are filtered out by `_DEAD_FEATURES` before use. New interaction features can be added here.

---

## 6. EV Score (Ranking Signal)

The tier label alone is too coarse for capital allocation (top-100, top-500 ranking). The multi-class probabilities produce a continuous ranking score:

```
tier_ev_score = P(tier=0) * 4000 + P(tier=1) * 2000 + P(tier=2) * 550 + P(tier=3) * 50 + P(tier=4) * 0
```

This is the primary ranking signal — Tier-VC@100 and Tier-VC@500 evaluate how well the top-k constraints by EV score capture actual shadow price value.

---

## 7. Metrics

### Group A — Blocking Gates (must pass to promote)

| Metric | What It Measures | Range |
|--------|-----------------|-------|
| **Tier-VC@100** | Fraction of total actual shadow price value captured by top 100 constraints ranked by `tier_ev_score` | [0, 1] |
| **Tier-VC@500** | Same as VC@100 but top 500 | [0, 1] |
| **Tier-NDCG** | Normalized DCG using actual shadow price as relevance, ranked by `tier_ev_score` | [0, 1] |
| **QWK** | Cohen's Quadratic Weighted Kappa — penalizes large tier mismatches quadratically | (-inf, 1] |

### Group B — Monitor Gates (tracked, don't block promotion)

| Metric | What It Measures | Range |
|--------|-----------------|-------|
| **Macro-F1** | Unweighted average F1 across all tiers | [0, 1] |
| **Tier-Accuracy** | Overall classification accuracy | [0, 1] |
| **Adjacent-Accuracy** | Fraction of predictions within 1 tier of actual | [0, 1] |
| **Tier-Recall@0** | Recall for tier 0 (heavily binding, $3000+) | [0, 1] |
| **Tier-Recall@1** | Recall for tier 1 (strongly binding, $1000-3000) | [0, 1] |
| **Value-QWK** | Value-Weighted QWK — tier 0 misclassifications penalized ~80x more than tier 3 | (-inf, 1] |

### Value-Weighted QWK

Standard QWK treats all tier misclassifications equally (by ordinal distance). Value-QWK weights each row of the confusion matrix by the tier's midpoint value:

```
Weight for tier 0 miss: 4000 / 6600 = 0.606  (60.6% of total penalty budget)
Weight for tier 1 miss: 2000 / 6600 = 0.303  (30.3%)
Weight for tier 2 miss:  550 / 6600 = 0.083   (8.3%)
Weight for tier 3 miss:   50 / 6600 = 0.008   (0.8%)
Weight for tier 4 miss:    0 / 6600 = 0.000   (0.0%)
```

This reflects the capital allocation reality: getting tier 0 wrong ($4000 shadow price) costs ~80x more than getting tier 3 wrong ($50).

**Status:** Added as Group B monitor with `pending_baseline: true`. Will be calibrated after v0 baseline completes.

### All Metrics Are Higher-Is-Better

No direction inversions needed (unlike stage 2 which had lower-is-better C-RMSE).

---

## 8. Gate System — Three-Layer Promotion Checks

### Three Layers

| Layer | Formula | Purpose |
|-------|---------|---------|
| **1. Mean Quality** | `mean(metric) >= floor` | Basic quality bar |
| **2. Tail Safety** | `count(months < tail_floor) <= 1` | No catastrophic months |
| **3. Tail Non-Regression** | `bottom_2_mean(new) >= bottom_2_mean(champ) - 0.02` | Worst months don't regress |

### Gate Floors (from v0 baseline, PRE-early-stopping)

These floors will be recalibrated after the current v0 baseline (with early stopping) completes.

| Gate | Floor | Tail Floor | Group |
|------|-------|------------|-------|
| Tier-VC@100 | 0.075 | 0.008 | A |
| Tier-VC@500 | 0.217 | 0.047 | A |
| Tier-NDCG | 0.767 | 0.629 | A |
| QWK | 0.359 | 0.184 | A |
| Macro-F1 | 0.369 | 0.288 | B |
| Tier-Accuracy | 0.943 | 0.931 | B |
| Adjacent-Accuracy | 0.975 | 0.961 | B |
| Tier-Recall@0 | 0.374 | 0.076 | B |
| Tier-Recall@1 | 0.098 | 0.026 | B |
| Value-QWK | pending | pending | B |

### Calibration Formula

From `ml/populate_v0_gates.py`:
- `floor = 0.9 * v0_mean` (10% below v0 mean)
- `tail_floor = v0_min` (worst month sets the absolute floor)
- `noise_tolerance = 0.02`
- `tail_max_failures = 1` (allow 1 month below tail floor)

---

## 9. Pipeline Architecture

### 6-Phase Pipeline (`ml/pipeline.py`)

```
Phase 1: Load train/val data (6+2 month lookback via Ray)
Phase 2: Compute interaction features, prepare feature matrices, compute tier labels
Phase 3: Train XGBoost classifier (with early stopping on val set)
Phase 4: Load target-month test data
Phase 5: Evaluate on test data (all metrics)
Phase 6: Return results
```

### Key Files

| File | Purpose | Who May Modify |
|------|---------|----------------|
| `ml/config.py` | TierConfig, PipelineConfig, GateConfig | Worker (features + monotone only) |
| `ml/features.py` | Feature prep, interaction features, tier labels, sample weights | Worker |
| `ml/train.py` | XGBoost training with early stopping | HUMAN ONLY |
| `ml/evaluate.py` | All metric computation | HUMAN ONLY |
| `ml/pipeline.py` | 6-phase orchestration | HUMAN ONLY |
| `ml/benchmark.py` | 12-month benchmark runner | HUMAN ONLY |
| `ml/data_loader.py` | Data loading via Ray | HUMAN ONLY |
| `ml/compare.py` | Gate comparison logic | HUMAN ONLY |
| `ml/populate_v0_gates.py` | Calibrate gate floors from v0 | HUMAN ONLY |
| `registry/gates.json` | Gate definitions | HUMAN ONLY |
| `memory/human_input.md` | Per-batch constraints | Human (before batch) |

### Data Loading

Data comes from `pbase` via Ray. The `load_data()` function:
1. Connects to Ray cluster at `ray://10.8.0.36:10001`
2. Loads MISO constraint data for the specified months
3. Returns polars DataFrames for train and val sets

---

## 10. Autonomous Loop Configuration

### Batch Structure

Each batch runs 3 iterations. Each iteration:
1. **Orchestrator plans** — generates 2 hypotheses from learnings + human guidance
2. **Worker implements** — screens both on 2 months (1 weak + 1 strong), runs full 12-month on winner
3. **Claude reviews** — independent code + gate review
4. **Codex reviews** — independent second review
5. **Orchestrator synthesizes** — merges reviews, decides promotion

### Current Constraint: Feature Engineering Only

The autonomous loop is currently constrained to:
- **ALLOWED:** Change `features` list, `monotone_constraints` list, `compute_interaction_features()` function
- **FORBIDDEN:** Any hyperparameter changes, class weight changes, tier bin/midpoint changes, training logic changes, evaluation changes

This is enforced via `memory/human_input.md` which all agent prompts reference.

### Candidate Interaction Features

From `memory/human_input.md`, these are potential new features to explore:

| Feature | Formula | Domain Rationale |
|---------|---------|-----------------|
| `overload_x_hist` | `expected_overload * hist_da` | Historical binding x overload signal |
| `prob110_x_hist` | `prob_exceed_110 * recent_hist_da` | Flow exceedance x recent price |
| `log1p_hist_da` | `log1p(hist_da)` | Compress long-tailed price history |
| `prob_range_high` | `prob_exceed_100 - prob_exceed_110` | Probability mass in 100-110% range |
| `tail_x_hist` | `tail_concentration * hist_da` | Density tail x price signal |
| `sf_x_overload` | `sf_max_abs * expected_overload` | Shift factor x overload |
| `density_range` | `density_entropy * density_cv` | Distribution spread indicator |

### Stage-2 FE Learnings

1. Adding `density_skewness`/`kurtosis`/`cv`, `season_hist_da_3`, `prob_below_85` was the biggest win (+9% EV-VC@100)
2. Pruning 5 dead features improved sampling efficiency (+5.2% EV-VC@100)
3. `flow_direction` had no effect (likely constant)
4. `value_weighted` had no effect

---

## 11. Infrastructure

### Environment

```bash
# Activate venv
cd /home/xyz/workspace/pmodel && source .venv/bin/activate

# Set project path
cd /home/xyz/workspace/research-qianli-v2/research-stage3-tier
export PYTHONPATH=.
```

### Ray Cluster

- Address: `ray://10.8.0.36:10001`
- Must be initialized BEFORE any data loading
- Benchmark.py auto-initializes Ray if not already connected

### Running Benchmarks

```bash
# Full 12-month benchmark
source /home/xyz/workspace/pmodel/.venv/bin/activate
cd /home/xyz/workspace/research-qianli-v2/research-stage3-tier
PYTHONPATH=. python ml/benchmark.py --version-id v0 --ptype f0 --class-type onpeak

# Smoke test (synthetic data, fast)
SMOKE_TEST=true PYTHONPATH=. python ml/benchmark.py --version-id v0 --ptype f0 --eval-months 2021-07

# Populate gate floors from v0
PYTHONPATH=. python ml/populate_v0_gates.py

# Run autonomous batch
bash agents/run_pipeline.sh --batch-name my-batch
```

### Running Tests

```bash
SMOKE_TEST=true PYTHONPATH=. python -m pytest ml/tests/ -x -q
```

---

## 12. Known Issues & Observations

### Tier Distribution Imbalance

- **Tier 3** (lightly binding, $0-100) dominates: ~93-95% of samples
- **Tier 2** (moderately binding): ~3-5%
- **Tier 1** (strongly binding): ~1-2%
- **Tier 0** (heavily binding): ~0.5-1%
- **Tier 4** (not binding): 0% in all months

This extreme imbalance is addressed via class weights (10/5/2/1/0.5) and is the primary challenge for the pipeline.

### Tier-Recall@1 is Low

Mean recall for tier 1 is only ~0.098 — the model misses most strongly binding constraints. This is a key area for improvement via feature engineering.

### Early Stopping Effectiveness

Observed in partial v0 runs: models use ~87-191 trees instead of 400, with significant metric improvements vs the pre-early-stopping baseline.

---

## 13. Glossary

| Term | Definition |
|------|-----------|
| **Shadow price** | The marginal cost of relaxing a transmission constraint by 1 MW. Higher = more valuable to trade. |
| **Tier** | Ordinal category binning shadow prices into 5 levels (0=highest, 4=lowest). |
| **EV score** | `tier_ev_score = sum(P(tier) * midpoint)` — continuous ranking signal from tier probabilities. |
| **VC@k** | Value Capture at k — fraction of total actual value in top-k ranked constraints. |
| **QWK** | Quadratic Weighted Kappa — agreement metric penalizing large ordinal mismatches. |
| **Value-QWK** | QWK weighted by tier midpoints — penalizes high-value tier misclassifications more. |
| **NDCG** | Normalized Discounted Cumulative Gain — ranking quality metric using actual shadow price as relevance. |
| **FTR** | Financial Transmission Right — the derivative product traded on congestion. |
| **MISO** | Midcontinent Independent System Operator — the RTO whose data we use. |
| **pbase** | Internal power trading library for data loading and analysis. |
| **pmodel** | Internal modeling library; provides the Python venv. |
