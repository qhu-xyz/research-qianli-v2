# Experiment Setup: Annual Constraint Tier Prediction

**Date:** 2026-03-08
**Goal:** Predict which MISO constraints will bind in each annual auction quarter (aq1-aq4), producing 5-tier rankings analogous to V6.1's output.

**Direct parallel to monthly stage5:** `/home/xyz/workspace/research-qianli-v2/research-stage5-tier/experiment-setup.md`

---

## 1. Problem Statement

For each MISO annual FTR auction quarter, rank ~400-900 constraints by their likelihood of binding. Output: 5 equal-sized tiers (tier 0 = most binding, tier 4 = least).

**Unit of prediction:** One (planning_year, aq_round) — e.g., 2024-06/aq1.

**MISO annual auction structure:**

| Round | Market months | Typical auction timing |
|-------|--------------|----------------------|
| aq1 | Jun-Aug | Before Jun 1 |
| aq2 | Sep-Nov | Before Sep 1 |
| aq3 | Dec-Feb | Before Dec 1 |
| aq4 | Mar-May | Before Mar 1 |

Planning years available: 2019-06 through 2025-06 (7 years x 4 rounds = **28 query groups**).

---

## 2. The V6.1 Signal — Our Row Definer and v0 Benchmark

> **CRITICAL NOTE: Parallel with monthly pipeline.**
> In the monthly pipeline, V6.2B monthly parquets define the rows and v0 benchmark.
> In the annual pipeline, V6.1 annual parquets serve the EXACT same role.
> The formula, column schema, and evaluation approach are identical.

### 2.1 Signal Location

**Production signal:** `Signal.MISO.SPICE_ANNUAL_V6.1`
**Path:** `/opt/data/xyz-dataset/signal_data/miso/constraints/Signal.MISO.SPICE_ANNUAL_V6.1/{year}/{aq}/onpeak/`
**Partitions:** year: 2019-06..2025-06, aq: aq1..aq4

> **CRITICAL NOTE: No generation code exists.**
> Unlike V4.5 (which has a generation notebook at `psignal/notebook/hz/.../spice5/4.get_signal.ipynb`),
> V6.1 has NO committed generation code in any repo (psignal, pmodel, pbase).
> It was likely generated ad-hoc by a teammate via an uncommitted notebook.
> However, the output parquets exist and the formula is fully verified.
> This is the SAME situation as V6.2B monthly — no generation code found there either.

### 2.2 V6.1 Schema (21 columns)

| Column | Type | Role | Notes |
|--------|------|------|-------|
| constraint_id | str | key | Unique identifier |
| flow_direction | i64 | key | 1 or -1 |
| branch_name | str | metadata/join key | Used to join with realized DA |
| equipment | str | metadata | = branch_name |
| bus_key, bus_key_group | str | metadata | |
| mean_branch_max | f64 | **feature** | Max branch loading forecast |
| ori_mean | f64 | **feature** | Mean flow, baseline scenario |
| mix_mean | f64 | **feature** | Mean flow, mixed scenario |
| density_mix_rank_value | f64 | **feature** | Percentile rank of mix flow (lower = more binding) |
| density_ori_rank_value | f64 | **feature** | Percentile rank of ori flow (lower = more binding) |
| shadow_price_da | f64 | **feature** | Historical DA shadow price (NOT realized — see below) |
| da_rank_value | f64 | redundant | = rank(shadow_price_da), Spearman = -1.0 with shadow_price_da |
| density_mix_rank | f64 | redundant | Integer version of density_mix_rank_value |
| mean_branch_max_fillna | f64 | redundant | = mean_branch_max with nulls filled |
| rank_ori | f64 | **v0 output** | Formula score (see 2.3) |
| rank | f64 | **v0 output** | = percentile_rank(rank_ori) |
| tier | i64 | **v0 output** | = quintile_bin(rank), 0-4 |
| shadow_sign | i64 | derived | = -flow_direction |
| shadow_price | f64 | derived | = shadow_price_da * shadow_sign |

### 2.3 V6.1 Formula (Verified — Exact Reproduction)

```
rank_ori = 0.60 * da_rank_value + 0.30 * density_mix_rank_value + 0.10 * density_ori_rank_value
rank = percentile_rank(rank_ori, method='dense')
tier = quintile_bin(rank)   # 5 equal groups (~85 each for 425 constraints)
```

**Reproduction verification (2024-06/aq1, 425 rows):**
- `rank_ori` reproduction: `max_abs_diff = 0.0` (exact match)
- Tier distribution: [84, 85, 85, 85, 86] (near-perfect quintiles)

> **CRITICAL NOTE: shadow_price_da is HISTORICAL, not realized.**
> Verified across multiple adjacent quarters:
> - Spearman ~0.72 between adjacent quarters (not 1.0 — proves it changes slowly)
> - Only 21-33% of values are identical across quarters
> - In monthly pipeline: Spearman ~0.81 with V6.4B hist_shadow, only ~0.36 with actual_shadow
> - This is a legitimate feature (backward-looking), NOT leakage
> - da_rank_value = rank(shadow_price_da), so using shadow_price_da directly is preferred (preserves magnitude)

### 2.4 V6.1 vs V4.5 (Production Signal) — Important Distinction

V6.1 and V4.5 are DIFFERENT signals with different formulas and constraint universes:

| | V6.1 (our benchmark) | V4.5 (pmodel production) |
|--|---|---|
| Path prefix | `Signal.MISO.SPICE_ANNUAL_V6.1` | `TEST.Signal.MISO.SPICE_ANNUAL_V4.5.R{round}` |
| Formula | `0.60*da_rank + 0.30*dmix + 0.10*dori` | `0.3*dev_max_rank + 0.2*dev_sum_rank + 0.5*shadow_rank` |
| Features | SPICE6 density ranks | SPICE5 flow percentile deviations |
| Tiers | Equal quintiles (5 x ~85) | Unequal: 12%, 16%, 20%, 24%, 28% |
| Rows (2024/aq1) | 425 | 678 |
| Equipment overlap | 243 in common | |
| Generation code | **NOT found** | Notebook in psignal |
| Used in pmodel? | NO | YES (ftr23/v3 production) |

> **CRITICAL NOTE: We use V6.1, not V4.5.**
> V6.1 has the same schema as V6.2B monthly (our proven monthly setup).
> V4.5 uses a completely different feature set and tier logic.
> If we later want to benchmark against V4.5 too, that's a separate exercise.

### 2.5 All Annual Signal Versions on Disk

```
Signal.MISO.SPICE_ANNUAL_V6.1           (production, no TEST prefix, no per-round suffix)
TEST.Signal.MISO.SPICE_ANNUAL_V3.1.R{1,2}
TEST.Signal.MISO.SPICE_ANNUAL_V3.2.R{1,2,3}
TEST.Signal.MISO.SPICE_ANNUAL_V3.3.R{1,2,3}
TEST.Signal.MISO.SPICE_ANNUAL_V4.1.R{1,2,3}
TEST.Signal.MISO.SPICE_ANNUAL_V4.2.R{1,2,3}
TEST.Signal.MISO.SPICE_ANNUAL_V4.3.R{1,2,3}
TEST.Signal.MISO.SPICE_ANNUAL_V4.4.R{1,2,3}
TEST.Signal.MISO.SPICE_ANNUAL_V4.5.R{1,2,3}
```

V6.1 is unique: no TEST prefix, no per-round suffix, uses SPICE6 density features.

---

## 3. Ground Truth (Target Label)

**Realized DA constraint shadow prices** for the target quarter, fetched independently from features.

> **CRITICAL NOTE: This is future information at backtest time.**
> For planning year 2022, aq1 (Jun-Aug), the features are computed BEFORE June 2022.
> The ground truth is the actual shadow prices realized DURING Jun-Aug 2022.
> This separation is what makes this a genuine predictive task.

### 3.1 Source API

```python
from pbase.analysis.tools.all_positions import MisoApTools
aptools = MisoApTools()

# For aq1 2024-06 (Jun-Aug 2024):
da_shadow = aptools.tools.get_da_shadow_by_peaktype(
    st=pd.Timestamp("2024-06-01", tz="US/Central"),
    et_ex=pd.Timestamp("2024-09-01", tz="US/Central"),  # exclusive end
    peak_type="onpeak",
)
# Returns: monitored_facility (constraint name), shadow_price, branch_name
```

**Requires Ray initialization.** Date range: 2017-06 to 2026-03.

### 3.2 Mapping Realized DA to V6.1 Constraints

1. `get_da_shadow_by_peaktype()` returns `monitored_facility` (MISO constraint name)
2. Map to `branch_name` via `MISO_SPICE_CONSTRAINT_INFO` parquet: `constraint_id -> branch_name`
3. V6.1 has `branch_name` / `equipment` column — join on this
4. Multiple `monitored_facility` may map to same `branch_name` — aggregate by sum

Reference implementation: `research-spice-shadow-price-pred-qianli/src/shadow_price_prediction/data_loader.py` lines 875-899.

### 3.3 Label Construction

For each (planning_year, aq_round):
1. Determine the 3 market months in the quarter (e.g., aq1 = Jun, Jul, Aug)
2. Fetch realized DA shadow prices for all 3 months via `get_da_shadow_by_peaktype()`
3. Map `monitored_facility` -> `branch_name` via spice constraint_info
4. Aggregate: `sum(abs(shadow_price))` per `branch_name` across ALL hours in the quarter
5. Left-join to V6.1 constraint universe on `branch_name`/`equipment`
6. Constraints with no realized shadow price -> label = 0.0 (did not bind)

This gives a continuous relevance label per constraint. Higher = more binding = should be ranked higher.

### 3.4 Quarter-to-Month Mapping

| Planning Year | Round | Market Months | Ground Truth Period |
|--------------|-------|---------------|-------------------|
| YYYY-06 | aq1 | Jun, Jul, Aug | YYYY-06-01 to YYYY-09-01 |
| YYYY-06 | aq2 | Sep, Oct, Nov | YYYY-09-01 to YYYY-12-01 |
| YYYY-06 | aq3 | Dec, Jan, Feb | YYYY-12-01 to (YYYY+1)-03-01 |
| YYYY-06 | aq4 | Mar, Apr, May | (YYYY+1)-03-01 to (YYYY+1)-06-01 |

---

## 4. Features

### 4.1 Leakage Rules

**STRICT TEMPORAL CUTOFF:** For a target quarter starting in month M, all features must use information available BEFORE month M.

| Feature source | Temporal relationship | Leakage? |
|---------------|----------------------|----------|
| `shadow_price_da` in V6.1 | Historical DA shadow prices (rolling lookback ending before auction) | **NO** — verified historical |
| `mean_branch_max`, `ori_mean`, `mix_mean` | SPICE6 forward simulation run before auction | **NO** — forecasts |
| `density_*_rank_value` | Within-signal percentile ranks of flow forecasts | **NO** — derived from forecasts |
| `prob_exceed_*` from density distribution | SPICE6 simulation output | **NO** — forecasts |
| `constraint_limit`, `rate_a` | Static network thermal ratings | **NO** |
| `rank_ori`, `rank`, `tier` | Derived from formula (composite of above) | **DO NOT USE** — opaque derived columns |
| `da_rank_value` | = rank(shadow_price_da) | **OK but redundant** — use shadow_price_da directly |
| `shadow_sign`, `shadow_price` | Signed transformation of shadow_price_da | **DO NOT USE** — redundant/derived |
| Realized DA shadow price for target quarter | Future information | **THIS IS THE LABEL, never a feature** |

### 4.2 Feature Sets

**Set A — V6.1 base (6 features, from pre-aggregated signal):**

| Feature | Source | Monotone | Description |
|---------|--------|----------|-------------|
| `shadow_price_da` | V6.1 parquet | +1 | Historical DA shadow price (higher = more historically binding) |
| `mean_branch_max` | V6.1 parquet | +1 | Max branch loading from simulation |
| `ori_mean` | V6.1 parquet | +1 | Mean flow, baseline scenario |
| `mix_mean` | V6.1 parquet | +1 | Mean flow, mixed scenario |
| `density_mix_rank_value` | V6.1 parquet | -1 | Rank of mix flow (lower = more binding) |
| `density_ori_rank_value` | V6.1 parquet | -1 | Rank of original flow (lower = more binding) |

> Note: `da_rank_value` is a monotonic transform of `shadow_price_da` (Spearman = -1.0).
> Use `shadow_price_da` directly — it preserves magnitude information that `da_rank_value` discards.
> The V6.1 formula uses `da_rank_value` (rank), but ML can learn better from the raw value.

**Set B — V6.1 + spice6 density (11 features):**

Set A plus:

| Feature | Source | Monotone | Description |
|---------|--------|----------|-------------|
| `prob_exceed_110` | DENSITY_SIGNAL_SCORE | +1 | P(flow > 110% of limit) |
| `prob_exceed_100` | DENSITY_SIGNAL_SCORE | +1 | P(flow > 100% of limit) |
| `prob_exceed_90` | DENSITY_SIGNAL_SCORE | +1 | P(flow > 90% of limit) |
| `prob_exceed_85` | DENSITY_SIGNAL_SCORE | +1 | P(flow > 85% of limit) |
| `prob_exceed_80` | DENSITY_SIGNAL_SCORE | +1 | P(flow > 80% of limit) |

**Set C — Full (13 features):**

Set B plus:

| Feature | Source | Monotone | Description |
|---------|--------|----------|-------------|
| `constraint_limit` | CONSTRAINT_LIMIT | 0 | MW thermal limit |
| `rate_a` | CONSTRAINT_INFO | 0 | Branch rating A (MW) |

### 4.3 Feature Aggregation for spice6 Features

Raw spice_data has per-(constraint_id, market_month, outage_date) granularity. For each (planning_year, aq_round):
1. Filter to the 3 market_months in the quarter
2. Aggregate across outage_dates AND market_months: `mean` per constraint_id
3. Join to V6.1 constraint universe on `constraint_id`

Spice6 data sources:
- `MISO_SPICE_DENSITY_SIGNAL_SCORE.parquet` — raw scores, filter `auction_type=annual`, ~23M annual rows
- `MISO_SPICE_CONSTRAINT_LIMIT.parquet` — per-constraint limits
- `MISO_SPICE_CONSTRAINT_INFO.parquet` — structural info (rate_a, branch_name mapping)

**Enrichment coverage: 99.7-100% of V6.1 constraints** have matching spice6 density data (verified across all 7 years).

Reference: stage4's `ml/spice6_loader.py` (does single-month aggregation; extend to 3-month for annual).

---

## 5. Data Paths

| Data | Path | Partitions |
|------|------|-----------|
| V6.1 annual signal | `/opt/data/xyz-dataset/signal_data/miso/constraints/Signal.MISO.SPICE_ANNUAL_V6.1/{year}/{aq}/onpeak/` | year: 2019-06..2025-06, aq: aq1..aq4 |
| V4.5 annual signal | `/opt/data/xyz-dataset/signal_data/miso/constraints/TEST.Signal.MISO.SPICE_ANNUAL_V4.5.R1/{year}/{aq}/onpeak/` | same |
| Density signal score | `/opt/data/xyz-dataset/spice_data/miso/MISO_SPICE_DENSITY_SIGNAL_SCORE.parquet` | auction_type=annual |
| Constraint limit | `/opt/data/xyz-dataset/spice_data/miso/MISO_SPICE_CONSTRAINT_LIMIT.parquet` | auction_type=annual |
| Constraint info | `/opt/data/xyz-dataset/spice_data/miso/MISO_SPICE_CONSTRAINT_INFO.parquet` | auction_type=annual |
| Density distribution | `/opt/data/xyz-dataset/spice_data/miso/MISO_SPICE_DENSITY_DISTRIBUTION.parquet` | auction_type=annual |
| Realized DA shadow | via `MisoApTools.tools.get_da_shadow_by_peaktype()` | requires Ray |

---

## 6. Data Volume

**V6.1 constraint counts per (planning_year, aq_round):**

| Planning year | aq1 | aq2 | aq3 | aq4 | Total |
|--------------|-----|-----|-----|-----|-------|
| 2019-06 | ~500 | ~500 | ~400 | ~500 | ~1,900 |
| 2020-06 | ~500 | ~500 | ~400 | ~500 | ~1,900 |
| 2021-06 | ~500 | ~500 | ~400 | ~500 | ~1,900 |
| 2022-06 | ~500 | ~500 | ~400 | ~500 | ~1,900 |
| 2023-06 | ~600 | ~700 | ~600 | ~600 | ~2,500 |
| 2024-06 | 425 | 483 | 315 | 476 | 1,699 |
| 2025-06 | ~500 | ~600 | ~400 | ~500 | ~2,000 |

**Total: 28 query groups, ~14,000 constraint-rows.**
**3,703 unique constraints** across all groups.
Cross-quarter constraint overlap: ~47-56% (universes differ significantly between quarters).
Year-to-year churn: ~65-70% per year (only 23% of 2019 constraints still present in 2025).

---

## 7. Train / Eval Split

### Rolling-year cross-validation

With only 7 planning years, use expanding-window forward validation:

| Split | Train years | Eval year | Train groups | Eval groups |
|-------|------------|-----------|-------------|-------------|
| 1 | 2019-2021 | 2022 | 12 | 4 |
| 2 | 2019-2022 | 2023 | 16 | 4 |
| 3 | 2019-2023 | 2024 | 20 | 4 |
| 4 | 2019-2024 | 2025 | 24 | 4 |

**Primary eval:** Splits 1-3 (2022-2024, **12 eval groups** total). 2025 held out for final validation.

**No validation split.** Same finding as stage4 monthly: folding val into train was the single biggest improvement. Use all train years for training.

### Within each query group

Each (planning_year, aq_round) = one LTR query group. The ~400-900 constraints provide ample pairwise comparisons for LambdaRank.

### All 4 rounds pooled

Train on all aq1-aq4 together (not separate models per round). The density features already capture seasonal variation. Add `aq_round` as a categorical feature only if needed.

---

## 8. Modeling

### v0: V6.1 Formula Baseline

Evaluate the existing V6.1 ranking (`rank` column) against realized DA shadow prices. **No training needed.** Just load V6.1 rank and evaluate against ground truth.

```
rank_ori = 0.60 * da_rank_value + 0.30 * density_mix_rank_value + 0.10 * density_ori_rank_value
```

This tells us: how well does the current signal predict actual binding?

### v1: LightGBM LambdaRank (Set A — 6 features)

Same 6 features available to V6.1, but learned weights via ML.
- Backend: LightGBM lambdarank
- Hyperparams: lr=0.05, 100 trees, 31 leaves, subsample=0.8, colsample=0.8
- Monotone constraints: as specified in feature table
- No early stopping (100 trees is small enough)

**Purpose:** Can ML beat the hand-tuned formula using the same inputs?

### v2: LightGBM LambdaRank (Set B — 11 features)

Add spice6 density exceedance probabilities.

**Purpose:** Do additional density features help beyond what V6.1 already captures?

### v3: LightGBM LambdaRank (Set A — 6 features, tiered labels)

Same as v1 but with tiered labels: 0=non-binding (58% of constraints), 1-4=quantile buckets of positive shadow prices. Avoids rank-transform noise where ~58% of constraints get arbitrary distinct ranks despite all having sp=0.

**Purpose:** Do tiered labels improve ranking quality?

### v4: LightGBM LambdaRank (Set AF — 7 features, rank labels)

Set A plus `rank_ori` (V6.1 formula output) as 7th feature, with raw rank labels. This lets the model see the formula's own ranking while learning to improve on it.

**Purpose:** Does providing the formula score as a feature help ML improve beyond it?

### v5: LightGBM LambdaRank (Set AF — 7 features, tiered labels)

Combines both improvements: formula-as-feature + tiered labels. Best of v3 and v4.

**Purpose:** Are the two improvements additive?

### v6+: Iterations (Planned)

Better baselines (alternative formulas, raw shadow_price_da), blending ML + formula (score blend, rank blend, RRF).

---

## 9. Evaluation Metrics

Same framework as monthly stage5 (reuse `ml/evaluate.py`):

**Group A — Blocking (must improve over v0):**

| Metric | Description |
|--------|-------------|
| VC@20 | Value captured by model's top-20 constraints / total value |
| VC@100 | Value captured by top-100 |
| Recall@20 | Overlap of model's top-20 with true top-20 |
| Recall@50 | Same for top-50 |
| Recall@100 | Same for top-100 |
| NDCG | Normalized discounted cumulative gain |

**Group B — Monitor:**

| Metric | Description |
|--------|-------------|
| VC@10, VC@25, VC@50, VC@200 | Value capture at other k |
| Recall@10 | Hit rate at top-10 |
| Spearman | Rank correlation |
| Tier0-AP | Average precision for top-20% |
| Tier01-AP | Average precision for top-40% |

### Gate System (3-layer, calibrated from v0)

| Layer | Formula | Purpose |
|-------|---------|---------|
| L1 Mean | `mean(metric) >= 0.9 * v0_mean` | Basic quality floor |
| L2 Tail | `count(groups < v0_min) <= 1` | No catastrophic groups |
| L3 Bot2 | `bot2_mean(new) >= bot2_mean(v0) - 0.02` | Worst groups don't regress |

---

## 10. Repo Structure

```
research-annual-signal/
    experiment-setup.md          # this file — problem spec and design
    mem.md                       # working memory — results, findings, next steps
    audit.md                     # 20-point integrity audit
    runbook.md                   # how to run everything
    ml/
        __init__.py
        config.py                # feature sets, data paths, eval groups, leaky feature guard
        data_loader.py           # load V6.1 + spice6 features (cached)
        ground_truth.py          # fetch realized DA shadow prices via MisoApTools (cached)
        evaluate.py              # all 13 metrics + aggregation
        compare.py               # 3-layer gate system + comparison tables
        benchmark.py             # multi-group runner (train per year, eval per quarter)
        train.py                 # LightGBM lambdarank + XGBoost (fallback)
        pipeline.py              # train-predict-evaluate workflow
        features.py              # prepare feature matrix, query groups
    scripts/
        run_v0_baseline.py       # evaluate V6.1 formula vs realized GT + calibrate gates
        run_v1_experiment.py     # ML with Set A (6 features)
        run_v2_experiment.py     # ML with Set B (11 features)
        run_v3_experiment.py     # ML with Set A + tiered labels
        run_v4_experiment.py     # ML with Set AF (7 features, formula-as-feature)
        run_v5_experiment.py     # ML with Set AF + tiered labels
        cache_all_ground_truth.py  # pre-cache all 28 ground truth files (requires Ray)
    cache/
        enriched/                # V6.1 + spice6 per (year, aq) — auto-created on first load
        ground_truth/            # realized DA shadow prices per (year, aq) — requires Ray
    registry/
        gates.json               # quality gates (calibrated from v0)
        champion.json            # current champion version
        comparisons/             # auto-generated comparison JSONs
        v0/ .. v5/               # version metrics, config, metadata
        v1_holdout/              # v1 holdout (2025) results
    reports/
        v1_comparison.md .. v5_comparison.md  # comparison reports
    docs/plans/
        2026-03-08-annual-tier-prediction-plan.md  # implementation plan
```

---

## 11. Execution History

| Step | Version | Status | Key Result |
|------|---------|--------|------------|
| 1 | Ground truth pipeline | DONE | 28 parquet files cached, ~31-42% binding rate |
| 2 | v0 baseline | DONE | VC@20=0.2323, gates calibrated |
| 3 | v1 ML (6 feat) | DONE | VC@20=0.2934 (+26%), ALL gates PASS |
| 4 | v2 ML (11 feat) | DONE | VC@20=0.2904, spice6 doesn't help annual |
| 5 | v1 holdout (2025) | DONE | VC@20=0.2152 (+38% vs v0 holdout) |
| 6 | v3 tiered labels | DONE | VC@20=0.2871, tiered hurts VC@20 alone |
| 7 | v4 formula-as-feature | DONE | VC@20=0.3030, +3.3% vs v1 |
| 8 | v5 both improvements | DONE | VC@20=0.3075, best on 5/9 metrics |
| 9 | Full audit | DONE | 20/20 checks PASS |
| 10 | Better baselines + blending | PLANNED | Explore alternative formulas and ML+formula blending |

---

## 12. Key Risks

1. **Small eval set.** Only 12 eval groups (3 years x 4 rounds) for primary evaluation. Statistical power is limited. Focus on consistent directional improvements, not small margins.

2. **Constraint mapping coverage.** Not all V6.1 constraints may have realized shadow prices (some may never bind). This is expected — constraints with no realized shadow price get label=0.

3. **V6.1 formula advantage.** 60% of V6.1's ranking weight is on `da_rank_value = rank(shadow_price_da)`. If historical DA correlates with realized DA (even partially), the formula has a natural baseline. ML must learn to weight features better, not just exploit correlation structure.

4. **Year-to-year regime changes.** Grid topology, generation mix, and load patterns change across years. A model trained on 2019-2022 may not generalize to 2024. The 65-70% year-over-year constraint churn amplifies this.

5. **No generation code for V6.1.** If we ever need to generate V6.1 for new planning years, we'd need to reconstruct the full pipeline. For now this is not blocking since data exists through 2025-06.

---

## 13. Reusable Code from Stage 4 Monthly

| Module | Action | Notes |
|--------|--------|-------|
| `ml/evaluate.py` | Copy | All metrics correct, no changes needed |
| `ml/compare.py` | Copy | 3-layer gate system |
| `ml/train.py` | Copy | LightGBM lambdarank training |
| `ml/features.py` | Copy+Modify | `prepare_features()` — adapt for annual query groups |
| `ml/spice6_loader.py` | Adapt | Single-month -> 3-month aggregation |
| `ml/v62b_formula.py` | Rename | Same formula, rename to `v61_formula.py` |
| `ml/config.py` | Rewrite | Annual-specific paths, feature sets, eval groups |
| `ml/data_loader.py` | Rewrite | Annual structure (year/aq partitions), 3-month spice6 aggregation |
| `ml/pipeline.py` | Rewrite | Annual train/eval splits, realized DA ground truth |
| `ml/benchmark.py` | Adapt | Multi-group runner for annual groups |

---

## 14. Monthly vs Annual — Side-by-Side Comparison

| | Monthly (stage5) | Annual (this project) |
|--|---|---|
| **Row definer** | V6.2B `SPICE_F0P_V6.2B.R1/{month}/f0/onpeak` | V6.1 `SPICE_ANNUAL_V6.1/{year}/{aq}/onpeak` |
| **v0 formula** | `0.60*da_rank + 0.30*dmix + 0.10*dori` | **Same formula** (verified diff=0.0) |
| **Features** | V6.2B + spice6 density + ml_pred | V6.1 + spice6 density (annual partition) |
| **Target** | Realized DA for that **month** | Realized DA for that **quarter** (3 months) |
| **Unit of eval** | 1 month = 1 query group | 1 (year, aq_round) = 1 query group |
| **Rows per group** | ~500-800 | ~315-767 |
| **Total eval groups** | 12 months | 12 groups (3 years x 4 rounds) |
| **Training** | Rolling 8-month window | Expanding-window (all prior years) |
| **v0 results** | VC@20=0.3336, Spearman=0.1964 | VC@20=0.2323, Spearman=0.3425 |
| **Generation code** | Not found | Not found |
| **Formula verified** | Yes (max_abs_diff=0) | Yes (max_abs_diff=0) |

---

## 15. References

- Monthly stage5 experiment: `/home/xyz/workspace/research-qianli-v2/research-stage5-tier/experiment-setup.md`
- Stage5 handoff: `/home/xyz/workspace/research-qianli-v2/research-stage5-tier/stage5-handoff.md`
- Shadow price prediction codebase: `/home/xyz/workspace/research-qianli-v2/research-spice-shadow-price-pred-qianli/src/shadow_price_prediction/data_loader.py`
- Stage4 ML modules: `/home/xyz/workspace/research-qianli-v2/research-stage4-tier/ml/`
- V6.1 signal generation notebook (V4.x/V5.x, NOT V6.1): `psignal/notebook/hz/2025-planning-year/nov/miso/spice5/4.get_signal.ipynb`
- pmodel annual signal usage: `pmodel/src/pmodel/base/ftr23/v3/params/prod/auc25annual/miso_models_a_prod_r2.py` (references V4.4/V4.5)
