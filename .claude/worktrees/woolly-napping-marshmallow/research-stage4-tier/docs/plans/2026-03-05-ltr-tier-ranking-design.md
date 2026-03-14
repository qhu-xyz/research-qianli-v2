# Stage 4: Learning-to-Rank Tier Prediction — Design Document

**Date:** 2026-03-05
**Status:** Approved

## 1. Objective

Replace V6.2B's hand-tuned ranking formula with an ML-based learning-to-rank model that better identifies high-value MISO transmission constraints. Beat V6.2B on VC@20, VC@100, Recall@20, Recall@100, and NDCG across 12 evaluation months spanning 2020-2023.

## 2. Why Stage 4 (from Stage 3)

Stage 3 used a 5-class XGBoost classifier on ~130k constraint samples per month. Three fundamental problems make it unrecoverable:

1. **Binary collapse** — The model degenerates to a binary classifier. Tier-Recall@1 ~ 0.05 means it essentially never predicts tier 1. With 93-95% of samples in tier 3, the class imbalance is too extreme for multi-class classification.
2. **Wrong constraint universe** — Stage 3 treats each (constraint_id x outage_date x direction) as an independent sample (~130k/month). V6.2B aggregates across outage dates and produces ~400-740 constraints/month.
3. **Classification vs ranking** — The downstream objective is capital allocation: rank constraints by expected shadow price so the top-k capture maximum value. This is a ranking problem, not a classification problem.

## 3. Constraint Universe

Use V6.2B's pre-built universe directly. No need to re-derive aggregation/dedup logic.

- **Source:** `/opt/data/xyz-dataset/signal_data/miso/constraints/TEST.TEST.Signal.MISO.SPICE_F0P_V6.2B.R1/{month}/f0/onpeak/`
- **Size:** ~400-740 constraints per month (varies by month)
- **Granularity:** One row per (constraint_id, flow_direction), already aggregated across 11 outage dates
- **Flow direction:** Preserved (both positive and negative, ~90%/10% split)
- **Ground truth:** `shadow_price_da` column (realized DA shadow price for the target month)
- **Coverage:** 106 months (2017-06 to 2026-03), zero nulls in shadow_price_da

### V6.2B Internals (Reverse-Engineered)

V6.2B does **not** use ML. Its `rank` column is a weighted formula (R²=0.96 linear fit):

```
rank ≈ 0.72 * da_rank_value + 0.51 * rank_ori + 0.23 * density_mix_rank_value
      + 0.09 * density_mix_rank + 0.07 * density_ori_rank_value - 0.67
```

Where:
- `da_rank_value` = historical DA shadow price rank (NOT look-ahead; correlation with realized shadow_price_da is only -0.28)
- `rank_ori` = original density rank
- `density_mix_rank_value` = mixed density rank value
- Tier assignment: equal-quantile of rank (20% each = 5 tiers)
- `rank` is the percentile position (i/n, 0=best, 1=worst)

## 4. V6.2B Baseline Performance

The floor to beat, computed on 12 eval months:

| Metric | Mean | Min | Max |
|--------|------|-----|-----|
| **VC@10** | 0.439 | 0.361 | 0.536 |
| **VC@20** | 0.519 | 0.432 | 0.625 |
| **VC@25** | 0.556 | 0.475 | 0.695 |
| **VC@50** | 0.682 | 0.567 | 0.825 |
| **VC@100** | 0.840 | 0.756 | 0.929 |
| **VC@200** | 0.963 | 0.936 | 0.994 |
| **Recall@10** | 0.467 | 0.300 | 0.700 |
| **Recall@20** | 0.488 | 0.300 | 0.650 |
| **Recall@50** | 0.552 | 0.420 | 0.660 |
| **Recall@100** | 0.706 | 0.550 | 0.780 |
| **NDCG** | 0.940 | 0.895 | 0.969 |

Key insight: VC@100 is 0.84 but Recall@100 is only 0.71. V6.2B misses ~30% of the true top-100 constraints, but the ones it finds are high-value enough to capture 84% of the value. This gap is where ML can win.

## 5. Model

XGBoost with `rank:pairwise` objective (LambdaMART-style pairwise ranking).

| Parameter | Value |
|-----------|-------|
| objective | `rank:pairwise` |
| n_estimators | 400 |
| max_depth | 5 |
| learning_rate | 0.05 |
| subsample | 0.8 |
| colsample_bytree | 0.8 |
| min_child_weight | 25 |
| early_stopping_rounds | 50 |
| monotone_constraints | per-feature (from stage 3) |

- **Query groups:** Each (auction_month, period_type) — rank within a month
- **Relevance labels:** `shadow_price_da` (continuous)
- **Output:** Continuous ranking score (higher = more likely to bind)
- **No tier labels needed** — tiers are derived post-hoc from score quantiles if needed

## 6. Features

### Source Data

Features computed directly from raw spice6 parquet, keyed to V6.2B's constraint set via `constraint_id` join (512/513 = 99.8% overlap; only `SO_MW_Transfer` interface missing).

| Data | Path | Format |
|------|------|--------|
| Density | `/opt/temp/tmp/pw_data/spice6/prod_f0p_model_miso/density/` | 77 flow percentile bins (-300% to +300%), ~14k constraints x 11 outage dates per month |
| Shift Factors | `/opt/temp/tmp/pw_data/spice6/prod_f0p_model_miso/sf/` | Per outage date, per constraint |
| Constraint Info | `/opt/temp/tmp/pw_data/spice6/prod_f0p_model_miso/constraint_info/` | branch_name, limit, type, class_type |

### Feature Computation

1. Read density across all 11 outage dates for the target month
2. Aggregate across outage dates (mean/max) per constraint_id
3. Compute features from the aggregated density CDF
4. Filter to V6.2B's constraint set via inner join on constraint_id

### Starting Feature Set (34 from Stage 3 + 6 from V6.2B)

**Flow probability (13):** prob_exceed_{80,85,90,95,100,105,110}, prob_below_{85,90,95,100}, prob_band_{95_100, 100_105}

**Historical price (7):** hist_da, hist_da_trend, recent_hist_da, hist_da_max_season, season_hist_da_{1,2,3}

**Shift factor (4):** sf_max_abs, sf_mean_abs, sf_std, sf_nonzero_frac

**Distribution shape (7):** density_mean, density_variance, density_entropy, density_skewness, density_kurtosis, density_cv, tail_concentration

**Constraint type (2):** is_interface, constraint_limit

**Overload (1):** expected_overload

**V6.2B columns (6):** mean_branch_max, ori_mean, mix_mean, density_mix_rank_value, density_ori_rank_value, da_rank_value

Total: 40 features.

## 7. Training Setup

| Parameter | Value |
|-----------|-------|
| Train window | 6 months |
| Val window | 2 months (early stopping) |
| Period type | f0 (forward month) |
| Class type | onpeak |

For each eval month: train on 6 prior months of V6.2B data, validate on next 2, predict on target month.

## 8. Data Split

| Set | Months | Purpose |
|-----|--------|---------|
| **Dev eval** | 12 months spread across 2020-06 to 2023-05 | Development iterations, gate calibration |
| **Holdout** | 2024-2025 | One-time final validation, never touched during development |

Specific eval months TBD — will sample 12 months spread across the 2020-06 to 2023-05 range, similar spacing to stage 3.

## 9. Metrics

### Group A — Blocking Gates

| Metric | Definition |
|--------|-----------|
| **VC@20** | Fraction of total actual shadow price value captured by model's top 20 constraints |
| **VC@100** | Same, top 100 |
| **Recall@20** | Of the true top-20 by shadow_price_da, how many appear in model's top 20 |
| **Recall@100** | Of the true top-100, how many in model's top 100 |
| **NDCG** | Normalized Discounted Cumulative Gain — full ranking quality, position-discounted |

### Group B — Monitor (tracked, don't block promotion)

| Metric | Definition |
|--------|-----------|
| VC@10, VC@25, VC@50, VC@200 | Value capture at other k values |
| Recall@10, Recall@50 | Hit rate at other k values |
| Spearman | Rank correlation between model scores and actual shadow prices |
| Tier0-AP | Average Precision for top-20% by actual shadow_price_da, using model score |
| Tier01-AP | Average Precision for top-40% by actual shadow_price_da, using model score |

All metrics are higher-is-better.

### Gate Calibration

Same 3-layer system from stage 3, calibrated from V6.2B baseline:

| Layer | Formula | Purpose |
|-------|---------|---------|
| Mean Quality | `mean(metric) >= floor` | Basic quality bar |
| Tail Safety | `count(months < tail_floor) <= 1` | No catastrophic months |
| Tail Non-Regression | `bottom_2_mean(new) >= bottom_2_mean(champ) - 0.02` | Worst months don't regress |

Calibration formula:
- `floor = 0.9 * V6.2B baseline mean`
- `tail_floor = V6.2B baseline min`
- `noise_tolerance = 0.02`
- `tail_max_failures = 1`

## 10. Registry and Promotion

Full infrastructure ported from stage 3:

- `registry/gates.json` — gate definitions calibrated from V6.2B baseline
- `registry/v0/` — V6.2B baseline metrics (no model, just their ranking evaluated)
- `registry/vN/` — each experiment version with config.json + metrics.json
- `ml/compare.py` — 3-layer gate check + promotion logic
- `registry/champion.json` — current best version

## 11. Workflow

Semi-autonomous with human approval:
1. I propose hypotheses (feature changes, hyperparameter adjustments)
2. Human approves
3. I implement, run benchmark, report results
4. Human decides next steps

No 3-iter autonomous loop. No agent team.

## 12. Directory Structure

```
research-stage4-tier/
├── ml/
│   ├── config.py          # LTR config (features, hyperparams)
│   ├── data_loader.py     # Load V6.2B universe + raw spice6 features
│   ├── features.py        # Feature computation from raw spice6 density/SF
│   ├── train.py           # XGBoost rank:pairwise training
│   ├── evaluate.py        # VC@k, Recall@k, NDCG, Spearman, AP
│   ├── pipeline.py        # Load -> features -> train -> predict -> evaluate
│   ├── benchmark.py       # N-month rolling benchmark
│   ├── compare.py         # Gate comparison + promotion
│   └── tests/
├── registry/
│   ├── gates.json
│   ├── champion.json
│   └── v0/               # V6.2B baseline
├── docs/
│   └── plans/
├── mem.md
└── README.md
```

## 13. Risks

| Risk | Mitigation |
|------|-----------|
| Feature computation from raw spice6 is complex | Join on constraint_id is verified (99.8% overlap). Start with V6.2B's own columns as features before computing from raw density. |
| V6.2B baseline is very strong (VC@100=0.84) | ML can improve worst months and Recall@k (where V6.2B is weaker at 0.71). Even marginal gains on top-20 are valuable for trading. |
| Small dataset (~500 constraints/month x 6 months training) | LTR handles small groups well. XGBoost pairwise works with hundreds of items per query group. |
| Monotone constraints may not all apply to ranking objective | Start with stage 3's monotone constraints, validate via ablation. |

## 14. Success Criteria

Beat V6.2B baseline on **at least 3 of 5 Group A metrics** (mean across eval months), with no regression on remaining metrics beyond noise tolerance (0.02).
