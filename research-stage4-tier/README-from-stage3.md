# Stage 4: Learning-to-Rank Tier Prediction

## Why We're Rebuilding (from Stage 3)

Stage 3 used a 5-class XGBoost classifier on ~130k constraint samples per month.
Three fundamental problems make it unrecoverable:

1. **Binary collapse** — The model degenerates to a binary classifier. Tier-Recall@1 ≈ 0.05
   means it essentially never predicts tier 1. With 70%+ of samples in tier 3/4, the class
   imbalance is too extreme for multi-class classification to work.

2. **Wrong constraint universe** — Stage 3 treats each (constraint_id × outage_date × direction)
   as an independent sample (~130k/month). The production signal system (psignal) aggregates
   across outage dates, preserves flow direction, and deduplicates by equipment to produce
   ~800-1,700 constraints/month. We were solving the wrong problem.

3. **Classification vs ranking** — The downstream objective is capital allocation: rank
   constraints by expected shadow price so the top-100 capture maximum value. This is a
   ranking problem, not a classification problem. Learning-to-rank (LambdaMART, pairwise
   XGBoost) directly optimizes NDCG/VC@k.

## Architecture: Learning-to-Rank

```
Raw spice6 data (density, SF, constraint_info)
    │
    ├── Preprocessing (ported from psignal)
    │     ├── Aggregate across outage dates (mean/max over simulations)
    │     ├── Preserve flow direction (positive/negative separately)
    │     ├── Equipment dedup (Chebyshev distance ≥ 0.15, correlation ≥ 0.05 in SF space)
    │     └── Produce ~800-1,700 constraints per (auction_month, period_type)
    │
    ├── Feature engineering
    │     ├── Density-derived: prob_exceed_*, prob_below_*, expected_overload, etc.
    │     ├── SF-derived: sf_max_abs, sf_mean_abs, sf_std, sf_nonzero_frac
    │     ├── Historical: hist_da, hist_da_trend, recent_hist_da, seasonal
    │     ├── Distribution: density_mean, variance, entropy, skewness, kurtosis, cv
    │     └── Interactions: overload_x_hist, prob110_x_recent_hist, tail_x_hist
    │
    ├── Learning-to-rank model
    │     ├── XGBoost rank:pairwise or rank:ndcg objective
    │     ├── Query groups = (auction_month, period_type) — rank within each month
    │     ├── Relevance labels = actual DA shadow prices (continuous)
    │     └── Monotone constraints preserved from stage 3
    │
    └── Evaluation
          ├── Primary: Tier-VC@100, Tier-VC@500 (value capture at top-k)
          ├── Ranking: NDCG, MAP
          ├── Tier-derived: Tier0-AP, Tier01-AP (using score quantiles for tier assignment)
          └── Baseline comparison against psignal V6.2B signal
```

## Ground Truth

Actual DA shadow prices for the realized market month. Available for ~98% of constraints
in the psignal universe via `MisoApTools.get_da_shadow_by_peaktype()`, mapped to branch_name
through spice constraint_info.

## Baseline: psignal Signal Versions

### V5.8 (spice5-based)
- Path: `/opt/data/xyz-dataset/signal_data/miso/constraints/TEST.Signal.MISO.SPICE_F0P_V5.8.R1/`
- ~1,732 constraints/month, 5 tiers with percentile-based cutoffs (12/28/48/72/100%)
- Ranking formula: `deviation_max_rank * 0.3 + deviation_sum_rank * 0.5 + shadow_rank * 0.2`
- Code: `/home/xyz/workspace/psignal/notebook/hz/2025-planning-year/nov/miso/spice5/4.get_signal.ipynb`
- 102 months available

### V6.2B (spice6-based) — Latest 
- Path: `/opt/data/xyz-dataset/signal_data/miso/constraints/TEST.TEST.Signal.MISO.SPICE_F0P_V6.2B.R1/`
- ~789 constraints/month, 5 equal-sized tiers (~158 each)
- Ranking: `density_mix_rank`, `density_ori_rank`, `da_rank` (different weights)
- Columns: constraint_id, flow_direction, mean_branch_max, ori_mean, branch_name, bus_key,
  mix_mean, shadow_price_da, density_mix_rank_value, density_ori_rank_value, da_rank_value,
  rank_ori, density_mix_rank, rank, tier, shadow_sign, shadow_price, equipment
- 106 months (2017-06 to 2026-03)
- Code: NOT found in psignal notebooks (only V5.8 notebook located)

### Which baseline?
V6.2B uses spice6 data matching our iso_configs, but has only ~789 constraints with equal
tier sizes — a very different setup from V5.8's ~1,732 with percentile cutoffs. The V6.2B
generation code was not found in psignal notebooks.

**Decision needed**: Use V6.2B as primary baseline (matches our data source) or V5.8
(well-understood code, larger constraint set, more representative of production).

## Relevant Repositories and Data Paths

### Repositories
| Repo | Path | Purpose |
|------|------|---------|
| research-qianli-v2 (this) | `/home/xyz/workspace/research-qianli-v2/` | ML research workspace |
| Stage 3 tier pipeline | `research-stage3-tier/ml/` | Previous classifier (reference only) |
| Source data loader | `research-spice-shadow-price-pred-qianli/src/shadow_price_prediction/` | MisoDataLoader, feature computation |
| psignal | `/home/xyz/workspace/psignal/` | Signal generation notebooks, dedup logic |
| pbase | (installed via venv) | Ray utilities, data access, config |
| pmodel | `/home/xyz/workspace/pmodel/` | Virtual environment, model infrastructure |

### Raw Data (spice6)
| Data | Path |
|------|------|
| Density (flow simulations) | `/opt/temp/tmp/pw_data/spice6/prod_f0p_model_miso/density/` |
| Shift factors | `/opt/temp/tmp/pw_data/spice6/prod_f0p_model_miso/sf/` |
| Constraint info | `/opt/temp/tmp/pw_data/spice6/prod_f0p_model_miso/constraint_info/` |
| (also spice5 if needed) | `/opt/temp/tmp/pw_data/spice5/prod_f0p_model_miso/` |

### Signal Data (baselines)
| Signal | Path |
|--------|------|
| V5.8 | `/opt/data/xyz-dataset/signal_data/miso/constraints/TEST.Signal.MISO.SPICE_F0P_V5.8.R1/` |
| V6.2B | `/opt/data/xyz-dataset/signal_data/miso/constraints/TEST.TEST.Signal.MISO.SPICE_F0P_V6.2B.R1/` |

## Key Differences from Stage 3

| Aspect | Stage 3 | Stage 4 |
|--------|---------|---------|
| Objective | Multi-class classification | Learning-to-rank |
| Loss | Softmax cross-entropy | Pairwise / LambdaMART |
| Samples/month | ~130,000 | ~800-1,700 |
| Aggregation | None (raw outage × constraint) | Outage-date aggregation |
| Flow direction | Collapsed (one per constraint) | Preserved (positive/negative) |
| Equipment dedup | None | Chebyshev + correlation dedup |
| Primary metric | Tier-Recall, QWK | VC@100, NDCG |
| Tier assignment | Fixed SP bins | Post-hoc from ranking scores |

## Development Plan

No 3-iteration-per-report loop needed. Phases:

1. **Port preprocessing** — Replicate psignal's constraint filtering, outage-date aggregation,
   flow direction preservation, and equipment dedup. Verify we reproduce their ~800-1,700
   constraint universe.

2. **Establish baseline** — Compute VC@100, VC@500, NDCG for psignal's own signal (V6.2B
   and/or V5.8) using actual DA shadow prices as ground truth.

3. **Build LTR pipeline** — XGBoost with `rank:pairwise` or `rank:ndcg` objective, query
   groups by month, relevance = actual shadow price.

4. **Iterate on features** — Same feature set from stage 3 as starting point, plus any new
   features enabled by the aggregated constraint representation.

5. **Evaluate** — Beat the psignal baseline on VC@100 and NDCG across the evaluation window.

## Environment

```bash
cd /home/xyz/workspace/pmodel && source .venv/bin/activate
export RAY_ADDRESS="ray://10.8.0.36:10001"
```
