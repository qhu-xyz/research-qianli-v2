# Annual FTR Constraint Tier Prediction

## Goal
Predict which MISO annual FTR constraints will bind in each auction quarter (aq1-aq4).
Produce 5-tier rankings (tier 0=most binding, tier 4=least) using ML on top of V6.1 formula baseline.

## Pipeline Architecture
```
V6.1 Signal (pre-auction forecasts)    Spice6 Density (forward simulation)
         \                                    /
          → cache/enriched/ (27 cols, ~276-632 rows/quarter)
                           |
                    [Features extracted]
                           |
                    LightGBM LambdaRank
                           |
                   Realized DA Shadow Prices ← MisoApTools (separate fetch)
                   cache/ground_truth/ (2 cols: branch_name, realized_shadow_price)
                           |
                   Evaluate: VC@K, Recall@K, NDCG, Spearman, Tier-AP
```
- **V6.1 annual signal** defines the constraint universe (~276-632 per quarter)
- **Ground truth**: Realized DA shadow prices, mapped via MISO_SPICE_CONSTRAINT_INFO bridge table
- **Eval splits**: split1 (train 2019-2021, eval 2022), split2 (+2022, eval 2023), split3 (+2023, eval 2024). 2025 held out.
- **12 eval groups**: 3 years x 4 quarters
- **Training**: expanding window, train once per eval year, evaluate 4 quarters

## Constraint Mapping Chain
```
DA shadow constraint_id (numeric MISO IDs)
  -> MISO_SPICE_CONSTRAINT_INFO constraint_id -> branch_name (short format)
  -> LEFT JOIN to V6.1 branch_name
```
- 0 null branch_names in cached ground truth
- ~31-42% of V6.1 rows are binding per quarter
- ~20-28% of total DA shadow value falls on V6.1 branches (rest is outside V6.1 universe)
- Both flow_direction rows of a binding branch get the same shadow price

## VC@20 Definition
VC@20 = sum(realized_shadow_price of our top-20 ranked) / sum(realized_shadow_price of ALL V6.1 constraints)
Denominator is V6.1 universe only — NOT all market DA shadow price.

---

## Results (12-group eval, 2022-2024)

| Metric | v0 (formula) | v0b (pure DA) | v5 (best ML) | blend_v7d_a70 |
|--------|-------------|--------------|-------------|---------------|
| VC@20 | 0.2329 | 0.2997 | 0.3075 | **0.3113** |
| VC@100 | 0.6573 | 0.6879 | 0.6792 | **0.6935** |
| Recall@20 | 0.2167 | 0.3208 | 0.3208 | **0.3417** |
| Recall@50 | 0.3817 | **0.4600** | 0.4367 | 0.4533 |
| Recall@100 | 0.5117 | 0.5208 | 0.5200 | **0.5308** |
| NDCG | 0.5889 | 0.6028 | **0.6098** | 0.5987 |
| Spearman | 0.3392 | 0.3678 | 0.3695 | **0.3715** |

### ML Variant Results (v7 rebase)

| Version | Features | VC@20 | vs v0b |
|---------|----------|-------|--------|
| v0b | Pure da_rank_value | 0.2997 | baseline |
| v7a | Set A (6f), tiered | 0.3024 | +0.9% |
| v7b | Lean (4f), tiered | 0.3072 | +2.5% |
| v7c | Lean+da_rank (5f) | 0.3025 | +0.9% |
| v7d | Set A+da_rank (7f) | 0.3033 | +1.2% |
| v5 | Set AF (7f), tiered | 0.3075 | +2.6% |
| **blend_v7d_a70** | **70% v7d + 30% v0b** | **0.3113** | **+3.9%** |

### Key Findings

1. **Density features hurt in formula**: Grid search found pure da_rank_value (alpha=1.0) optimal. Density_mix/ori add noise (-28.7% VC@20).
2. **ML marginal value is small**: Best ML (v5) = +2.6% VC@20 over v0b. Best blend = +3.9%. Original claim of +32% vs v0 was misleading.
3. **Feature count doesn't matter**: v7b (4 lean features) matches v5 (7 features) at 0.3072 vs 0.3075.
4. **Score blending is best strategy**: 70% ML + 30% v0b gives best VC@20 (0.3113) and Recall@20 (0.3417).
5. **Recall@100 tradeoff is fundamental**: All versions that sharpen top-k sacrifice worst-case Recall@100.
6. **No version passes all v0b gates**: Recall@100 L3 tail regression is the blocker.

### Recommendation
Use v0b (pure da_rank_value) as primary signal. ML adds at most +3.9% via blending — marginal given complexity. Density components should be removed from annual formula.

---

## Holdout Results (2025, 4 quarters)

| Metric | v0 (formula) | v1 (ML) | Delta |
|--------|-------------|---------|-------|
| VC@20 | 0.1559 | **0.2152** | +38.0% |
| VC@100 | 0.5784 | 0.5812 | +0.5% |
| Recall@20 | 0.2500 | **0.3125** | +25.0% |
| Recall@100 | 0.4675 | 0.4875 | +4.3% |
| NDCG | 0.5043 | **0.5218** | +3.5% |
| Spearman | 0.3872 | 0.3906 | +0.9% |
| Tier0-AP | — | 0.3914 | — |
| Tier01-AP | — | 0.5443 | — |

- v1 beats v0 on VC@20 (+38%), Recall@20 (+25%), NDCG (+3.5%) on holdout
- Note: 2025/aq4 has only 76/418 (18.2%) binding — likely partial/incomplete data (Mar-May 2026)
- Holdout VC@20 (0.215) is ~30% below dev eval (0.307) — expected degradation

---

## Per-Year Analysis (v0 vs v5)

| Year | v0 VC@20 | v5 VC@20 | Delta | v0 NDCG | v5 NDCG | Delta |
|------|----------|----------|-------|---------|---------|-------|
| 2022 | 0.2870 | 0.3817 | +33.0% | 0.5831 | 0.6111 | +4.8% |
| 2023 | 0.2590 | 0.2934 | +13.3% | 0.6234 | 0.6117 | -1.9% |
| 2024 | 0.1508 | 0.2475 | +64.1% | 0.5711 | 0.6067 | +6.2% |

### Caveats
1. **ML doesn't win every group**: v5 loses 4/12 groups on VC@20 (2022/aq3, 2022/aq4, 2023/aq1, 2023/aq4)
2. **2023 NDCG regresses**: ML loses NDCG vs formula in 2023 (-1.9%)
3. **Gains are lumpy**: driven by a few strong quarters (2022/aq2: +0.30, 2024/aq1: +0.15)
4. **Small N**: only 12 eval groups, one outlier quarter swings the mean
5. **Top-K only**: VC@100+ flat or down — ML sharpens the head, doesn't improve full ranking
6. **Holdout degradation**: ~30% below dev eval on VC@20
7. **Limited eval years**: only 3 eval years (2022-2024)

---

## Next Steps

### Completed (2026-03-09)
- Grid search over formula weights: pure da_rank_value (alpha=1.0) is optimal (+28.7% VC@20)
- Raw shadow_price_da tested: identical to da_rank_value (monotonic transform)
- Recall@100 tie-breaking fix: cap true set to positive-value rows only
- Ground truth mapping fix committed (partition-filtered, was already cached correctly)
- Gate recalibration from v0b: no version passes all gates due to Recall@100 tail tradeoff
- ML rebase (v7a-v7d): tested lean/full features with tiered labels against v0b
- Blending experiments: score blend, rank blend, RRF — best is score_blend_v7d_a70 (+3.9%)

### Remaining
- Test blend_v7d_a70 on holdout (2025) to confirm dev-eval gains hold
- Consider relaxing Recall@100 tail gate (fundamental top-k vs breadth tradeoff)
- Communicate finding: annual formula density components should be removed/downweighted

---

## Feature Sets
- **Set A (v1)**: shadow_price_da, mean_branch_max, ori_mean, mix_mean, density_mix_rank_value, density_ori_rank_value
- **Set AF (v4/v5)**: Set A + rank_ori (formula score as feature)
- **Set B (v2)**: Set A + prob_exceed_80/85/90/100/110
- **Set C (unused)**: Set B + constraint_limit, rate_a

## V6.1 Formula
```
rank_ori = 0.60 * da_rank_value + 0.30 * density_mix_rank_value + 0.10 * density_ori_rank_value
```
- da_rank_value = rank(shadow_price_da) — historical DA, NOT realized
- Lower rank_ori = more binding. Score = 1.0 - rank_ori for evaluation.

## Model Config
- Backend: LightGBM lambdarank
- Hyperparams: lr=0.05, 100 trees, 31 leaves, subsample=0.8, colsample=0.8, min_data_in_leaf=25
- num_threads=4 (CRITICAL — prevents 64-core thread contention, 570x speedup)
- Monotone constraints enforced for all features
- Walltime: ~3s per 12-group run

## Data
- V6.1: `/opt/data/xyz-dataset/signal_data/miso/constraints/Signal.MISO.SPICE_ANNUAL_V6.1`
- Spice6 density: `/opt/data/xyz-dataset/spice_data/miso/MISO_SPICE_DENSITY_DISTRIBUTION.parquet`
- Ground truth cache: `cache/ground_truth/` (28 parquet files, all pre-fetched)
- Enriched cache: `cache/enriched/` (V6.1 + spice6 per (year, aq))

## Registry
- `registry/v0/` — V6.1 formula baseline (0.60/0.30/0.10 weights)
- `registry/v0b/` — Pure da_rank_value (alpha=1.0) — **best formula**
- `registry/v0c/` — Raw shadow_price_da (identical to v0b)
- `registry/v1/`..`v5/` — ML experiments (see version descriptions above)
- `registry/v7a/`..`v7d/` — ML rebase experiments (lean/full features)
- `registry/v7_blending/` — Blending experiment results (score/rank/RRF)
- `registry/v1_holdout/` — v1 holdout (2025) results
- `registry/gates.json` — calibrated from v0
- `registry/gates_v0b.json` — calibrated from v0b (stricter)
- `registry/champion.json` — currently v0