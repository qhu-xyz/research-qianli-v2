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

| Metric | v0 (formula) | v0b (pure DA) | v1 (ML, 6f) | v5 (best ML) |
|--------|-------------|--------------|-------------|-------------|
| VC@20 | 0.2329 | **0.2997** | 0.2934 | **0.3075** |
| VC@100 | 0.6573 | 0.6879 | 0.6854 | 0.6792 |
| Recall@20 | 0.2167 | **0.3208** | 0.2708 | **0.3208** |
| Recall@50 | 0.3817 | **0.4600** | 0.4383 | 0.4367 |
| Recall@100 | 0.5117 | 0.5208 | **0.5292** | 0.5200 |
| NDCG | 0.5889 | **0.6028** | 0.6071 | **0.6098** |
| Spearman | 0.3392 | **0.3678** | 0.3642 | **0.3695** |

### Version Descriptions
- **v0**: V6.1 formula baseline (rank_ori = 0.60*da_rank + 0.30*density_mix + 0.10*density_ori)
- **v0b**: Pure da_rank_value (alpha=1.0, beta=0.0, gamma=0.0) — **best formula from grid search**
- **v0c**: Raw shadow_price_da (identical to v0b — monotonic transform of da_rank_value)
- **v1**: LightGBM lambdarank, 6 V6.1 features (Set A), raw rank labels
- **v2**: v1 + spice6 density features (11 features, Set B), raw rank labels
- **v3**: v1 with tiered labels (0=non-binding, 1-4=quantile buckets)
- **v4**: v1 + rank_ori as 7th feature (formula-as-feature, Set AF), raw rank labels
- **v5**: v4 with tiered labels (both improvements combined) — **best ML**

### Key Findings (Baseline Recalibration, 2026-03-09)
- **Density features actively hurt**: Grid search found optimal weights are alpha=1.0 (pure da_rank_value). Density_mix and density_ori add noise.
- **v0b (+28.7% VC@20 vs v0)**: Simply dropping density components nearly matches v1 ML (+26.0% vs v0)
- **ML marginal value is much smaller than thought**: v5 vs v0b is only +2.6% VC@20 (was +32% vs v0)
- **v5 still best overall** but the gap over a properly-tuned formula is narrow
- **Recall@100 tradeoff**: v0b's worst-case Recall@100 (0.42) is worse than v0's (0.475) — sharper head, weaker tail

### Gate Status (v0-calibrated gates)
- After Recall@100 tie-breaking fix, NO version passes all v0 gates (Recall@100 tail gate too tight)
- With v0b-calibrated gates: still NO version passes all gates (Recall@100 L3 tail regression)
- Root cause: sharpening top-k rankings sacrifices worst-case Recall@100 — fundamental tradeoff

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

### Remaining
- Re-run ML (v1-v5) with v0b as the baseline feature (rank_ori = pure da_rank_value)
- Blending: score blend, rank blend, RRF between ML and formula
- Consider relaxing Recall@100 tail gate (fundamental top-k vs breadth tradeoff)
- Test on holdout with v0b baseline

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
- `registry/v0/` — V6.1 formula baseline (0.60/0.30/0.10 weights, calibrates original gates)
- `registry/v0b/` — Pure da_rank_value (alpha=1.0) — **best formula from grid search**
- `registry/v0c/` — Raw shadow_price_da (identical to v0b)
- `registry/v1/` — LightGBM Set A
- `registry/v2/` — LightGBM Set B
- `registry/v3/` — Tiered labels
- `registry/v4/` — Formula-as-feature
- `registry/v5/` — Both improvements (best ML)
- `registry/v1_holdout/` — v1 holdout (2025) results
- `registry/gates.json` — calibrated from v0
- `registry/gates_v0b.json` — calibrated from v0b (stricter)
- `registry/champion.json` — currently v0