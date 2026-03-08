# Annual FTR Constraint Tier Prediction

## Goal
Predict which MISO annual FTR constraints will bind in each auction quarter (aq1-aq4).
Produce 5-tier rankings (tier 0=most binding, tier 4=least) using ML on top of V6.1 formula baseline.

## Pipeline Architecture
- **V6.1 annual signal** defines the constraint universe (~300-500 per quarter)
- **Ground truth**: Realized DA shadow prices, mapped via MISO_SPICE_CONSTRAINT_INFO bridge table
- **Eval splits**: split1 (train 2019-2021, eval 2022), split2 (+2022, eval 2023), split3 (+2023, eval 2024). 2025 held out.
- **12 eval groups**: 3 years x 4 quarters

## Constraint Mapping Chain
```
DA shadow constraint_id (numeric MISO IDs)
  -> MISO_SPICE_CONSTRAINT_INFO constraint_id -> branch_name (short format)
  -> LEFT JOIN to V6.1 branch_name
```
- 0 null branch_names in cached ground truth
- ~31-35% of V6.1 rows are binding per quarter
- ~20-28% of total DA shadow value falls on V6.1 branches (rest is outside V6.1 universe)
- Both flow_direction rows of a binding branch get the same shadow price

## VC@20 Definition
VC@20 = sum(realized_shadow_price of our top-20 ranked) / sum(realized_shadow_price of ALL V6.1 constraints)
Denominator is V6.1 universe only — NOT all market DA shadow price.

## Results (12-group eval, 2022-2024)

| Metric | v0 (formula) | v1 (ML, 6f) | v2 (11f) | v3 (tiered) | v4 (+formula) | v5 (both) |
|--------|-------------|-------------|----------|-------------|---------------|-----------|
| VC@20 | 0.2323 | 0.2934 | 0.2904 | 0.2871 | 0.3030 | **0.3075** |
| VC@100 | 0.6518 | **0.6854** | 0.6861 | 0.6697 | 0.6686 | 0.6792 |
| Recall@20 | 0.2208 | 0.2708 | 0.2667 | 0.2875 | 0.3042 | **0.3208** |
| Recall@50 | 0.3783 | 0.4383 | 0.4300 | **0.4417** | 0.4233 | 0.4367 |
| Recall@100 | 0.5075 | **0.5292** | 0.5208 | 0.5225 | 0.5133 | 0.5200 |
| NDCG | 0.5925 | 0.6071 | 0.5978 | 0.6078 | 0.6024 | **0.6098** |
| Spearman | 0.3425 | 0.3642 | 0.3604 | 0.3678 | 0.3641 | **0.3695** |
| Tier0-AP | 0.4154 | 0.4564 | 0.4567 | 0.4566 | 0.4531 | **0.4628** |
| Tier01-AP | 0.5641 | **0.5860** | 0.5843 | 0.5885 | 0.5805 | 0.5865 |

### Version Descriptions
- **v0**: V6.1 formula baseline (rank_ori = 0.60*da_rank + 0.30*density_mix + 0.10*density_ori)
- **v1**: LightGBM lambdarank, 6 V6.1 features, raw rank labels
- **v2**: v1 + spice6 density features (11 features)
- **v3**: v1 with tiered labels (0=non-binding, 1-4=quantile buckets)
- **v4**: v1 + rank_ori as 7th feature (formula-as-feature), raw rank labels
- **v5**: v4 with tiered labels (both improvements combined)

### Key Findings
- **v5 is new champion**: tiered labels + formula-as-feature, best on VC@20, Recall@20, NDCG, Spearman, Tier0-AP
- **Formula-as-feature** is the bigger contributor: +3.3% VC@20, +12.3% Recall@20 individually
- **Tiered labels** alone hurt VC@20 (-2.1%) but help Recall@20 (+6.2%) — sharper top-k focus
- **Combined effect is additive**: Recall@20 +18.5% vs v1
- VC@100 and Recall@100 slightly down — tradeoff for sharper top-k precision
- LightGBM lambdarank, simple params (lr=0.05, 31 leaves, 100 trees), walltime ~3s per 12-group run

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

## Feature Sets
- **Set A (v1)**: shadow_price_da, mean_branch_max, ori_mean, mix_mean, density_mix_rank_value, density_ori_rank_value
- **Set B (v2)**: Set A + prob_exceed_80/85/90/100/110

## Data
- V6.1: `/opt/data/xyz-dataset/signal_data/miso/constraints/Signal.MISO.SPICE_ANNUAL_V6.1`
- Spice6 density: `/opt/temp/tmp/pw_data/spice6/prod_f0p_model_miso/density/`
- Ground truth cache: `cache/ground_truth/` (28 parquet files, all pre-fetched)

## Registry
- `registry/v0/` — V6.1 formula baseline
- `registry/v1/` — LightGBM Set A (champion)
- `registry/v2/` — LightGBM Set B
- `registry/v1_holdout/` — v1 holdout (2025) results
- `registry/gates.json` — calibrated from v0
- `registry/champion.json` — currently v0 (needs promotion to v1)

## Caching
- `cache/enriched/` — V6.1 + spice6 data per (year, aq), avoids re-scanning 18GB density parquet on NFS
- `cache/ground_truth/` — realized DA shadow prices per (year, aq)
- Benchmark trains once per eval year, evaluates 4 quarters — not 4 separate trains