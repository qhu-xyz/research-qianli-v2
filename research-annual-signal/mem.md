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

| Metric | v0 (formula) | v1 (ML, 6 feat) | v2 (ML, 11 feat) |
|--------|-------------|-----------------|------------------|
| VC@20 | 0.2323 | **0.2934** (+26.3%) | 0.2904 (+25.0%) |
| VC@100 | 0.6518 | 0.6854 (+5.2%) | **0.6861** (+5.3%) |
| Recall@20 | 0.2208 | **0.2708** (+22.6%) | 0.2667 (+20.8%) |
| Recall@50 | 0.3783 | **0.4383** (+15.9%) | 0.4300 (+13.7%) |
| Recall@100 | 0.5075 | **0.5292** (+4.3%) | 0.5208 (+2.6%) |
| NDCG | 0.5925 | **0.6071** (+2.5%) | 0.5978 (+0.9%) |
| Spearman | 0.3425 | **0.3642** (+6.3%) | 0.3604 (+5.2%) |
| Tier0-AP | 0.4154 | 0.4564 (+9.9%) | **0.4567** (+9.9%) |
| Tier01-AP | 0.6847 | **0.7041** (+2.8%) | 0.7037 (+2.8%) |
| Gates | PASS | **PASS** | FAIL (Recall@100 tail) |

### Key Findings
- **v1 is champion**: 6 V6.1 features with ML-learned weights beats formula on ALL metrics
- **v2 (+ spice6 density) doesn't help**: marginal difference, fails Recall@100 tail gate
- Biggest gain: VC@20 +26.3% — ML much better at identifying top binding constraints
- LightGBM lambdarank, simple params (lr=0.05, 31 leaves, 100 trees)

## Feature Sets
- **Set A (v1)**: da_rank_value, density_mix_rank_value, density_ori_rank_value, mean_branch_max, mean_branch_max_fillna, ori_mean
- **Set B (v2)**: Set A + prob_exceed_80/85/90/100/110

## Data
- V6.1: `/opt/data/xyz-dataset/signal_data/miso/constraints/TEST.TEST.Signal.MISO.SPICE_F0P_V6.2B.R1`
- Spice6 density: `/opt/temp/tmp/pw_data/spice6/prod_f0p_model_miso/density/`
- Ground truth cache: `cache/ground_truth/` (28 parquet files, all pre-fetched)

## Registry
- `registry/v0/` — V6.1 formula baseline
- `registry/v1/` — LightGBM Set A (champion)
- `registry/v2/` — LightGBM Set B
- `registry/gates.json` — calibrated from v0
- `registry/champion.json` — currently v0 (needs promotion to v1)
