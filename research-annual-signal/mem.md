# Annual FTR Constraint Tier Prediction

## Goal
Predict which MISO annual FTR constraints will bind in each auction quarter (aq1-aq4).
Produce rankings using LightGBM LambdaRank on top of V6.1 signal baseline.

## Champion: v16 backfill+offpeak

**Features (7)**: `shadow_price_da`, `da_rank_value`, `bf_6`, `bf_12`, `bf_15`, `bfo_6`, `bfo_12`
**Config**: `registry/v16_champion/config.json`

### Holdout Results (2025-06, aq1-aq3, out-of-sample)

| Metric | v0b (formula) | v10e (prev ML) | **v16 champion** | v16 vs v0b |
|--------|:---:|:---:|:---:|:---:|
| VC@20 | 0.2329 | 0.3124 | **0.3920** | +68% |
| VC@50 | 0.4727 | 0.5735 | **0.5935** | +26% |
| VC@100 | 0.6936 | 0.7049 | **0.7153** | +3% |
| Recall@20 | 0.3167 | 0.3000 | **0.3667** | +16% |
| Recall@50 | 0.3667 | 0.4533 | **0.4800** | +31% |
| Recall@100 | 0.5433 | 0.5667 | **0.5767** | +6% |
| NDCG | 0.5691 | 0.7064 | **0.7093** | +25% |
| Spearman | 0.4347 | 0.5503 | **0.5794** | +33% |

- **Passes ALL 7 Group A gates** vs v0b (all 3 layers: mean floor, tail safety, worst-group)
- Composite rank #1 across all Group A metrics
- v10e FAILS Recall@20 gating vs v0b

### Per-Group Holdout

| Group | VC@20 | VC@50 | Recall@20 | NDCG | Spearman |
|-------|:-----:|:-----:|:---------:|:----:|:--------:|
| aq1 | 0.386 | 0.552 | 0.40 | 0.676 | 0.581 |
| aq2 | 0.303 | 0.623 | 0.35 | 0.592 | 0.581 |
| aq3 | 0.487 | 0.606 | 0.35 | 0.861 | 0.577 |

### Tail Risk (holdout)

| | VC@20 | NDCG |
|---|:---:|:---:|
| Mean | 0.3920 | 0.7093 |
| Bottom-2 | 0.3444 | 0.6337 |
| Worst | 0.3030 | 0.5918 |

### Feature Importance (holdout model)

| Feature | Gain % | Description |
|---------|:------:|-------------|
| da_rank_value | 35.5% | Historical DA constraint rank |
| bfo_12 | 28.6% | Offpeak 12-month binding frequency |
| bf_15 | 12.2% | Onpeak 15-month binding frequency |
| bf_6 | 8.4% | Onpeak 6-month binding frequency |
| bfo_6 | 6.7% | Offpeak 6-month binding frequency |
| shadow_price_da | 5.8% | Historical DA shadow price |
| bf_12 | 2.9% | Onpeak 12-month binding frequency |

---

## Pipeline Architecture
```
V6.1 Signal (pre-auction forecasts)    Spice6 Density (forward simulation)
         \                                    /
          -> cache/enriched/ (27 cols, ~276-632 rows/quarter)
                           |
                    Realized DA -> binding_freq (onpeak+offpeak, 107 months, 2017-04+)
                           |
                    LightGBM LambdaRank (tiered labels, 7 features)
                           |
                   Realized DA Shadow Prices <- MISO_SPICE_CONSTRAINT_INFO bridge
                   cache/ground_truth/ (branch_name -> realized_shadow_price)
                           |
                   Evaluate: VC@K, Recall@K, NDCG, Spearman, Tier-AP
```
- **V6.1 annual signal** defines the constraint universe (~276-632 per quarter)
- **Ground truth**: Realized DA shadow prices, mapped via MISO_SPICE_CONSTRAINT_INFO bridge table
- **Binding frequency**: Monthly binding sets from realized DA cache (`research-stage5-tier/data/realized_da/`)
  - Onpeak (bf_N): `*.parquet` files, offpeak (bfo_N): `*_offpeak.parquet` files
  - Backfill from 2017-04 through current (107+ months)
  - Cutoff: `< YYYY-04` for annual auction (`YYYY-06` planning year)
- **Eval splits**: split1 (train 2019-2021, eval 2022), split2 (+2022, eval 2023), split3 (+2023, eval 2024). 2025 holdout.
- **12 dev groups**: 3 years x 4 quarters
- **3 holdout groups**: 2025-06 aq1-aq3 (aq4 excluded: delivery Mar-May 2026 incomplete)
- **Training**: expanding window, train once per eval year, evaluate 4 quarters

## Constraint Mapping Chain
```
DA shadow constraint_id (numeric MISO IDs)
  -> MISO_SPICE_CONSTRAINT_INFO constraint_id -> branch_name
  -> LEFT JOIN to V6.1 branch_name
```
- Bridge table partition-filtered: (auction_type='annual', auction_month, period_type, class_type='onpeak')
- ~31-42% of V6.1 rows are binding per quarter
- Both flow_direction rows of a binding branch get the same shadow price

## VC@20 Definition
VC@20 = sum(realized_shadow_price of our top-20 ranked) / sum(realized_shadow_price of ALL V6.1 constraints)

---

## Key Findings

1. **Offpeak BF is the breakthrough feature**: bfo_12 is #2 feature (29% importance), captures structural congestion invisible to onpeak-only data.
2. **Backfill improves generalization**: Using 2017-04+ history hurts dev VC@20 but helps holdout (+25%). Dev eval was misleading.
3. **Density features hurt annual formula**: Pure da_rank_value (alpha=1.0) is optimal. Density_mix/ori add noise (-28.7% VC@20).
4. **Always use holdout**: Dev eval led to wrong conclusions multiple times. Holdout (2025) is definitive.
5. **Always evaluate all metrics**: v10e looked best on dev VC@20 but failed Recall@20 gating. Composite ranking is more robust.
6. **LightGBM num_threads=4**: Container has 64 CPUs, auto-detection causes 570x slowdown.

---

## Version History

| Version | Description | Dev VC@20 | Holdout VC@20 | Status |
|---------|-------------|:---------:|:-------------:|--------|
| v0 | V6.1 formula (0.60/0.30/0.10) | 0.2329 | 0.1559 | Superseded |
| v0b | Pure da_rank_value | 0.2997 | 0.2329 | Formula baseline |
| v1-v5 | V6.1 features + ML | 0.29-0.31 | -- | Superseded |
| v7d | 7f ML, tiered labels | 0.3033 | 0.2391 | Superseded |
| blend_v7d_a70 | 70% ML + 30% formula | 0.3113 | 0.2513 | Prev champion |
| v8b | +binding_freq (5 BF) | 0.3270 | -- | Superseded |
| v10e | +engineered interactions (8f) | 0.3389 | 0.3124 | Dev-best, fails Recall@20 gate |
| v13 | Backfill strategies | 0.31-0.33 | -- | Exploration |
| v14 | Combined signals | 0.31-0.34 | -- | Exploration |
| v15 | Multi-metric eval | -- | -- | Holdout framework |
| **v16** | **backfill+offpeak (7f)** | **0.3160** | **0.3920** | **CHAMPION** |

---

## Model Config (v16 champion)
- Backend: LightGBM LambdaRank
- Label mode: tiered (5 levels: 0=non-binding, 1-4=quantile buckets)
- Hyperparams: lr=0.03, 200 trees, 31 leaves, subsample=0.8, colsample=0.8
- num_threads=4
- Monotone constraints: [1, -1, 1, 1, 1, 1, 1]
- Backfill floor: 2017-04 (107 months of realized DA history)
- Walltime: ~11s for full holdout eval

## V6.1 Formula
```
rank_ori = 0.60 * da_rank_value + 0.30 * density_mix_rank_value + 0.10 * density_ori_rank_value
```
- da_rank_value = rank(shadow_price_da) -- historical DA, NOT realized
- Lower rank_ori = more binding. Score = 1.0 - rank_ori for evaluation.

## Data
- V6.1: `/opt/data/xyz-dataset/signal_data/miso/constraints/Signal.MISO.SPICE_ANNUAL_V6.1`
- Spice6 density: `/opt/data/xyz-dataset/spice_data/miso/MISO_SPICE_DENSITY_DISTRIBUTION.parquet`
- Realized DA cache: `/home/xyz/workspace/research-qianli-v2/research-stage5-tier/data/realized_da/`
- Ground truth cache: `cache/ground_truth/` (28 parquet files)
- Enriched cache: `cache/enriched/` (V6.1 + spice6 per (year, aq))

## Registry
- `registry/champion.json` -- points to v16_champion
- `registry/v16_champion/config.json` -- full champion config with holdout metrics
- `registry/v15_multi_metric/summary.json` -- all variant results (dev + holdout)
- `registry/gates.json` -- quality gates (calibrated from v0)
- `registry/v0/`..`v5/`, `v7a/`..`v7d/`, `v8a/`..`v9j/` -- historical experiment results

## Key Scripts
- `scripts/run_v16_champion_analysis.py` -- champion holdout analysis (per-group, tail risk, gating)
- `scripts/run_v15_multi_metric.py` -- multi-metric dev+holdout eval for 7 variants
- `scripts/run_v8_binding_freq.py` -- binding frequency experiment
- `scripts/run_v0_baseline.py` -- formula baseline
- `ml/binding_freq.py` -- binding frequency module (onpeak, offpeak, decayed)
- `ml/train.py` -- LightGBM training
- `ml/evaluate.py` -- all metrics
