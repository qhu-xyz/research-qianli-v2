# Stage 5 Summary (Reference)

This is a consolidated reference for the stage 5 work that preceded stage 6.
For full details, see `registry/v10e-lag1/NOTES.md`.

---

## What We Built

A LightGBM regression model (v10e-lag1) that ranks ~600-800 MISO transmission
constraints by predicted binding severity for f0/onpeak. It replaces the V6.2B
production formula (v0), a hand-tuned 3-feature weighted sum.

## Results

### Dev (36 months, 2020-06 to 2023-05)

| Metric | v0 (formula) | v10e-lag1 (ML) | vs v0 |
|--------|-------------|---------------|-------|
| VC@20 | 0.2817 | **0.4137** | +47% |
| VC@100 | 0.6008 | **0.7195** | +20% |
| Recall@20 | 0.1833 | **0.3278** | +79% |
| NDCG | 0.4423 | **0.5837** | +32% |
| Spearman | 0.2045 | **0.2989** | +46% |

### Holdout (24 months, 2024-2025)

| Metric | v0 (formula) | v10e-lag1 (ML) | vs v0 |
|--------|-------------|---------------|-------|
| VC@20 | 0.1835 | **0.3529** | +92% |
| VC@100 | 0.5924 | **0.6807** | +15% |
| Recall@20 | 0.1500 | **0.3021** | +101% |
| NDCG | 0.4224 | **0.5497** | +30% |
| Spearman | 0.1946 | **0.3226** | +66% |

## Model: v10e-lag1 (9 features)

| Feature | Source | Importance | Monotone |
|---------|--------|-----------|----------|
| binding_freq_12 | Realized DA | 36.4% | +1 |
| v7_formula_score | V6.2B | 19.4% | -1 |
| binding_freq_15 | Realized DA | 16.3% | +1 |
| da_rank_value | V6.2B | 10.0% | -1 |
| binding_freq_6 | Realized DA | 9.0% | +1 |
| binding_freq_1 | Realized DA | 2.9% | +1 |
| binding_freq_3 | Realized DA | 2.7% | +1 |
| prob_exceed_110 | Spice6 | 2.3% | +1 |
| constraint_limit | Spice6 | 0.9% | 0 |

Config: LightGBM regression, tiered labels (0/1/2/3), 8-month rolling window,
0 validation, 1-month production lag, lr=0.05, 31 leaves, 100 trees, num_threads=4.

`v7_formula_score = 0.85 * da_rank_value + 0.15 * density_ori_rank_value`

## V6.2B Formula (v0 Baseline)

```
rank_ori = 0.60 * da_rank_value + 0.30 * density_mix_rank_value + 0.10 * density_ori_rank_value
```

Verified exact match (max_abs_diff=0.0 all months).

## Ground Truth

Realized DA shadow prices for the delivery month:

```python
MisoApTools().tools.get_da_shadow_by_peaktype(st, et_ex, peak_type="onpeak")
```

Aggregation: `abs(sum(shadow_price))` per constraint per month.
Cached: `data/realized_da/{YYYY-MM}.parquet` (79 months).
~12% binding rate within V6.2B universe (~68-81 binding per ~600 constraints).

## Temporal Leakage Discovery

v9-v10e were inflated 6-20% by using month M-1 data not available at f0 submission
(~mid of M-1). v10e-lag1 fixes this by shifting everything back 1 month:

| Component | Leaky (v10e) | Safe (v10e-lag1) |
|-----------|-------------|-----------------|
| Training months | M-8..M-1 | M-9..M-2 |
| bf for month T | months < T | months < T-1 |
| bf for test M | months < M | months < M-1 |

## Pipeline Audit (19/19 PASS)

1. Ground truth (6 checks): realized DA cache correct, abs(sum()) aggregation, not circular
2. Feature leakage (5 checks): leaky set defined, no leaky features used, da_rank_value is clean
3. Data mapping (4 checks): constraint_id String type, joins correct, spice6 99%+ overlap
4. Training labels (3 checks): per-month labels, tiered (0-3), correct defaults
5. Score direction (1 check): higher score = more binding throughout pipeline

External codex audit found: Tier0-AP/Tier01-AP degenerate (removed), Recall@100
tie-contaminated (demoted), FEATURES_V3 dead code (acknowledged).

## Data Paths

| Data | Path |
|------|------|
| V6.2B | `/opt/data/xyz-dataset/signal_data/miso/constraints/TEST.TEST.Signal.MISO.SPICE_F0P_V6.2B.R1/{month}/f0/onpeak` |
| Spice6 density | `/opt/temp/tmp/pw_data/spice6/prod_f0p_model_miso/density/` |
| Spice6 ml_pred | `/opt/temp/tmp/pw_data/spice6/prod_f0p_model_miso/ml_pred/` |
| Realized DA | `data/realized_da/{YYYY-MM}.parquet` (cached from Ray) |

## Key ML Learnings

- binding_freq is #1 feature; multi-window (1/3/6/12/15) beats single window
- Longer bf windows more robust to production lag (bf_12/bf_15 dominate in lag1)
- LightGBM regression > lambdarank for top-k precision on sparse targets (88% zeros)
- Tiered labels fix rank-transform noise: +36% VC@20
- Formula-as-feature consistently helps (+10-18% VC@20)
- Feature pruning: 9 features beats 14 (less noise)
- 8mo train / 0 val >> 6mo train / 2 val
- LightGBM deadlocks with 64 threads in containers — always num_threads=4

## Version History (Archived in registry/archive/)

- **v0**: V6.2B formula baseline
- **v1-v4**: Feature set and label experiments
- **v5-v6c**: Regression vs lambdarank comparison (v6b was prev champion)
- **v9**: First binding_freq (+34% VC@20 dev, but leaky)
- **v10-v10g**: Feature pruning and multi-window bf (all leaky)
- **v10e-lag1**: Production-safe champion (current)
