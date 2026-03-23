# NB-hist-12 Experiment V2: Per-Ctype Results (2026-03-23)

## V1 → V2 Changes

V1 had multiple bugs: combined v0c (wrong BF + combined da_rank_value), onpeak-only V4.4 comparison, combined NB model, global LambdaRank labels, V4.4 as model feature. See `docs/superpowers/plans/2026-03-23-nb-experiment-v2-restructure.md` for full diff table.

## Experimental Setup

- **2 NB models per eval year**: onpeak (`bf_12==0`, `onpeak_sp` target) and offpeak (`bfo_12==0`, `offpeak_sp` target)
- **8 features** (class-agnostic): `bin_80/90/100/110_cid_max`, `rt_max`, `count_active_cids`, `shadow_price_da`, `da_rank_value`
- **v0c**: Fully class-specific via `build_class_model_table` — class-specific `da_rank_value`, `shadow_price_da`, `bf_12`/`bfo_12`
- **V4.4**: Loaded per-ctype as benchmark (not as feature). Branches without V4.4 scored -inf.
- **Rolling CV**: train 2021-22→eval 2023, train 2021-23→eval 2024, train 2021-24→eval 2025
- **Fixed K**: All configs produce exactly K=200 or K=400 branches via v0c backfill
- **Two masks**: `is_dormant = (class_bf == 0)` for candidate pool, `is_nb_binder = is_dormant & (sp > 0)` for binder metrics
- **Per-group LambdaRank labels**: Tertiles within each (PY, aq) group, not global

## Part 1: NB-Only Metrics (Per-Ctype Dormant Universe)

### Onpeak

| Year | K | ML_nb VC | V4.4 VC | v0c VC | ML_nb Rec | V4.4 Rec |
|------|---|---------|---------|--------|-----------|----------|
| 2024 | 50 | **0.096** | 0.013 | 0.057 | **0.075** | 0.040 |
| 2024 | 100 | **0.163** | 0.070 | 0.097 | **0.148** | 0.096 |
| 2025 | 50 | **0.248** | 0.108 | 0.140 | **0.101** | 0.069 |
| 2025 | 100 | **0.290** | 0.211 | 0.217 | **0.150** | 0.137 |

### Offpeak

| Year | K | ML_nb VC | V4.4 VC | v0c VC | ML_nb Rec | V4.4 Rec |
|------|---|---------|---------|--------|-----------|----------|
| 2024 | 50 | **0.103** | 0.012 | 0.032 | **0.087** | 0.034 |
| 2024 | 100 | **0.204** | 0.059 | 0.050 | **0.141** | 0.093 |
| 2025 | 50 | 0.106 | **0.128** | 0.143 | **0.084** | 0.059 |
| 2025 | 100 | 0.192 | **0.195** | **0.252** | **0.143** | 0.098 |

**Key finding**: ML_nb dominates V4.4 on onpeak NB-only for both years at all K levels. On offpeak, ML_nb wins 2024 convincingly but 2025 is mixed (v0c actually leads at K=100 VC). V4.4 is inconsistent — near-zero on 2024 for both ctypes.

## Part 2: Full Universe Aggregate

### Onpeak K=200

| Config | VC | Abs | Rec | NDCG | NB_in | NB_b | NB_SP | NB_VC | D20 | Fill |
|--------|---:|----:|----:|-----:|------:|-----:|------:|------:|----:|-----:|
| pure_v0c | 0.579 | 0.535 | 0.312 | 0.835 | 5 | 0.4 | $175 | 0.000 | 9/13 | - |
| pure_v44 | 0.417 | 0.384 | 0.194 | 0.759 | 62 | 8.1 | $41K | 0.115 | 6/13 | - |
| R30_nb | 0.562 | 0.519 | 0.301 | 0.835 | 33 | 6.3 | $21K | 0.079 | 9/13 | 30/30 |
| R30_v44 | 0.572 | 0.527 | 0.299 | 0.835 | 33 | 5.6 | $34K | 0.094 | 9/13 | 30/30 |
| R50_nb | 0.554 | 0.511 | 0.284 | 0.835 | 52 | 9.6 | $38K | 0.143 | 8/13 | 50/50 |
| R50_v44 | 0.553 | 0.510 | 0.276 | 0.835 | 52 | 6.7 | $39K | 0.104 | 8/13 | 50/50 |

### Onpeak K=400

| Config | VC | Abs | Rec | NDCG | NB_in | NB_b | NB_SP | NB_VC | D20 | Fill |
|--------|---:|----:|----:|-----:|------:|-----:|------:|------:|----:|-----:|
| pure_v0c | 0.699 | 0.644 | 0.444 | 0.830 | 72 | 7.3 | $43K | 0.141 | 10/13 | - |
| pure_v44 | 0.594 | 0.545 | 0.337 | 0.754 | 167 | 21.6 | $93K | 0.310 | 9/13 | - |
| **R30_nb** | **0.703** | **0.647** | **0.446** | 0.830 | 94 | 13.2 | **$56K** | 0.211 | 10/13 | 50/50 |
| R30_v44 | 0.704 | 0.648 | 0.443 | 0.830 | 94 | 12.2 | $60K | 0.183 | 10/13 | 50/50 |
| R50_nb | 0.695 | 0.640 | 0.435 | 0.831 | 123 | 18.4 | $76K | 0.268 | 10/13 | 100/100 |
| R50_v44 | 0.689 | 0.635 | 0.428 | 0.831 | 123 | 15.9 | $71K | 0.226 | 10/13 | 100/100 |

### Offpeak K=200

| Config | VC | Abs | Rec | NDCG | NB_in | NB_b | NB_SP | NB_VC | D20 | Fill |
|--------|---:|----:|----:|-----:|------:|-----:|------:|------:|----:|-----:|
| pure_v0c | 0.674 | 0.631 | 0.348 | 0.846 | 5 | 0.6 | $2K | 0.010 | 9/12 | - |
| pure_v44 | 0.500 | 0.469 | 0.212 | 0.768 | 65 | 6.9 | $33K | 0.112 | 7/12 | - |
| R30_nb | 0.657 | 0.615 | 0.333 | 0.846 | 32 | 6.0 | $15K | 0.079 | 9/12 | 30/30 |
| R30_v44 | 0.663 | 0.620 | 0.330 | 0.846 | 32 | 5.1 | $26K | 0.083 | 9/12 | 30/30 |
| R50_nb | 0.637 | 0.596 | 0.315 | 0.846 | 51 | 8.8 | $24K | 0.117 | 9/12 | 50/50 |
| R50_v44 | 0.641 | 0.599 | 0.306 | 0.847 | 51 | 5.9 | $32K | 0.108 | 9/12 | 50/50 |

### Offpeak K=400

| Config | VC | Abs | Rec | NDCG | NB_in | NB_b | NB_SP | NB_VC | D20 | Fill |
|--------|---:|----:|----:|-----:|------:|-----:|------:|------:|----:|-----:|
| pure_v0c | 0.788 | 0.736 | 0.491 | 0.837 | 81 | 9.4 | $46K | 0.174 | 10/12 | - |
| pure_v44 | 0.657 | 0.613 | 0.364 | 0.759 | 180 | 20.9 | $80K | 0.300 | 9/12 | - |
| R30_nb | 0.782 | 0.732 | 0.486 | 0.838 | 103 | 14.8 | $50K | 0.210 | 10/12 | 50/50 |
| R30_v44 | 0.782 | 0.730 | 0.475 | 0.839 | 103 | 11.3 | $52K | 0.185 | 10/12 | 50/50 |
| R50_nb | 0.760 | 0.710 | 0.471 | 0.839 | 131 | 18.6 | $57K | 0.274 | 10/12 | 100/100 |
| R50_v44 | 0.755 | 0.705 | 0.461 | 0.840 | 131 | 14.9 | $56K | 0.193 | 10/12 | 100/100 |

## Part 3: Delta vs pure_v0c

### Onpeak

| Config | K=200 dVC | K=200 dNB_SP | K=400 dVC | K=400 dNB_SP |
|--------|--------:|------------:|--------:|------------:|
| R30_nb | -1.8pp | +$21K | **+0.4pp** | +$13K |
| R30_v44 | -0.8pp | +$34K | **+0.5pp** | +$16K |
| R50_nb | -2.6pp | +$38K | -0.4pp | +$33K |
| R50_v44 | -2.7pp | +$39K | -0.9pp | +$27K |

### Offpeak

| Config | K=200 dVC | K=200 dNB_SP | K=400 dVC | K=400 dNB_SP |
|--------|--------:|------------:|--------:|------------:|
| R30_nb | -1.7pp | +$13K | -0.5pp | +$3K |
| R30_v44 | -1.0pp | +$23K | -0.6pp | +$6K |
| R50_nb | -3.7pp | +$22K | -2.7pp | +$11K |
| R50_v44 | -3.3pp | +$30K | -3.2pp | +$10K |

## Part 4: Fill-Rate Analysis

All reserved-slot configs achieved 100% fill rate across both ctypes and both K levels. The dormant population (~1,800 onpeak, ~2,000 offpeak per quarter) is large enough that neither ML_nb nor V4.4 ever needs v0c backfill at the tested reservation sizes (30-100 slots).

## Conclusions

### 1. ML_nb wins NB-only on onpeak, mixed on offpeak
ML_nb dominates V4.4 on NB-only onpeak (VC@50: 0.096 vs 0.013 in 2024, 0.248 vs 0.108 in 2025). On offpeak, ML_nb wins 2024 but 2025 is mixed with v0c actually competitive.

### 2. At K=400 onpeak, R30 is a near-free lunch
R30_nb: +0.4pp VC + $13K NB_SP. R30_v44: +0.5pp VC + $16K NB_SP. Both improve overall VC while adding NB capture.

### 3. At K=400 offpeak, all reservations cost VC
Unlike onpeak, offpeak reservations all cost VC (-0.5pp to -3.2pp). The NB model doesn't find enough valuable offpeak dormant binders to compensate for displacing v0c picks.

### 4. V4.4 as NB scorer captures more NB_SP than ML at K=200
At K=200, R30_v44 captures $34K NB_SP vs R30_nb's $21K (onpeak). V4.4's per-ctype ranking is more targeted for NB dollar capture, while ML_nb finds more NB binders but with less SP each.

### 5. Recommended configs
- **Onpeak K=400**: R30_nb or R30_v44 — both positive VC delta
- **Onpeak K=200**: R30_v44 — smallest VC cost (-0.8pp) with most NB_SP (+$34K)
- **Offpeak K=400**: R30_nb — smallest VC cost (-0.5pp)
- **Offpeak K=200**: R30_v44 — smallest VC cost (-1.0pp)

## Scripts and Artifacts

| File | Purpose |
|------|---------|
| `scripts/nb_experiment_v2.py` | Main experiment: 2 NB models, class-specific eval, V4.4 benchmark |
| `tests/test_nb_experiment_v2.py` | 12 unit tests |
| `registry/onpeak/nb_v2/` | Onpeak: config, metrics, nb_only_metrics, case_studies |
| `registry/offpeak/nb_v2/` | Offpeak: same |
| `docs/superpowers/plans/2026-03-23-nb-experiment-v2-restructure.md` | Implementation plan |

Runtime: 259s.
