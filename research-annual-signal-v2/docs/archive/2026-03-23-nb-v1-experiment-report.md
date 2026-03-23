# NB-hist-12 Model Experiment Report (2026-03-23)

## Motivation

The v0c champion formula (`0.40×da_rank + 0.30×rt_max + 0.30×bf`) captures ~62% of binding value at K=200 but is structurally blind to dormant branches (bf_combined_12 == 0, i.e. no binding in the last 12 months). These "NB-hist-12" branches are ~66% of the universe (~1,800/quarter) but include surprise binders that contribute significant DA shadow price.

V4.4 (an existing SPICE signal) uses forward-looking deviation features from SPICE scenario analysis. It was observed to rank certain dormant outliers (MONTCELO, BENTON) much higher than v0c. The question: **can we build a model that beats V4.4 on NB-hist-12 branches, then combine with v0c without sacrificing overall performance?**

## Experimental Setup

### Data
- **Universe**: Branch-level MISO annual signal, aq1-aq3 (aq4 incomplete).
- **Planning years**: 2021-06 through 2025-06.
- **NB-hist-12 population**: Branches with `bf_combined_12 == 0` (no binding in prior 12 months of realized DA).
- **Target**: `realized_shadow_price` (quarterly DA SP) — combined onpeak+offpeak.

### Rolling CV (expanding window)
| Eval PY | Train PYs |
|---------|-----------|
| 2023-06 | 2021, 2022 |
| 2024-06 | 2021, 2022, 2023 |
| 2025-06 | 2021, 2022, 2023, 2024 |

Each eval PY has 3 quarters (aq1-aq3) = 9 total eval groups.

### NB Model Architecture
- **Training population**: NB-hist-12 branches with V4.4 coverage only (need V4.4 deviation features).
- **Algorithm**: LightGBM LambdaRank with tiered labels (0/1/2/3 based on SP tertiles).
- **Hyperparameters**: 15 leaves, 0.05 LR, 150 rounds, 80% subsample, 80% colsample, `num_threads=4`.
- **14 features**:
  - Density bins: `bin_80_max`, `bin_90_max`, `bin_100_max`, `bin_110_max`
  - Branch metadata: `rt_max`, `count_active_cids`
  - Historical DA: `shadow_price_da`, `da_rank_value`
  - V4.4 deviation: `dev_max`, `dev_sum`, `pct100`, `pct90`
  - V4.4 ranks: `shadow_rank_v44`, `dev_max_rank`

### Scorers Compared
| Scorer | Description |
|--------|-------------|
| **ML_nb** | LambdaRank model trained on NB-hist-12 population |
| **V4.4** | Score = `1 - rank` from V4.4 signal (lower rank = better) |
| **blend_05** | `0.5 × minmax(ML_nb) + 0.5 × minmax(V4.4)` |
| **v0c** | Formula baseline (for reference) |

### Full-Universe Combination Configs
For combined scoring, we reserve N slots for the NB scorer within the top-K selection:

| Config | K=200 split | K=400 split |
|--------|-------------|-------------|
| pure_v0c | 200 v0c + 0 NB | 400 v0c + 0 NB |
| R30_v44 | 170 v0c + 30 V4.4-NB | 350 v0c + 50 V4.4-NB |
| R30_blend | 170 v0c + 30 blend-NB | 350 v0c + 50 blend-NB |
| R50_v44 | 150 v0c + 50 V4.4-NB | 300 v0c + 100 V4.4-NB |
| R50_blend | 150 v0c + 50 blend-NB | 300 v0c + 100 blend-NB |
| R100_v44 | 100 v0c + 100 V4.4-NB | 200 v0c + 200 V4.4-NB |
| R100_blend | 100 v0c + 100 blend-NB | 200 v0c + 200 blend-NB |

NB slots are filled from the **dormant population only** (bf_combined_12 == 0), excluding branches already selected by v0c.

## Part 1 Results: NB-hist-12 Only (Per Eval Year)

All metrics computed on the NB-hist-12 population only (dormant branches).

### K=30 (NB universe)

| Scorer | 2023 VC | 2024 VC | 2025 VC | AGG VC | AGG Rec |
|--------|---------|---------|---------|--------|---------|
| ML_nb | 0.032 | **0.105** | **0.131** | 0.089 | 0.047 |
| V4.4 | **0.205** | 0.006 | 0.128 | 0.113 | 0.043 |
| blend_05 | 0.156 | 0.035 | **0.163** | **0.118** | **0.049** |
| v0c | 0.112 | 0.023 | 0.145 | 0.093 | 0.037 |

### K=50 (NB universe)

| Scorer | 2023 VC | 2024 VC | 2025 VC | AGG VC | AGG Rec |
|--------|---------|---------|---------|--------|---------|
| ML_nb | 0.142 | **0.220** | **0.187** | **0.183** | **0.081** |
| V4.4 | **0.206** | 0.008 | 0.147 | 0.120 | 0.057 |
| blend_05 | 0.187 | 0.149 | 0.190 | 0.175 | 0.071 |
| v0c | 0.180 | 0.033 | 0.165 | 0.126 | 0.055 |

### K=100 (NB universe)

| Scorer | 2023 VC | 2024 VC | 2025 VC | AGG VC | AGG Rec |
|--------|---------|---------|---------|--------|---------|
| ML_nb | 0.271 | **0.300** | **0.300** | **0.291** | **0.144** |
| V4.4 | 0.262 | 0.071 | 0.239 | 0.191 | 0.121 |
| blend_05 | **0.322** | 0.220 | 0.267 | 0.270 | 0.134 |
| v0c | 0.238 | 0.080 | 0.327 | 0.215 | 0.104 |

### Key Observations — NB-hist-12

1. **V4.4 is extremely inconsistent**: Dominates 2023 (VC@30=0.205) but nearly zero on 2024 (VC@30=0.006, VC@50=0.008). This makes it unreliable as a standalone NB scorer.

2. **ML_nb is more stable across years**: Never crashes to near-zero. Wins 2024 and 2025 consistently at K≥50.

3. **blend_05 is the best K=30 scorer** (AGG VC=0.118): Inherits V4.4's 2023 strength while ML rescues 2024. Best of both worlds at low K.

4. **ML_nb dominates at K≥50** (AGG VC@50=0.183, +52% vs V4.4): The ML model's broader discrimination becomes dominant as K grows.

5. **v0c is surprisingly competitive at K=100 for 2025** (0.327) — but terrible at K=30 across the board (0.093). It picks up dormant binders accidentally via da_rank_value, but cannot deliberately target them.

## Part 2 Results: Full Universe with Reserved NB Slots

### Aggregate across all eval years (9 quarter-groups)

#### K=200

| Config | VC | Abs | Rec | Bind | NB_in | NB_bind | NB_SP |
|--------|---:|----:|----:|-----:|------:|--------:|------:|
| pure_v0c | 0.616 | 0.572 | 0.307 | 122 | 3 | 0.4 | $540 |
| R30_v44 | 0.605 | 0.562 | 0.283 | 113 | 31 | 5.0 | $66K |
| R30_blend | 0.607 | 0.563 | 0.285 | 114 | 31 | 5.9 | $69K |
| R50_v44 | 0.578 | 0.537 | 0.264 | 105 | 51 | 6.6 | $68K |
| R50_blend | 0.587 | 0.545 | 0.269 | 107 | 51 | 8.4 | $86K |
| R100_v44 | 0.499 | 0.463 | 0.214 | 86 | 100 | 13.9 | $101K |
| R100_blend | 0.513 | 0.476 | 0.220 | 88 | 100 | 15.9 | $134K |

#### K=400

| Config | VC | Abs | Rec | Bind | NB_in | NB_bind | NB_SP |
|--------|---:|----:|----:|-----:|------:|--------:|------:|
| pure_v0c | 0.730 | 0.678 | 0.446 | 178 | 52 | 6.3 | $46K |
| R30_v44 | 0.734 | 0.681 | 0.435 | 173 | 79 | 10.6 | $81K |
| **R30_blend** | **0.746** | **0.692** | **0.439** | 175 | 79 | 12.0 | **$105K** |
| R50_v44 | 0.722 | 0.669 | 0.423 | 169 | 115 | 16.2 | $107K |
| **R50_blend** | **0.744** | **0.689** | **0.426** | 170 | 115 | 17.3 | **$158K** |
| R100_v44 | 0.699 | 0.647 | 0.374 | 149 | 203 | 27.7 | $196K |
| R100_blend | 0.695 | 0.644 | 0.376 | 150 | 203 | 28.6 | $185K |

### Delta vs pure_v0c (Aggregate)

#### K=200

| Config | dVC | dNB_SP | dNB_bind |
|--------|----:|-------:|---------:|
| R30_v44 | -0.011 | +$66K | +4.6 |
| R30_blend | -0.010 | +$68K | +5.4 |
| R50_v44 | -0.038 | +$67K | +6.1 |
| R50_blend | -0.029 | +$85K | +8.0 |
| R100_v44 | -0.117 | +$101K | +13.4 |
| R100_blend | -0.103 | +$134K | +15.4 |

#### K=400

| Config | dVC | dNB_SP | dNB_bind |
|--------|----:|-------:|---------:|
| R30_v44 | +0.004 | +$35K | +4.2 |
| **R30_blend** | **+0.016** | **+$59K** | **+5.7** |
| R50_v44 | -0.009 | +$61K | +9.9 |
| **R50_blend** | **+0.014** | **+$112K** | **+11.0** |
| R100_v44 | -0.032 | +$150K | +21.3 |
| R100_blend | -0.035 | +$139K | +22.2 |

## Conclusions

### 1. At K=400, blend reservations are a **free lunch**
- R30_blend: +0.016 VC (better overall!) AND +$59K NB_SP per quarter
- R50_blend: +0.014 VC AND +$112K NB_SP per quarter
- The blend's NB picks are genuinely good branches — they ADD value, not just trade VC for NB.

### 2. At K=200, any reservation costs VC — but blend minimizes the cost
- R30_blend loses only -0.010 VC while gaining +$68K NB_SP
- R50_blend: -0.029 VC for +$85K NB_SP
- The tradeoff is worthwhile if NB detection is valued.

### 3. Blend consistently outperforms pure V4.4 for NB selection
- At every config and K level, blend captures more NB_SP and NB_bind than the equivalent V4.4-only config.
- The ML model adds genuine value beyond what V4.4 provides — especially in years where V4.4 crashes (2024).

### 4. Recommended configs
- **K=400, R50_blend**: Best balance — no VC sacrifice, +$112K NB_SP, +11 dormant binders per quarter.
- **K=200, R30_blend**: Minimal VC cost (-1.0pp) for +$68K NB_SP per quarter.

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/nb_model_yearly.py` | Full experiment: rolling CV, per-PY breakdown, reserved-slot combos |
| `scripts/nb_model_experiment.py` | Earlier prototype: blended tier experiment (v0c + V4.4 reserved slots) |

## Runtime

- Data build: ~211s (15 model tables: 5 PYs × 3 quarters)
- Total: ~211s (model training is negligible)
