# Stage 5 Experiment Setup: LTR Constraint Ranking

**Date:** 2026-03-08
**Predecessor:** Stage 4 (abandoned — circular evaluation)
**Working dir:** `/home/xyz/workspace/research-qianli-v2/research-stage4-tier/`

---

## 0. Glossary

| Term | Meaning |
|------|---------|
| f0 | First monthly period type — auction month = delivery month |
| onpeak | On-peak hours class type |
| constraint_id | String identifier for a transmission constraint (93% numeric MISO IDs like "72691", 7% SPICE-style like "1026FG") |
| flow_direction | Int64 (1 or -1) — direction of power flow on the constraint |
| shadow_price | $/MWh cost of congestion on a binding constraint |
| LTR | Learning-to-rank (LightGBM lambdarank) |
| V6.2B | The production constraint ranking signal (`TEST.TEST.Signal.MISO.SPICE_F0P_V6.2B.R1`) |
| spice6 | The density simulation model that produces flow forecasts and exceedance probabilities |
| query group | In LTR, one month's constraints form a single group — the model learns to rank within groups |

---

## 1. Problem Statement

Rank MISO FTR constraints by expected binding severity for a future auction month. The output is a tier signal (5 tiers) consumed by the trading workflow.

**Scope:** f0/onpeak only. One ranking per auction month, ~500-800 constraints.

---

## 2. Why Stage 4 Failed

Stage 4 used `shadow_price_da` from V6.2B parquet as **both**:
- **Feature source**: `da_rank_value = rank(shadow_price_da)` — 60% of the formula
- **Ground truth**: `y = shadow_price_da` — the evaluation target (`ml/pipeline.py` lines 52, 84)

This is circular: `Spearman(da_rank_value, shadow_price_da) = -1.0000` (exact inverse rank).

**The fix:** Use **realized DA shadow prices** for the delivery month as ground truth. These are fetched independently from MISO market data and are NOT available at prediction time.

### Circularity Evidence

```
Spearman(da_rank_value, shadow_price_da)  = -1.0000  # feature IS target (inverted rank)
Spearman(V6.2B_formula, shadow_price_da)  = +0.9125  # circular evaluation
Spearman(V6.2B_formula, realized_DA)      = +0.2045  # correct evaluation (12-mo mean)
```

### Clarification: shadow_price_da and da_rank_value Are NOT Leaky Features

`shadow_price_da` is a **60-month historical lookback** of DA shadow prices (up to the month before the auction). It is legitimate as a feature — it captures long-term binding propensity and correlates +0.37 with realized DA for the delivery month. Adjacent months share ~65-70% identical values.

`da_rank_value` = within-month percentile rank of shadow_price_da. Also legitimate as a feature.

**What was wrong:** Using shadow_price_da as the *evaluation target*. Not using it as a *feature*.

**For ML models:** da_rank_value (or shadow_price_da itself) should be AVAILABLE as a feature. Whether to include it is an experimental question (v1 vs v1b below).

### What Carries Forward from Stage 4

- **Hyperparams**: 8mo train / 0 val, LightGBM lambdarank, lr=0.05, 100 trees, 31 leaves
- **LightGBM >> XGBoost**: ~22x faster, similar quality
- **Spice6 density features help**: prob_exceed_* consistently adds value
- **Engineered features don't help**: LightGBM handles non-linear combinations internally. The `_add_engineered_features()` in data_loader.py (Tiers 1-9, ~37 derived features) added no value. Skip them.
- **Monotone constraints matter**: flow features +1, rank features -1, prob_exceed +1
- **Reusable code**: see Section 12

### WARNING: Registry State Is Corrupted

The entire `registry/` directory contains results from the circular evaluation:
- `registry/v0/metrics.json` — circular v0 (VC@20~0.50, Spearman~0.91). WRONG.
- `registry/gates.json` — gates calibrated from circular v0 (Spearman floor=0.82). UNUSABLE.
- `registry/v5/` through `registry/v9_screen/` — all invalid (trained/evaluated against shadow_price_da).
- `registry/champion.json` — points to circular v0.

**Action:** Before running any `compare.py` gate checks, you MUST either:
1. Delete the registry and rebuild from correct v0, or
2. Rewrite `scripts/run_v0_formula_baseline.py` to use realized DA, re-run it, and let it overwrite gates.json.

---

## 3. Data Landscape

### 3.1 V6.2B Parquet (Constraint Universe — Defines the Rows)

**Path:** `/opt/data/xyz-dataset/signal_data/miso/constraints/TEST.TEST.Signal.MISO.SPICE_F0P_V6.2B.R1/{month}/f0/onpeak`
**Coverage:** 106 months (2017-06 to 2026-03), ~450-780 constraints per month.

V6.2B is the row-defining dataset. All other features join onto it by `constraint_id`.

#### Full Schema

| Column | Type | Role | Notes |
|--------|------|------|-------|
| constraint_id | str | **key** | String — mostly numeric MISO IDs ("72691"), some SPICE-style ("1026FG") |
| flow_direction | i64 | **key** | 1 or -1 |
| mean_branch_max | f64 | **feature** | Max branch loading forecast |
| ori_mean | f64 | **feature** | Mean flow, baseline scenario |
| mix_mean | f64 | **feature** | Mean flow, mixed scenario |
| density_mix_rank_value | f64 | **feature** | Within-month percentile rank of mix flow (lower=more binding) |
| density_ori_rank_value | f64 | **feature** | Within-month percentile rank of ori flow (lower=more binding) |
| da_rank_value | f64 | **feature** | Within-month percentile rank of shadow_price_da (lower=more binding). Historical, NOT leaky. |
| shadow_price_da | f64 | **feature (raw)** | 60-month historical DA lookback sum. Can use directly or via da_rank_value. |
| rank_ori | f64 | derived | = 0.60*da_rank + 0.30*dmix + 0.10*dori |
| rank | f64 | derived | = dense_rank(rank_ori) normalized to [0,1] |
| tier | i64 | derived | = quintile(rank), 0-4 |
| shadow_sign | i64 | metadata | = -flow_direction |
| shadow_price | f64 | derived | = shadow_price_da * shadow_sign (signed version) |
| branch_name | str | metadata | |
| bus_key, bus_key_group | str | metadata | |
| equipment | str | metadata | = branch_name |
| mean_branch_max_fillna | f64 | redundant | = mean_branch_max with nulls filled |
| density_mix_rank | f64 | redundant | integer version of density_mix_rank_value |

#### V6.2B Formula (Verified — reproduces `rank` exactly, max_abs_diff=0 all 12 months)

```
rank_ori = 0.60 * da_rank_value + 0.30 * density_mix_rank_value + 0.10 * density_ori_rank_value
rank = dense_rank(rank_ori) / max(dense_rank(rank_ori))
tier = quintile(rank)   # 5 tiers, 20% each
```

Note: lower rank_value = more binding. Formula score is NEGATED for evaluation (higher score = more binding).

### 3.2 Spice6 Density (Extra Features — Same Constraint Space)

**Path:** `/opt/temp/tmp/pw_data/spice6/prod_f0p_model_miso/density/`
**Coverage:** 106 months (2017-06 to 2026-03)
**Overlap with V6.2B: 100%** (verified across 4 months)

Loaded by `ml/spice6_loader.py` — aggregates `score_df.parquet` (mean across outage dates) and `limit.parquet` per (constraint_id, flow_direction).

Directory structure: `density/auction_month={YYYY-MM}/market_month={YYYY-MM}/market_round=1/outage_date={YYYY-MM-DD}/score_df.parquet`

| Feature | Monotone | Description |
|---------|----------|-------------|
| prob_exceed_110 | +1 | P(flow > 110% of limit) |
| prob_exceed_100 | +1 | P(flow > 100% of limit) |
| prob_exceed_90 | +1 | P(flow > 90% of limit) |
| prob_exceed_85 | +1 | P(flow > 85% of limit) |
| prob_exceed_80 | +1 | P(flow > 80% of limit) |
| constraint_limit | 0 | MW thermal limit |

**Status: CLEAN.** These are SPICE6 model forecasts, not derived from realized outcomes.

### 3.3 Spice6 ML Predictions (New Features — Same Constraint Space)

**Path:** `/opt/temp/tmp/pw_data/spice6/prod_f0p_model_miso/ml_pred/`
**Coverage:** 92 months (2018-06 to 2026-01)
**Overlap with V6.2B: ~100%** (99.8%, verified across 4 months)

These are outputs from the V6.7B shadow price prediction pipeline (`research-spice-shadow-price-pred-qianli`). Per-constraint ML predictions available before the auction.

| Feature | Monotone | Spearman w/ shadow_price_da | Description |
|---------|----------|---------------------------|-------------|
| predicted_shadow_price | +1 | 0.507 | ML model's predicted shadow price |
| binding_probability | +1 | 0.406 | Probability of constraint binding |
| binding_probability_scaled | +1 | 0.508 | Scaled version of binding_probability |
| prob_exceed_95 | +1 | — | Extra threshold (not in density) |
| prob_exceed_105 | +1 | — | Extra threshold (not in density) |
| hist_da | +1 | **0.998** | = shadow_price_da (same data, redundant) |

**Key findings:**
- `predicted_shadow_price` and `binding_probability` are **genuinely new signals**. They correlate only ~0.5 and ~0.4 with shadow_price_da, and ~0.04-0.14 with flow features.
- `predicted_shadow_price` and `binding_probability` correlate 0.76 with each other (related but not redundant).
- `hist_da` ≈ `shadow_price_da` (Spearman 0.998). Redundant — use one or the other.

**DO NOT USE from ml_pred:**
- `actual_shadow_price` — IS the target (always 0 in ml_pred anyway)
- `actual_binding` — IS the target
- `abs_error`, `error` — require actuals
- `model_used` — metadata

### 3.4 Spice6 Constraint Info (Structural Features)

**Path:** `/opt/temp/tmp/pw_data/spice6/prod_f0p_model_miso/constraint_info/`
**Coverage:** 106 months, **Overlap with V6.2B: 100%**

Contains structural/network information per constraint: `base_case_limit`, `contingency_case_limit`, `rate_a/b/c`, `B/R/X` (impedance), `factor`, `is_monitored`, `type`, `device_type`. 33K rows (full network), V6.2B's ~500-800 are a strict subset.

### 3.5 V6.4B Signal (Different Constraint Universe — DO NOT USE)

**Path:** `/opt/data/xyz-dataset/signal_data/miso/constraints/TEST.TEST.Signal.MISO.SPICE_F0P_V6.4B.R1/`
**Overlap with V6.2B: only 23%.** Different constraint selection. Not usable.

---

## 4. Constraint Space Summary

| Data Source | Constraints (2022-06) | Overlap with V6.2B | Usable? |
|------------|----------------------|-------------------|---------|
| **V6.2B** (rows) | 578 | — | Defines the universe |
| spice6 density | 14,282 | **100%** | Yes — extra features |
| spice6 ml_pred | 13,158 | **99.8%** | Yes — extra features (92mo) |
| spice6 constraint_info | 14,361 | **100%** | Yes — structural features |
| V6.4B | 891 | **23%** | No — different universe |

---

## 5. Feature Set

All features must be available **before the auction** for month M.

### Group A: V6.2B Flow Forecasts (5 features, 106 months)

| # | Feature | Monotone | Source |
|---|---------|----------|--------|
| 1 | mean_branch_max | +1 | V6.2B parquet |
| 2 | ori_mean | +1 | V6.2B parquet |
| 3 | mix_mean | +1 | V6.2B parquet |
| 4 | density_mix_rank_value | -1 | V6.2B parquet |
| 5 | density_ori_rank_value | -1 | V6.2B parquet |

### Group B: Spice6 Density (6 features, 106 months)

| # | Feature | Monotone | Source |
|---|---------|----------|--------|
| 6 | prob_exceed_110 | +1 | density/score_df |
| 7 | prob_exceed_100 | +1 | density/score_df |
| 8 | prob_exceed_90 | +1 | density/score_df |
| 9 | prob_exceed_85 | +1 | density/score_df |
| 10 | prob_exceed_80 | +1 | density/score_df |
| 11 | constraint_limit | 0 | density/limit |

### Group C: Historical DA Signal (1-2 features, 106 months)

| # | Feature | Monotone | Source |
|---|---------|----------|--------|
| 12 | da_rank_value | -1 | V6.2B parquet (historical 60mo lookback, NOT leaky) |
| 13 | shadow_price_da | +1 | V6.2B parquet (raw value, alternative to da_rank_value) |

### Group D: ML Predictions (2-3 features, 92 months)

| # | Feature | Monotone | Source |
|---|---------|----------|--------|
| 14 | predicted_shadow_price | +1 | ml_pred/final_results |
| 15 | binding_probability | +1 | ml_pred/final_results |
| 16 | binding_probability_scaled | +1 | ml_pred/final_results |

### Group E: Custom Historical DA (4 features, computed via Ray)

| # | Feature | Monotone | Source |
|---|---------|----------|--------|
| 17 | hist_da_recent | +1 | get_da_shadow_by_peaktype(), 3-month lookback |
| 18 | hist_da_season_1 | +1 | Same-season 1 year ago |
| 19 | hist_da_season_2 | +1 | Same-season 2 years ago |
| 20 | hist_da_season_3 | +1 | Same-season 3 years ago |

Temporal cutoff following V6.7B pattern:
```python
cutoff_month = auction_month - MonthBegin(1)  # strict cutoff BEFORE auction
recent_months = [cutoff-1, cutoff-2, cutoff-3]
season_months = same_season_1yr_ago, 2yr_ago, 3yr_ago
```

### Columns to NEVER Use as Features

| Column | Source | Why |
|--------|--------|-----|
| rank_ori, rank, tier | V6.2B | Derived outputs (formula result, not input) |
| shadow_sign, shadow_price | V6.2B | Derived/metadata |
| hist_da | ml_pred | = shadow_price_da (Spearman 0.998), redundant |
| actual_shadow_price | ml_pred | IS the target |
| actual_binding | ml_pred | IS the target |

---

## 6. Ground Truth

### What

The **realized DA constraint shadow price** for month M, fetched independently from MISO market data.

### How

```python
import os
os.environ["RAY_ADDRESS"] = "ray://10.8.0.36:10001"
from pbase.config.ray import init_ray
import pmodel
init_ray(extra_modules=[pmodel])

import pandas as pd
from pbase.analysis.tools.all_positions import MisoApTools
aptools = MisoApTools()

# Fetch realized DA shadow prices for the TARGET month
st = pd.Timestamp("2022-06-01")
et = pd.Timestamp("2022-07-01")
da_shadow = aptools.tools.get_da_shadow_by_peaktype(st=st, et_ex=et, peak_type="onpeak")

# Aggregate per constraint: absolute sum of shadow prices for the month
# Using abs(sum) — net directional shadow price magnitude
realized = da_shadow.groupby("constraint_id")["shadow_price"].sum().reset_index()
realized.columns = ["constraint_id", "realized_sp"]
realized["realized_sp"] = realized["realized_sp"].abs()
```

**Note on aggregation:** We use `abs(sum(shadow_price))` not `sum(abs(shadow_price))`. The sum preserves directional netting within the month, then abs gives the magnitude. This aligns with how shadow_price_da is computed in the V6.2B signal.

### Alignment and Constraint ID Mapping (Verified 2026-03-08)

- V6.2B: ~450-780 constraints per month
- Realized DA: ~317 unique binding constraints per month (across full MISO network)
- Of those 317, only ~68 overlap with V6.2B's ~578 (**11-13% of V6.2B universe binds**)
- Non-binding V6.2B constraints get `realized_sp = 0`
- Realized DA constraints not in V6.2B are ignored (outside model universe)
- Join: `V6.2B.merge(realized, on="constraint_id", how="left")`

#### No Bridge Table Needed (Unlike Annual V6.1)

The annual agent (V6.1) requires a bridge table (`MISO_SPICE_CONSTRAINT_INFO`) because V6.1 uses `branch_name` as the join key and the naming format differs between DA and V6.1. **We do NOT have this problem:**

| | Annual (V6.1) | Us (V6.2B) |
|---|---|---|
| **Join key** | `branch_name` (format mismatch) | `constraint_id` (direct match) |
| **DA format** | Long: `"BARKRC BARKRC_BOGLSA6 A (LN/CLEC/EES)"` | Numeric: `"225824"` |
| **Signal format** | Short: `"ALTIMX_WODWRD3 A"` | Numeric: `"225824"` (93%) or SPICE: `"1026FG"` (7%) |
| **Needs bridge?** | YES | NO |

V6.2B has 540/578 (93%) numeric MISO constraint_ids → direct match with DA. The 38 FG-suffix constraints (7%) have SPICE-style IDs that can't match DA directly. These get `realized_sp = 0` (treated as non-binding). If this becomes a concern, we could investigate mapping FG IDs through constraint_info, but 7% is small and FG constraints may simply not appear in DA.

**Key insight:** Most constraints (87-89%) don't bind. The ranking task is primarily about separating the ~68 binding constraints from the ~500+ non-binding ones, and ranking the binding ones by severity.

---

## 7. Pipeline Architecture

```
For eval_month M (e.g., "2022-06"):

  ROWS: V6.2B constraint universe for month M (~500-800 constraints)
         Loaded from: SPICE_F0P_V6.2B.R1/{M}/f0/onpeak

  FEATURES per row (all available before M):
  ├── Group A: V6.2B flow forecasts (5 cols from same parquet)
  ├── Group B: spice6 density (6 cols joined by constraint_id+flow_direction)
  ├── Group C: da_rank_value (1 col from same parquet, historical)
  ├── Group D: ml_pred predictions (2-3 cols joined by constraint_id)
  └── Group E: custom hist DA (4 cols, computed from Ray lookback)

  GROUND TRUTH for month M:
  └── realized_sp = abs(sum(DA shadow prices during month M))
      Fetched via get_da_shadow_by_peaktype(st=M_start, et_ex=M+1_start)
      Joined to V6.2B by constraint_id. Missing = 0.

  TRAINING (for ML models, not v0):
  ├── For months M-8 through M-1:
  │   ├── Features: V6.2B + spice6 + ml_pred for that month
  │   └── Labels: realized DA for that SAME month (not M!)
  │       Each training month's labels = actual binding in that month
  ├── Each month = one query group
  └── LightGBM lambdarank learns to rank within groups

  EVALUATION:
  └── Month M: model predicts scores, compare ranking vs realized DA ranking
```

### Training Data Construction (Detailed)

For eval_month `M = "2022-06"` with 8-month training:

| Training month | Features from | Labels from | Query group |
|----------------|--------------|-------------|-------------|
| 2021-10 | V6.2B/2021-10 + spice6/2021-10 | realized DA Oct 2021 | group 1 |
| 2021-11 | V6.2B/2021-11 + spice6/2021-11 | realized DA Nov 2021 | group 2 |
| ... | ... | ... | ... |
| 2022-05 | V6.2B/2022-05 + spice6/2022-05 | realized DA May 2022 | group 8 |

Test month: features from V6.2B/2022-06 + spice6/2022-06, labels from realized DA Jun 2022.

**Critical: each training month needs its own realized DA labels.** This requires fetching realized shadow prices for every training month, not just the eval month. Cache these to parquet.

---

## 8. v0 Baseline: V6.2B Formula vs Realized DA — DONE

V6.2B formula ranking evaluated against realized DA shadow prices:

```
score = -(0.60 * da_rank_value + 0.30 * density_mix_rank_value + 0.10 * density_ori_rank_value)
```
(Negated because lower rank_value = more binding, but evaluation expects higher score = better)

**Verified:** Formula reproduces stored `rank` column exactly (`max_abs_diff=0`, all 12 months).
**Code:** `ml/v62b_formula.py`, `ml/reproduce_v62b.py`

**WARNING about `scripts/run_v0_formula_baseline.py`:** This script evaluates against `shadow_price_da` (line 32: `actual = df["shadow_price_da"]`), NOT realized DA. It produces the CIRCULAR v0 numbers. The correct v0 results below were obtained via ad-hoc inline evaluation against realized DA fetched through Ray. This script must be rewritten to use cached realized DA before it can be trusted.

### Aggregate Results (12-month eval)

| Metric | Mean | Min | Max |
|--------|------|-----|-----|
| VC@20 | 0.2817 | 0.0251 | 0.6023 |
| VC@100 | 0.6008 | 0.2876 | 0.8155 |
| Recall@20 | 0.1833 | 0.0000 | 0.3500 |
| Recall@50 | 0.2300 | 0.1400 | 0.3600 |
| Recall@100 | 0.2392 | 0.1600 | 0.3200 |
| Spearman | 0.2045 | 0.1303 | 0.2658 |

### Per-Month v0 Results

| Month | n_rows | n_binding | VC@20 | VC@100 | R@20 | R@50 | R@100 | Spearman |
|-------|--------|-----------|-------|--------|------|------|-------|----------|
| 2020-09 | 715 | 67 | 0.4358 | 0.8085 | 0.2000 | 0.2600 | 0.2100 | 0.2002 |
| 2020-12 | 550 | 72 | 0.4166 | 0.8155 | 0.3500 | 0.3600 | 0.3200 | 0.2606 |
| 2021-03 | 669 | 69 | 0.1849 | 0.7075 | 0.1500 | 0.2400 | 0.2700 | 0.1923 |
| 2021-06 | 556 | 68 | 0.2887 | 0.5309 | 0.2000 | 0.2600 | 0.2600 | 0.2658 |
| 2021-09 | 621 | 68 | 0.6023 | 0.7011 | 0.3500 | 0.2200 | 0.2000 | 0.1992 |
| 2021-12 | 448 | 50 | 0.3363 | 0.6551 | 0.1000 | 0.1400 | 0.1600 | 0.1577 |
| 2022-03 | 717 | 84 | 0.3434 | 0.6001 | 0.2000 | 0.1400 | 0.2400 | 0.2090 |
| 2022-06 | 578 | 68 | 0.0509 | 0.4870 | 0.0500 | 0.2200 | 0.2200 | 0.1992 |
| 2022-09 | 600 | 83 | 0.1475 | 0.4777 | 0.1500 | 0.1800 | 0.3100 | 0.2198 |
| 2022-12 | 525 | 70 | 0.0251 | 0.4140 | 0.0000 | 0.2200 | 0.1900 | 0.1303 |
| 2023-03 | 780 | 81 | 0.1444 | 0.2876 | 0.2000 | 0.2200 | 0.2000 | 0.1637 |
| 2023-05 | 655 | 85 | 0.4041 | 0.7247 | 0.2500 | 0.3000 | 0.2900 | 0.2566 |

**Key observations:**
- High variance in VC@20 (0.03 to 0.60) — some months the formula captures the top binding constraints, some months it completely misses
- Spearman consistently low (0.13-0.27) — the formula has weak correlation with realized binding
- VC@100 = 0.60 is decent — the top-100 captures most binding value, but that's ~18% of universe
- This is the correct difficulty level: predicting future binding from forecasts is genuinely hard

---

## 9. Versioning Plan

### v0: V6.2B Formula Baseline — DONE
See Section 8.

### v1: LTR with Pure Forecasts (11 features, Groups A+B)

Features: mean_branch_max, ori_mean, mix_mean, density_mix_rank_value, density_ori_rank_value + prob_exceed_110/100/90/85/80 + constraint_limit.
No historical DA signal. No ML predictions.
LightGBM lambdarank, 8mo train / 0 val, lr=0.05, 100 trees, 31 leaves.

**Purpose:** Can pure flow forecasts and density probabilities predict realized binding?

### v1b: v1 + Historical DA Signal (12-13 features, Groups A+B+C)

Add da_rank_value (and/or shadow_price_da) to v1.

**Purpose:** Does the 60-month historical binding signal help predict next-month binding? This is the most important question — da_rank_value is 60% of the V6.2B formula.

### v2: v1b + Custom Historical DA (16-17 features, Groups A+B+C+E)

Add independently computed hist_da_recent, hist_da_season_1/2/3.

**Purpose:** Does structured temporal historical DA (recent vs seasonal) beat the raw 60-month aggregate?

### v3: v2 + ML Predictions (18-20 features, Groups A+B+C+D+E)

Add predicted_shadow_price, binding_probability from ml_pred.

**Purpose:** Do V6.7B ML predictions add signal beyond raw features?
**Note:** Coverage drops from 106 to 92 months. Restrict eval to months where all 8 training months have ml_pred.

### v4+: Engineering & Tuning

Only after v1-v3 established. Ideas:
- Structural features from constraint_info
- Hyperparameter search
- Monotone constraint tuning

---

## 10. Evaluation Framework

### Metrics

| Metric | Group | Definition |
|--------|-------|------------|
| VC@20 | A (blocking) | Value captured by model's top-20 / total value |
| VC@100 | A | Same for top-100 |
| Recall@20 | A | Overlap of model's top-20 with true top-20 |
| Recall@50 | A | Same for top-50 |
| Recall@100 | A | Same for top-100 |
| NDCG | A | Normalized discounted cumulative gain |
| Spearman | B (monitor) | Rank correlation |
| Tier0-AP | B | Average precision for top-20% constraints |

### Gate System (3-layer, MUST be recalibrated)

| Layer | Formula | Purpose |
|-------|---------|---------|
| L1 Mean | `mean(metric) >= 0.9 * v0_mean` | Basic quality floor |
| L2 Tail | `count(months < v0_min) <= 1` | No catastrophic months |
| L3 Bot2 | `bot2_mean(new) >= bot2_mean(v0) - 0.02` | Worst months don't regress |

**WARNING: `registry/gates.json` is from the CIRCULAR v0 and is UNUSABLE.** It has Spearman floor=0.82 but the correct v0 Spearman is 0.20. If you run `compare.py` against these gates, everything will fail. You MUST recalibrate gates from the correct v0 numbers (Section 8) before using the gate system.

Similarly, `registry/v0/metrics.json` contains the circular v0 metrics (VC@20~0.50, Spearman~0.91). These are WRONG. The correct v0 metrics are in Section 8 of this document.

### Eval Months

| Tier | Months | Use |
|------|--------|-----|
| Screen | 4 (2020-12, 2021-09, 2022-06, 2023-03) | Fast hypothesis test |
| Eval | 12 (quarterly 2020-09 to 2023-05) | Primary iteration |
| Full | 36 (2020-06 to 2023-05) | Final validation |

### Training Setup

- Rolling window: for eval month M, train on M-8 to M-1 (8 months)
- No validation split (val_months=0)
- Each month = one query group
- Ground truth labels per training month = realized DA for that month

---

## 11. Data Paths

| Data | Path | Ray? | Coverage |
|------|------|------|----------|
| V6.2B signal | `/opt/data/xyz-dataset/signal_data/miso/constraints/TEST.TEST.Signal.MISO.SPICE_F0P_V6.2B.R1/{month}/f0/onpeak` | No | 106 mo |
| Spice6 density | `/opt/temp/tmp/pw_data/spice6/prod_f0p_model_miso/density/` | No | 106 mo |
| Spice6 ml_pred | `/opt/temp/tmp/pw_data/spice6/prod_f0p_model_miso/ml_pred/` | No | 92 mo |
| Spice6 constraint_info | `/opt/temp/tmp/pw_data/spice6/prod_f0p_model_miso/constraint_info/` | No | 106 mo |
| Realized DA | `MisoApTools().tools.get_da_shadow_by_peaktype()` | **Yes** | full history |

### Ray Initialization (Required for Realized DA)

```python
import os
os.environ["RAY_ADDRESS"] = "ray://10.8.0.36:10001"
from pbase.config.ray import init_ray
import pmodel
init_ray(extra_modules=[pmodel])
```

Python venv: `/home/xyz/workspace/pmodel/.venv/bin/python`
Activate: `cd /home/xyz/workspace/pmodel && source .venv/bin/activate`

---

## 12. Reusable Code from Stage 4

**Location:** `/home/xyz/workspace/research-qianli-v2/research-stage4-tier/ml/`

| Module | Action | Notes |
|--------|--------|-------|
| `evaluate.py` | **Reuse as-is** | VC@k, Recall@k, NDCG, Spearman, Tier0-AP. All correct. |
| `compare.py` | **Reuse as-is** | 3-layer gate system, markdown tables |
| `train.py` | **Reuse as-is** | LightGBM lambdarank + XGBoost fallback. Label rank-transform included. |
| `features.py` | **Reuse as-is** | prepare_features (extracts numpy from polars), compute_query_groups |
| `spice6_loader.py` | **Reuse as-is** | Loads density/score_df, aggregates prob_exceed by (constraint_id, flow_direction) |
| `v62b_formula.py` | **Reuse as-is** | v62b_score(), dense_rank_normalized(), v62b_rank_from_columns() |
| `benchmark.py` | **Reuse as-is** | Multi-month benchmark runner with screen/eval/full modes |
| `config.py` | **Modify** | Fix misleading comment (line 18 says "da_rank_value is pure leakage" — WRONG). Add da_rank_value to Group C feature list. Fix PipelineConfig defaults to train_months=8, val_months=0. |
| `data_loader.py` | **Rewrite** | Replace shadow_price_da ground truth with realized DA. Remove _add_engineered_features() (37 useless derived features). Add realized DA fetching + caching. |
| `pipeline.py` | **Rewrite** | Lines 52 (y_train), 59 (y_val), 84 (actual_sp): all use `shadow_price_da` → change to `realized_sp`. |
| `scripts/run_v0_formula_baseline.py` | **Rewrite** | Currently evaluates against shadow_price_da (CIRCULAR). Must be rewritten to use cached realized DA. |
| `baseline_v62b.py` | **Delete** | Replaced by `scripts/run_v0_formula_baseline.py` |

### Key Code Locations

| What | File | Line/Function |
|------|------|---------------|
| v0 formula evaluation | `scripts/run_v0_formula_baseline.py` | `evaluate_formula_month()` |
| Formula reproduction | `ml/v62b_formula.py` | `v62b_score()`, `v62b_rank_from_columns()` |
| LightGBM training | `ml/train.py` | `_train_lightgbm()` |
| Label rank transform | `ml/train.py:16` | `_rank_transform_labels()` |
| All LTR metrics | `ml/evaluate.py` | `evaluate_ltr()` |
| Spice6 loading | `ml/spice6_loader.py` | `load_spice6_density()` |
| Feature extraction | `ml/features.py` | `prepare_features()` |
| Query group computation | `ml/features.py` | `compute_query_groups()` |
| Benchmark runner | `ml/benchmark.py` | `run_benchmark()` |
| Gate comparison | `ml/compare.py` | `check_gates_multi_month()` |
| Config/features | `ml/config.py` | `LTRConfig`, `PipelineConfig`, `_LEAKY_FEATURES` |

---

## 13. Key Risks

1. **Sparse ground truth**: Only 11-13% of constraints bind. Most labels are 0.
   - *Mitigation*: VC@k and Recall@k handle sparse relevance. LTR lambdarank is designed for this.

2. **Slow ground truth fetching**: `get_da_shadow_by_peaktype()` via Ray takes ~5-10s per month.
   - *Mitigation*: Cache all realized DA to local parquet upfront (one-time cost: ~3-5 min for 36 months).

3. **Lower absolute performance**: v0 Spearman = 0.20 vs stage 4's 0.91. This is correct.
   - *Context*: Random would give ~0. v0 = 0.20 means historical signal has modest predictive power.

4. **ml_pred coverage gap**: starts 2018-06 (92 months) vs V6.2B from 2017-06 (106 months).
   - *Mitigation*: For v3, restrict eval to months where all 8 training months have ml_pred (eval >= 2019-02).

5. **Feature leakage traps**: Stage 4 already fell into this once.
   - *Mitigation*: Checklist before every experiment: (1) Is any feature derived from the delivery month's data? (2) Is the ground truth independent of all features?

---

## 14. Recommended Steps for Next AI

### Step 0: Fix Registry and v0 Script (Priority: CRITICAL)

The registry and v0 script are corrupted (circular evaluation). Before anything else:

1. Rewrite `scripts/run_v0_formula_baseline.py` to use cached realized DA instead of `shadow_price_da`
2. Re-run it to overwrite `registry/v0/metrics.json`, `registry/gates.json`, and `registry/champion.json`
3. Verify the new v0 numbers match Section 8 of this document

This depends on Step 1 (caching realized DA), so do Step 1 first, then come back to Step 0.

### Step 1: Cache Realized DA Ground Truth (Priority: HIGH)

Fetch and cache realized DA shadow prices for ALL months needed (training + eval):
- Eval months: 12 months (2020-09 to 2023-05)
- Training months: 8 months before each eval month
- Total unique months needed: ~2017-01 to 2023-05 (~77 months)

Save to: `data/realized_da/{month}.parquet` with columns `[constraint_id, realized_sp]`.

This is a one-time operation requiring Ray. After caching, the rest of the pipeline runs without Ray.

### Step 2: Modify data_loader.py

1. Remove `_add_engineered_features()` entirely (37 useless features).
2. Add `load_realized_da(month)` that reads from cached parquet.
3. In `load_train_val_test()`, replace the ground truth column:
   - OLD: `shadow_price_da` from V6.2B parquet
   - NEW: `realized_sp` from cached realized DA (joined by constraint_id)
4. Keep spice6 enrichment as-is.

### Step 3: Modify pipeline.py

1. Line 52: `y_train = train_df["realized_sp"]` (was `shadow_price_da`)
2. Line 59: `y_val = val_df["realized_sp"]` (was `shadow_price_da`)
3. Line 84: `actual_sp = test_df["realized_sp"]` (was `shadow_price_da`)
4. Everything else stays the same.

**Also:** `PipelineConfig` defaults are `train_months=6, val_months=2`, but Stage 4 found `train_months=8, val_months=0` is optimal. Override defaults when creating PipelineConfig.

### Step 4: Update config.py

1. Fix misleading comment on line 17-18: "The 60% da_rank_value component is pure leakage" → should say "da_rank_value is a historical 60-month lookback, legitimate as a feature."
2. `_LEAKY_FEATURES` is already correct (da_rank_value is NOT in it). No change needed.
3. Add `da_rank_value` to a new feature group (Group C in this doc) — currently it's not in any feature list.
4. Change `PipelineConfig` defaults: `train_months=8, val_months=0` (currently 6, 2).

### Step 5: Run v1 (11 features, Groups A+B, no historical DA)

```python
features = ["mean_branch_max", "ori_mean", "mix_mean",
            "density_mix_rank_value", "density_ori_rank_value",
            "prob_exceed_110", "prob_exceed_100", "prob_exceed_90",
            "prob_exceed_85", "prob_exceed_80", "constraint_limit"]
monotone = [1, 1, 1, -1, -1, 1, 1, 1, 1, 1, 0]
```

Run on screen (4 months) first. If promising, run eval (12 months).

### Step 6: Run v1b (12 features, Groups A+B+C, add da_rank_value)

Add `da_rank_value` with monotone -1. Compare vs v1 to measure historical DA signal value.

### Step 7: Continue with v2, v3

See Section 9 versioning plan.

### Step 8: Recalibrate Gates

After v1/v1b, recalibrate gates from whichever is the new champion (or keep v0 if nothing beats it).

---

## Appendix A: Stage 4 Results (Invalid — For Reference Only)

### v6/v7 (da_rank_value as feature, shadow_price_da as GT — CIRCULAR)

| Metric | v0 (formula) | v6 (ML 6feat) | v7 (ML 12feat) |
|--------|-------------|---------------|----------------|
| VC@20 | 0.5169 | 0.6089 | 0.5854 |
| VC@100 | 0.8240 | 0.8485 | 0.8800 |
| Recall@20 | 0.6917 | 0.9125 | 0.8125 |
| Recall@50 | 0.7500 | 0.8983 | 0.9250 |
| Recall@100 | 0.7833 | 0.7617 | 0.8925 |
| NDCG | 0.9370 | 0.9666 | 0.9465 |
| Spearman | 0.3809 | 0.4512 | 0.5150 |
| Tier0-AP | 0.7543 | 0.8087 | 0.9239 |

**Invalid** — model had rank(target) as feature, evaluated against target. Spearman 0.91 was artifact.

### Circularity Proof

```
Spearman(da_rank_value, shadow_price_da) = -1.0000  # EXACT inverse rank of target
Spearman(V6.2B_formula, shadow_price_da)  = +0.9125  # circular: feature IS target
Spearman(V6.2B_formula, realized_DA)      = +0.2045  # correct: 12-month mean
```
