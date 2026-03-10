# PJM V7.0 Constraint Tier — Comprehensive Handoff for AI Agent

## 1. Goal

Build a **V7.0b constraint tier system for PJM**, analogous to MISO's V7.0. The V7.0 hybrid signal replaces the V6.2B formula with ML-scored tiers for f0 and f1, while passing through V6.2B unchanged for f2–f11.

**Scope of changes**: only `f0` and `f1` get ML scoring. Everything else (`f2`–`f11`) is V6.2B passthrough.

---

## 2. Background: What MISO V7.0 Did

MISO's V7.0 (built in `research-stage5-tier/`) proved that a LightGBM LambdaRank model with 9 features beats the V6.2B formula by +43–92% on holdout VC@20 across f0/f1 × onpeak/offpeak slices. The approach:

1. **V6.2B formula baseline** (v0): `rank_ori = 0.60*da_rank_value + 0.30*density_mix_rank_value + 0.10*density_ori_rank_value`
2. **ML model** (v2): LightGBM LambdaRank with tiered labels, 9 features including binding frequency windows and spice6 density
3. **Hybrid signal**: f0/f1 = ML-scored, f2+ = V6.2B passthrough, all under one signal name

PJM follows the **exact same architecture** with adaptations for PJM's market structure.

---

## 3. PJM vs MISO: Key Differences

| Dimension | MISO | PJM |
|-----------|------|-----|
| **Class types** | onpeak, offpeak (2) | onpeak, dailyoffpeak, wkndonpeak (3) |
| **Period types** | f0, f1, f2, f3 (4) | f0–f11 (up to 12, varies by month) |
| **ML slices** | 4 (f0×2 + f1×2) | **6** (f0×3 + f1×3) |
| **Passthrough slices** | f2–f3 | f2–f11 |
| **constraint_id format** | Mostly numeric MISO IDs (93%), some SPICE-style | Compound `"monitored:contingency"` format |
| **DA shadow price join** | Direct on constraint_id | Split constraint_id on `":"` → first part = monitored_facility |
| **SO_MW_Transfer exception** | Yes (tier forced to 0) | **NO** — not applicable to PJM |
| **Spice6 wkndonpeak** | N/A (MISO has no wkndonpeak) | **Missing** — spice6 ml_pred has onpeak + dailyoffpeak only |
| **Signal names (pmodel)** | `Signal.MISO.DA_V7.16` (f0), `Signal.MISO.DZ_F0P_V5` (f1+) | `Signal.PJM.DA_V7.16` (f0), `Signal.PJM.DZ_F0P_V5` (f1+) |

### Critical: wkndonpeak spice6 gap

PJM's spice6 density data at `/opt/temp/tmp/pw_data/spice6/prod_f0p_model_pjm/` has `ml_pred/final_results.parquet` with class types **onpeak** and **dailyoffpeak** only. There is **no wkndonpeak** spice6 data. This means:

- For wkndonpeak ML slices, spice6 features (`prob_exceed_110`, `constraint_limit`, etc.) will be all zeros
- You may need to evaluate whether wkndonpeak benefits from ML at all, or if it should remain V6.2B passthrough
- Alternatively, investigate if onpeak spice6 can proxy for wkndonpeak (both are peak-hour related)

---

## 4. Data Sources and Paths

### 4.1 V6.2B Signal Data

```
/opt/data/xyz-dataset/signal_data/pjm/constraints/TEST.TEST.Signal.PJM.SPICE_F0P_V6.2B.R1/
```

**Structure**: `{auction_month}/{period_type}/{class_type}` — each leaf is a parquet file.

**Columns** (21, identical to MISO):
```
constraint_id, flow_direction, shadow_price_da, da_rank_value,
density_ori_value, density_ori_rank_value, density_mix_value, density_mix_rank_value,
ori_mean, ori_median, ori_max, ori_min, ori_value_count,
mix_mean, mix_median, mix_max, mix_min, mix_value_count,
rank_ori, rank, tier
```

**Data range**: 2017-06 to 2026-03 (varies by period type and month).

**Period type availability** (varies by auction month):
- May: f0 only
- June: f0–f11 (all 12)
- July–April: varies (generally f0–f3 always present, higher fN sparse)

**Class types**: onpeak, dailyoffpeak, wkndonpeak (always all 3 when period type exists).

### 4.2 Realized DA Shadow Prices (Ground Truth)

**API**:
```python
from pbase.data.pjm.ap_tools import PjmApTools

tools = PjmApTools().tools
df = tools.get_da_shadow_by_peaktype(start_date, end_date_exclusive, peak_type)
```

**Peak type values**: `onpeak`, `dailyoffpeak`, `wkndonpeak` (from `pbase.data.pjm.enums.ClassType`).

**Returned columns**: `monitored_facility`, `contingency_facility`, `shadow_price`, `constraint_full`, plus date columns.

**Aggregation**: Sum `shadow_price` over the delivery month by `(monitored_facility, peak_type)` to get the realized shadow price per constraint.

### 4.3 constraint_id ↔ DA Shadow Price Join

**This is PJM's biggest structural difference from MISO.**

V6.2B `constraint_id` is a compound string like `"MONITORED_NAME:CONTINGENCY_NAME"`. DA shadow prices use `monitored_facility` as the key.

**Join strategy**:
```python
# Extract monitored_facility from V6.2B constraint_id
df = df.with_columns(
    pl.col("constraint_id").str.split(":").list.first().alias("monitored_facility")
)
# Join to DA shadow prices on monitored_facility
df = df.join(realized_da, on="monitored_facility", how="left")
```

**Verified**: 98.8% overlap (499/505 monitored_facilities match between V6.2B and DA data).

**Important**: Multiple V6.2B rows can share the same `monitored_facility` (different contingencies). When joining realized DA (which is per monitored_facility), all rows with the same monitored_facility get the same realized shadow price. This is correct — the realized DA shadow price is a property of the monitored facility, not the contingency.

### 4.4 Spice6 Density Data

```
/opt/temp/tmp/pw_data/spice6/prod_f0p_model_pjm/
```

**Structure**: `{market_month}/` containing:
- `density/{class_type}/` — density parquets
- `score.parquet` — columns: `score`, `constraint_id`, `flow_direction`
- `ml_pred/final_results.parquet` — columns include: `prob_exceed_80`, `prob_exceed_85`, `prob_exceed_90`, `prob_exceed_100`, `prob_exceed_110`, `actual_shadow_price`, `predicted_shadow_price`, `binding_probability`, `constraint_limit`

**market_month** = delivery month (= auction_month + N for period type fN).

**Class types in ml_pred**: onpeak, dailyoffpeak only (**no wkndonpeak**).

### 4.5 Scale Factor (SF) Data

```
/opt/data/xyz-dataset/signal_data/pjm/sf/MANUAL.TEST.TEST.Signal.PJM.SPICE_F0P_V6.2B.R1/
```

Only a MANUAL version exists (sparse). This is informational — V7.0 does not modify SF logic.

---

## 5. V6.2B Formula (Verified Exact for PJM)

```python
rank_ori = 0.60 * da_rank_value + 0.30 * density_mix_rank_value + 0.10 * density_ori_rank_value
```

**Rank computation**:
```python
def dense_rank_normalized(values):
    """Lower rank_ori → lower rank value → more binding."""
    unique_sorted = np.sort(np.unique(values))
    dense_rank = np.searchsorted(unique_sorted, values) + 1  # 1..K
    K = int(dense_rank.max())  # number of unique dense ranks, NOT row count
    return dense_rank / K
```

**Tier computation** (V6.2B):
```python
tier = np.clip(np.floor(rank * 5).astype(int), 0, 4)
```

Tiers 0–4 where 0 = most binding (lowest rank_ori).

**Verified**: `max_abs_diff = 0.0` for all test months. Formula is identical to MISO's.

**PJM has NO SO_MW_Transfer exception** (that's MISO-specific where certain constraints are forced to tier 0).

---

## 6. V7.0 Tier Computation (ML Slices)

For f0/f1 ML-scored slices, V7.0 uses **row-percentile tiering with V6.2B tie-breaking**:

1. ML model produces a score per constraint (higher = more binding)
2. Sort by ML score descending; break ties using V6.2B `rank_ori` (lower rank_ori = more binding)
3. Assign tier based on row percentile: `tier = ceil(row_position / total_rows * 5) - 1`, clamped to [0, 4]

This means tier 0 gets the top ~20% of constraints, tier 1 the next ~20%, etc.

**rank_ori semantics change**: V6.2B rank_ori has lower = more binding. V7.0 ML score has higher = more binding. The deployment script must handle this sign flip when constructing the output DataFrame.

---

## 7. Temporal Lag Rules (CRITICAL — Data Leakage Prevention)

### 7.1 The Core Rule

For period type fN, the signal for auction month M is submitted **~mid of month M-1**. At submission time, only realized DA data through month **M-2** is complete.

**BF_LAG = 1** for both f0 and f1 (keyed on auction month, not delivery month).

### 7.2 Training Window

For eval month M with 8 training months:
- **Train months**: M-9 through M-2 (shift back by 1 from naive M-8..M-1)
- **Test month**: M

Implementation: call `load_train_val_test(prev_month(eval_month), 8, 0)`, discard its test split, then load eval_month separately as the real test set.

### 7.3 Binding Frequency Features

Binding frequency for training month T uses realized DA from months **< T-1** (not < T).

```python
# For bf_N feature at auction_month M:
# Available realized DA: months < M-1
# bf_1 = binding freq over 1 month ending at M-2
# bf_3 = binding freq over 3 months ending at M-2
# bf_6, bf_12, bf_15 = same pattern
```

### 7.4 General Lag Formula

| Period Type | Lag | Training Window (8mo) | BF Cutoff |
|------------|:---:|----------------------|-----------|
| f0 | 1 | M-9..M-2 | months < M-1 |
| f1 | 1 | M-9..M-2 | months < M-1 |
| f2+ | N/A | passthrough | passthrough |

**Why BF_LAG=1 for both f0 AND f1**: The lag is about when the signal is submitted (mid M-1) and what DA data is available at that time (through M-2). This is the same regardless of whether we're scoring f0 or f1.

### 7.5 What is NOT Affected by Lag

- V6.2B signal features (`da_rank_value`, `ori_mean`, etc.) — generated by signal pipeline, not realized DA
- Spice6 features (`prob_exceed_110`, `constraint_limit`) — forward-looking model outputs
- Test month label (`realized_sp` for delivery month) — this is ground truth we evaluate against, not a feature

---

## 8. Feature Engineering

### 8.1 Features (9 total, same as MISO V7.0)

| # | Feature | Source | Description |
|---|---------|--------|-------------|
| 1 | `bf_1` | Realized DA | Binding frequency, 1-month window |
| 2 | `bf_3` | Realized DA | Binding frequency, 3-month window |
| 3 | `bf_6` | Realized DA | Binding frequency, 6-month window |
| 4 | `bf_12` | Realized DA | Binding frequency, 12-month window |
| 5 | `bf_15` | Realized DA | Binding frequency, 15-month window |
| 6 | `v7_formula_score` | V6.2B | The V6.2B formula `rank_ori` value |
| 7 | `prob_exceed_110` | Spice6 | Probability of shadow price exceeding 110% of limit |
| 8 | `constraint_limit` | Spice6 | Constraint thermal limit from spice6 |
| 9 | `da_rank_value` | V6.2B | Dense-rank-normalized DA shadow price |

### 8.2 Binding Frequency Computation

For each constraint at auction month M with lag L=1:
```python
# Available realized DA months: all months < M-1 (i.e., through M-2)
# For bf_N: count months where constraint was binding in the N months ending at M-2
# binding = realized_sp > 0 for that (constraint_id, month)
bf_N = count_binding_months / N
```

**PJM adaptation**: The binding frequency join uses `monitored_facility` (extracted from constraint_id by splitting on ":"), not the full compound constraint_id.

### 8.3 Spice6 Feature Join

Spice6 data is keyed by `market_month` = delivery month = auction_month + N for fN.

```python
# For f0 auction_month 2025-01: delivery_month = 2025-01, market_month = 2025-01
# For f1 auction_month 2025-01: delivery_month = 2025-02, market_month = 2025-02
spice6 = load_spice6_density(delivery_month, period_type="f0")  # always f0 in spice6 path
joined = df.join(spice6, on=["constraint_id", "flow_direction"], how="left")
```

**wkndonpeak gap**: Spice6 ml_pred only has onpeak and dailyoffpeak. For wkndonpeak, either:
- Fill spice6 features with 0 (baseline approach)
- Proxy from onpeak data (requires validation)
- Skip ML for wkndonpeak entirely (keep V6.2B passthrough)

### 8.4 Tiered Labels (Ground Truth)

Labels for LambdaRank training are tiered, not binary:

| Label | Condition |
|:-----:|-----------|
| 0 | Non-binding (`realized_sp == 0`) |
| 1 | Binding, bottom 50% of shadow prices |
| 2 | Binding, top 50% of shadow prices |
| 3 | Binding, top 20% of shadow prices |

The `realized_sp` for each constraint comes from the DA shadow price join (see §4.3).

---

## 9. ML Model Configuration

### 9.1 LightGBM LambdaRank

```python
params = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [20],
    "learning_rate": 0.05,
    "num_leaves": 31,
    "n_estimators": 100,
    "min_data_in_leaf": 25,      # mapped from min_child_weight in config
    "reg_alpha": 1.0,
    "reg_lambda": 1.0,
    "num_threads": 4,            # CRITICAL: prevents deadlock on 64-CPU container
    "seed": 42,
    "verbose": -1,
}
```

### 9.2 Query Groups

LambdaRank requires query groups. Each auction month = one query group. The `group` parameter to `lgb.train()` is an array of group sizes (number of constraints per month).

```python
def compute_query_groups(df: pl.DataFrame) -> np.ndarray:
    """Return array of group sizes for LambdaRank training."""
    return df.group_by("query_month", maintain_order=True).len()["len"].to_numpy()
```

### 9.3 Training Configuration

- **Train window**: 8 months
- **Validation**: 0 months (no validation split)
- **Eval metric**: VC@20 (value capture at top 20 constraints)
- **Backend**: LightGBM LambdaRank with tiered labels (0/1/2/3)

### 9.4 Blend Weights (Starting Point)

MISO's optimized blend weights per slice:

| Slice | ML weight | Formula weight | Blend weight |
|-------|:---------:|:--------------:|:------------:|
| f0/onpeak | 0.85 | 0.15 | (0.85, 0.00, 0.15) |
| f0/offpeak | 0.85 | 0.15 | (0.85, 0.00, 0.15) |
| f1/onpeak | 0.70 | 0.30 | (0.70, 0.00, 0.30) |
| f1/offpeak | 0.80 | 0.20 | (0.80, 0.00, 0.20) |

For PJM, these are **starting points only**. You must run a blend search for each of the 6 PJM slices (f0×3 + f1×3). The blend format is `(w_ml, w_blend, w_formula)` where `w_blend` is typically 0.

---

## 10. Registry Organization

### 10.1 Layout

```
registry/
  f0/
    onpeak/
      v0/          # V6.2B formula baseline
        metrics.json
        NOTES.md
      v1/          # blend search
      v2/          # ML model
    dailyoffpeak/
      v0/
      v1/
      v2/
    wkndonpeak/
      v0/
      v1/
      v2/
    gates.json
    champion.json
  f1/
    onpeak/
    dailyoffpeak/
    wkndonpeak/
    gates.json
    champion.json
holdout/
  (same structure as registry/)
```

Each `(period_type, class_type)` combo gets its own model and metrics. Never mix training data across period types or class types.

### 10.2 Gates and Promotion

Each slice has quality gates. A new version must beat the champion on gated metrics to be promoted. Use `ml.registry_paths` helpers to construct paths — never hardcode.

### 10.3 Metrics Schema

```json
{
  "version": "v2",
  "period_type": "f0",
  "class_type": "onpeak",
  "eval_months": ["2022-07", "2022-08", ...],
  "vc_at_20_mean": 0.35,
  "vc_at_20_std": 0.12,
  "vc_at_20_by_month": {"2022-07": 0.38, ...},
  "recall_at_100_mean": 0.72,
  "ndcg_at_20_mean": 0.45
}
```

---

## 11. Pipeline Steps

### Step 1: Data Discovery

- Verify V6.2B data exists at the path in §4.1
- Enumerate available (auction_month, period_type, class_type) combinations
- Determine usable months: need at least 8+1 months of history (8 train + 1 eval)
- For f1: check which months have f1 data (May/June often missing for higher fN)

```python
from pathlib import Path

V62B_BASE = "/opt/data/xyz-dataset/signal_data/pjm/constraints/TEST.TEST.Signal.PJM.SPICE_F0P_V6.2B.R1"

def list_available_months(period_type: str, class_type: str) -> list[str]:
    base = Path(V62B_BASE)
    months = []
    for month_dir in sorted(base.iterdir()):
        if (month_dir / period_type / class_type).exists():
            months.append(month_dir.name)
    return months
```

### Step 2: Realized DA Cache

Build a realized DA cache analogous to MISO's:

```python
# Cache location
REALIZED_DA_CACHE = "/opt/temp/qianli/spice_data/pjm/realized_da/"

# Structure: {cache_dir}/{YYYY-MM}_{peak_type}.parquet
# Columns: monitored_facility, realized_sp
```

**Fetch function**:
```python
def fetch_and_cache_month(month: str, peak_type: str, cache_dir: str) -> pl.DataFrame:
    """Fetch PJM DA shadow prices for a month, aggregate, and cache."""
    cache_path = Path(cache_dir) / f"{month}_{peak_type}.parquet"
    if cache_path.exists():
        return pl.read_parquet(str(cache_path))

    # Fetch from API
    st = f"{month}-01"
    et = (pd.Timestamp(st) + pd.DateOffset(months=1)).strftime("%Y-%m-%d")
    df = PjmApTools().tools.get_da_shadow_by_peaktype(st, et, peak_type)

    # Aggregate: sum shadow_price by monitored_facility
    result = (
        pl.from_pandas(df)
        .group_by("monitored_facility")
        .agg(pl.col("shadow_price").sum().alias("realized_sp"))
    )

    # Write with atomic temp+rename
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(".tmp")
    result.write_parquet(str(tmp))
    tmp.rename(cache_path)

    return result
```

### Step 3: Build Data Loader

Adapt `ml/data_loader.py` from stage5-tier:

**Key changes**:
- `V62B_SIGNAL_BASE` → PJM path
- Realized DA join: use `monitored_facility` instead of `constraint_id`
- Extract `monitored_facility` from compound `constraint_id` (split on ":")
- Support 3 class types: onpeak, dailyoffpeak, wkndonpeak
- Spice6 loader: use PJM path, handle missing wkndonpeak

```python
def load_v62b_month(auction_month, period_type, class_type, cache_dir=None):
    # Load V6.2B parquet
    # Extract monitored_facility from constraint_id
    # Enrich with spice6 density (handle wkndonpeak gap)
    # Join realized DA on monitored_facility
    # Return enriched DataFrame
    ...
```

### Step 4: V0 — Formula Baseline

Run the V6.2B formula as v0 for all 6 slices. This establishes the baseline to beat.

For each `(ptype, ctype)` in `{f0, f1} × {onpeak, dailyoffpeak, wkndonpeak}`:
1. Load eval months with 8-month train windows (respecting lag)
2. Score using V6.2B formula (rank_ori already in data)
3. Evaluate VC@20, Recall@100, NDCG@20
4. Save to `registry/{ptype}/{ctype}/v0/metrics.json`

### Step 5: Build Binding Frequency Features

For each eval month, compute bf_1, bf_3, bf_6, bf_12, bf_15:

```python
def compute_binding_freq(constraint_monitored_facility, auction_month, window, lag=1):
    """
    Count fraction of months where constraint was binding.
    Uses realized DA months < auction_month - lag + 1 (i.e., through M-2 for lag=1).
    """
    cutoff = prev_month(auction_month)  # M-1 for lag=1
    # Look back `window` months from cutoff
    # For each month, check if realized_sp > 0
    # Return count / window
```

### Step 6: V2 — ML Model

Train LightGBM LambdaRank for each slice:

1. For each eval month M:
   a. Load train data: months M-9..M-2 (lag-adjusted)
   b. Compute binding frequency features for each train month
   c. Load test data: month M
   d. Compute binding frequency features for test month
   e. Build tiered labels from realized_sp
   f. Train LambdaRank with query groups
   g. Score test month
   h. Evaluate VC@20

2. Aggregate metrics across eval months
3. Save to `registry/{ptype}/{ctype}/v2/metrics.json`

### Step 7: Blend Search

For each slice, search for optimal `(w_ml, 0, w_formula)` blend:

```python
for alpha in [0.50, 0.55, 0.60, ..., 0.95, 1.00]:
    blended_score = alpha * ml_score + (1 - alpha) * formula_score
    evaluate(blended_score)
```

Save best blend to `registry/{ptype}/{ctype}/v1/`.

### Step 8: Holdout Evaluation

Re-run best version on held-out months (2024–2025) that were never used during development. Save to `holdout/{ptype}/{ctype}/{version}/metrics.json`.

---

## 12. Evaluation Metric: VC@20

**Value Capture at top 20** — the primary metric:

```python
def vc_at_20(df, score_col, value_col="realized_sp", k=20):
    """Fraction of total realized shadow price captured by top-k ranked constraints."""
    top_k = df.sort(score_col, descending=True).head(k)
    total_value = df[value_col].sum()
    if total_value == 0:
        return 0.0
    return top_k[value_col].sum() / total_value
```

Also track: Recall@100 (fraction of binding constraints in top 100), NDCG@20.

---

## 13. Code to Carry Over from MISO Stage5

The following modules from `research-stage5-tier/ml/` should be adapted:

| Module | What it does | PJM adaptations needed |
|--------|-------------|----------------------|
| `config.py` | Paths, LTRConfig, `delivery_month()`, `collect_usable_months()` | Change V62B path, class types, period type ranges |
| `data_loader.py` | Load V6.2B + spice6 + realized DA | Change join key to monitored_facility, handle wkndonpeak |
| `realized_da.py` | Fetch + cache realized DA | Use PjmApTools, aggregate by monitored_facility |
| `spice6_loader.py` | Load spice6 density features | Change path to PJM, handle missing wkndonpeak |
| `v62b_formula.py` | V6.2B formula reproduction | No changes (formula identical) |
| `features.py` | Binding frequency computation | Join on monitored_facility, not constraint_id |
| `train.py` | LightGBM LambdaRank training | No changes |
| `evaluate.py` | VC@20, Recall, NDCG metrics | No changes |
| `benchmark.py` | Walk-forward evaluation harness | Add wkndonpeak to slice list |
| `registry_paths.py` | Registry path helpers | No changes |

### What NOT to carry over
- MISO-specific `SO_MW_Transfer` exception handling
- MISO blend weights (run fresh search for PJM)
- MISO spice6 paths (use PJM paths)

---

## 14. Usable Month Estimation

### f0
V6.2B f0 data starts ~2017-06. With 8-month train + 1 lag, earliest eval month ≈ 2018-03.
Realistic dev window: 2018-03 to ~2023-12. Holdout: 2024-01 to 2025-12.

### f1
f1 data is sparser (May/June typically missing). Enumerate available months first.
Expected: fewer eval months than f0, similar to MISO's 30 dev / 19 holdout split.

Use `collect_usable_months()` pattern from MISO:
```python
def collect_usable_months(period_type, class_type, min_train=8, lag=1):
    """Return list of months that have enough training history."""
    available = list_available_months(period_type, class_type)
    usable = []
    for i, month in enumerate(available):
        # Need min_train + lag months before this one
        if i >= min_train + lag:
            usable.append(month)
    return usable
```

---

## 15. Score-to-DataFrame Alignment (CRITICAL)

When scoring with ML, the model returns scores in the order of the input array. When writing the output signal, the DataFrame may have a different row order (especially if using pandas for I/O).

**Rule**: Always join scores back on `constraint_id`, never assign positionally.

```python
# WRONG — positional assignment
df["ml_score"] = model.predict(features)

# RIGHT — join on constraint_id
scores_df = pl.DataFrame({
    "constraint_id": test_constraint_ids,
    "ml_score": model.predict(features),
})
df = df.join(scores_df, on="constraint_id", how="left")
```

---

## 16. Deployment Script Outline

```python
def generate_v70_signal(auction_month: str):
    """Generate V7.0 hybrid signal for all period types and class types."""

    for period_type in get_available_ptypes(auction_month):
        for class_type in ["onpeak", "dailyoffpeak", "wkndonpeak"]:

            if period_type in ("f0", "f1"):
                # ML-scored path
                df = load_v62b_month(auction_month, period_type, class_type)
                features = build_features(df, auction_month, period_type, class_type)
                model = load_model(period_type, class_type)

                # Score and join on constraint_id
                scores = model.predict(features)
                df = df.with_columns(pl.Series("ml_score", scores))

                # Blend with formula
                w_ml, _, w_formula = get_blend_weights(period_type, class_type)
                df = df.with_columns(
                    (w_ml * pl.col("ml_score") + w_formula * pl.col("v7_formula_score"))
                    .alias("blended_score")
                )

                # Row-percentile tiering with V6.2B tie-breaking
                df = assign_v70_tiers(df, score_col="blended_score")
            else:
                # Passthrough: use V6.2B rank/tier unchanged
                df = load_v62b_month(auction_month, period_type, class_type)

            # Write output in V6.2B format
            save_signal(df, auction_month, period_type, class_type)
```

---

## 17. Virtual Environment and Runtime

```bash
cd /home/xyz/workspace/pmodel && source .venv/bin/activate
```

Run all scripts from the `research-pjm-stage0-tier/` directory.

**Ray initialization** (required for PjmApTools data access):
```python
import os
os.environ["RAY_ADDRESS"] = "ray://10.8.0.36:10001"
from pbase.config.ray import init_ray
import pmodel
init_ray(extra_modules=[pmodel])
```

**Memory rules**: Use polars over pandas, scan_parquet for large files, `mem_mb()` at each stage, free intermediates with `del df; gc.collect()`. See parent CLAUDE.md for full memory budget.

**LightGBM**: Always `num_threads=4` to prevent deadlock on 64-CPU container.

---

## 18. Checklist for the Implementing Agent

- [ ] **Step 1**: Enumerate PJM V6.2B data — list all (month, ptype, ctype) combos, confirm column schema
- [ ] **Step 2**: Verify constraint_id split join — confirm 98%+ overlap with DA shadow prices
- [ ] **Step 3**: Build realized DA cache at `/opt/temp/qianli/spice_data/pjm/realized_da/`
- [ ] **Step 4**: Build data loader with monitored_facility join + spice6 enrichment
- [ ] **Step 5**: Run v0 formula baseline for all 6 ML slices, establish gates
- [ ] **Step 6**: Build binding frequency features (respect lag=1, join on monitored_facility)
- [ ] **Step 7**: Run v2 ML model for all 6 slices
- [ ] **Step 8**: Run blend search for all 6 slices
- [ ] **Step 9**: Run holdout evaluation
- [ ] **Step 10**: Decide wkndonpeak strategy (ML with zero spice6? passthrough? proxy from onpeak?)
- [ ] **Step 11**: Write deployment script with row-percentile tiering

---

## 19. Reference Files

| What | Path |
|------|------|
| MISO V7.0 handoff | `research-miso-signal7/docs/v70-deployment-handoff.md` |
| MISO stage5 ML code | `research-stage5-tier/ml/` |
| MISO v10e-lag1 run script | `research-stage5-tier/scripts/run_v10e_lagged.py` |
| MISO formula reproduction | `research-stage5-tier/ml/v62b_formula.py` |
| MISO benchmark harness | `research-stage5-tier/ml/benchmark.py` |
| MISO config (LTRConfig) | `research-stage5-tier/ml/config.py` |
| Parent CLAUDE.md | `research-qianli-v2/CLAUDE.md` |
| Stage5 CLAUDE.md | `research-stage5-tier/CLAUDE.md` |
| PJM V6.2B data | `/opt/data/xyz-dataset/signal_data/pjm/constraints/TEST.TEST.Signal.PJM.SPICE_F0P_V6.2B.R1/` |
| PJM spice6 | `/opt/temp/tmp/pw_data/spice6/prod_f0p_model_pjm/` |
| PJM SF data | `/opt/data/xyz-dataset/signal_data/pjm/sf/MANUAL.TEST.TEST.Signal.PJM.SPICE_F0P_V6.2B.R1/` |
| User requirements | `research-pjm-stage0-tier/human-input/memory.md` |
