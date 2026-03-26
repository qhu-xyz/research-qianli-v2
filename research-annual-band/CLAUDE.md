# CLAUDE.md — research-annual-band

Inherits rules from parent: `/home/xyz/workspace/research-qianli-v2/CLAUDE.md`

## Band Reporting Standard (MANDATORY)

### Primary Metric: Clearing Probability

All trades are buy trades. **Pxx = the bid price at which there is an xx% chance of clearing
on training data.** The clearing rate for a bid at price P is:

```
clearing_rate = P(MCP <= P)
```

**Pxx is a SINGLE bid price, NOT a pair.** Each level is an independent bid:
- P20 = bid price with 20% clearing chance (cheap, you rarely clear)
- P50 = bid price with 50% clearing chance (median)
- P80 = bid price with 80% clearing chance (expensive, you usually clear)
- P95 = bid price with 95% clearing chance (very expensive, almost always clear)

```python
# Calibration: Pxx bid = baseline + quantile(residual, xx/100) on training data
P20_bid = baseline + np.quantile(train_residual, 0.20)
P95_bid = baseline + np.quantile(train_residual, 0.95)
```

### Granularity

Always report at **(round, planning_year, bin, flow_type)** granularity:
- **Round**: R1-R4 (PJM), R1-R3 (MISO)
- **Planning year**: the test PY in temporal CV
- **Bin**: q1-q5 (quantile bins by |baseline|)
- **Flow type**: `prevail` (baseline > 0) or `counter` (baseline < 0)

### Flow Type Definition

```python
flow_type = "prevail" if baseline > 0 else "counter"
```

Prevail = you pay to acquire the position (positive expected value).
Counter = you get paid to take the position (negative expected value).

### What to Report

For each (round, year, bin, flow_type) cell:
1. **N** — number of paths in the cell
2. **Actual clearing rate at Pxx** — P(MCP <= Pxx_bid) on test data
3. **Miss** — actual_clearing_rate - xx% (positive = over-clearing, negative = under-clearing)
4. **Bias** — mean(mcp - baseline) on test data (positive = baseline under-predicts)
5. **MAE** — mean(|mcp - baseline|)

### Red Flag Thresholds

| Shortfall | Severity |
|-----------|----------|
| > -3pp | OK |
| -3pp to -5pp | WATCH |
| -5pp to -10pp | CONCERN |
| < -10pp | FLAG |

### Aggregation Order

When summarizing, drill down from coarse to fine:
1. Overall per round (all PYs, all bins, all flow types)
2. Per PY per round (identify worst years)
3. Per bin per round (identify worst bins)
4. Per flow_type per round (identify prevail vs counter differences)
5. Full grid: (round, PY, bin, flow_type) for flagged cells only

### Do NOT

- Do NOT report two-sided coverage as a primary metric. Two-sided coverage adds buy-miss and sell-miss rates — misleading because no single trade is exposed to both.
- Do NOT call a two-sided number "catastrophic" without decomposing it into buy clearing rate.
- Do NOT use "sell clearing rate" — all trades are buy.

## Path Sign Convention (CRITICAL)

**FTR path value = sink - source.** This applies to ALL path-level quantities:
- `mcp` = sink_mcp - source_mcp
- `nodal_f0` = sink_f0 - source_f0
- `1(rev)` = sink_node_DA_congestion - source_node_DA_congestion
- `mtm_1st_mean` = sink - source (already correct in M2M data)

**A bug was found (2026-03-20):** the 1(rev) computation in `r1_1rev_option_b.parquet` used
`source_cong - sink_cong` (wrong sign). Correlation with f0 was -0.81 instead of +0.81.
77% of paths had opposite signs between f0 and 1(rev) because of this.

**Fix:** `1_rev_corrected = -1 * 1_rev` or recompute with `sink_cong - source_cong`.

## No Silent Fallbacks (MANDATORY)

**Never silently fall back** for class_type, period_type, round, or any categorical dimension.
If a required column or groupby key produces NaN, empty, or unexpected values — **raise**, do not substitute defaults.

```python
# WRONG — hides bugs
if cls not in bin_pairs[label]:
    pairs = pooled_pairs  # silent fallback

# RIGHT — explicit, auditable
if cls not in bin_pairs[label]:
    raise ValueError(f"No calibration data for {label}/{cls}. Check data pipeline.")
```

The ONLY acceptable fallback is when the design explicitly requires it (e.g., per-class → pooled
when a cell has fewer rows than threshold). In that case:
1. Log a warning with cell identity and row count
2. Document the fallback in the metrics output
3. If fallback rate > 20% of cells for any round/quarter, raise instead

For any data loading step: if a column that should exist has > 5% NaN, **raise** — do not fill or drop silently.

## Scale Convention (MANDATORY)

**All prices are in native auction scale. Never use monthly scale.**

- **MISO annual**: quarterly scale (×3). Use `mcp` (quarterly clearing price). **Do NOT use `mcp_mean`** — it is deprecated and has caused unit bugs.
- **PJM annual**: annual scale (×12). Use `mcp` directly.
- Baselines, band edges, residuals, widths, bid prices — all in native scale.
- If a column is monthly and you need quarterly: multiply by 3 explicitly and rename (e.g., `baseline_q = nodal_f0 * 3`).
- Never mix scales in the same DataFrame. If you see monthly and quarterly columns side by side, raise.

## Exhaustive Post-Step Inspection (MANDATORY)

After every data loading, transformation, or modeling step, **print and inspect aggressively**.
The goal is to catch errors as early as possible. Be abundant — any dimension you can think of, check it.

### After data loading:
```python
print(f"Shape: {df.shape}")
print(f"Columns: {sorted(df.columns)}")
print(f"Dtypes:\n{df.dtypes}")
print(f"Null counts:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
print(f"Unique counts: {df.nunique()}")

# Key columns — print stats
for col in ['mcp', 'mtm_1st_mean', 'baseline', 'planning_year', 'round', 'class_type', 'period_type']:
    if col in df.columns:
        print(f"\n{col}: min={df[col].min()}, max={df[col].max()}, "
              f"mean={df[col].mean():.2f}, null={df[col].isnull().sum()}, "
              f"unique={df[col].nunique()}")

# Groupby counts
print(df.groupby(['round', 'period_type', 'planning_year', 'class_type']).size()
      .reset_index(name='n').to_string())
```

### After joins / merges:
```python
# Row count before and after
print(f"Before: {n_before:,}, After: {n_after:,}, Ratio: {n_after/n_before:.3f}")

# Check for unintended fan-out (duplicates created by join)
assert n_after <= n_before * 1.01, f"Join fan-out: {n_after/n_before:.2f}x"

# Check join coverage
n_matched = df['joined_col'].notna().sum()
print(f"Join coverage: {n_matched}/{n_after} = {n_matched/n_after*100:.1f}%")
```

### After feature computation:
```python
# For each new feature column
for col in new_feature_cols:
    s = df[col]
    print(f"{col}: min={s.min():.2f}, p25={s.quantile(0.25):.2f}, "
          f"median={s.median():.2f}, p75={s.quantile(0.75):.2f}, max={s.max():.2f}, "
          f"null={s.isnull().sum()}, zero={( s == 0).sum()}")
```

### After band calibration:
```python
# Per-cell: N, lower, upper, width, and sanity checks
for cell in cells:
    assert lower < upper, f"Inverted band: {cell}"
    assert n >= MIN_CELL_ROWS, f"Underpopulated cell: {cell}, n={n}"

# Coverage monotonicity: P10 ⊂ P30 ⊂ ... ⊂ P99
for path in sample_paths:
    for i in range(len(levels) - 1):
        assert width[levels[i]] <= width[levels[i+1]]
```

### After any modeling / evaluation step:
```python
# Print full results table, not just summary
# Include worst cells, not just averages
# Compare to prior run if available
```

**The spirit of this rule:** if something is wrong, the print output should make it obvious
within seconds of looking at it. Do not wait until a downstream step fails to discover
an upstream data issue.

## Job Submission (MANDATORY)

Use the Ray cluster for all data loading and parallel work. Follow the dual-mode CLI pattern
from the parent CLAUDE.md (`run` for local, `submit` for Ray Job). See `/parallel-with-ray` skill
for patterns including `ray_map_bounded`, `@ray.remote(scheduling_strategy="SPREAD")`, and
`build_runtime_env`.
