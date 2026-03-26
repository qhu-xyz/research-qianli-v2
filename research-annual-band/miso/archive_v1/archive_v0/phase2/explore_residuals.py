"""
Explore residual patterns in MISO annual FTR nodal_f0 baseline.
Focuses on aq1 baseline file + the all_residuals_v2.parquet.

Memory-safe: uses polars lazy scans, stays well under 10 GiB.
"""

import resource
import gc
import numpy as np
import polars as pl

def mem_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

print(f"[start] mem = {mem_mb():.0f} MB")

# ============================================================
# 1.  Load aq1 baselines, compute residual = mcp_mean - nodal_f0
# ============================================================
print("\n" + "=" * 70)
print("SECTION 1: Basic residual statistics for aq1 (mcp_mean - nodal_f0)")
print("=" * 70)

aq1 = (
    pl.scan_parquet("/opt/temp/qianli/annual_research/crossproduct_work/aq1_all_baselines.parquet")
    .filter(pl.col("nodal_f0").is_not_null() & pl.col("mcp_mean").is_not_null())
    .with_columns(
        (pl.col("mcp_mean") - pl.col("nodal_f0")).alias("residual"),
        (pl.col("mcp_mean") - pl.col("nodal_f0")).abs().alias("abs_residual"),
        pl.concat_str([pl.col("source_id"), pl.col("sink_id")], separator="->").alias("path"),
    )
    .collect()
)

print(f"  Valid rows: {len(aq1):,}")
print(f"  [after load] mem = {mem_mb():.0f} MB")

# --- 1a. Overall distribution ---
print("\n--- 1a. Overall residual distribution ---")
stats = aq1.select(
    pl.col("residual").mean().alias("mean"),
    pl.col("residual").median().alias("median"),
    pl.col("residual").std().alias("std"),
    pl.col("residual").min().alias("min"),
    pl.col("residual").max().alias("max"),
    pl.col("residual").quantile(0.05).alias("p5"),
    pl.col("residual").quantile(0.25).alias("p25"),
    pl.col("residual").quantile(0.75).alias("p75"),
    pl.col("residual").quantile(0.95).alias("p95"),
    pl.col("abs_residual").mean().alias("mae"),
    (pl.col("residual") ** 2).mean().sqrt().alias("rmse"),
)
print(stats)

# --- 1b. By planning year ---
print("\n--- 1b. Residual stats by planning_year ---")
by_py = (
    aq1.group_by("planning_year")
    .agg(
        pl.len().alias("n"),
        pl.col("residual").mean().alias("mean"),
        pl.col("residual").median().alias("median"),
        pl.col("residual").std().alias("std"),
        pl.col("abs_residual").mean().alias("mae"),
        (pl.col("residual") ** 2).mean().sqrt().alias("rmse"),
        pl.col("residual").quantile(0.05).alias("p5"),
        pl.col("residual").quantile(0.95).alias("p95"),
    )
    .sort("planning_year")
)
print(by_py)

# --- 1c. By class_type ---
print("\n--- 1c. Residual stats by class_type ---")
by_class = (
    aq1.group_by("class_type")
    .agg(
        pl.len().alias("n"),
        pl.col("residual").mean().alias("mean"),
        pl.col("residual").median().alias("median"),
        pl.col("residual").std().alias("std"),
        pl.col("abs_residual").mean().alias("mae"),
        (pl.col("residual") ** 2).mean().sqrt().alias("rmse"),
    )
    .sort("class_type")
)
print(by_class)

# --- 1d. By magnitude bins of |nodal_f0| ---
print("\n--- 1d. Residual stats by |nodal_f0| magnitude bins ---")
aq1_binned = aq1.with_columns(
    pl.when(pl.col("nodal_f0").abs() < 1)
    .then(pl.lit("A: |f0| < 1"))
    .when(pl.col("nodal_f0").abs() < 5)
    .then(pl.lit("B: 1 <= |f0| < 5"))
    .when(pl.col("nodal_f0").abs() < 20)
    .then(pl.lit("C: 5 <= |f0| < 20"))
    .when(pl.col("nodal_f0").abs() < 50)
    .then(pl.lit("D: 20 <= |f0| < 50"))
    .otherwise(pl.lit("E: |f0| >= 50"))
    .alias("f0_mag_bin")
)
by_mag = (
    aq1_binned.group_by("f0_mag_bin")
    .agg(
        pl.len().alias("n"),
        pl.col("residual").mean().alias("mean"),
        pl.col("residual").median().alias("median"),
        pl.col("residual").std().alias("std"),
        pl.col("abs_residual").mean().alias("mae"),
        (pl.col("residual") ** 2).mean().sqrt().alias("rmse"),
        # bias ratio: mean_residual / mean |f0|
        (pl.col("residual").mean() / pl.col("nodal_f0").abs().mean()).alias("bias_ratio"),
    )
    .sort("f0_mag_bin")
)
print(by_mag)

# --- 1e. By residual magnitude bins (how big are the errors?) ---
print("\n--- 1e. Distribution of |residual| magnitude ---")
aq1_rbinned = aq1.with_columns(
    pl.when(pl.col("abs_residual") < 0.5)
    .then(pl.lit("A: <0.5"))
    .when(pl.col("abs_residual") < 2)
    .then(pl.lit("B: 0.5-2"))
    .when(pl.col("abs_residual") < 5)
    .then(pl.lit("C: 2-5"))
    .when(pl.col("abs_residual") < 10)
    .then(pl.lit("D: 5-10"))
    .when(pl.col("abs_residual") < 25)
    .then(pl.lit("E: 10-25"))
    .when(pl.col("abs_residual") < 50)
    .then(pl.lit("F: 25-50"))
    .otherwise(pl.lit("G: >=50"))
    .alias("resid_mag_bin")
)
by_rmag = (
    aq1_rbinned.group_by("resid_mag_bin")
    .agg(
        pl.len().alias("n"),
        (pl.len() / len(aq1) * 100).alias("pct"),
        pl.col("residual").mean().alias("mean_resid"),
        pl.col("nodal_f0").abs().mean().alias("mean_abs_f0"),
    )
    .sort("resid_mag_bin")
)
print(by_rmag)

# --- 1f. By class_type x |nodal_f0| magnitude ---
print("\n--- 1f. MAE by class_type x |nodal_f0| magnitude ---")
by_class_mag = (
    aq1_binned.group_by("class_type", "f0_mag_bin")
    .agg(
        pl.len().alias("n"),
        pl.col("abs_residual").mean().alias("mae"),
        pl.col("residual").mean().alias("bias"),
    )
    .sort("class_type", "f0_mag_bin")
)
print(by_class_mag)

del aq1_binned, aq1_rbinned
gc.collect()
print(f"\n  [after section 1] mem = {mem_mb():.0f} MB")

# ============================================================
# 2.  Autocorrelation: do residuals persist for same path across PYs?
# ============================================================
print("\n" + "=" * 70)
print("SECTION 2: Autocorrelation of residuals across planning years")
print("=" * 70)

# For each (path, class_type), pivot by PY to get residual per year
path_py = (
    aq1.select("path", "class_type", "planning_year", "residual")
    .group_by("path", "class_type", "planning_year")
    .agg(pl.col("residual").mean())
)

# Pivot to wide format
pys = sorted(path_py["planning_year"].unique().to_list())
pivot = path_py.pivot(on="planning_year", index=["path", "class_type"], values="residual")

# Compute year-over-year correlations
print("\n--- 2a. Correlation of path residuals between consecutive PYs ---")
py_cols = [str(py) for py in pys]
for i in range(len(py_cols) - 1):
    c1, c2 = py_cols[i], py_cols[i + 1]
    if c1 in pivot.columns and c2 in pivot.columns:
        valid = pivot.filter(pl.col(c1).is_not_null() & pl.col(c2).is_not_null())
        if len(valid) > 10:
            corr = valid.select(pl.corr(c1, c2)).item()
            print(f"  PY {c1} -> PY {c2}: r = {corr:.4f}  (n = {len(valid):,})")

# Also compute correlation with 2-year lag
print("\n--- 2b. Correlation with 2-year lag ---")
for i in range(len(py_cols) - 2):
    c1, c2 = py_cols[i], py_cols[i + 2]
    if c1 in pivot.columns and c2 in pivot.columns:
        valid = pivot.filter(pl.col(c1).is_not_null() & pl.col(c2).is_not_null())
        if len(valid) > 10:
            corr = valid.select(pl.corr(c1, c2)).item()
            print(f"  PY {c1} -> PY {c2}: r = {corr:.4f}  (n = {len(valid):,})")

# Average autocorrelation across all lag-1 pairs
print("\n--- 2c. Overall lag-1 autocorrelation (pooled approach) ---")
# Self-join: match same path across consecutive PYs
resid_long = aq1.select("path", "class_type", "planning_year", "residual")
lag1 = (
    resid_long.join(
        resid_long.rename({"residual": "residual_prev", "planning_year": "py_prev"}),
        on=["path", "class_type"],
    )
    .filter(pl.col("planning_year") == pl.col("py_prev") + 1)
)
if len(lag1) > 0:
    overall_corr = lag1.select(pl.corr("residual", "residual_prev")).item()
    print(f"  Pooled lag-1 correlation: r = {overall_corr:.4f}  (n = {len(lag1):,})")

    # By class_type
    for ct in ["onpeak", "offpeak"]:
        sub = lag1.filter(pl.col("class_type") == ct)
        if len(sub) > 10:
            c = sub.select(pl.corr("residual", "residual_prev")).item()
            print(f"  Lag-1 for {ct}: r = {c:.4f}  (n = {len(sub):,})")

del pivot, lag1, resid_long
gc.collect()
print(f"\n  [after section 2] mem = {mem_mb():.0f} MB")

# ============================================================
# 3.  Correlation with available features
# ============================================================
print("\n" + "=" * 70)
print("SECTION 3: Correlation of residual with available features")
print("=" * 70)

# Feature columns to test
feature_cols = [
    "nodal_f0",
    "mcp_mean",
    "mtm_1st_mean",
    "f0_path_corr",
    "prior_r3_path",
    "prior_r2_path",
    "prior_r1_path",
    "f1_path",
]

print("\n--- 3a. Pearson correlation of residual with each feature ---")
for fcol in feature_cols:
    if fcol in aq1.columns:
        valid = aq1.filter(pl.col(fcol).is_not_null())
        if len(valid) > 100:
            corr = valid.select(pl.corr("residual", fcol)).item()
            print(f"  corr(residual, {fcol:20s}) = {corr:+.4f}  (n = {len(valid):,})")

# Also test derived features
print("\n--- 3b. Correlation with derived features ---")
aq1_ext = aq1.with_columns(
    pl.col("nodal_f0").abs().alias("abs_nodal_f0"),
    (pl.col("mcp_mean") - pl.col("mtm_1st_mean")).alias("mcp_minus_mtm1"),
    # f0 - path level forward = nodal correction
    (pl.col("nodal_f0") - pl.col("f0_path_corr")).alias("nodal_correction"),
)

derived_features = ["abs_nodal_f0", "mcp_minus_mtm1", "nodal_correction"]
for fcol in derived_features:
    valid = aq1_ext.filter(pl.col(fcol).is_not_null())
    if len(valid) > 100:
        corr = valid.select(pl.corr("residual", fcol)).item()
        print(f"  corr(residual, {fcol:20s}) = {corr:+.4f}  (n = {len(valid):,})")

# --- 3c. By class_type ---
print("\n--- 3c. Key correlations by class_type ---")
for ct in ["onpeak", "offpeak"]:
    print(f"\n  [{ct}]")
    sub = aq1_ext.filter(pl.col("class_type") == ct)
    for fcol in ["nodal_f0", "abs_nodal_f0", "f0_path_corr", "nodal_correction", "mtm_1st_mean"]:
        valid = sub.filter(pl.col(fcol).is_not_null())
        if len(valid) > 100:
            corr = valid.select(pl.corr("residual", fcol)).item()
            print(f"    corr(residual, {fcol:20s}) = {corr:+.4f}  (n = {len(valid):,})")

del aq1_ext
gc.collect()
print(f"\n  [after section 3] mem = {mem_mb():.0f} MB")

# ============================================================
# 4.  How much does the path-level correction help?
# ============================================================
print("\n" + "=" * 70)
print("SECTION 4: Comparing baselines — does nodal correction help?")
print("=" * 70)

# Compare MAE of nodal_f0 vs f0_path_corr vs f1_path vs prior round paths
baselines = {
    "nodal_f0": "nodal_f0",
    "f0_path_corr": "f0_path_corr",
    "f1_path": "f1_path",
    "prior_r3_path": "prior_r3_path",
    "prior_r2_path": "prior_r2_path",
    "prior_r1_path": "prior_r1_path",
    "mtm_1st_mean": "mtm_1st_mean",
}

print("\n--- 4a. Overall MAE / RMSE / Bias for each baseline ---")
for name, col in baselines.items():
    if col in aq1.columns:
        valid = aq1.filter(pl.col(col).is_not_null())
        stats = valid.select(
            (pl.col("mcp_mean") - pl.col(col)).abs().mean().alias("mae"),
            ((pl.col("mcp_mean") - pl.col(col)) ** 2).mean().sqrt().alias("rmse"),
            (pl.col("mcp_mean") - pl.col(col)).mean().alias("bias"),
        )
        row = stats.row(0)
        print(f"  {name:20s}  MAE={row[0]:.4f}  RMSE={row[1]:.4f}  Bias={row[2]:+.4f}  (n={len(valid):,})")

# --- 4b. By class_type ---
print("\n--- 4b. MAE by class_type for key baselines ---")
for ct in ["onpeak", "offpeak"]:
    print(f"\n  [{ct}]")
    sub = aq1.filter(pl.col("class_type") == ct)
    for name, col in baselines.items():
        if col in aq1.columns:
            valid = sub.filter(pl.col(col).is_not_null())
            if len(valid) > 0:
                mae = valid.select((pl.col("mcp_mean") - pl.col(col)).abs().mean()).item()
                rmse = valid.select(((pl.col("mcp_mean") - pl.col(col)) ** 2).mean().sqrt()).item()
                print(f"    {name:20s}  MAE={mae:.4f}  RMSE={rmse:.4f}  (n={len(valid):,})")

# --- 4c. By PY ---
print("\n--- 4c. nodal_f0 MAE by PY ---")
for py in sorted(aq1["planning_year"].unique().to_list()):
    sub = aq1.filter(pl.col("planning_year") == py)
    mae = sub.select(pl.col("abs_residual").mean()).item()
    rmse = sub.select((pl.col("residual") ** 2).mean().sqrt()).item()
    bias = sub.select(pl.col("residual").mean()).item()
    print(f"  PY {py}: MAE={mae:.4f}  RMSE={rmse:.4f}  Bias={bias:+.4f}  (n={len(sub):,})")

print(f"\n  [after section 4] mem = {mem_mb():.0f} MB")

# ============================================================
# 5.  Explore the all_residuals_v2 file for broader patterns
# ============================================================
print("\n" + "=" * 70)
print("SECTION 5: Patterns in all_residuals_v2.parquet")
print("=" * 70)

# Lazy scan with filters to keep memory low
resid_all = pl.scan_parquet("/opt/temp/qianli/annual_research/all_residuals_v2.parquet")

# --- 5a. Overall stats by period_type ---
print("\n--- 5a. Residual stats by period_type (all rounds) ---")
by_pt = (
    resid_all
    .group_by("period_type")
    .agg(
        pl.len().alias("n"),
        pl.col("residual").mean().alias("mean"),
        pl.col("residual").median().alias("median"),
        pl.col("residual").std().alias("std"),
        pl.col("abs_residual").mean().alias("mae"),
        (pl.col("residual") ** 2).mean().sqrt().alias("rmse"),
    )
    .sort("period_type")
    .collect()
)
print(by_pt)

# --- 5b. By round ---
print("\n--- 5b. Residual stats by round ---")
by_round = (
    resid_all
    .group_by("round")
    .agg(
        pl.len().alias("n"),
        pl.col("residual").mean().alias("mean"),
        pl.col("abs_residual").mean().alias("mae"),
        (pl.col("residual") ** 2).mean().sqrt().alias("rmse"),
    )
    .sort("round")
    .collect()
)
print(by_round)

# --- 5c. By PY x period_type (round 1 only) ---
print("\n--- 5c. MAE by PY x period_type (round 1 only) ---")
by_py_pt = (
    resid_all
    .filter(pl.col("round") == 1)
    .group_by("planning_year", "period_type")
    .agg(
        pl.len().alias("n"),
        pl.col("abs_residual").mean().alias("mae"),
        pl.col("residual").mean().alias("bias"),
    )
    .sort("period_type", "planning_year")
    .collect()
)
print(by_py_pt)

# --- 5d. Autocorrelation in the residuals file ---
print("\n--- 5d. Autocorrelation of residuals (round 1, aq1) from residuals file ---")
resid_r1_aq1 = (
    resid_all
    .filter((pl.col("round") == 1) & (pl.col("period_type") == "aq1"))
    .select("path", "class_type", "planning_year", "residual")
    .group_by("path", "class_type", "planning_year")
    .agg(pl.col("residual").mean())
    .collect()
)
print(f"  Loaded {len(resid_r1_aq1):,} path-class-PY combinations")

# Pivot and correlate
pys_r = sorted(resid_r1_aq1["planning_year"].unique().to_list())
pivot_r = resid_r1_aq1.pivot(on="planning_year", index=["path", "class_type"], values="residual")

print("\n  Lag-1 correlations (from residuals file):")
py_cols_r = [str(py) for py in pys_r]
for i in range(len(py_cols_r) - 1):
    c1, c2 = py_cols_r[i], py_cols_r[i + 1]
    if c1 in pivot_r.columns and c2 in pivot_r.columns:
        valid = pivot_r.filter(pl.col(c1).is_not_null() & pl.col(c2).is_not_null())
        if len(valid) > 10:
            corr = valid.select(pl.corr(c1, c2)).item()
            print(f"    PY {c1} -> PY {c2}: r = {corr:.4f}  (n = {len(valid):,})")

del resid_r1_aq1, pivot_r
gc.collect()

# --- 5e. Top paths by residual magnitude ---
print("\n--- 5e. Top 20 paths by average |residual| (round 1, aq1) ---")
top_paths = (
    resid_all
    .filter((pl.col("round") == 1) & (pl.col("period_type") == "aq1"))
    .group_by("path", "class_type")
    .agg(
        pl.len().alias("n_obs"),
        pl.col("abs_residual").mean().alias("mean_abs_resid"),
        pl.col("residual").mean().alias("mean_resid"),
        pl.col("mcp").mean().alias("mean_mcp"),
    )
    .filter(pl.col("n_obs") >= 3)  # at least 3 years
    .sort("mean_abs_resid", descending=True)
    .head(20)
    .collect()
)
print(top_paths)

# --- 5f. Persistent bias paths (same sign residual across years) ---
print("\n--- 5f. Paths with consistently positive/negative residuals (round 1, aq1) ---")
path_signs = (
    resid_all
    .filter((pl.col("round") == 1) & (pl.col("period_type") == "aq1"))
    .group_by("path", "class_type", "planning_year")
    .agg(pl.col("residual").mean().alias("resid"))
    .group_by("path", "class_type")
    .agg(
        pl.len().alias("n_years"),
        (pl.col("resid") > 0).sum().alias("n_positive"),
        (pl.col("resid") < 0).sum().alias("n_negative"),
        pl.col("resid").mean().alias("mean_resid"),
        pl.col("resid").std().alias("std_resid"),
    )
    .filter(pl.col("n_years") >= 4)
    .collect()
)

# Paths where sign is consistent (>= 80% same sign)
consistent_pos = path_signs.filter(
    pl.col("n_positive") >= (pl.col("n_years") * 0.8)
).sort("mean_resid", descending=True)

consistent_neg = path_signs.filter(
    pl.col("n_negative") >= (pl.col("n_years") * 0.8)
).sort("mean_resid")

print(f"\n  Paths with >= 80% positive residuals: {len(consistent_pos):,} of {len(path_signs):,}")
print(f"  Paths with >= 80% negative residuals: {len(consistent_neg):,} of {len(path_signs):,}")
print(f"  Neutral paths: {len(path_signs) - len(consistent_pos) - len(consistent_neg):,}")

# Summary of persistent bias
if len(consistent_pos) > 0:
    print(f"\n  Persistent positive (model underestimates MCP):")
    print(f"    Mean residual: {consistent_pos['mean_resid'].mean():.4f}")
    print(f"    Mean std: {consistent_pos['std_resid'].mean():.4f}")
if len(consistent_neg) > 0:
    print(f"\n  Persistent negative (model overestimates MCP):")
    print(f"    Mean residual: {consistent_neg['mean_resid'].mean():.4f}")
    print(f"    Mean std: {consistent_neg['std_resid'].mean():.4f}")

del path_signs, consistent_pos, consistent_neg
gc.collect()

# ============================================================
# 6.  Check if residual correlates with volume / cleared_mwh
# ============================================================
print("\n" + "=" * 70)
print("SECTION 6: Residual vs volume/liquidity (from all_residuals_v2)")
print("=" * 70)

vol_corr = (
    resid_all
    .filter((pl.col("round") == 1) & (pl.col("cleared_volume").is_not_null()))
    .select("residual", "abs_residual", "cleared_volume", "cleared_mwh", "class_type", "period_type", "mcp")
    .collect()
)
print(f"  Loaded {len(vol_corr):,} rows with volume data")

# Overall correlations
print("\n--- 6a. Correlations with volume ---")
for vcol in ["cleared_volume", "cleared_mwh"]:
    valid = vol_corr.filter(pl.col(vcol).is_not_null())
    if len(valid) > 100:
        c1 = valid.select(pl.corr("residual", vcol)).item()
        c2 = valid.select(pl.corr("abs_residual", vcol)).item()
        print(f"  corr(residual, {vcol:20s}) = {c1:+.4f}")
        print(f"  corr(|residual|, {vcol:20s}) = {c2:+.4f}")

# Volume bins
print("\n--- 6b. MAE by cleared_volume decile ---")
vol_corr_q = vol_corr.with_columns(
    pl.col("cleared_volume").rank("ordinal").alias("vol_rank"),
)
n_total = len(vol_corr_q)
vol_corr_q = vol_corr_q.with_columns(
    (pl.col("vol_rank") * 10 / n_total).cast(pl.Int32).clip(0, 9).alias("vol_decile")
)
by_vol = (
    vol_corr_q.group_by("vol_decile")
    .agg(
        pl.len().alias("n"),
        pl.col("cleared_volume").mean().alias("mean_vol"),
        pl.col("abs_residual").mean().alias("mae"),
        pl.col("residual").mean().alias("bias"),
        pl.col("mcp").abs().mean().alias("mean_abs_mcp"),
    )
    .sort("vol_decile")
)
print(by_vol)

del vol_corr, vol_corr_q
gc.collect()

# ============================================================
# 7.  Cross-quarter patterns: load all 4 baselines for consistent paths
# ============================================================
print("\n" + "=" * 70)
print("SECTION 7: Cross-quarter comparison of nodal_f0 residuals")
print("=" * 70)

all_q_stats = []
for q in [1, 2, 3, 4]:
    path = f"/opt/temp/qianli/annual_research/crossproduct_work/aq{q}_all_baselines.parquet"
    df_q = (
        pl.scan_parquet(path)
        .filter(pl.col("nodal_f0").is_not_null() & pl.col("mcp_mean").is_not_null())
        .with_columns(
            (pl.col("mcp_mean") - pl.col("nodal_f0")).alias("residual"),
            (pl.col("mcp_mean") - pl.col("nodal_f0")).abs().alias("abs_residual"),
        )
        .group_by("class_type")
        .agg(
            pl.len().alias("n"),
            pl.col("residual").mean().alias("bias"),
            pl.col("abs_residual").mean().alias("mae"),
            (pl.col("residual") ** 2).mean().sqrt().alias("rmse"),
        )
        .with_columns(pl.lit(f"aq{q}").alias("quarter"))
        .collect()
    )
    all_q_stats.append(df_q)

cross_q = pl.concat(all_q_stats).sort("quarter", "class_type")
print("\n--- 7a. nodal_f0 residual stats by quarter x class_type ---")
print(cross_q)

# Also compare f0_path_corr where available
print("\n--- 7b. f0_path_corr MAE by quarter (where available) ---")
for q in [1, 2, 3, 4]:
    path = f"/opt/temp/qianli/annual_research/crossproduct_work/aq{q}_all_baselines.parquet"
    stats = (
        pl.scan_parquet(path)
        .filter(pl.col("f0_path_corr").is_not_null() & pl.col("mcp_mean").is_not_null())
        .select(
            pl.len().alias("n"),
            (pl.col("mcp_mean") - pl.col("f0_path_corr")).abs().mean().alias("mae"),
            (pl.col("mcp_mean") - pl.col("f0_path_corr")).mean().alias("bias"),
        )
        .collect()
    )
    row = stats.row(0)
    print(f"  aq{q}: MAE={row[1]:.4f}  Bias={row[2]:+.4f}  (n={row[0]:,})")

# ============================================================
# 8.  Shrinkage opportunity: what if we blend nodal_f0 toward zero?
# ============================================================
print("\n" + "=" * 70)
print("SECTION 8: Shrinkage analysis — blending nodal_f0 toward zero")
print("=" * 70)

# Reload aq1 with all needed columns
aq1 = (
    pl.scan_parquet("/opt/temp/qianli/annual_research/crossproduct_work/aq1_all_baselines.parquet")
    .filter(pl.col("nodal_f0").is_not_null() & pl.col("mcp_mean").is_not_null())
    .collect()
)

print("\n--- 8a. MAE with shrinkage factor alpha (prediction = alpha * nodal_f0) ---")
alphas = [0.0, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2]
for alpha in alphas:
    pred = aq1["nodal_f0"] * alpha
    resid = aq1["mcp_mean"] - pred
    mae = resid.abs().mean()
    rmse = (resid ** 2).mean() ** 0.5
    bias = resid.mean()
    print(f"  alpha={alpha:.2f}  MAE={mae:.4f}  RMSE={rmse:.4f}  Bias={bias:+.4f}")

# By class_type
print("\n--- 8b. Optimal shrinkage by class_type ---")
for ct in ["onpeak", "offpeak"]:
    sub = aq1.filter(pl.col("class_type") == ct)
    print(f"\n  [{ct}] (n={len(sub):,})")
    best_mae = 1e12
    best_alpha = 0
    for alpha in [x / 100 for x in range(0, 150, 5)]:
        pred = sub["nodal_f0"] * alpha
        resid = sub["mcp_mean"] - pred
        mae = resid.abs().mean()
        if mae < best_mae:
            best_mae = mae
            best_alpha = alpha
    # Print neighborhood around optimum
    for alpha in [best_alpha - 0.1, best_alpha - 0.05, best_alpha, best_alpha + 0.05, best_alpha + 0.1]:
        if 0 <= alpha <= 2:
            pred = sub["nodal_f0"] * alpha
            resid = sub["mcp_mean"] - pred
            mae = resid.abs().mean()
            rmse = (resid ** 2).mean() ** 0.5
            tag = " <-- best" if abs(alpha - best_alpha) < 0.01 else ""
            print(f"    alpha={alpha:.2f}  MAE={mae:.4f}  RMSE={rmse:.4f}{tag}")

# --- 8c. Optimal shrinkage by PY ---
print("\n--- 8c. Optimal shrinkage by PY ---")
for py in sorted(aq1["planning_year"].unique().to_list()):
    sub = aq1.filter(pl.col("planning_year") == py)
    best_mae = 1e12
    best_alpha = 0
    for alpha in [x / 100 for x in range(0, 150, 5)]:
        pred = sub["nodal_f0"] * alpha
        resid = sub["mcp_mean"] - pred
        mae = resid.abs().mean()
        if mae < best_mae:
            best_mae = mae
            best_alpha = alpha
    # Also get baseline MAE (alpha=1)
    base_mae = (sub["mcp_mean"] - sub["nodal_f0"]).abs().mean()
    print(f"  PY {py}: optimal alpha={best_alpha:.2f}  MAE={best_mae:.4f}  (baseline MAE={base_mae:.4f}, improvement={100*(base_mae - best_mae)/base_mae:.1f}%)")

# ============================================================
# 9.  Blending nodal_f0 with f0_path_corr
# ============================================================
print("\n" + "=" * 70)
print("SECTION 9: Blending nodal_f0 with f0_path_corr")
print("=" * 70)

aq1_blend = aq1.filter(pl.col("f0_path_corr").is_not_null())
print(f"  Rows with both nodal_f0 and f0_path_corr: {len(aq1_blend):,}")

print("\n--- 9a. MAE with blend (w * nodal_f0 + (1-w) * f0_path_corr) ---")
for w in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    pred = w * aq1_blend["nodal_f0"] + (1 - w) * aq1_blend["f0_path_corr"]
    resid = aq1_blend["mcp_mean"] - pred
    mae = resid.abs().mean()
    rmse = (resid ** 2).mean() ** 0.5
    print(f"  w={w:.1f}  MAE={mae:.4f}  RMSE={rmse:.4f}")

del aq1, aq1_blend
gc.collect()

print(f"\n[done] mem = {mem_mb():.0f} MB")
print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
