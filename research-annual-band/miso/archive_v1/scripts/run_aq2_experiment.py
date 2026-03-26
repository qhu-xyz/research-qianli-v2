# DEPRECATED — archived 2026-03-17. Old data sources. See task_plan.md for porting plan.
import warnings as _w; _w.warn("This script is DEPRECATED (archived 2026-03-17). Use canonical get_trades_of_given_duration().", DeprecationWarning, stacklevel=2)

#!/usr/bin/env python3
"""
aq2 (Sep-Nov) R1 baseline experiment — comprehensive comparison.

Tests all available baseline sources for predicting R1 aq2 MCP:
  1. H              — historical DA congestion (current production)
  2. Prior year R3   — PY-1 R3 aq2 path MCP
  3. Prior year R2   — PY-1 R2 aq2 path MCP
  4. Prior year R1   — PY-1 R1 aq2 path MCP
  5. Nodal f0        — nodal stitched from 3-month avg, corrected year (PY-1)
  6. f0 path corr    — prior year f0 path avg for Sep/Oct/Nov (corrected)
  7. f1 path prior   — prior year f1 avg for Sep/Oct/Nov
  8. q2 path         — prior year q2 quarterly forward MCP (NEW for aq2)

Buggy baselines (PY-2 year mapping) are dropped — bug proven in aq1.
q1 does not exist, but q2 DOES: auctioned July of prior year, delivers Sep/Oct/Nov.

Year mapping:
  PY 2024 aq2 delivers Sep/Oct/Nov 2024.
  "Prior year same months" = Sep/Oct/Nov 2023 → dy = PY - 1.

Usage:
    cd /home/xyz/workspace/pmodel && source .venv/bin/activate
    python /home/xyz/workspace/research-qianli-v2/research-annual/scripts/run_aq2_experiment.py
"""

import gc
import os
import sys
import warnings

import polars as pl

# Add scripts/ to path for shared module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baseline_utils import (
    mem_mb, eval_baseline, print_main_table, print_mae_table,
    build_py_fwd_map, get_all_aliases, print_cascade_stats,
    CASCADE_HEADER, CASCADE_SEP,
)

warnings.filterwarnings("ignore")

DATA_DIR = "/opt/temp/qianli/annual_research"
WORK_DIR = f"{DATA_DIR}/crossproduct_work"
AQ2_MONTHS = [9, 10, 11]


# ======================================================================
# STEP 1: Load aq2 R1 paths from existing Phase 2 output
# ======================================================================
print(f"{'='*80}")
print(f"STEP 1: Load aq2 R1 paths | mem: {mem_mb():.0f} MB")
print(f"{'='*80}")

r1_all = pl.read_parquet(f"{WORK_DIR}/r1_with_nodal.parquet")
aq2 = r1_all.filter(pl.col("period_type") == "aq2").clone()
del r1_all
gc.collect()

# Drop nodal_r3 (buggy column from Phase 2, bug proven in aq1)
if "nodal_r3" in aq2.columns:
    aq2 = aq2.drop("nodal_r3")

total_n = aq2.height
pys = sorted(aq2["planning_year"].unique().to_list())
n_pys = len(pys)
n_pys_with_prior = len([py for py in pys if int(py) > 2019])
print(f"  aq2 rows: {total_n:,}")
print(f"  PYs: {pys} ({n_pys} total, {n_pys_with_prior} with prior-year data)")
print(f"  Existing coverage:")
for col in ["mtm_1st_mean", "prior_r3_path"]:
    if col in aq2.columns:
        cov = aq2.filter(pl.col(col).is_not_null()).height
        print(f"    {col}: {cov:,} / {total_n:,} ({cov/total_n*100:.1f}%)")
print(f"  mem: {mem_mb():.0f} MB")


# ======================================================================
# STEP 2: Add prior year R2 and R1 path MCPs
# ======================================================================
print(f"\n{'='*80}")
print(f"STEP 2: Prior year R2 and R1 path MCPs | mem: {mem_mb():.0f} MB")
print(f"{'='*80}")

cols = ["planning_year", "round", "period_type", "class_type", "source_id", "sink_id", "mcp_mean"]
all_res = pl.read_parquet(f"{DATA_DIR}/all_residuals_v2.parquet", columns=cols)
print(f"  all_residuals: {all_res.height:,} rows")

paths_aq2 = (
    all_res.filter(pl.col("period_type") == "aq2")
    .group_by(["planning_year", "round", "class_type", "source_id", "sink_id"])
    .agg(pl.col("mcp_mean").first())
)
del all_res
gc.collect()

r2_prior = (
    paths_aq2.filter(pl.col("round") == 2)
    .select(
        (pl.col("planning_year") + 1).alias("planning_year"),
        "class_type", "source_id", "sink_id",
        pl.col("mcp_mean").alias("prior_r2_path"),
    )
)
r1_prior = (
    paths_aq2.filter(pl.col("round") == 1)
    .select(
        (pl.col("planning_year") + 1).alias("planning_year"),
        "class_type", "source_id", "sink_id",
        pl.col("mcp_mean").alias("prior_r1_path"),
    )
)
del paths_aq2
gc.collect()

aq2 = aq2.cast({"planning_year": pl.Int64})
r2_prior = r2_prior.cast({"planning_year": pl.Int64})
r1_prior = r1_prior.cast({"planning_year": pl.Int64})

join_keys = ["planning_year", "class_type", "source_id", "sink_id"]
pre = aq2.height
aq2 = aq2.join(r2_prior, on=join_keys, how="left")
assert aq2.height == pre, f"Row count changed after R2 join: {pre} -> {aq2.height}"
aq2 = aq2.join(r1_prior, on=join_keys, how="left")
assert aq2.height == pre, f"Row count changed after R1 join: {pre} -> {aq2.height}"
del r2_prior, r1_prior
gc.collect()

for col in ["prior_r3_path", "prior_r2_path", "prior_r1_path"]:
    if col in aq2.columns:
        cov = aq2.filter(pl.col(col).is_not_null()).height
        print(f"  {col}: {cov:,} / {total_n:,} ({cov/total_n*100:.1f}%)")
print(f"  mem: {mem_mb():.0f} MB")


# ======================================================================
# STEP 3: f0 path-level with corrected year mapping
# ======================================================================
print(f"\n{'='*80}")
print(f"STEP 3: f0 path-level cross-product | mem: {mem_mb():.0f} MB")
print(f"{'='*80}")

f0 = (
    pl.scan_parquet(f"{DATA_DIR}/f0p_cleared_all.parquet")
    .filter(
        (pl.col("period_type") == "f0")
        & pl.col("split_market_month").dt.month().is_in(AQ2_MONTHS)
    )
    .select(
        pl.col("class_type").cast(pl.String),
        "source_id", "sink_id", "mcp",
        pl.col("split_market_month").dt.month().cast(pl.Int8).alias("dm_num"),
        pl.col("split_market_month").dt.year().cast(pl.Int16).alias("dy"),
        "round",
    )
    .collect()
)
print(f"  f0 Sep/Oct/Nov rows: {f0.height:,}")

f0m = (
    f0.sort("round")
    .group_by(["dy", "dm_num", "class_type", "source_id", "sink_id"])
    .agg(pl.col("mcp").last().alias("f0_mcp"))
)
del f0
gc.collect()

dy_range = sorted(f0m["dy"].unique().to_list())
print(f"  f0 deduped: {f0m.height:,}, delivery years: {dy_range}")

# Year mapping: for aq2, delivery months 9/10/11 >= 6 → dy = PY - 1
print(f"\n  Year mapping verification:")
print(f"  {'PY':>6} | {'Corrected (PY-1)':>16} | Delivery months")
for py in pys:
    corr = int(py) - 1
    avail = "Y" if corr in dy_range else "N"
    print(f"  {int(py):>6} | {corr:>12} [{avail}] | Sep/Oct/Nov {corr}")

map_corrected = []
for py in pys:
    for m in AQ2_MONTHS:
        map_corrected.append({"planning_year": int(py), "dy": int(py) - 1, "dm_num": int(m)})

mdf_c = pl.DataFrame(map_corrected).cast({"dy": pl.Int16, "dm_num": pl.Int8})
f0m_cast = f0m.cast({"dy": pl.Int16, "dm_num": pl.Int8})

f0_avg_c = (
    mdf_c.join(f0m_cast, on=["dy", "dm_num"], how="inner")
    .group_by(["planning_year", "class_type", "source_id", "sink_id"])
    .agg(pl.col("f0_mcp").mean().alias("f0_path_corr"), pl.col("f0_mcp").count().alias("_nc"))
)
del f0m, f0m_cast, mdf_c
gc.collect()

print(f"  f0 CORRECTED: {f0_avg_c.height:,} path averages")
print(f"  Month counts:\n{f0_avg_c['_nc'].value_counts().sort('_nc')}")

f0_avg_c = f0_avg_c.cast({"planning_year": pl.Int64})

pre = aq2.height
aq2 = aq2.join(f0_avg_c.drop("_nc"), on=join_keys, how="left")
assert aq2.height == pre, f"Row count changed after f0 join: {pre} -> {aq2.height}"
del f0_avg_c
gc.collect()

print(f"\n  f0 path coverage by PY:")
for py in pys:
    sub = aq2.filter(pl.col("planning_year") == py)
    cc = sub.filter(pl.col("f0_path_corr").is_not_null()).height
    print(f"    PY {int(py)}: corr={cc:,}/{sub.height:,} ({cc/sub.height*100:.1f}%)")
print(f"  mem: {mem_mb():.0f} MB")


# ======================================================================
# STEP 4: f1 prior year path-level
# ======================================================================
print(f"\n{'='*80}")
print(f"STEP 4: f1 prior year | mem: {mem_mb():.0f} MB")
print(f"{'='*80}")

f1 = (
    pl.scan_parquet(f"{DATA_DIR}/f0p_cleared_all.parquet")
    .filter(
        (pl.col("period_type") == "f1")
        & pl.col("split_market_month").dt.month().is_in(AQ2_MONTHS)
    )
    .select(
        pl.col("class_type").cast(pl.String),
        "source_id", "sink_id", "mcp",
        pl.col("split_market_month").dt.month().cast(pl.Int8).alias("dm_num"),
        pl.col("split_market_month").dt.year().cast(pl.Int16).alias("dy"),
        "round",
    )
    .collect()
)
print(f"  f1 Sep/Oct/Nov rows: {f1.height:,}")

if f1.height > 0:
    f1m = (
        f1.sort("round")
        .group_by(["dy", "dm_num", "class_type", "source_id", "sink_id"])
        .agg(pl.col("mcp").last().alias("f1_mcp"))
    )
    del f1
    gc.collect()

    map_f1 = []
    for py in pys:
        for m in AQ2_MONTHS:
            map_f1.append({"planning_year": int(py), "dy": int(py) - 1, "dm_num": int(m)})
    mdf_f1 = pl.DataFrame(map_f1).cast({"dy": pl.Int16, "dm_num": pl.Int8})
    f1_avg = (
        mdf_f1.join(f1m.cast({"dy": pl.Int16, "dm_num": pl.Int8}), on=["dy", "dm_num"], how="inner")
        .group_by(["planning_year", "class_type", "source_id", "sink_id"])
        .agg(pl.col("f1_mcp").mean().alias("f1_path"))
    )
    del f1m, mdf_f1
    gc.collect()

    pre = aq2.height
    aq2 = aq2.join(f1_avg.cast({"planning_year": pl.Int64}), on=join_keys, how="left")
    assert aq2.height == pre, f"Row count changed after f1 join: {pre} -> {aq2.height}"
    del f1_avg
    gc.collect()

    cov = aq2.filter(pl.col("f1_path").is_not_null()).height
    print(f"  f1 path coverage: {cov:,} / {total_n:,} ({cov/total_n*100:.1f}%)")
else:
    aq2 = aq2.with_columns(pl.lit(None).cast(pl.Float64).alias("f1_path"))
    print(f"  No f1 data for aq2 months")
    del f1
    gc.collect()
print(f"  mem: {mem_mb():.0f} MB")


# ======================================================================
# STEP 5: Ray session — q2 quarterly forward + Nodal f0 stitching
# ======================================================================
print(f"\n{'='*80}")
print(f"STEP 5: Ray session (q2 + Nodal f0) | mem: {mem_mb():.0f} MB")
print(f"{'='*80}")
print(f"  Loading Ray + MisoCalculator + MisoNodalReplacement + MisoApTools...")

from pbase.config.ray import init_ray
import pmodel
init_ray(address="ray://10.8.0.36:10001", extra_modules=[pmodel])

from pbase.data.m2m.calculator import MisoCalculator
from pbase.data.dataset.replacement import MisoNodalReplacement
from pbase.analysis.tools.all_positions import MisoApTools

import pandas as pd

# --- 5a: q2 quarterly forward MCPs ---
print(f"\n  --- Step 5a: q2 quarterly forward ---")

miso_aptools = MisoApTools()
q2_slices: list[pl.DataFrame] = []

for py in pys:
    prior_year = int(py) - 1
    if prior_year < 2019:
        print(f"  PY {int(py)}: skip q2 (prior year {prior_year} < 2019)")
        continue

    # q2 is auctioned in July of the prior year, delivers Sep/Oct/Nov of the prior year.
    # Load trades from July through end of December of prior year to capture q2.
    trades = miso_aptools.get_all_cleared_trades(
        start_date=pd.Timestamp(f"{prior_year}-07-01"),
        end_date=pd.Timestamp(f"{prior_year}-12-31"),
    )
    q2_trades = trades[trades["period_type"] == "q2"].copy()
    miso_aptools.tools.cast_category_to_str(q2_trades)
    del trades
    gc.collect()

    n_q2 = len(q2_trades)
    print(f"  PY {int(py)}: {n_q2:,} q2 trades from {prior_year}")

    if n_q2 > 0:
        # Vectorized: average MCP per path (across split months), then convert to polars
        agg_pdf = (
            q2_trades
            .groupby(["class_type", "source_id", "sink_id"])["mcp"]
            .mean()
            .reset_index()
            .rename(columns={"mcp": "q2_path"})
        )
        agg_pdf["planning_year"] = int(py)
        py_slice = pl.from_pandas(
            agg_pdf[["planning_year", "class_type", "source_id", "sink_id", "q2_path"]]
        )
        q2_slices.append(py_slice)
        del agg_pdf

    del q2_trades
    gc.collect()

if q2_slices:
    q2_df = pl.concat(q2_slices).cast({
        "planning_year": pl.Int64,
        "class_type": pl.String,
        "source_id": pl.String,
        "sink_id": pl.String,
    })
    del q2_slices
    gc.collect()

    pre = aq2.height
    aq2 = aq2.join(q2_df, on=join_keys, how="left")
    assert aq2.height == pre, f"Row count changed after q2 join: {pre} -> {aq2.height}"
    del q2_df
    gc.collect()

    cov = aq2.filter(pl.col("q2_path").is_not_null()).height
    print(f"  q2 path coverage: {cov:,} / {total_n:,} ({cov/total_n*100:.1f}%)")
else:
    del q2_slices
    gc.collect()
    aq2 = aq2.with_columns(pl.lit(None).cast(pl.Float64).alias("q2_path"))
    print(f"  No q2 data found")
print(f"  mem: {mem_mb():.0f} MB")


# --- 5b: Nodal f0 stitching (corrected year, 3-month avg) ---
print(f"\n  --- Step 5b: Nodal f0 stitching ---")

repl_raw = MisoNodalReplacement().load_data()
print(f"  Replacement records: {len(repl_raw):,}")


calc = MisoCalculator()

nodal_rows: list[tuple] = []

for py in pys:
    prior_year = int(py) - 1
    if prior_year < 2019:
        print(f"  PY {int(py)}: skip nodal (prior year {prior_year} < 2019)")
        continue

    # For aq2, target date = {PY}-11 (last delivery month of the quarter)
    target_date = f"{int(py)}-11"
    fwd_map_py = build_py_fwd_map(repl_raw, target_date)
    rev_map_py: dict[str, list[str]] = {}
    for fn, tn in fwd_map_py.items():
        rev_map_py.setdefault(tn, []).append(fn)
    print(f"  PY {int(py)}: {len(fwd_map_py)} active replacements at {target_date}")

    for m in AQ2_MONTHS:
        mm = f"{prior_year}-{m:02d}"
        try:
            mcp_df, _ = calc.get_mcp_df(market_month=mm, fillna=True)
            if mcp_df is None or len(mcp_df) == 0:
                print(f"    {mm}: no data")
                continue
        except Exception as e:
            print(f"    {mm}: ERR {e}")
            continue

        for ct in ["onpeak", "offpeak"]:
            try:
                ct_data = mcp_df.xs(ct, level="class_type")
                # Column 0 = f0 monthly forward (same-month, round 1).
                # Column -1 = annual R1 (WRONG — was the bug).
                raw = ct_data[ct_data.columns[0]].dropna()
                for nid, val in raw.items():
                    aliases = get_all_aliases(str(nid), fwd_map_py, rev_map_py)
                    for alias in aliases:
                        nodal_rows.append((int(py), ct, alias, float(val), m))
            except (KeyError, IndexError):
                pass

        del mcp_df

    gc.collect()
    n_py = sum(1 for r in nodal_rows if r[0] == py)
    print(f"  PY {int(py)}: {n_py:,} nodal entries (3 months), mem: {mem_mb():.0f} MB")

del repl_raw
gc.collect()

# Shutdown Ray — all remote calls done
import ray
ray.shutdown()
del calc, miso_aptools
gc.collect()
print(f"  Ray shutdown, mem: {mem_mb():.0f} MB")

# Build nodal lookup: average across 3 months per (py, ct, node)
nodal_df = pl.DataFrame(
    {
        "planning_year": [r[0] for r in nodal_rows],
        "class_type": [r[1] for r in nodal_rows],
        "node_id": [r[2] for r in nodal_rows],
        "node_mcp": [r[3] for r in nodal_rows],
        "month": [r[4] for r in nodal_rows],
    },
    schema={
        "planning_year": pl.Int64, "class_type": pl.String,
        "node_id": pl.String, "node_mcp": pl.Float64, "month": pl.Int8,
    },
)
del nodal_rows
gc.collect()

# Dedupe per (py, ct, node, month) then average across months
nodal_dedup = (
    nodal_df.unique(subset=["planning_year", "class_type", "node_id", "month"], keep="last")
)
nodal_avg = (
    nodal_dedup.group_by(["planning_year", "class_type", "node_id"])
    .agg(
        pl.col("node_mcp").mean().alias("node_mcp_avg"),
        pl.col("node_mcp").count().alias("n_months"),
    )
)
del nodal_df, nodal_dedup
gc.collect()

month_dist = nodal_avg["n_months"].value_counts().sort("n_months")
print(f"  Nodal avg lookup: {nodal_avg.height:,} nodes")
print(f"  Month coverage:\n{month_dist}")

# Vectorized stitch: join source, then sink, compute path MCP
src_lookup = nodal_avg.select(
    "planning_year", "class_type",
    pl.col("node_id").alias("source_id"),
    pl.col("node_mcp_avg").alias("_src_f0"),
)
snk_lookup = nodal_avg.select(
    "planning_year", "class_type",
    pl.col("node_id").alias("sink_id"),
    pl.col("node_mcp_avg").alias("_snk_f0"),
)
del nodal_avg
gc.collect()

nj_keys = ["planning_year", "class_type"]
pre = aq2.height
aq2 = aq2.join(src_lookup, on=nj_keys + ["source_id"], how="left")
assert aq2.height == pre, f"Row count changed after src join: {pre} -> {aq2.height}"
aq2 = aq2.join(snk_lookup, on=nj_keys + ["sink_id"], how="left")
assert aq2.height == pre, f"Row count changed after snk join: {pre} -> {aq2.height}"
del src_lookup, snk_lookup
gc.collect()

aq2 = aq2.with_columns(
    pl.when(pl.col("_src_f0").is_not_null() & pl.col("_snk_f0").is_not_null())
    .then(pl.col("_snk_f0") - pl.col("_src_f0"))
    .otherwise(None)
    .alias("nodal_f0")
).drop(["_src_f0", "_snk_f0"])

cov = aq2.filter(pl.col("nodal_f0").is_not_null()).height
print(f"\n  Nodal f0 (corrected, 3-mo):  {cov:,} / {total_n:,} ({cov/total_n*100:.1f}%)")

# Per-PY coverage
print(f"  Nodal f0 coverage by PY:")
for py in pys:
    sub = aq2.filter(pl.col("planning_year") == py)
    nc = sub.filter(pl.col("nodal_f0").is_not_null()).height
    print(f"    PY {int(py)}: {nc:,}/{sub.height:,} ({nc/sub.height*100:.1f}%)")
print(f"  mem: {mem_mb():.0f} MB")


# ======================================================================
# STEP 5c: VERIFY — nodal f0 stitching matches f0 path on matched paths
# ======================================================================
print(f"\n{'='*80}")
print(f"STEP 5c: Nodal f0 verification | mem: {mem_mb():.0f} MB")
print(f"{'='*80}")

import numpy as np

both = aq2.filter(pl.col("f0_path_corr").is_not_null() & pl.col("nodal_f0").is_not_null())
n_both = both.height
print(f"  Paths with both f0 path and nodal f0: {n_both:,}")

if n_both > 0:
    f0v = both["f0_path_corr"].to_numpy()
    nv = both["nodal_f0"].to_numpy()
    diff = np.abs(f0v - nv)

    exact_01 = int((diff < 0.01).sum())
    close_1 = int((diff < 1.0).sum())
    close_10 = int((diff < 10.0).sum())

    print(f"  Exact match (diff < $0.01):  {exact_01:,} ({exact_01/n_both*100:.1f}%)")
    print(f"  Close match (diff < $1.00):  {close_1:,} ({close_1/n_both*100:.1f}%)")
    print(f"  Close match (diff < $10.00): {close_10:,} ({close_10/n_both*100:.1f}%)")
    print(f"  Mean |diff|: ${np.mean(diff):.2f}")
    print(f"  Median |diff|: ${np.median(diff):.2f}")
    print(f"  p95 |diff|: ${np.percentile(diff, 95):.2f}")
    print(f"  Max |diff|: ${np.max(diff):.2f}")

    # Check outliers for .AZ nodes
    outlier_mask = diff >= 10.0
    n_outliers = int(outlier_mask.sum())
    if n_outliers > 0:
        outlier_df = both.filter(pl.lit(outlier_mask))
        has_az = outlier_df.filter(
            pl.col("source_id").str.contains(".AZ") | pl.col("sink_id").str.contains(".AZ")
        ).height
        print(f"  Outliers (diff >= $10): {n_outliers:,}, of which .AZ nodes: {has_az:,} ({has_az/n_outliers*100:.0f}%)")

    # Win rate: which baseline is closer to actual?
    actual_both = both["mcp_mean"].to_numpy()
    ae_f0 = np.abs(actual_both - f0v)
    ae_nodal = np.abs(actual_both - nv)
    f0_wins = int((ae_f0 < ae_nodal).sum())
    nodal_wins = int((ae_nodal < ae_f0).sum())
    ties = n_both - f0_wins - nodal_wins
    print(f"\n  Win rate (which is closer to actual MCP):")
    print(f"    f0 path wins:  {f0_wins:,} ({f0_wins/n_both*100:.1f}%)")
    print(f"    Nodal f0 wins: {nodal_wins:,} ({nodal_wins/n_both*100:.1f}%)")
    print(f"    Tied:          {ties:,} ({ties/n_both*100:.1f}%)")

    del f0v, nv, diff, actual_both, ae_f0, ae_nodal
    del both
    gc.collect()


# ======================================================================
# STEP 6: RESULTS — Comprehensive Comparison
# ======================================================================
print(f"\n{'='*80}")
print(f"RESULTS: aq2 (Sep-Nov) R1 Baseline Comparison")
print(f"{'='*80}")
print(f"  Total paths: {total_n:,}")
print(f"  PYs tested: {pys} ({n_pys} years, {n_pys_with_prior} with prior-year data)")
print(f"  Note: |MCP| in 'Dir>50', 'Dir>100' means ACTUAL aq2 MCP for that path")

baselines = [
    ("mtm_1st_mean",  "H (production)"),
    ("prior_r3_path", "Prior R3 path"),
    ("prior_r2_path", "Prior R2 path"),
    ("prior_r1_path", "Prior R1 path"),
    ("nodal_f0",      "Nodal f0 (PY-1,3mo)"),
    ("f0_path_corr",  "f0 path corr (PY-1)"),
    ("f1_path",       "f1 path (PY-1)"),
    ("q2_path",       "q2 path (PY-1)"),
]

# --- Section A: Overall (PY 2020-2025 only) ---
aq2_ex19 = aq2.filter(pl.col("planning_year") >= 2020)
total_ex19 = aq2_ex19.height

results_all = []
for col, label in baselines:
    if col in aq2_ex19.columns:
        results_all.append(eval_baseline(aq2_ex19["mcp_mean"], aq2_ex19[col], label, total_ex19))

print_main_table(results_all, f"Section A: Overall PY 2020-2025 ({total_ex19:,} paths, excludes PY 2019)")
print_mae_table(results_all, "Section A (cont): MAE by |MCP| bin")

# --- Section B: By PY ---
print(f"\n{'='*80}")
print(f"Section B: By Planning Year")
print(f"{'='*80}")

for py in pys:
    sub = aq2.filter(pl.col("planning_year") == py)
    results_py = []
    for col, label in baselines:
        if col in sub.columns:
            results_py.append(eval_baseline(sub["mcp_mean"], sub[col], label, sub.height))
    print_main_table(results_py, f"  PY {int(py)} ({sub.height:,} paths)")

# --- Section C: Head-to-head on matched paths ---
print(f"\n{'='*80}")
print(f"Section C: Head-to-head (same paths)")
print(f"{'='*80}")

# C1: Nodal f0 vs H
nh = aq2.filter(pl.col("nodal_f0").is_not_null() & pl.col("mtm_1st_mean").is_not_null())
if nh.height > 0:
    results_c1 = [
        eval_baseline(nh["mcp_mean"], nh["nodal_f0"], "Nodal f0 (corr)", nh.height),
        eval_baseline(nh["mcp_mean"], nh["mtm_1st_mean"], "H", nh.height),
    ]
    print_main_table(results_c1, f"  C1: Nodal f0 vs H ({nh.height:,} overlap)")
    print_mae_table(results_c1, f"  C1 (cont): MAE by |MCP| bin")

# C2: f0 path vs nodal f0 vs H (on f0-path-covered paths)
f0h = aq2.filter(
    pl.col("f0_path_corr").is_not_null()
    & pl.col("nodal_f0").is_not_null()
    & pl.col("mtm_1st_mean").is_not_null()
)
if f0h.height > 0:
    results_c2 = [
        eval_baseline(f0h["mcp_mean"], f0h["f0_path_corr"], "f0 path corr", f0h.height),
        eval_baseline(f0h["mcp_mean"], f0h["nodal_f0"], "Nodal f0", f0h.height),
        eval_baseline(f0h["mcp_mean"], f0h["mtm_1st_mean"], "H", f0h.height),
    ]
    print_main_table(results_c2, f"  C2: f0 path vs Nodal f0 vs H ({f0h.height:,} all-covered)")
    print_mae_table(results_c2, f"  C2 (cont): MAE by |MCP| bin")

# C3: q2 vs f0 path vs nodal f0 (where all 3 are available)
q2h = aq2.filter(
    pl.col("q2_path").is_not_null()
    & pl.col("f0_path_corr").is_not_null()
    & pl.col("nodal_f0").is_not_null()
)
if q2h.height > 100:
    results_c3 = [
        eval_baseline(q2h["mcp_mean"], q2h["q2_path"], "q2 path", q2h.height),
        eval_baseline(q2h["mcp_mean"], q2h["f0_path_corr"], "f0 path corr", q2h.height),
        eval_baseline(q2h["mcp_mean"], q2h["nodal_f0"], "Nodal f0", q2h.height),
        eval_baseline(q2h["mcp_mean"], q2h["mtm_1st_mean"], "H", q2h.height),
    ]
    print_main_table(results_c3, f"  C3: q2 vs f0 vs Nodal f0 vs H ({q2h.height:,} all-covered)")
    print_mae_table(results_c3, f"  C3 (cont): MAE by |MCP| bin")

# C4: All non-buggy baselines on fully-matched paths
mask_all = pl.lit(True)
for c, _ in baselines:
    if c in aq2.columns:
        mask_all = mask_all & pl.col(c).is_not_null()
all_match = aq2.filter(mask_all)
if all_match.height > 100:
    results_c4 = []
    for col, label in baselines:
        if col in all_match.columns:
            results_c4.append(eval_baseline(all_match["mcp_mean"], all_match[col], label, all_match.height))
    print_main_table(results_c4, f"  C4: All baselines, fully matched ({all_match.height:,} paths)")
    print_mae_table(results_c4, f"  C4 (cont): MAE by |MCP| bin")

# --- Section D: By class_type ---
print(f"\n{'='*80}")
print(f"Section D: By class_type")
print(f"{'='*80}")

for ct in ["onpeak", "offpeak"]:
    sub = aq2.filter(pl.col("class_type") == ct)
    results_ct = []
    for col, label in baselines:
        if col in sub.columns:
            results_ct.append(eval_baseline(sub["mcp_mean"], sub[col], label, sub.height))
    print_main_table(results_ct, f"  {ct} ({sub.height:,} paths)")
    print_mae_table(results_ct, f"  {ct} MAE by |MCP| bin")

# --- Section E: By |MCP| magnitude ---
print(f"\n{'='*80}")
print(f"Section E: By |MCP| magnitude bin")
print(f"{'='*80}")

bins = [
    (0, 50, "tiny (<50)"),
    (50, 250, "small (50-250)"),
    (250, 1000, "medium (250-1k)"),
    (1000, 999999, "large (1k+)"),
]
key_baselines = [
    ("mtm_1st_mean", "H"),
    ("prior_r3_path", "R3 path"),
    ("nodal_f0", "Nodal f0"),
    ("f0_path_corr", "f0 path corr"),
    ("f1_path", "f1 path"),
    ("q2_path", "q2 path"),
]
for lo, hi, bin_label in bins:
    sub = aq2.filter((pl.col("mcp_mean").abs() >= lo) & (pl.col("mcp_mean").abs() < hi))
    if sub.height == 0:
        continue
    results_bin = []
    for col, label in key_baselines:
        if col in sub.columns:
            results_bin.append(eval_baseline(sub["mcp_mean"], sub[col], label, sub.height))
    print_main_table(results_bin, f"  |MCP| {bin_label} ({sub.height:,} paths)")


# ======================================================================
# STEP 7: CASCADE ANALYSIS
# ======================================================================
print(f"\n{'='*80}")
print(f"Section F: Cascade comparison (PY 2020-2025)")
print(f"{'='*80}")

import numpy as np

actual_arr = aq2_ex19["mcp_mean"].to_numpy().astype(float)
nodal_arr = aq2_ex19["nodal_f0"].to_numpy().astype(float)
f0p_arr = aq2_ex19["f0_path_corr"].to_numpy().astype(float)
h_arr = aq2_ex19["mtm_1st_mean"].to_numpy().astype(float)

print(CASCADE_HEADER)
print(CASCADE_SEP)

# A: Nodal f0 → H (recommended 2-tier)
cascade_a = np.where(~np.isnan(nodal_arr), nodal_arr,
            np.where(~np.isnan(h_arr), h_arr, np.nan))
print_cascade_stats(actual_arr, cascade_a, "A: Nodal f0 → H (2-tier)", total_ex19)

# B: f0 path → Nodal f0 → H (old 3-tier)
cascade_b = np.where(~np.isnan(f0p_arr), f0p_arr,
            np.where(~np.isnan(nodal_arr), nodal_arr,
            np.where(~np.isnan(h_arr), h_arr, np.nan)))
print_cascade_stats(actual_arr, cascade_b, "B: f0 path → Nodal → H (3-tier)", total_ex19)

# C: Nodal f0 only (no fallback)
print_cascade_stats(actual_arr, nodal_arr, "C: Nodal f0 only", total_ex19)

# D: H only (production)
print_cascade_stats(actual_arr, h_arr, "D: H only (production)", total_ex19)

# Tier allocation
n_nodal = int((~np.isnan(nodal_arr)).sum())
n_h_only = total_ex19 - n_nodal
print(f"\n  Tier allocation (2-tier cascade):")
print(f"    Tier 1 (Nodal f0): {n_nodal:,} ({n_nodal/total_ex19*100:.1f}%)")
print(f"    Tier 2 (H):        {n_h_only:,} ({n_h_only/total_ex19*100:.1f}%)")

# Marginal value of f0 path
print(f"\n  Marginal value of f0 path tier:")
f0_covered = aq2_ex19.filter(pl.col("f0_path_corr").is_not_null() & pl.col("nodal_f0").is_not_null())
n_fc = f0_covered.height
a_fc = f0_covered["mcp_mean"].to_numpy()
f0_fc = f0_covered["f0_path_corr"].to_numpy()
n_fc_vals = f0_covered["nodal_f0"].to_numpy()
ae_f0_fc = np.abs(a_fc - f0_fc)
ae_n_fc = np.abs(a_fc - n_fc_vals)
print(f"    On {n_fc:,} paths where both exist:")
print(f"      f0 path:  MAE {np.mean(ae_f0_fc):.0f}, Med {np.median(ae_f0_fc):.0f}, p95 {np.percentile(ae_f0_fc, 95):.0f}")
print(f"      Nodal f0: MAE {np.mean(ae_n_fc):.0f}, Med {np.median(ae_n_fc):.0f}, p95 {np.percentile(ae_n_fc, 95):.0f}")
print(f"      Delta:    MAE {np.mean(ae_f0_fc) - np.mean(ae_n_fc):+.0f}")

del actual_arr, nodal_arr, f0p_arr, h_arr, cascade_a, cascade_b
del f0_covered, a_fc, f0_fc, n_fc_vals, ae_f0_fc, ae_n_fc
gc.collect()


# ======================================================================
# SAVE
# ======================================================================
out_path = f"{WORK_DIR}/aq2_all_baselines.parquet"
aq2.write_parquet(out_path)
print(f"\nSaved: {out_path} ({aq2.shape})")
print(f"Columns: {aq2.columns}")
print(f"Final mem: {mem_mb():.0f} MB")
print("DONE")
