#!/usr/bin/env python3
"""
aq4 (Mar-May) R1 baseline experiment — comprehensive comparison.

Tests all available baseline sources for predicting R1 aq4 MCP:
  1. H              — historical DA congestion (current production)
  2. Prior year R3   — PY-1 R3 aq4 path MCP
  3. Prior year R2   — PY-1 R2 aq4 path MCP
  4. Prior year R1   — PY-1 R1 aq4 path MCP
  5. Nodal f0        — nodal stitched from 3-month avg, corrected year
  6. f0 path corr    — prior year f0 path avg for Mar/Apr/May (corrected)
  7. f1 path prior   — prior year f1 avg for Mar/Apr/May
  8. q4 path         — prior year q4 quarterly forward MCP

NOTE: aq4 delivers Mar/Apr/May of PY+1 calendar year.
  PY 2024 aq4 delivers Mar/Apr/May 2025.
  Year mapping: all months < 6, so dy = PY for all three.
    Mar (month 3, < 6): dy = PY → Mar 2024 for PY 2024
    Apr (month 4, < 6): dy = PY → Apr 2024 for PY 2024
    May (month 5, < 6): dy = PY → May 2024 for PY 2024
  Replacement target date: {PY+1}-05 (last delivery month in PY+1).
  q4 IS in f0p_cleared_all.parquet (unlike q2/q3).

SPECIAL: aq4 only gets ~1 month of DA data (March) due to April auction
cutoff in fill_mtm_1st_period_with_hist_revenue(). Direction accuracy for
H is expected to be ~62% (worst among all quarters).

Usage:
    cd /home/xyz/workspace/pmodel && source .venv/bin/activate
    python /home/xyz/workspace/research-qianli-v2/research-annual/scripts/run_aq4_experiment.py
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
AQ4_MONTHS = [3, 4, 5]


# ======================================================================
# STEP 1: Load aq4 R1 paths from existing Phase 2 output
# ======================================================================
print(f"{'='*80}")
print(f"STEP 1: Load aq4 R1 paths | mem: {mem_mb():.0f} MB")
print(f"{'='*80}")

r1_all = pl.read_parquet(f"{WORK_DIR}/r1_with_nodal.parquet")
aq4 = r1_all.filter(pl.col("period_type") == "aq4").clone()
del r1_all
gc.collect()

# Drop nodal_r3 (buggy column from Phase 2, bug proven in aq1)
if "nodal_r3" in aq4.columns:
    aq4 = aq4.drop("nodal_r3")

total_n = aq4.height
pys = sorted(aq4["planning_year"].unique().to_list())
n_pys = len(pys)
n_pys_with_prior = len([py for py in pys if int(py) > 2019])
print(f"  aq4 rows: {total_n:,}")
print(f"  PYs: {pys} ({n_pys} total, {n_pys_with_prior} with prior-year data)")
print(f"  Existing coverage:")
for col in ["mtm_1st_mean", "prior_r3_path"]:
    if col in aq4.columns:
        cov = aq4.filter(pl.col(col).is_not_null()).height
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

paths_aq4 = (
    all_res.filter(pl.col("period_type") == "aq4")
    .group_by(["planning_year", "round", "class_type", "source_id", "sink_id"])
    .agg(pl.col("mcp_mean").first())
)
del all_res
gc.collect()

r2_prior = (
    paths_aq4.filter(pl.col("round") == 2)
    .select(
        (pl.col("planning_year") + 1).alias("planning_year"),
        "class_type", "source_id", "sink_id",
        pl.col("mcp_mean").alias("prior_r2_path"),
    )
)
r1_prior = (
    paths_aq4.filter(pl.col("round") == 1)
    .select(
        (pl.col("planning_year") + 1).alias("planning_year"),
        "class_type", "source_id", "sink_id",
        pl.col("mcp_mean").alias("prior_r1_path"),
    )
)
del paths_aq4
gc.collect()

aq4 = aq4.cast({"planning_year": pl.Int64})
r2_prior = r2_prior.cast({"planning_year": pl.Int64})
r1_prior = r1_prior.cast({"planning_year": pl.Int64})

join_keys = ["planning_year", "class_type", "source_id", "sink_id"]
pre = aq4.height
aq4 = aq4.join(r2_prior, on=join_keys, how="left")
assert aq4.height == pre, f"Row count changed after R2 join: {pre} -> {aq4.height}"
aq4 = aq4.join(r1_prior, on=join_keys, how="left")
assert aq4.height == pre, f"Row count changed after R1 join: {pre} -> {aq4.height}"
del r2_prior, r1_prior
gc.collect()

for col in ["prior_r3_path", "prior_r2_path", "prior_r1_path"]:
    if col in aq4.columns:
        cov = aq4.filter(pl.col(col).is_not_null()).height
        print(f"  {col}: {cov:,} / {total_n:,} ({cov/total_n*100:.1f}%)")
print(f"  mem: {mem_mb():.0f} MB")


# ======================================================================
# STEP 3: f0 path-level with corrected year mapping
# ======================================================================
print(f"\n{'='*80}")
print(f"STEP 3: f0 path-level cross-product | mem: {mem_mb():.0f} MB")
print(f"{'='*80}")

# aq4 delivers Mar/Apr/May. All months < 6, so dy = PY.
f0 = (
    pl.scan_parquet(f"{DATA_DIR}/f0p_cleared_all.parquet")
    .filter(
        (pl.col("period_type") == "f0")
        & pl.col("split_market_month").dt.month().is_in(AQ4_MONTHS)
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
print(f"  f0 Mar/Apr/May rows: {f0.height:,}")

f0m = (
    f0.sort("round")
    .group_by(["dy", "dm_num", "class_type", "source_id", "sink_id"])
    .agg(pl.col("mcp").last().alias("f0_mcp"))
)
del f0
gc.collect()

dy_range = sorted(f0m["dy"].unique().to_list())
print(f"  f0 deduped: {f0m.height:,}, delivery years: {dy_range}")

# Year mapping: for aq4, delivery months 3/4/5 < 6 → dy = PY
print(f"\n  Year mapping verification:")
print(f"  {'PY':>6} | {'dy (=PY)':>10} | Delivery months")
for py in pys:
    dy = int(py)  # All months < 6, so dy = PY
    avail = "Y" if dy in dy_range else "N"
    print(f"  {int(py):>6} | {dy:>6} [{avail}] | Mar/Apr/May {int(py)+1}")

map_corrected = []
for py in pys:
    for m in AQ4_MONTHS:
        # All months < 6 → dy = PY
        map_corrected.append({"planning_year": int(py), "dy": int(py), "dm_num": int(m)})

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

pre = aq4.height
aq4 = aq4.join(f0_avg_c.drop("_nc"), on=join_keys, how="left")
assert aq4.height == pre, f"Row count changed after f0 join: {pre} -> {aq4.height}"
del f0_avg_c
gc.collect()

print(f"\n  f0 path coverage by PY:")
for py in pys:
    sub = aq4.filter(pl.col("planning_year") == py)
    cc = sub.filter(pl.col("f0_path_corr").is_not_null()).height
    print(f"    PY {int(py)}: corr={cc:,}/{sub.height:,} ({cc/sub.height*100:.1f}%)")
print(f"  mem: {mem_mb():.0f} MB")


# ======================================================================
# STEP 4: f1 prior year path-level + q4 from f0p parquet
# ======================================================================
print(f"\n{'='*80}")
print(f"STEP 4: f1 + q4 prior year | mem: {mem_mb():.0f} MB")
print(f"{'='*80}")

# --- 4a: f1 ---
f1 = (
    pl.scan_parquet(f"{DATA_DIR}/f0p_cleared_all.parquet")
    .filter(
        (pl.col("period_type") == "f1")
        & pl.col("split_market_month").dt.month().is_in(AQ4_MONTHS)
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
print(f"  f1 Mar/Apr/May rows: {f1.height:,}")

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
        for m in AQ4_MONTHS:
            map_f1.append({"planning_year": int(py), "dy": int(py), "dm_num": int(m)})
    mdf_f1 = pl.DataFrame(map_f1).cast({"dy": pl.Int16, "dm_num": pl.Int8})
    f1_avg = (
        mdf_f1.join(f1m.cast({"dy": pl.Int16, "dm_num": pl.Int8}), on=["dy", "dm_num"], how="inner")
        .group_by(["planning_year", "class_type", "source_id", "sink_id"])
        .agg(pl.col("f1_mcp").mean().alias("f1_path"))
    )
    del f1m, mdf_f1
    gc.collect()

    pre = aq4.height
    aq4 = aq4.join(f1_avg.cast({"planning_year": pl.Int64}), on=join_keys, how="left")
    assert aq4.height == pre, f"Row count changed after f1 join: {pre} -> {aq4.height}"
    del f1_avg
    gc.collect()

    cov = aq4.filter(pl.col("f1_path").is_not_null()).height
    print(f"  f1 path coverage: {cov:,} / {total_n:,} ({cov/total_n*100:.1f}%)")
else:
    aq4 = aq4.with_columns(pl.lit(None).cast(pl.Float64).alias("f1_path"))
    print(f"  No f1 data for aq4 months")
    del f1
    gc.collect()

# --- 4b: q4 from f0p_cleared_all.parquet (q4 IS in this file!) ---
print(f"\n  --- Step 4b: q4 quarterly forward (from parquet) ---")

q4 = (
    pl.scan_parquet(f"{DATA_DIR}/f0p_cleared_all.parquet")
    .filter(
        (pl.col("period_type") == "q4")
        & pl.col("split_market_month").dt.month().is_in(AQ4_MONTHS)
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
print(f"  q4 Mar/Apr/May rows: {q4.height:,}")

if q4.height > 0:
    q4m = (
        q4.sort("round")
        .group_by(["dy", "dm_num", "class_type", "source_id", "sink_id"])
        .agg(pl.col("mcp").last().alias("q4_mcp"))
    )
    del q4
    gc.collect()

    # Same year mapping as f0: dy = PY
    map_q4 = []
    for py in pys:
        for m in AQ4_MONTHS:
            map_q4.append({"planning_year": int(py), "dy": int(py), "dm_num": int(m)})
    mdf_q4 = pl.DataFrame(map_q4).cast({"dy": pl.Int16, "dm_num": pl.Int8})
    q4_avg = (
        mdf_q4.join(q4m.cast({"dy": pl.Int16, "dm_num": pl.Int8}), on=["dy", "dm_num"], how="inner")
        .group_by(["planning_year", "class_type", "source_id", "sink_id"])
        .agg(pl.col("q4_mcp").mean().alias("q4_path"))
    )
    del q4m, mdf_q4
    gc.collect()

    pre = aq4.height
    aq4 = aq4.join(q4_avg.cast({"planning_year": pl.Int64}), on=join_keys, how="left")
    assert aq4.height == pre, f"Row count changed after q4 join: {pre} -> {aq4.height}"
    del q4_avg
    gc.collect()

    cov = aq4.filter(pl.col("q4_path").is_not_null()).height
    print(f"  q4 path coverage: {cov:,} / {total_n:,} ({cov/total_n*100:.1f}%)")
else:
    aq4 = aq4.with_columns(pl.lit(None).cast(pl.Float64).alias("q4_path"))
    print(f"  No q4 data for aq4 months")
    del q4
    gc.collect()

print(f"  mem: {mem_mb():.0f} MB")


# ======================================================================
# STEP 5: Ray session — Nodal f0 stitching only (q4 already loaded above)
# ======================================================================
print(f"\n{'='*80}")
print(f"STEP 5: Ray session (Nodal f0) | mem: {mem_mb():.0f} MB")
print(f"{'='*80}")
print(f"  Loading Ray + MisoCalculator + MisoNodalReplacement...")

from pbase.config.ray import init_ray
import pmodel
init_ray(address="ray://10.8.0.36:10001", extra_modules=[pmodel])

from pbase.data.m2m.calculator import MisoCalculator
from pbase.data.dataset.replacement import MisoNodalReplacement

import pandas as pd


repl_raw = MisoNodalReplacement().load_data()
print(f"  Replacement records: {len(repl_raw):,}")


calc = MisoCalculator()

nodal_rows: list[tuple] = []

for py in pys:
    # For aq4: dy = PY for all months. Prior year f0 data is from {PY}-03/04/05.
    # But we need PY's f0 data to exist. PY maps to the same calendar year.
    # The check is whether f0 for Mar/Apr/May of year PY exists.
    # For PY 2019: f0 data for Mar/Apr/May 2019 should exist (these are early 2019 auctions).
    # However, PY 2019 has no prior PY annual data. The f0 data itself may or may not exist.
    # We skip PY 2019 anyway in the final analysis.
    dy = int(py)
    if dy < 2019:
        print(f"  PY {int(py)}: skip nodal (dy {dy} < 2019)")
        continue

    # For aq4, target date = {PY+1}-05 (last delivery month)
    target_date = f"{int(py) + 1}-05"
    fwd_map_py = build_py_fwd_map(repl_raw, target_date)
    rev_map_py: dict[str, list[str]] = {}
    for fn, tn in fwd_map_py.items():
        rev_map_py.setdefault(tn, []).append(fn)
    print(f"  PY {int(py)}: {len(fwd_map_py)} active replacements at {target_date}")

    for m in AQ4_MONTHS:
        # dy = PY for all months
        mm = f"{dy}-{m:02d}"
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
del calc
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
# Keep nodal_dedup for per-month verification; delete nodal_df
del nodal_df
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
pre = aq4.height
aq4 = aq4.join(src_lookup, on=nj_keys + ["source_id"], how="left")
assert aq4.height == pre, f"Row count changed after src join: {pre} -> {aq4.height}"
aq4 = aq4.join(snk_lookup, on=nj_keys + ["sink_id"], how="left")
assert aq4.height == pre, f"Row count changed after snk join: {pre} -> {aq4.height}"
del src_lookup, snk_lookup
gc.collect()

aq4 = aq4.with_columns(
    pl.when(pl.col("_src_f0").is_not_null() & pl.col("_snk_f0").is_not_null())
    .then(pl.col("_snk_f0") - pl.col("_src_f0"))
    .otherwise(None)
    .alias("nodal_f0")
).drop(["_src_f0", "_snk_f0"])

cov = aq4.filter(pl.col("nodal_f0").is_not_null()).height
print(f"\n  Nodal f0 (corrected, 3-mo):  {cov:,} / {total_n:,} ({cov/total_n*100:.1f}%)")

# Per-PY coverage
print(f"  Nodal f0 coverage by PY:")
for py in pys:
    sub = aq4.filter(pl.col("planning_year") == py)
    nc = sub.filter(pl.col("nodal_f0").is_not_null()).height
    print(f"    PY {int(py)}: {nc:,}/{sub.height:,} ({nc/sub.height*100:.1f}%)")
print(f"  mem: {mem_mb():.0f} MB")


# ======================================================================
# STEP 5c: VERIFY — per-month nodal f0 stitching + 3-month avg comparison
# ======================================================================
print(f"\n{'='*80}")
print(f"STEP 5c: Nodal f0 verification (per-month + averaged) | mem: {mem_mb():.0f} MB")
print(f"{'='*80}")

import numpy as np

# --- Per-month verification: sink_node_f0 - source_node_f0 == f0_path_mcp ---
print(f"  --- Per-month verification ---")
f0_monthly = (
    pl.scan_parquet(f"{DATA_DIR}/f0p_cleared_all.parquet")
    .filter(
        (pl.col("period_type") == "f0")
        & pl.col("split_market_month").dt.month().is_in(AQ4_MONTHS)
    )
    .select(
        pl.col("class_type").cast(pl.String),
        "source_id", "sink_id",
        pl.col("split_month_mcp"),
        pl.col("split_market_month").dt.month().cast(pl.Int8).alias("dm_num"),
        pl.col("split_market_month").dt.year().cast(pl.Int16).alias("dy"),
    )
    .collect()
)

f0_monthly_dedup = (
    f0_monthly
    .group_by(["dy", "dm_num", "class_type", "source_id", "sink_id"])
    .agg(pl.col("split_month_mcp").first().alias("path_mcp"))
)
del f0_monthly
gc.collect()

src_monthly = nodal_dedup.select(
    "planning_year", "class_type", "month",
    pl.col("node_id").alias("source_id"),
    pl.col("node_mcp").alias("src_mcp"),
)
snk_monthly = nodal_dedup.select(
    "planning_year", "class_type", "month",
    pl.col("node_id").alias("sink_id"),
    pl.col("node_mcp").alias("snk_mcp"),
)

total_per_month = 0
total_exact = 0
for m in AQ4_MONTHS:
    for py in pys:
        # aq4: dy = PY for all months
        dy = int(py)
        if dy < 2019:
            continue

        f0_m = f0_monthly_dedup.filter(
            (pl.col("dy") == dy) & (pl.col("dm_num") == m)
        )
        if f0_m.height == 0:
            continue

        src_m = src_monthly.filter(
            (pl.col("planning_year") == int(py)) & (pl.col("month") == m)
        ).select("class_type", "source_id", "src_mcp")
        snk_m = snk_monthly.filter(
            (pl.col("planning_year") == int(py)) & (pl.col("month") == m)
        ).select("class_type", "sink_id", "snk_mcp")

        check = (
            f0_m
            .join(src_m, on=["class_type", "source_id"], how="inner")
            .join(snk_m, on=["class_type", "sink_id"], how="inner")
        )
        if check.height == 0:
            continue

        stitched = (check["snk_mcp"] - check["src_mcp"]).to_numpy()
        path_actual = check["path_mcp"].to_numpy()
        diff_abs = np.abs(stitched - path_actual)
        n_exact = int((diff_abs < 0.015).sum())
        n_check = check.height
        total_per_month += n_check
        total_exact += n_exact
        pct = n_exact / n_check * 100 if n_check > 0 else 0
        if pct < 99.0:
            print(f"    WARNING: PY {int(py)} month {m} ({dy}-{m:02d}): {n_exact:,}/{n_check:,} ({pct:.1f}%) exact match")
        else:
            print(f"    PY {int(py)} month {m} ({dy}-{m:02d}): {n_exact:,}/{n_check:,} ({pct:.1f}%) exact match")

if total_per_month > 0:
    pct_total = total_exact / total_per_month * 100
    print(f"\n  PER-MONTH TOTAL: {total_exact:,}/{total_per_month:,} ({pct_total:.1f}%) exact match (within $0.015)")
else:
    print(f"\n  PER-MONTH: no matched paths to verify")

del f0_monthly_dedup, src_monthly, snk_monthly
gc.collect()

# --- 3-month averaged verification ---
print(f"\n  --- 3-month averaged verification ---")
both = aq4.filter(pl.col("f0_path_corr").is_not_null() & pl.col("nodal_f0").is_not_null())
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

del nodal_dedup
gc.collect()


# ======================================================================
# STEP 6: RESULTS — Comprehensive Comparison
# ======================================================================
print(f"\n{'='*80}")
print(f"RESULTS: aq4 (Mar-May) R1 Baseline Comparison")
print(f"{'='*80}")
print(f"  Total paths: {total_n:,}")
print(f"  PYs tested: {pys} ({n_pys} years, {n_pys_with_prior} with prior-year data)")
print(f"  Note: aq4 delivers Mar/Apr/May of PY+1; only 1 month DA data (H is weak)")

baselines = [
    ("mtm_1st_mean",  "H (production)"),
    ("prior_r3_path", "Prior R3 path"),
    ("prior_r2_path", "Prior R2 path"),
    ("prior_r1_path", "Prior R1 path"),
    ("nodal_f0",      "Nodal f0 (PY,3mo)"),
    ("f0_path_corr",  "f0 path corr (PY)"),
    ("f1_path",       "f1 path (PY)"),
    ("q4_path",       "q4 path (PY)"),
]

# --- Section A: Overall (PY 2020-2025 only) ---
aq4_ex19 = aq4.filter(pl.col("planning_year") >= 2020)
total_ex19 = aq4_ex19.height

results_all = []
for col, label in baselines:
    if col in aq4_ex19.columns:
        results_all.append(eval_baseline(aq4_ex19["mcp_mean"], aq4_ex19[col], label, total_ex19))

print_main_table(results_all, f"Section A: Overall PY 2020-2025 ({total_ex19:,} paths, excludes PY 2019)")
print_mae_table(results_all, "Section A (cont): MAE by |MCP| bin")

# --- Section B: By PY ---
print(f"\n{'='*80}")
print(f"Section B: By Planning Year")
print(f"{'='*80}")

for py in pys:
    sub = aq4.filter(pl.col("planning_year") == py)
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
nh = aq4.filter(pl.col("nodal_f0").is_not_null() & pl.col("mtm_1st_mean").is_not_null())
if nh.height > 0:
    results_c1 = [
        eval_baseline(nh["mcp_mean"], nh["nodal_f0"], "Nodal f0 (corr)", nh.height),
        eval_baseline(nh["mcp_mean"], nh["mtm_1st_mean"], "H", nh.height),
    ]
    print_main_table(results_c1, f"  C1: Nodal f0 vs H ({nh.height:,} overlap)")
    print_mae_table(results_c1, f"  C1 (cont): MAE by |MCP| bin")

# C2: f0 path vs nodal f0 vs H (on f0-path-covered paths)
f0h = aq4.filter(
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

# C3: q4 vs f0 path vs nodal f0 (where all 3 are available)
q4h = aq4.filter(
    pl.col("q4_path").is_not_null()
    & pl.col("f0_path_corr").is_not_null()
    & pl.col("nodal_f0").is_not_null()
)
if q4h.height > 100:
    results_c3 = [
        eval_baseline(q4h["mcp_mean"], q4h["q4_path"], "q4 path", q4h.height),
        eval_baseline(q4h["mcp_mean"], q4h["f0_path_corr"], "f0 path corr", q4h.height),
        eval_baseline(q4h["mcp_mean"], q4h["nodal_f0"], "Nodal f0", q4h.height),
        eval_baseline(q4h["mcp_mean"], q4h["mtm_1st_mean"], "H", q4h.height),
    ]
    print_main_table(results_c3, f"  C3: q4 vs f0 vs Nodal f0 vs H ({q4h.height:,} all-covered)")
    print_mae_table(results_c3, f"  C3 (cont): MAE by |MCP| bin")

# C4: All non-buggy baselines on fully-matched paths
mask_all = pl.lit(True)
for c, _ in baselines:
    if c in aq4.columns:
        mask_all = mask_all & pl.col(c).is_not_null()
all_match = aq4.filter(mask_all)
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
    sub = aq4.filter(pl.col("class_type") == ct)
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
    ("q4_path", "q4 path"),
]
for lo, hi, bin_label in bins:
    sub = aq4.filter((pl.col("mcp_mean").abs() >= lo) & (pl.col("mcp_mean").abs() < hi))
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

actual_arr = aq4_ex19["mcp_mean"].to_numpy().astype(float)
nodal_arr = aq4_ex19["nodal_f0"].to_numpy().astype(float)
f0p_arr = aq4_ex19["f0_path_corr"].to_numpy().astype(float)
h_arr = aq4_ex19["mtm_1st_mean"].to_numpy().astype(float)

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
f0_covered = aq4_ex19.filter(pl.col("f0_path_corr").is_not_null() & pl.col("nodal_f0").is_not_null())
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
out_path = f"{WORK_DIR}/aq4_all_baselines.parquet"
aq4.write_parquet(out_path)
print(f"\nSaved: {out_path} ({aq4.shape})")
print(f"Columns: {aq4.columns}")
print(f"Final mem: {mem_mb():.0f} MB")
print("DONE")
