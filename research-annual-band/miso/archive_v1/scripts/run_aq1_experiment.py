# DEPRECATED — archived 2026-03-17. Old data sources. See task_plan.md for porting plan.
import warnings as _w; _w.warn("This script is DEPRECATED (archived 2026-03-17). Use canonical get_trades_of_given_duration().", DeprecationWarning, stacklevel=2)

#!/usr/bin/env python3
"""
aq1 (Jun-Aug) R1 baseline experiment v2 — comprehensive comparison.

Tests all available baseline sources for predicting R1 aq1 MCP:
  1. H              — historical DA congestion (current production)
  2. Prior year R3   — PY-1 R3 aq1 path MCP (auctioned ~May PY-1)
  3. Prior year R2   — PY-1 R2 aq1 path MCP (auctioned ~Apr PY-1)
  4. Prior year R1   — PY-1 R1 aq1 path MCP (auctioned ~Apr PY-1)
  5. Nodal buggy     — existing Phase 2 nodal (1 month, PY-2 year bug)
  6. Nodal f0        — NEW: nodal stitched from 3-month avg, corrected year (PY-1)
  7. f0 path corr    — prior year f0 path avg for Jun/Jul/Aug (year bug fixed)
  8. f0 path buggy   — same but old buggy mapping (PY-2, for comparison)
  9. f1 path prior   — prior year f1 avg for Jun/Jul/Aug

No quarterly forward for aq1: q1 doesn't exist as a MISO product.

Year mapping:
  PY 2024 aq1 delivers Jun/Jul/Aug 2024.
  "Prior year same months" = Jun/Jul/Aug 2023 → dy = PY - 1.
  BUGGY code used dy = PY - 2 (2 years stale).

Usage:
    cd /home/xyz/workspace/pmodel && source .venv/bin/activate
    python /home/xyz/workspace/research-qianli-v2/research-annual/scripts/run_aq1_experiment.py
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
    build_py_fwd_map, get_all_aliases,
)

warnings.filterwarnings("ignore")

DATA_DIR = "/opt/temp/qianli/annual_research"
WORK_DIR = f"{DATA_DIR}/crossproduct_work"
AQ1_MONTHS = [6, 7, 8]


# ======================================================================
# STEP 1: Load aq1 R1 paths from existing Phase 2 output
# ======================================================================
print(f"{'='*80}")
print(f"STEP 1: Load aq1 R1 paths | mem: {mem_mb():.0f} MB")
print(f"{'='*80}")

r1_all = pl.read_parquet(f"{WORK_DIR}/r1_with_nodal.parquet")
aq1 = r1_all.filter(pl.col("period_type") == "aq1").clone()
del r1_all
gc.collect()

# Rename existing nodal column to make the buggy year clear
aq1 = aq1.rename({"nodal_r3": "nodal_buggy"})

total_n = aq1.height
pys = sorted(aq1["planning_year"].unique().to_list())
n_pys = len(pys)
n_pys_with_prior = len([py for py in pys if int(py) > 2019])
print(f"  aq1 rows: {total_n:,}")
print(f"  PYs: {pys} ({n_pys} total, {n_pys_with_prior} with prior-year data)")
print(f"  Existing coverage:")
for col in ["mtm_1st_mean", "prior_r3_path", "nodal_buggy"]:
    cov = aq1.filter(pl.col(col).is_not_null()).height
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

paths_aq1 = (
    all_res.filter(pl.col("period_type") == "aq1")
    .group_by(["planning_year", "round", "class_type", "source_id", "sink_id"])
    .agg(pl.col("mcp_mean").first())
)
del all_res
gc.collect()

r2_prior = (
    paths_aq1.filter(pl.col("round") == 2)
    .select(
        (pl.col("planning_year") + 1).alias("planning_year"),
        "class_type", "source_id", "sink_id",
        pl.col("mcp_mean").alias("prior_r2_path"),
    )
)
r1_prior = (
    paths_aq1.filter(pl.col("round") == 1)
    .select(
        (pl.col("planning_year") + 1).alias("planning_year"),
        "class_type", "source_id", "sink_id",
        pl.col("mcp_mean").alias("prior_r1_path"),
    )
)
del paths_aq1
gc.collect()

aq1 = aq1.cast({"planning_year": pl.Int64})
r2_prior = r2_prior.cast({"planning_year": pl.Int64})
r1_prior = r1_prior.cast({"planning_year": pl.Int64})

join_keys = ["planning_year", "class_type", "source_id", "sink_id"]
pre = aq1.height
aq1 = aq1.join(r2_prior, on=join_keys, how="left")
assert aq1.height == pre
aq1 = aq1.join(r1_prior, on=join_keys, how="left")
assert aq1.height == pre
del r2_prior, r1_prior
gc.collect()

for col in ["prior_r3_path", "prior_r2_path", "prior_r1_path"]:
    cov = aq1.filter(pl.col(col).is_not_null()).height
    print(f"  {col}: {cov:,} / {total_n:,} ({cov/total_n*100:.1f}%)")
print(f"  mem: {mem_mb():.0f} MB")


# ======================================================================
# STEP 3: f0 path-level with corrected and buggy year mapping
# ======================================================================
print(f"\n{'='*80}")
print(f"STEP 3: f0 path-level cross-product | mem: {mem_mb():.0f} MB")
print(f"{'='*80}")

f0 = (
    pl.scan_parquet(f"{DATA_DIR}/f0p_cleared_all.parquet")
    .filter(
        (pl.col("period_type") == "f0")
        & pl.col("split_market_month").dt.month().is_in(AQ1_MONTHS)
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
print(f"  f0 Jun/Jul/Aug rows: {f0.height:,}")

f0m = (
    f0.sort("round")
    .group_by(["dy", "dm_num", "class_type", "source_id", "sink_id"])
    .agg(pl.col("mcp").last().alias("f0_mcp"))
)
del f0
gc.collect()

dy_range = sorted(f0m["dy"].unique().to_list())
print(f"  f0 deduped: {f0m.height:,}, delivery years: {dy_range}")

print(f"\n  Year mapping verification:")
print(f"  {'PY':>6} | {'Corrected (PY-1)':>16} | {'Buggy (PY-2)':>14} | Delivery months")
for py in pys:
    corr, bug = int(py) - 1, int(py) - 2
    avail_c = "Y" if corr in dy_range else "N"
    avail_b = "Y" if bug in dy_range else "N"
    print(f"  {int(py):>6} | {corr:>12} [{avail_c}] | {bug:>10} [{avail_b}] | Jun/Jul/Aug {corr}")

map_corrected, map_buggy = [], []
for py in pys:
    for m in AQ1_MONTHS:
        map_corrected.append({"planning_year": int(py), "dy": int(py) - 1, "dm_num": int(m)})
        map_buggy.append({"planning_year": int(py), "dy": int(py) - 2, "dm_num": int(m)})

mdf_c = pl.DataFrame(map_corrected).cast({"dy": pl.Int16, "dm_num": pl.Int8})
mdf_b = pl.DataFrame(map_buggy).cast({"dy": pl.Int16, "dm_num": pl.Int8})
f0m_cast = f0m.cast({"dy": pl.Int16, "dm_num": pl.Int8})

f0_avg_c = (
    mdf_c.join(f0m_cast, on=["dy", "dm_num"], how="inner")
    .group_by(["planning_year", "class_type", "source_id", "sink_id"])
    .agg(pl.col("f0_mcp").mean().alias("f0_path_corr"), pl.col("f0_mcp").count().alias("_nc"))
)
f0_avg_b = (
    mdf_b.join(f0m_cast, on=["dy", "dm_num"], how="inner")
    .group_by(["planning_year", "class_type", "source_id", "sink_id"])
    .agg(pl.col("f0_mcp").mean().alias("f0_path_buggy"), pl.col("f0_mcp").count().alias("_nb"))
)
del f0m, f0m_cast, mdf_c, mdf_b
gc.collect()

print(f"  f0 CORRECTED: {f0_avg_c.height:,} path averages")
print(f"  Month counts:\n{f0_avg_c['_nc'].value_counts().sort('_nc')}")
print(f"  f0 BUGGY:     {f0_avg_b.height:,} path averages")

f0_avg_c = f0_avg_c.cast({"planning_year": pl.Int64})
f0_avg_b = f0_avg_b.cast({"planning_year": pl.Int64})

pre = aq1.height
aq1 = aq1.join(f0_avg_c.drop("_nc"), on=join_keys, how="left")
assert aq1.height == pre
aq1 = aq1.join(f0_avg_b.drop("_nb"), on=join_keys, how="left")
assert aq1.height == pre
del f0_avg_c, f0_avg_b
gc.collect()

print(f"\n  f0 path coverage by PY:")
for py in pys:
    sub = aq1.filter(pl.col("planning_year") == py)
    cc = sub.filter(pl.col("f0_path_corr").is_not_null()).height
    cb = sub.filter(pl.col("f0_path_buggy").is_not_null()).height
    print(f"    PY {int(py)}: corr={cc:,}/{sub.height:,} ({cc/sub.height*100:.1f}%), buggy={cb:,} ({cb/sub.height*100:.1f}%)")
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
        & pl.col("split_market_month").dt.month().is_in(AQ1_MONTHS)
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
print(f"  f1 Jun/Jul/Aug rows: {f1.height:,}")

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
        for m in AQ1_MONTHS:
            map_f1.append({"planning_year": int(py), "dy": int(py) - 1, "dm_num": int(m)})
    mdf_f1 = pl.DataFrame(map_f1).cast({"dy": pl.Int16, "dm_num": pl.Int8})
    f1_avg = (
        mdf_f1.join(f1m.cast({"dy": pl.Int16, "dm_num": pl.Int8}), on=["dy", "dm_num"], how="inner")
        .group_by(["planning_year", "class_type", "source_id", "sink_id"])
        .agg(pl.col("f1_mcp").mean().alias("f1_path"))
    )
    del f1m, mdf_f1
    gc.collect()

    pre = aq1.height
    aq1 = aq1.join(f1_avg.cast({"planning_year": pl.Int64}), on=join_keys, how="left")
    assert aq1.height == pre
    del f1_avg
    gc.collect()

    cov = aq1.filter(pl.col("f1_path").is_not_null()).height
    print(f"  f1 path coverage: {cov:,} / {total_n:,} ({cov/total_n*100:.1f}%)")
else:
    aq1 = aq1.with_columns(pl.lit(None).cast(pl.Float64).alias("f1_path"))
    print(f"  No f1 data for aq1 months")
    del f1
    gc.collect()
print(f"  mem: {mem_mb():.0f} MB")


# ======================================================================
# STEP 5: Nodal f0 stitching via Ray (corrected year, 3-month avg)
# ======================================================================
print(f"\n{'='*80}")
print(f"STEP 5: Nodal f0 stitching (Ray) | mem: {mem_mb():.0f} MB")
print(f"{'='*80}")
print(f"  Loading Ray + MisoCalculator + MisoNodalReplacement...")

from pbase.config.ray import init_ray
import pmodel
init_ray(address="ray://10.8.0.36:10001", extra_modules=[pmodel])

from pbase.data.m2m.calculator import MisoCalculator
from pbase.data.dataset.replacement import MisoNodalReplacement

import pandas as pd

# Load replacement data WITH effective dates for year-aware filtering
repl_raw = MisoNodalReplacement().load_data()
print(f"  Replacement records: {len(repl_raw):,}")


calc = MisoCalculator()

# For each PY, call get_mcp_df for Jun, Jul, Aug of PY-1 (corrected year)
# Then average across 3 months per node
nodal_rows: list[tuple] = []  # (py, ct, node_id, mcp, month)

for py in pys:
    prior_year = int(py) - 1
    if prior_year < 2019:
        print(f"  PY {int(py)}: skip (prior year {prior_year} < 2019)")
        continue

    # Build date-filtered replacement map for this PY's delivery period.
    # For aq1, all 3 delivery months (Jun/Jul/Aug) are in the same calendar year,
    # so a single target date works. Use the LAST delivery month to capture any
    # renames that become effective mid-quarter.
    # NOTE: For aq3 (Dec/Jan/Feb) and aq4 (Mar-May), delivery months cross into PY+1,
    # so those scripts MUST use per-month target dates or the latest delivery month.
    target_date = f"{int(py)}-08"
    fwd_map_py = build_py_fwd_map(repl_raw, target_date)
    rev_map_py: dict[str, list[str]] = {}
    for fn, tn in fwd_map_py.items():
        rev_map_py.setdefault(tn, []).append(fn)
    print(f"  PY {int(py)}: {len(fwd_map_py)} active replacements at {target_date}")

    for m in AQ1_MONTHS:
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
                # get_mcp_df returns columns ordered: monthly(f0), monthly(f1), ..., annual(R3), annual(R2), annual(R1).
                raw = ct_data[ct_data.columns[0]].dropna()
                for nid, val in raw.items():
                    # Register MCP under ALL equivalent names in the chain
                    # e.g. A->B->C: register under A, B, AND C
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

# Shutdown Ray
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
pre = aq1.height
aq1 = aq1.join(src_lookup, on=nj_keys + ["source_id"], how="left")
assert aq1.height == pre
aq1 = aq1.join(snk_lookup, on=nj_keys + ["sink_id"], how="left")
assert aq1.height == pre
del src_lookup, snk_lookup
gc.collect()

aq1 = aq1.with_columns(
    pl.when(pl.col("_src_f0").is_not_null() & pl.col("_snk_f0").is_not_null())
    .then(pl.col("_snk_f0") - pl.col("_src_f0"))
    .otherwise(None)
    .alias("nodal_f0")
).drop(["_src_f0", "_snk_f0"])

cov = aq1.filter(pl.col("nodal_f0").is_not_null()).height
cov_buggy = aq1.filter(pl.col("nodal_buggy").is_not_null()).height
print(f"\n  Nodal f0 (corrected, 3-mo):  {cov:,} / {total_n:,} ({cov/total_n*100:.1f}%)")
print(f"  Nodal buggy (Phase 2, 1-mo): {cov_buggy:,} / {total_n:,} ({cov_buggy/total_n*100:.1f}%)")
print(f"  mem: {mem_mb():.0f} MB")


# ======================================================================
# STEP 6: RESULTS — Comprehensive Comparison
# ======================================================================
print(f"\n{'='*80}")
print(f"RESULTS: aq1 (Jun-Aug) R1 Baseline Comparison")
print(f"{'='*80}")
print(f"  Total paths: {total_n:,}")
print(f"  PYs tested: {pys} ({n_pys} years, {n_pys_with_prior} with prior-year data)")
print(f"  Note: |MCP| in 'Dir>50', 'Dir>100' means ACTUAL aq1 MCP for that path")

baselines = [
    ("mtm_1st_mean",  "H (production)"),
    ("prior_r3_path", "Prior R3 path"),
    ("prior_r2_path", "Prior R2 path"),
    ("prior_r1_path", "Prior R1 path"),
    ("nodal_buggy",   "Nodal buggy (PY-2)"),
    ("nodal_f0",      "Nodal f0 (PY-1,3mo)"),
    ("f0_path_corr",  "f0 path corr (PY-1)"),
    ("f0_path_buggy", "f0 path buggy(PY-2)"),
    ("f1_path",       "f1 path (PY-1)"),
]

# --- Section A: Overall (PY 2020-2025 only) ---
# Exclude PY 2019: no prior-year data exists (MCP data starts 2019), so ALL
# non-H baselines are structurally 0% for PY 2019. Including it unfairly dilutes coverage.
aq1_ex19 = aq1.filter(pl.col("planning_year") >= 2020)
total_ex19 = aq1_ex19.height

results_all = []
for col, label in baselines:
    if col in aq1_ex19.columns:
        results_all.append(eval_baseline(aq1_ex19["mcp_mean"], aq1_ex19[col], label, total_ex19))

print_main_table(results_all, f"Section A: Overall PY 2020-2025 ({total_ex19:,} paths, excludes PY 2019)")
print_mae_table(results_all, "Section A (cont): MAE by |MCP| bin")

# --- Section B: By PY ---
print(f"\n{'='*80}")
print(f"Section B: By Planning Year")
print(f"{'='*80}")

for py in pys:
    sub = aq1.filter(pl.col("planning_year") == py)
    results_py = []
    for col, label in baselines:
        if col in sub.columns:
            results_py.append(eval_baseline(sub["mcp_mean"], sub[col], label, sub.height))
    print_main_table(results_py, f"  PY {int(py)} ({sub.height:,} paths)")

# --- Section C: Head-to-head on matched paths ---
print(f"\n{'='*80}")
print(f"Section C: Head-to-head (same paths)")
print(f"{'='*80}")

# C1: Nodal f0 vs Nodal buggy
nb = aq1.filter(pl.col("nodal_f0").is_not_null() & pl.col("nodal_buggy").is_not_null())
if nb.height > 0:
    results_c1 = [
        eval_baseline(nb["mcp_mean"], nb["nodal_f0"], "Nodal f0 (corr)", nb.height),
        eval_baseline(nb["mcp_mean"], nb["nodal_buggy"], "Nodal buggy", nb.height),
        eval_baseline(nb["mcp_mean"], nb["mtm_1st_mean"], "H", nb.height),
    ]
    print_main_table(results_c1, f"  C1: Nodal f0 vs Nodal buggy vs H ({nb.height:,} overlap)")
    print_mae_table(results_c1, f"  C1 (cont): MAE by |MCP| bin")

# C2: f0 path corrected vs buggy
fb = aq1.filter(pl.col("f0_path_corr").is_not_null() & pl.col("f0_path_buggy").is_not_null())
if fb.height > 0:
    results_c2 = [
        eval_baseline(fb["mcp_mean"], fb["f0_path_corr"], "f0 path corr", fb.height),
        eval_baseline(fb["mcp_mean"], fb["f0_path_buggy"], "f0 path buggy", fb.height),
    ]
    print_main_table(results_c2, f"  C2: f0 path corrected vs buggy ({fb.height:,} overlap)")

# C3: Nodal f0 vs f0 path corrected vs H (on f0-path-covered paths)
f0h = aq1.filter(
    pl.col("f0_path_corr").is_not_null()
    & pl.col("nodal_f0").is_not_null()
    & pl.col("mtm_1st_mean").is_not_null()
)
if f0h.height > 0:
    results_c3 = [
        eval_baseline(f0h["mcp_mean"], f0h["f0_path_corr"], "f0 path corr", f0h.height),
        eval_baseline(f0h["mcp_mean"], f0h["nodal_f0"], "Nodal f0", f0h.height),
        eval_baseline(f0h["mcp_mean"], f0h["mtm_1st_mean"], "H", f0h.height),
    ]
    print_main_table(results_c3, f"  C3: f0 path vs Nodal f0 vs H ({f0h.height:,} all-covered)")
    print_mae_table(results_c3, f"  C3 (cont): MAE by |MCP| bin")

# C4: All baselines on fully-matched paths
non_buggy = [c for c, _ in baselines if "buggy" not in c]
mask_all = pl.lit(True)
for c in non_buggy:
    if c in aq1.columns:
        mask_all = mask_all & pl.col(c).is_not_null()
all_match = aq1.filter(mask_all)
if all_match.height > 100:
    results_c4 = []
    for col, label in baselines:
        if col in all_match.columns and "buggy" not in col:
            results_c4.append(eval_baseline(all_match["mcp_mean"], all_match[col], label, all_match.height))
    print_main_table(results_c4, f"  C4: All non-buggy baselines, fully matched ({all_match.height:,} paths)")
    print_mae_table(results_c4, f"  C4 (cont): MAE by |MCP| bin")

# --- Section D: By class_type ---
print(f"\n{'='*80}")
print(f"Section D: By class_type")
print(f"{'='*80}")

for ct in ["onpeak", "offpeak"]:
    sub = aq1.filter(pl.col("class_type") == ct)
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
]
for lo, hi, bin_label in bins:
    sub = aq1.filter((pl.col("mcp_mean").abs() >= lo) & (pl.col("mcp_mean").abs() < hi))
    if sub.height == 0:
        continue
    results_bin = []
    for col, label in key_baselines:
        if col in sub.columns:
            results_bin.append(eval_baseline(sub["mcp_mean"], sub[col], label, sub.height))
    print_main_table(results_bin, f"  |MCP| {bin_label} ({sub.height:,} paths)")


# ======================================================================
# SAVE
# ======================================================================
out_path = f"{WORK_DIR}/aq1_all_baselines.parquet"
aq1.write_parquet(out_path)
print(f"\nSaved: {out_path} ({aq1.shape})")
print(f"Columns: {aq1.columns}")
print(f"Final mem: {mem_mb():.0f} MB")
print("DONE")
