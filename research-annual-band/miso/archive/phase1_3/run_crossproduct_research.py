#!/usr/bin/env python3
"""
Cross-product research pipeline — memory-optimized with polars.

Uses polars for data processing (2-4x less memory than pandas).
Saves intermediates to parquet; skips phases where output already exists.
Target peak memory: <3 GiB.

Usage:
    python run_crossproduct_research.py                # skip completed phases
    python run_crossproduct_research.py --force        # re-run everything
    python run_crossproduct_research.py --from-phase 3 # re-run from phase 3+

Phases:
  1  Path-level R1 + PY-1 R3 path-match           (no Ray)
  2  Nodal MCP stitching via vectorized join        (Ray required)
  3  f0p cross-product match for delivery months    (no Ray)
  4  Cascade baseline + statistics                  (no Ray)
"""

import gc
import os
import resource
import sys
import warnings

import numpy as np
import polars as pl

warnings.filterwarnings("ignore")

DATA_DIR = "/opt/temp/qianli/annual_research"
WORK_DIR = f"{DATA_DIR}/crossproduct_work"
os.makedirs(WORK_DIR, exist_ok=True)

# --------------- CLI args ---------------
force_all = "--force" in sys.argv
from_phase = 1
for i, arg in enumerate(sys.argv[1:], 1):
    if arg == "--from-phase" and i < len(sys.argv) - 1:
        from_phase = int(sys.argv[i + 1])
    elif arg.startswith("--from-phase="):
        from_phase = int(arg.split("=")[1])


# --------------- Helpers ---------------
def mem_mb() -> float:
    """Current peak RSS in MB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def should_run(phase: int, output: str) -> bool:
    """Phase runs if forced, at/after from_phase, or output missing."""
    if force_all:
        return True
    if phase >= from_phase:
        return True
    return not os.path.exists(output)


def eval_baseline(mcp: pl.Series, baseline: pl.Series, label: str = "") -> dict:
    """Compute bias, MAE, p95, direction accuracy."""
    mask = baseline.is_not_null() & mcp.is_not_null()
    n = int(mask.sum())
    if n == 0:
        return {"label": label, "n": 0, "bias": 0, "mean_abs": 0, "p95_abs": 0, "dir_acc": float("nan")}
    m, b = mcp.filter(mask), baseline.filter(mask)
    res = m - b
    abs_res = res.abs()
    dir_mask = (m != 0) & (b != 0)
    dn = int(dir_mask.sum())
    dir_acc = (
        round(float((m.filter(dir_mask).sign() == b.filter(dir_mask).sign()).mean()) * 100, 1)
        if dn > 0
        else float("nan")
    )
    return {
        "label": label,
        "n": n,
        "bias": round(float(res.mean()), 1),
        "mean_abs": round(float(abs_res.mean()), 1),
        "p95_abs": round(float(abs_res.quantile(0.95)), 0),
        "dir_acc": dir_acc,
    }


def fmt(s: dict) -> str:
    return f"n={s['n']:,}  p95={s['p95_abs']:.0f}  dir={s['dir_acc']:.1f}%  bias={s['bias']:+.0f}"


AQ_MONTHS = {"aq1": [6, 7, 8], "aq2": [9, 10, 11], "aq3": [12, 1, 2], "aq4": [3, 4, 5]}
AQ_TO_MONTH = {"aq1": 6, "aq2": 9, "aq3": 12, "aq4": 3}


# ================================================================
# PHASE 1: Path-level R1 + PY-1 R3 path-match (no Ray)
# ================================================================
phase1_out = f"{WORK_DIR}/r1_base.parquet"

if should_run(1, phase1_out):
    print(f"{'=' * 70}")
    print(f"PHASE 1: Path-level R1 + PY-1 R3 path-match | mem: {mem_mb():.0f} MB")
    print(f"{'=' * 70}")

    cols = [
        "planning_year", "round", "period_type", "class_type",
        "source_id", "sink_id", "mcp_mean", "mtm_1st_mean",
    ]
    df = pl.read_parquet(f"{DATA_DIR}/all_residuals_v2.parquet", columns=cols)
    print(f"  Loaded {df.height:,} rows, mem: {mem_mb():.0f} MB")

    # Aggregate to path level (one row per unique path)
    paths = df.group_by(
        ["planning_year", "round", "period_type", "class_type", "source_id", "sink_id"]
    ).agg(pl.col("mcp_mean").first(), pl.col("mtm_1st_mean").first())
    del df
    gc.collect()
    print(f"  Path-level: {paths.height:,}, mem: {mem_mb():.0f} MB")

    r1 = paths.filter(pl.col("round") == 1)
    # Shift R3 PY+1 so it aligns with next year's R1
    r3 = (
        paths.filter(pl.col("round") == 3)
        .select(
            (pl.col("planning_year") + 1).alias("planning_year"),
            "period_type", "class_type", "source_id", "sink_id",
            pl.col("mcp_mean").alias("prior_r3_path"),
        )
    )
    del paths
    gc.collect()

    r1 = r1.join(
        r3,
        on=["planning_year", "period_type", "class_type", "source_id", "sink_id"],
        how="left",
    )
    del r3
    gc.collect()

    cov = r1.filter(pl.col("prior_r3_path").is_not_null()).height
    print(f"  R3 path coverage: {cov:,} / {r1.height:,} ({cov / r1.height * 100:.1f}%)")

    r1.write_parquet(phase1_out)
    print(f"  Saved: {r1.shape}, mem: {mem_mb():.0f} MB")
    del r1
    gc.collect()
else:
    print(f"PHASE 1: SKIP (output exists)")


# ================================================================
# PHASE 2: Nodal MCP stitching (requires Ray)
# ================================================================
phase2_out = f"{WORK_DIR}/r1_with_nodal.parquet"

if should_run(2, phase2_out):
    assert os.path.exists(phase1_out), f"Phase 1 output missing: {phase1_out}"
    print(f"\n{'=' * 70}")
    print(f"PHASE 2: Nodal MCP stitching | mem: {mem_mb():.0f} MB")
    print(f"{'=' * 70}")

    from pbase.config.ray import init_ray
    import pmodel

    init_ray(address="ray://10.8.0.36:10001", extra_modules=[pmodel])

    from pbase.data.m2m.calculator import MisoCalculator
    from pbase.data.dataset.replacement import MisoNodalReplacement

    # Build forward replacement map
    repl_df = MisoNodalReplacement().load_data()
    fwd_map = dict(zip(repl_df["from_node"], repl_df["to_node"]))
    print(f"  Replacement entries: {len(fwd_map):,}")
    del repl_df
    gc.collect()

    def resolve_forward(nid):
        seen = set()
        while nid in fwd_map and nid not in seen:
            seen.add(nid)
            nid = fwd_map[nid]
        return nid

    r1 = pl.read_parquet(phase1_out)
    pys = sorted(r1["planning_year"].unique().to_list())
    print(f"  R1: {r1.shape}, PYs: {pys}")

    def get_market_month(py: int, pt: str) -> str:
        m = AQ_TO_MONTH[pt]
        return f"{py - 1 if m >= 6 else py}-{m:02d}"

    # Collect nodal MCPs: one get_mcp_df call per (prior_py, qt), split by ct
    calc = MisoCalculator()
    nodal_rows: list[tuple] = []  # (py, qt, ct, node_id, mcp)

    for py in pys:
        prior_py = int(py) - 1
        if prior_py < 2019:
            continue

        for qt in ["aq1", "aq2", "aq3", "aq4"]:
            mm = get_market_month(prior_py, qt)
            try:
                mcp_df, _ = calc.get_mcp_df(market_month=mm, fillna=True)
                if mcp_df is None or len(mcp_df) == 0:
                    continue
            except Exception as e:
                print(f"    ERR {mm}: {e}")
                continue

            for ct in ["onpeak", "offpeak"]:
                try:
                    ct_data = mcp_df.xs(ct, level="class_type")
                    raw = ct_data[ct_data.columns[-1]].dropna()
                    for nid, val in raw.items():
                        nodal_rows.append((int(py), qt, ct, str(nid), float(val)))
                        fwd_id = resolve_forward(nid)
                        if fwd_id != nid:
                            nodal_rows.append((int(py), qt, ct, str(fwd_id), float(val)))
                except (KeyError, IndexError):
                    pass

            del mcp_df

        gc.collect()
        n_py = sum(1 for r in nodal_rows if r[0] == py)
        print(f"    PY {py}: {n_py:,} nodal entries, mem: {mem_mb():.0f} MB")

    # Shutdown Ray early to free memory
    import ray

    ray.shutdown()
    del calc
    gc.collect()
    print(f"  Ray shutdown, mem: {mem_mb():.0f} MB")

    # Build lookup DataFrame, deduplicate
    nodal_df = (
        pl.DataFrame(
            {
                "planning_year": [r[0] for r in nodal_rows],
                "period_type": [r[1] for r in nodal_rows],
                "class_type": [r[2] for r in nodal_rows],
                "node_id": [r[3] for r in nodal_rows],
                "node_mcp": [r[4] for r in nodal_rows],
            },
            schema={
                "planning_year": pl.Int64,
                "period_type": pl.String,
                "class_type": pl.String,
                "node_id": pl.String,
                "node_mcp": pl.Float64,
            },
        )
        .unique(subset=["planning_year", "period_type", "class_type", "node_id"], keep="last")
    )
    del nodal_rows
    gc.collect()
    print(f"  Nodal lookup: {nodal_df.shape}")

    # Ensure compatible dtypes
    r1 = r1.cast({"planning_year": pl.Int64})

    # Vectorized stitch: join for source, then for sink
    join_keys = ["planning_year", "period_type", "class_type"]

    src_lookup = nodal_df.rename({"node_id": "source_id", "node_mcp": "_src_mcp"})
    r1 = r1.join(src_lookup, on=join_keys + ["source_id"], how="left")
    del src_lookup
    gc.collect()

    snk_lookup = nodal_df.rename({"node_id": "sink_id", "node_mcp": "_snk_mcp"})
    r1 = r1.join(snk_lookup, on=join_keys + ["sink_id"], how="left")
    del snk_lookup, nodal_df
    gc.collect()

    r1 = r1.with_columns(
        pl.when(pl.col("_src_mcp").is_not_null() & pl.col("_snk_mcp").is_not_null())
        .then(pl.col("_snk_mcp") - pl.col("_src_mcp"))
        .otherwise(None)
        .alias("nodal_r3")
    ).drop(["_src_mcp", "_snk_mcp"])

    cov = r1.filter(pl.col("nodal_r3").is_not_null()).height
    print(f"  Nodal R3 coverage: {cov:,} / {r1.height:,} ({cov / r1.height * 100:.1f}%)")

    # Quick accuracy comparison
    sub = r1.filter(pl.col("nodal_r3").is_not_null() & pl.col("mtm_1st_mean").is_not_null())
    if sub.height > 0:
        sn = eval_baseline(sub["mcp_mean"], sub["nodal_r3"], "Nodal")
        sh = eval_baseline(sub["mcp_mean"], sub["mtm_1st_mean"], "H")
        print(f"    Nodal: {fmt(sn)}")
        print(f"    H:     {fmt(sh)}")

    r1.write_parquet(phase2_out)
    print(f"  Saved: {r1.shape}, mem: {mem_mb():.0f} MB")
    del r1
    gc.collect()
else:
    print(f"\nPHASE 2: SKIP (output exists)")


# ================================================================
# PHASE 3: f0p cross-product match (no Ray)
# ================================================================
phase3_out = f"{WORK_DIR}/r1_final.parquet"

if should_run(3, phase3_out):
    assert os.path.exists(phase2_out), f"Phase 2 output missing: {phase2_out}"
    print(f"\n{'=' * 70}")
    print(f"PHASE 3: f0p cross-product | mem: {mem_mb():.0f} MB")
    print(f"{'=' * 70}")

    r1 = pl.read_parquet(phase2_out)
    print(f"  R1: {r1.shape}")

    # Load only f0 rows via lazy scan + filter (avoids full materialization)
    f0 = (
        pl.scan_parquet(f"{DATA_DIR}/f0p_cleared_all.parquet")
        .filter(pl.col("period_type") == "f0")
        .select(
            pl.col("class_type").cast(pl.String),
            "source_id",
            "sink_id",
            "mcp",
            pl.col("split_market_month").dt.month().cast(pl.Int8).alias("dm_num"),
            pl.col("split_market_month").dt.year().cast(pl.Int16).alias("dy"),
            "round",
        )
        .collect()
    )
    print(f"  f0 loaded: {f0.shape}, mem: {mem_mb():.0f} MB")

    # Keep highest round per path-month, take its MCP
    f0m = (
        f0.sort("round")
        .group_by(["dy", "dm_num", "class_type", "source_id", "sink_id"])
        .agg(pl.col("mcp").last().alias("f0_mcp"))
    )
    del f0
    gc.collect()
    print(f"  f0 deduped: {f0m.shape}, mem: {mem_mb():.0f} MB")

    # Path overlap
    r1_ps = r1.select("source_id", "sink_id").unique()
    f0_ps = f0m.select("source_id", "sink_id").unique()
    overlap = r1_ps.join(f0_ps, on=["source_id", "sink_id"], how="inner").height
    print(f"  Path overlap: {overlap:,} / {r1_ps.height:,} ({overlap / r1_ps.height * 100:.1f}%)")
    del r1_ps, f0_ps
    gc.collect()

    # Build month mapping: annual (PY, quarter) → prior year's monthly f0 delivery months
    map_rows = []
    for py in sorted(r1["planning_year"].unique().to_list()):
        pp = int(py) - 1
        if pp < 2019:
            continue
        for qt, months in AQ_MONTHS.items():
            for m in months:
                if qt == "aq3":
                    dy = pp - 1 if m == 12 else pp
                elif qt == "aq4":
                    dy = pp
                else:
                    dy = pp - 1
                map_rows.append({
                    "planning_year": int(py),
                    "period_type": qt,
                    "dy": int(dy),
                    "dm_num": int(m),
                })

    mdf = pl.DataFrame(map_rows).cast({"dy": pl.Int16, "dm_num": pl.Int8})
    print(f"  Month mapping: {mdf.shape}")

    # Map annual quarters to monthly f0 data
    f0_mapped = mdf.join(
        f0m.cast({"dy": pl.Int16, "dm_num": pl.Int8}),
        on=["dy", "dm_num"],
        how="inner",
    )
    del f0m, mdf
    gc.collect()
    print(f"  f0 mapped: {f0_mapped.shape}, mem: {mem_mb():.0f} MB")

    # Average f0 MCP across months for each annual path
    f0_avg = f0_mapped.group_by(
        ["planning_year", "period_type", "class_type", "source_id", "sink_id"]
    ).agg(
        pl.col("f0_mcp").mean().alias("f0_avg_mcp"),
        pl.col("f0_mcp").count().alias("n_months"),
    )
    del f0_mapped
    gc.collect()
    print(f"  f0 averages: {f0_avg.shape}")
    print(f"  Months/path:\n{f0_avg['n_months'].value_counts().sort('n_months')}")

    # Ensure compatible dtypes for join
    join_cols = ["planning_year", "period_type", "class_type", "source_id", "sink_id"]
    r1 = r1.cast({"planning_year": pl.Int64})
    f0_avg = f0_avg.cast({"planning_year": pl.Int64})

    pre_rows = r1.height
    r1 = r1.join(
        f0_avg.select(*join_cols, "f0_avg_mcp"),
        on=join_cols,
        how="left",
    )
    assert r1.height == pre_rows, f"Row count changed in join: {pre_rows} -> {r1.height}"
    del f0_avg
    gc.collect()

    cov = r1.filter(pl.col("f0_avg_mcp").is_not_null()).height
    print(f"\n  f0 Coverage: {cov:,} / {r1.height:,} ({cov / r1.height * 100:.1f}%)")
    for qt in ["aq1", "aq2", "aq3", "aq4"]:
        sub = r1.filter(pl.col("period_type") == qt)
        c = sub.filter(pl.col("f0_avg_mcp").is_not_null()).height
        print(f"    {qt}: {c:,} / {sub.height:,} ({c / sub.height * 100:.1f}%)")

    r1.write_parquet(phase3_out)
    print(f"  Saved: {r1.shape}, mem: {mem_mb():.0f} MB")
    del r1
    gc.collect()
else:
    print(f"\nPHASE 3: SKIP (output exists)")


# ================================================================
# PHASE 4: Cascade baseline + statistics (always runs)
# ================================================================
assert os.path.exists(phase3_out), f"Phase 3 output missing: {phase3_out}"
print(f"\n{'=' * 70}")
print(f"PHASE 4: Statistics | mem: {mem_mb():.0f} MB")
print(f"{'=' * 70}")

r1 = pl.read_parquet(phase3_out)
print(f"  Loaded: {r1.shape}")

# --- f0 vs H comparison ---
sub_f0 = r1.filter(pl.col("f0_avg_mcp").is_not_null() & pl.col("mtm_1st_mean").is_not_null())
if sub_f0.height > 0:
    sf = eval_baseline(sub_f0["mcp_mean"], sub_f0["f0_avg_mcp"], "f0")
    sh = eval_baseline(sub_f0["mcp_mean"], sub_f0["mtm_1st_mean"], "H")
    print(f"\n### f0 vs H on {sub_f0.height:,} paths")
    print(f"  f0: {fmt(sf)}")
    print(f"  H:  {fmt(sh)}")

# --- Triple comparison (paths with all three baselines) ---
sub_tri = r1.filter(
    pl.col("f0_avg_mcp").is_not_null()
    & pl.col("mtm_1st_mean").is_not_null()
    & pl.col("prior_r3_path").is_not_null()
)
if sub_tri.height > 0:
    print(f"\n### Triple comparison on {sub_tri.height:,} paths")
    for col, label in [("mtm_1st_mean", "H"), ("prior_r3_path", "R3 path"), ("f0_avg_mcp", "f0 cross")]:
        s = eval_baseline(sub_tri["mcp_mean"], sub_tri[col], label)
        print(f"  {label:10s}: {fmt(s)}")

# --- f0 vs H by quarter ---
print(f"\n### f0 vs H by Quarter")
print(f"| Qt  |     n | H p95  | f0 p95 | H Dir  | f0 Dir |")
print(f"|-----|-------|--------|--------|--------|--------|")
for qt in ["aq1", "aq2", "aq3", "aq4"]:
    sub = r1.filter(
        (pl.col("period_type") == qt)
        & pl.col("f0_avg_mcp").is_not_null()
        & pl.col("mtm_1st_mean").is_not_null()
    )
    if sub.height > 0:
        sh = eval_baseline(sub["mcp_mean"], sub["mtm_1st_mean"])
        sf = eval_baseline(sub["mcp_mean"], sub["f0_avg_mcp"])
        print(
            f"| {qt} | {sub.height:>5,} | {sh['p95_abs']:>6.0f} | {sf['p95_abs']:>6.0f} "
            f"| {sh['dir_acc']:>5.1f}% | {sf['dir_acc']:>5.1f}% |"
        )

# --- Coverage summary ---
print(f"\n### Coverage Summary")
for col, label in [
    ("mtm_1st_mean", "H"),
    ("prior_r3_path", "R3 path"),
    ("nodal_r3", "Nodal R3"),
    ("f0_avg_mcp", "f0 cross"),
]:
    if col in r1.columns:
        n = r1.filter(pl.col(col).is_not_null()).height
        print(f"  {label:10s}: {n:>8,} ({n / r1.height * 100:.1f}%)")

# --- Cascade baseline ---
r1 = r1.with_columns(
    pl.col("prior_r3_path")
    .fill_null(pl.col("f0_avg_mcp"))
    .fill_null(pl.col("nodal_r3"))
    .fill_null(pl.col("mtm_1st_mean"))
    .alias("cascade")
)

n1 = r1.filter(pl.col("prior_r3_path").is_not_null()).height
n2 = r1.filter(pl.col("prior_r3_path").is_null() & pl.col("f0_avg_mcp").is_not_null()).height
n3 = r1.filter(
    pl.col("prior_r3_path").is_null() & pl.col("f0_avg_mcp").is_null() & pl.col("nodal_r3").is_not_null()
).height
n4 = r1.height - n1 - n2 - n3

print(f"\n### Cascade breakdown")
print(f"  Path R3:    {n1:>8,} ({n1 / r1.height * 100:5.1f}%)")
print(f"  + f0:       {n2:>8,} ({n2 / r1.height * 100:5.1f}%)")
print(f"  + Nodal R3: {n3:>8,} ({n3 / r1.height * 100:5.1f}%)")
print(f"  + H:        {n4:>8,} ({n4 / r1.height * 100:5.1f}%)")

# Overall cascade vs H
sub_c = r1.filter(pl.col("cascade").is_not_null())
sub_h = r1.filter(pl.col("mtm_1st_mean").is_not_null())
sc = eval_baseline(sub_c["mcp_mean"], sub_c["cascade"], "Cascade")
sh = eval_baseline(sub_h["mcp_mean"], sub_h["mtm_1st_mean"], "H only")
print(f"\n### Overall")
print(f"  Cascade: {fmt(sc)}")
print(f"  H only:  {fmt(sh)}")

# By quarter
print(f"\n### By Quarter")
print(f"| Qt  | H p95  | Casc p95 | H Dir  | Casc Dir |")
print(f"|-----|--------|----------|--------|----------|")
for qt in ["aq1", "aq2", "aq3", "aq4"]:
    sub = r1.filter(pl.col("period_type") == qt)
    sub_c2 = sub.filter(pl.col("cascade").is_not_null())
    sub_h2 = sub.filter(pl.col("mtm_1st_mean").is_not_null())
    sc2 = eval_baseline(sub_c2["mcp_mean"], sub_c2["cascade"])
    sh2 = eval_baseline(sub_h2["mcp_mean"], sub_h2["mtm_1st_mean"])
    if sc2["n"] > 0 and sh2["n"] > 0:
        print(
            f"| {qt} | {sh2['p95_abs']:>6.0f} | {sc2['p95_abs']:>8.0f} "
            f"| {sh2['dir_acc']:>5.1f}% | {sc2['dir_acc']:>7.1f}% |"
        )

# Per-source accuracy
print(f"\n### Per-source accuracy")
r1 = r1.with_columns(
    pl.when(pl.col("prior_r3_path").is_not_null())
    .then(pl.lit("path_r3"))
    .when(pl.col("f0_avg_mcp").is_not_null())
    .then(pl.lit("f0_cross"))
    .when(pl.col("nodal_r3").is_not_null())
    .then(pl.lit("nodal_r3"))
    .otherwise(pl.lit("H"))
    .alias("src_label")
)

print(f"| Source   |       n |   p95 |   Dir |   Bias |")
print(f"|----------|---------|-------|-------|--------|")
for src in ["path_r3", "f0_cross", "nodal_r3", "H"]:
    sub = r1.filter(
        (pl.col("src_label") == src)
        & pl.col("cascade").is_not_null()
        & pl.col("mcp_mean").is_not_null()
    )
    if sub.height > 0:
        s = eval_baseline(sub["mcp_mean"], sub["cascade"], src)
        print(f"| {src:8s} | {s['n']:>7,} | {s['p95_abs']:>5.0f} | {s['dir_acc']:>4.1f}% | {s['bias']:>+6.0f} |")

print(f"\nFinal mem: {mem_mb():.0f} MB")
print("DONE")
