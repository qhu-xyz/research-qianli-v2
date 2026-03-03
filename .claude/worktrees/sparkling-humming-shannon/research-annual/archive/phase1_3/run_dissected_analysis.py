#!/usr/bin/env python3
"""
Dissected analysis of R1 baselines by MCP magnitude.

Breaks down direction accuracy, bias, absolute error, and other stats
by |mcp| bins, quarter, PY, class_type, and baseline source.

Reads from: crossproduct_work/r1_final.parquet (output of Phase 3)
Peak memory: ~200 MB (636K rows, 11 cols)
"""

import gc
import os
import resource

import numpy as np
import polars as pl

DATA_DIR = "/opt/temp/qianli/annual_research"
WORK_DIR = f"{DATA_DIR}/crossproduct_work"
r1_path = f"{WORK_DIR}/r1_final.parquet"

assert os.path.exists(r1_path), f"Missing: {r1_path}"


def mem_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def compute_stats(df: pl.DataFrame, mcp_col: str, baseline_col: str) -> dict:
    """Compute all stats for a (mcp, baseline) pair. Returns dict."""
    mask = df[baseline_col].is_not_null() & df[mcp_col].is_not_null()
    n = int(mask.sum())
    if n == 0:
        return {"n": 0}
    sub = df.filter(mask)
    mcp = sub[mcp_col]
    bl = sub[baseline_col]
    res = mcp - bl
    abs_res = res.abs()

    dir_mask = (mcp != 0) & (bl != 0)
    dn = int(dir_mask.sum())
    if dn > 0:
        dir_acc = float((mcp.filter(dir_mask).sign() == bl.filter(dir_mask).sign()).mean()) * 100
    else:
        dir_acc = float("nan")

    return {
        "n": n,
        "bias": round(float(res.mean()), 1),
        "mean_abs": round(float(abs_res.mean()), 1),
        "median_abs": round(float(abs_res.median()), 1),
        "p75_abs": round(float(abs_res.quantile(0.75)), 0),
        "p90_abs": round(float(abs_res.quantile(0.90)), 0),
        "p95_abs": round(float(abs_res.quantile(0.95)), 0),
        "p99_abs": round(float(abs_res.quantile(0.99)), 0),
        "dir_acc": round(dir_acc, 1),
    }


def fmt_row(label: str, s: dict, extra_cols: list[str] | None = None) -> str:
    if s["n"] == 0:
        return f"| {label} | 0 | — | — | — | — | — | — |"
    base = (
        f"| {label} | {s['n']:>7,} | {s['bias']:>+7.0f} | {s['mean_abs']:>7.0f} "
        f"| {s['p95_abs']:>7.0f} | {s['p99_abs']:>7.0f} | {s['dir_acc']:>5.1f}% |"
    )
    return base


# ================================================================
# Load data
# ================================================================
print(f"Loading data | mem: {mem_mb():.0f} MB")
r1 = pl.read_parquet(r1_path)
print(f"  Shape: {r1.shape}, mem: {mem_mb():.0f} MB")

# Build cascade column
r1 = r1.with_columns(
    pl.col("prior_r3_path")
    .fill_null(pl.col("f0_avg_mcp"))
    .fill_null(pl.col("nodal_r3"))
    .fill_null(pl.col("mtm_1st_mean"))
    .alias("cascade")
)

# Source label
r1 = r1.with_columns(
    pl.when(pl.col("prior_r3_path").is_not_null()).then(pl.lit("path_r3"))
    .when(pl.col("f0_avg_mcp").is_not_null()).then(pl.lit("f0_cross"))
    .when(pl.col("nodal_r3").is_not_null()).then(pl.lit("nodal_r3"))
    .otherwise(pl.lit("H"))
    .alias("src_label")
)

# Bins by |mcp|
r1 = r1.with_columns(
    pl.when(pl.col("mcp_mean").abs() < 50).then(pl.lit("A: |mcp|<50"))
    .when(pl.col("mcp_mean").abs() < 100).then(pl.lit("B: 50-100"))
    .when(pl.col("mcp_mean").abs() < 250).then(pl.lit("C: 100-250"))
    .when(pl.col("mcp_mean").abs() < 1000).then(pl.lit("D: 250-1k"))
    .otherwise(pl.lit("E: 1k+"))
    .alias("mcp_bin")
)

# Also bins by |baseline| (H)
r1 = r1.with_columns(
    pl.when(pl.col("mtm_1st_mean").abs() < 50).then(pl.lit("A: |H|<50"))
    .when(pl.col("mtm_1st_mean").abs() < 100).then(pl.lit("B: 50-100"))
    .when(pl.col("mtm_1st_mean").abs() < 250).then(pl.lit("C: 100-250"))
    .when(pl.col("mtm_1st_mean").abs() < 1000).then(pl.lit("D: 250-1k"))
    .otherwise(pl.lit("E: 1k+"))
    .alias("h_bin")
)

output_lines = []


def p(line=""):
    print(line)
    output_lines.append(line)


baselines = [
    ("mtm_1st_mean", "H (DA congestion)"),
    ("prior_r3_path", "Prior R3 path MCP"),
    ("nodal_r3", "Nodal stitched"),
    ("f0_avg_mcp", "f0 cross-product"),
    ("cascade", "Cascade (best avail)"),
]


# ================================================================
# 1. Overall stats per baseline
# ================================================================
p("# Dissected R1 Baseline Analysis")
p()
p("## 1. Overall Stats per Baseline")
p()
p("| Baseline | n | Bias | MAE | p95 | p99 | Dir Acc |")
p("|----------|---|------|-----|-----|-----|---------|")
for col, label in baselines:
    s = compute_stats(r1, "mcp_mean", col)
    p(fmt_row(label, s))


# ================================================================
# 2. Direction accuracy by |MCP| magnitude
# ================================================================
p()
p("## 2. Direction Accuracy by |MCP| Magnitude")
p()
p("Only paths where |actual MCP| exceeds threshold. Higher |MCP| = more money at stake.")
p()

for col, label in baselines:
    p(f"### {label}")
    p("| Filter | n | Bias | MAE | p95 | p99 | Dir Acc |")
    p("|--------|---|------|-----|-----|-----|---------|")
    for thr_label, thr in [("All", 0), ("|mcp|>50", 50), ("|mcp|>100", 100), ("|mcp|>250", 250), ("|mcp|>1000", 1000)]:
        sub = r1.filter(pl.col("mcp_mean").abs() > thr)
        s = compute_stats(sub, "mcp_mean", col)
        p(fmt_row(thr_label, s))
    p()


# ================================================================
# 3. Stats by |MCP| bin (full breakdown)
# ================================================================
p("## 3. Stats by |MCP| Bin")
p()
p("Paths grouped by actual |MCP| magnitude.")
p()

for col, label in baselines:
    p(f"### {label}")
    p("| |MCP| Bin | n | Bias | MAE | p75 | p90 | p95 | p99 | Dir |")
    p("|----------|---|------|-----|-----|-----|-----|-----|-----|")
    for b in ["A: |mcp|<50", "B: 50-100", "C: 100-250", "D: 250-1k", "E: 1k+"]:
        sub = r1.filter(pl.col("mcp_bin") == b)
        s = compute_stats(sub, "mcp_mean", col)
        if s["n"] > 0:
            p(
                f"| {b} | {s['n']:>7,} | {s['bias']:>+7.0f} | {s['mean_abs']:>7.0f} "
                f"| {s['p75_abs']:>5.0f} | {s['p90_abs']:>5.0f} | {s['p95_abs']:>5.0f} "
                f"| {s['p99_abs']:>5.0f} | {s['dir_acc']:>5.1f}% |"
            )
    p()


# ================================================================
# 4. Stats by |H| bin (baseline magnitude)
# ================================================================
p("## 4. Stats by |H| Bin (baseline magnitude)")
p()
p("Paths grouped by how large the H baseline is. Shows how well each baseline")
p("performs for small-H vs large-H paths.")
p()

for col, label in baselines:
    p(f"### {label}")
    p("| |H| Bin | n | Bias | MAE | p95 | p99 | Dir |")
    p("|--------|---|------|-----|-----|-----|-----|")
    for b in ["A: |H|<50", "B: 50-100", "C: 100-250", "D: 250-1k", "E: 1k+"]:
        sub = r1.filter(pl.col("h_bin") == b)
        s = compute_stats(sub, "mcp_mean", col)
        if s["n"] > 0:
            p(
                f"| {b} | {s['n']:>7,} | {s['bias']:>+7.0f} | {s['mean_abs']:>7.0f} "
                f"| {s['p95_abs']:>5.0f} | {s['p99_abs']:>5.0f} | {s['dir_acc']:>5.1f}% |"
            )
    p()


# ================================================================
# 5. By quarter
# ================================================================
p("## 5. By Quarter")
p()

for col, label in baselines:
    p(f"### {label}")
    p("| Qt | n | Bias | MAE | p95 | p99 | Dir |")
    p("|----|---|------|-----|-----|-----|-----|")
    for qt in ["aq1", "aq2", "aq3", "aq4"]:
        sub = r1.filter(pl.col("period_type") == qt)
        s = compute_stats(sub, "mcp_mean", col)
        if s["n"] > 0:
            p(
                f"| {qt} | {s['n']:>7,} | {s['bias']:>+7.0f} | {s['mean_abs']:>7.0f} "
                f"| {s['p95_abs']:>5.0f} | {s['p99_abs']:>5.0f} | {s['dir_acc']:>5.1f}% |"
            )
    p()


# ================================================================
# 6. By planning year
# ================================================================
p("## 6. By Planning Year")
p()

for col, label in [("mtm_1st_mean", "H"), ("cascade", "Cascade")]:
    p(f"### {label}")
    p("| PY | n | Bias | MAE | p95 | Dir |")
    p("|----|---|------|-----|-----|-----|")
    for py in sorted(r1["planning_year"].unique().to_list()):
        sub = r1.filter(pl.col("planning_year") == py)
        s = compute_stats(sub, "mcp_mean", col)
        if s["n"] > 0:
            p(
                f"| {py} | {s['n']:>7,} | {s['bias']:>+7.0f} | {s['mean_abs']:>7.0f} "
                f"| {s['p95_abs']:>5.0f} | {s['dir_acc']:>5.1f}% |"
            )
    p()


# ================================================================
# 7. By class type (onpeak vs offpeak)
# ================================================================
p("## 7. By Class Type")
p()
p("| Class | Baseline | n | Bias | MAE | p95 | Dir |")
p("|-------|----------|---|------|-----|-----|-----|")
for ct in ["onpeak", "offpeak"]:
    for col, label in [("mtm_1st_mean", "H"), ("cascade", "Cascade")]:
        sub = r1.filter(pl.col("class_type") == ct)
        s = compute_stats(sub, "mcp_mean", col)
        if s["n"] > 0:
            p(
                f"| {ct} | {label} | {s['n']:>7,} | {s['bias']:>+7.0f} | {s['mean_abs']:>7.0f} "
                f"| {s['p95_abs']:>5.0f} | {s['dir_acc']:>5.1f}% |"
            )
p()


# ================================================================
# 8. Cascade source quality by |MCP| bin
# ================================================================
p("## 8. Cascade Source Quality by |MCP| Bin")
p()
p("For paths where cascade is used, which source contributes and how well?")
p()
p("| |MCP| Bin | Source | n | Bias | MAE | p95 | Dir |")
p("|----------|--------|---|------|-----|-----|-----|")
for b in ["A: |mcp|<50", "B: 50-100", "C: 100-250", "D: 250-1k", "E: 1k+"]:
    for src in ["path_r3", "f0_cross", "nodal_r3", "H"]:
        sub = r1.filter((pl.col("mcp_bin") == b) & (pl.col("src_label") == src))
        s = compute_stats(sub, "mcp_mean", "cascade")
        if s["n"] > 0:
            p(
                f"| {b} | {src:8s} | {s['n']:>7,} | {s['bias']:>+7.0f} | {s['mean_abs']:>7.0f} "
                f"| {s['p95_abs']:>5.0f} | {s['dir_acc']:>5.1f}% |"
            )
p()


# ================================================================
# 9. Direction accuracy by quarter x |MCP| threshold
# ================================================================
p("## 9. Direction Accuracy Grid: Quarter x |MCP| Threshold")
p()
p("Cascade baseline only.")
p()
p("| Qt | All | >50 | >100 | >250 | >1000 |")
p("|----|-----|-----|------|------|-------|")
for qt in ["aq1", "aq2", "aq3", "aq4"]:
    cells = [qt]
    for thr in [0, 50, 100, 250, 1000]:
        sub = r1.filter((pl.col("period_type") == qt) & (pl.col("mcp_mean").abs() > thr))
        s = compute_stats(sub, "mcp_mean", "cascade")
        if s["n"] > 0:
            cells.append(f"{s['dir_acc']:.1f}% ({s['n']:,})")
        else:
            cells.append("—")
    p("| " + " | ".join(cells) + " |")
p()


# ================================================================
# 10. MCP distribution stats (independent of baselines)
# ================================================================
p("## 10. MCP Distribution (actual clearing prices)")
p()
mcp = r1["mcp_mean"]
abs_mcp = mcp.abs()
p(f"- Total paths: {r1.height:,}")
p(f"- Mean MCP: {float(mcp.mean()):.1f}")
p(f"- Median |MCP|: {float(abs_mcp.median()):.1f}")
p(f"- p75 |MCP|: {float(abs_mcp.quantile(0.75)):.0f}")
p(f"- p90 |MCP|: {float(abs_mcp.quantile(0.90)):.0f}")
p(f"- p95 |MCP|: {float(abs_mcp.quantile(0.95)):.0f}")
p(f"- p99 |MCP|: {float(abs_mcp.quantile(0.99)):.0f}")
p(f"- % positive: {float((mcp > 0).mean()) * 100:.1f}%")
p(f"- % |MCP| < 50: {float((abs_mcp < 50).mean()) * 100:.1f}%")
p(f"- % |MCP| < 100: {float((abs_mcp < 100).mean()) * 100:.1f}%")
p(f"- % |MCP| > 1000: {float((abs_mcp > 1000).mean()) * 100:.1f}%")
p()

p("### By quarter")
p("| Qt | n | Mean | Med |MCP| | p95 |MCP| | % pos | % |MCP|>1k |")
p("|----|---|------|---------|-----------|-------|-----------|")
for qt in ["aq1", "aq2", "aq3", "aq4"]:
    sub = r1.filter(pl.col("period_type") == qt)
    m = sub["mcp_mean"]
    am = m.abs()
    p(
        f"| {qt} | {sub.height:>7,} | {float(m.mean()):>+7.1f} | {float(am.median()):>7.1f} "
        f"| {float(am.quantile(0.95)):>9.0f} | {float((m > 0).mean()) * 100:>5.1f}% "
        f"| {float((am > 1000).mean()) * 100:>9.1f}% |"
    )
p()


# ================================================================
# Write findings doc
# ================================================================
findings_path = "/home/xyz/workspace/research-qianli-v2/research-annual/findings_dissected.md"
with open(findings_path, "w") as f:
    f.write("\n".join(output_lines) + "\n")
print(f"\nWritten to: {findings_path}")
print(f"Final mem: {mem_mb():.0f} MB")
print("DONE")
