"""Shared utilities for annual FTR R1 baseline experiments.

Extracted from run_aq{1,2,3,4}_experiment.py to eliminate duplication.
"""

import resource

import numpy as np
import pandas as pd
import polars as pl


def mem_mb() -> float:
    """Current RSS in MB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def eval_baseline(mcp: pl.Series, baseline: pl.Series, label: str = "", total_n: int = 0) -> dict:
    """Comprehensive baseline evaluation: coverage, bias, MAE, direction accuracy, binned MAE."""
    mask = baseline.is_not_null() & mcp.is_not_null()
    n = int(mask.sum())
    empty = {
        "label": label, "n": 0, "coverage_pct": 0.0,
        "bias": 0, "mae": 0, "median_ae": 0, "p95_ae": 0, "p99_ae": 0,
        "dir_all": float("nan"), "n_dir": 0,
        "dir_50": float("nan"), "n_50": 0,
        "dir_100": float("nan"), "n_100": 0,
        "mae_tiny": float("nan"), "mae_small": float("nan"),
        "mae_med": float("nan"), "mae_large": float("nan"),
    }
    if n == 0:
        return empty

    m, b = mcp.filter(mask), baseline.filter(mask)
    res = m - b
    abs_res = res.abs()
    abs_m = m.abs()

    def dir_acc(ms, bs):
        valid = (ms != 0) & (bs != 0)
        vn = int(valid.sum())
        if vn == 0:
            return float("nan"), 0
        acc = float((ms.filter(valid).sign() == bs.filter(valid).sign()).mean()) * 100
        return round(acc, 1), vn

    da_all, dn_all = dir_acc(m, b)

    mask_50 = abs_m > 50
    n_50 = int(mask_50.sum())
    da_50 = dir_acc(m.filter(mask_50), b.filter(mask_50))[0] if n_50 > 0 else float("nan")

    mask_100 = abs_m > 100
    n_100 = int(mask_100.sum())
    da_100 = dir_acc(m.filter(mask_100), b.filter(mask_100))[0] if n_100 > 0 else float("nan")

    def mae_bin(lo, hi):
        bm = (abs_m >= lo) & (abs_m < hi)
        bn = int(bm.sum())
        return round(float(abs_res.filter(bm).mean()), 0) if bn > 100 else float("nan")

    return {
        "label": label, "n": n,
        "coverage_pct": round(n / total_n * 100, 1) if total_n > 0 else 0.0,
        "bias": round(float(res.mean()), 1),
        "mae": round(float(abs_res.mean()), 1),
        "median_ae": round(float(abs_res.median()), 1),
        "p95_ae": round(float(abs_res.quantile(0.95)), 0),
        "p99_ae": round(float(abs_res.quantile(0.99)), 0),
        "dir_all": da_all, "n_dir": dn_all,
        "dir_50": da_50, "n_50": n_50,
        "dir_100": da_100, "n_100": n_100,
        "mae_tiny": mae_bin(0, 50),
        "mae_small": mae_bin(50, 250),
        "mae_med": mae_bin(250, 1000),
        "mae_large": mae_bin(1000, 999999),
    }


def print_main_table(results: list[dict], title: str = ""):
    """Print summary table: coverage, bias, MAE, p95, direction accuracy."""
    if title:
        print(f"\n{title}")
    hdr = (
        f"  {'Source':<22} | {'n':>8} | {'Cov%':>5} | {'Bias':>7} | {'MAE':>6} "
        f"| {'MedAE':>6} | {'p95':>6} | {'p99':>6} | {'Dir%':>5} | {'D>50':>5} | {'D>100':>5}"
    )
    sep = f"  {'-'*22}-+-{'-'*8}-+-{'-'*5}-+-{'-'*7}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*5}-+-{'-'*5}-+-{'-'*5}"
    print(hdr)
    print(sep)
    for r in results:
        if r["n"] == 0:
            print(f"  {r['label']:<22} |      n/a |")
            continue
        print(
            f"  {r['label']:<22} | {r['n']:>8,} | {r['coverage_pct']:>5.1f} "
            f"| {r['bias']:>+7.0f} | {r['mae']:>6.0f} | {r['median_ae']:>6.0f} "
            f"| {r['p95_ae']:>6.0f} | {r['p99_ae']:>6.0f} "
            f"| {r['dir_all']:>5.1f} | {r['dir_50']:>5.1f} | {r['dir_100']:>5.1f}"
        )


def print_mae_table(results: list[dict], title: str = ""):
    """Print MAE breakdown by |MCP| bin."""
    if title:
        print(f"\n{title}")
    hdr = (
        f"  {'Source':<22} | {'MAE all':>8} | {'<50':>8} | {'50-250':>8} "
        f"| {'250-1k':>8} | {'1k+':>8}"
    )
    sep = f"  {'-'*22}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}"
    print(hdr)
    print(sep)
    for r in results:
        if r["n"] == 0:
            continue

        def fmt_mae(v):
            return f"{v:>8.0f}" if v == v else f"{'n/a':>8}"

        print(
            f"  {r['label']:<22} | {r['mae']:>8.0f} | {fmt_mae(r['mae_tiny'])} | {fmt_mae(r['mae_small'])} "
            f"| {fmt_mae(r['mae_med'])} | {fmt_mae(r['mae_large'])}"
        )


def build_py_fwd_map(repl_df: pd.DataFrame, target_date_str: str) -> dict[str, str]:
    """Forward replacement map filtered to replacements active at target_date.

    Handles tz-aware (EST) dates in the replacement data.
    """
    td = pd.Timestamp(target_date_str)
    start_col = repl_df["effective_start_date"]
    if hasattr(start_col.dt, "tz") and start_col.dt.tz is not None:
        td = td.tz_localize(start_col.dt.tz)
    active = repl_df[
        (start_col <= td) & (repl_df["effective_end_date"] >= td)
    ]
    return dict(zip(active["from_node"].astype(str), active["to_node"].astype(str)))


def get_all_aliases(nid: str, fwd_map: dict, rev_map: dict) -> set[str]:
    """BFS through forward AND reverse replacement chains."""
    aliases = set()
    queue = [nid]
    while queue:
        curr = queue.pop()
        if curr in aliases:
            continue
        aliases.add(curr)
        if curr in fwd_map:
            queue.append(fwd_map[curr])
        if curr in rev_map:
            queue.extend(rev_map[curr])
    return aliases


def print_cascade_stats(
    actual_arr: np.ndarray,
    pred: np.ndarray,
    label: str,
    total_n: int,
):
    """Print one row of cascade comparison table."""
    valid = ~np.isnan(pred)
    n = int(valid.sum())
    a = actual_arr[valid]
    p = pred[valid]
    res = a - p
    ae = np.abs(res)
    m = (a != 0) & (p != 0)
    d = float(np.mean(np.sign(a[m]) == np.sign(p[m]))) * 100 if m.sum() > 0 else 0
    print(
        f"  {label:<30} | {n:>8,} | {n/total_n*100:>5.1f}% "
        f"| {np.mean(res):>+7.0f} | {np.mean(ae):>7.0f} | {np.median(ae):>7.0f} "
        f"| {np.percentile(ae, 95):>7.0f} | {d:>5.1f}%"
    )


CASCADE_HEADER = (
    "  {:<30} | {:>8} | {:>6} | {:>7} | {:>7} | {:>7} | {:>7} | {:>6}".format(
        "Cascade", "n", "Cov%", "Bias", "MAE", "Med", "p95", "Dir%"
    )
)
CASCADE_SEP = f"  {'-'*30}-+-{'-'*8}-+-{'-'*6}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*6}"
