# DEPRECATED — archived 2026-03-17. Old data sources. See task_plan.md for porting plan.
import warnings as _w; _w.warn("This script is DEPRECATED (archived 2026-03-17). Use canonical get_trades_of_given_duration().", DeprecationWarning, stacklevel=2)

"""Phase 2: Improve the nodal f0 baseline with alpha scaling + prior-year residual correction.

Experiments:
  A. Alpha scaling (LOO per PY): find optimal multiplier
  B. Prior-year residual correction: adjusted = nodal_f0 + beta * prior_residual
  C. Combined: adjusted = alpha * nodal_f0 + beta * prior_residual
  D. Cross-quarter summary

Data: aq{1,2,3,4}_all_baselines.parquet (already on disk, no Ray needed).
"""

import gc
import sys
import os

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baseline_utils import mem_mb, eval_baseline, print_main_table

# ── Config ──────────────────────────────────────────────────────────────
DATA_DIR = "/opt/temp/qianli/annual_research/crossproduct_work"
QUARTERS = ["aq1", "aq2", "aq3", "aq4"]
PYS = [2020, 2021, 2022, 2023, 2024, 2025]  # exclude 2019 (no nodal_f0)
ALPHA_GRID = [1.0, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35, 1.40, 1.45, 1.50, 1.55, 1.60]
BETA_GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60]


# ── Helpers ─────────────────────────────────────────────────────────────
def load_quarter(q: str) -> pl.DataFrame:
    """Load baseline parquet, keep only rows with valid nodal_f0 and excl PY 2019."""
    path = f"{DATA_DIR}/{q}_all_baselines.parquet"
    df = (
        pl.scan_parquet(path)
        .filter(
            (pl.col("planning_year") >= 2020)
            & pl.col("nodal_f0").is_not_null()
            & pl.col("mcp_mean").is_not_null()
        )
        .select([
            "planning_year", "class_type", "source_id", "sink_id",
            "mcp_mean", "nodal_f0", "mtm_1st_mean",
        ])
        .collect()
    )
    # Add path key for joining prior-year residuals
    df = df.with_columns(
        (pl.col("source_id") + "," + pl.col("sink_id")).alias("path")
    )
    return df


def compute_mae(mcp: np.ndarray, pred: np.ndarray) -> float:
    valid = ~np.isnan(pred) & ~np.isnan(mcp)
    if valid.sum() == 0:
        return np.inf
    return float(np.abs(mcp[valid] - pred[valid]).mean())


def loo_optimal_alpha(df: pl.DataFrame, test_py: int) -> tuple[float, float]:
    """Find alpha that minimizes MAE on all PYs except test_py."""
    train = df.filter(pl.col("planning_year") != test_py)
    mcp_tr = train["mcp_mean"].to_numpy()
    f0_tr = train["nodal_f0"].to_numpy()
    best_alpha, best_mae = 1.0, np.inf
    for a in ALPHA_GRID:
        mae = compute_mae(mcp_tr, f0_tr * a)
        if mae < best_mae:
            best_mae = mae
            best_alpha = a
    return best_alpha, best_mae


def loo_optimal_alpha_beta(
    df: pl.DataFrame, test_py: int
) -> tuple[float, float, float]:
    """Find (alpha, beta) that minimizes MAE on training PYs.

    Model: adjusted = alpha * nodal_f0 + beta * prior_residual
    Only uses rows where prior_residual is available.
    """
    train = df.filter(
        (pl.col("planning_year") != test_py)
        & pl.col("prior_residual").is_not_null()
    )
    if len(train) == 0:
        return 1.0, 0.0, np.inf
    mcp_tr = train["mcp_mean"].to_numpy()
    f0_tr = train["nodal_f0"].to_numpy()
    pr_tr = train["prior_residual"].to_numpy()
    best_a, best_b, best_mae = 1.0, 0.0, np.inf
    for a in ALPHA_GRID:
        for b in BETA_GRID:
            pred = f0_tr * a + pr_tr * b
            mae = compute_mae(mcp_tr, pred)
            if mae < best_mae:
                best_mae = mae
                best_a, best_b = a, b
    return best_a, best_b, best_mae


def eval_series(mcp: pl.Series, pred: pl.Series, label: str, total_n: int) -> dict:
    """Thin wrapper around eval_baseline for convenience."""
    return eval_baseline(mcp, pred, label=label, total_n=total_n)


def print_loo_table(rows: list[dict], title: str):
    """Print LOO results per PY."""
    print(f"\n{title}")
    print(f"  {'PY':>4} | {'Alpha':>5} | {'Beta':>5} | {'MAE orig':>8} | {'MAE adj':>8} "
          f"| {'Impr%':>6} | {'Bias orig':>9} | {'Bias adj':>9} | {'Dir% orig':>9} | {'Dir% adj':>9} | {'Cov%':>5}")
    print(f"  {'-'*4}-+-{'-'*5}-+-{'-'*5}-+-{'-'*8}-+-{'-'*8}-+-{'-'*6}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}-+-{'-'*5}")
    for r in rows:
        impr = (1 - r["mae_adj"] / r["mae_orig"]) * 100 if r["mae_orig"] > 0 else 0
        print(
            f"  {r['py']:>4} | {r['alpha']:>5.2f} | {r['beta']:>5.2f} | {r['mae_orig']:>8.1f} | {r['mae_adj']:>8.1f} "
            f"| {impr:>+5.1f}% | {r['bias_orig']:>+9.1f} | {r['bias_adj']:>+9.1f} "
            f"| {r['dir_orig']:>8.1f}% | {r['dir_adj']:>8.1f}% | {r['cov_pct']:>5.1f}"
        )
    # Averages
    avg_impr = np.mean([(1 - r["mae_adj"] / r["mae_orig"]) * 100 for r in rows])
    avg_mae_o = np.mean([r["mae_orig"] for r in rows])
    avg_mae_a = np.mean([r["mae_adj"] for r in rows])
    print(f"  {'AVG':>4} | {'':>5} | {'':>5} | {avg_mae_o:>8.1f} | {avg_mae_a:>8.1f} | {avg_impr:>+5.1f}%")


# ── Main ────────────────────────────────────────────────────────────────
print(f"Phase 2 improvement experiments. Memory: {mem_mb():.0f} MB")
print("=" * 100)

# Store cross-quarter summary
summary_rows_A = []  # alpha-only
summary_rows_B = []  # residual-only
summary_rows_C = []  # combined

for q in QUARTERS:
    print(f"\n{'='*100}")
    print(f"  QUARTER: {q.upper()}")
    print(f"{'='*100}")

    df = load_quarter(q)
    n_total = len(df)
    print(f"  Loaded {n_total:,} rows.  Memory: {mem_mb():.0f} MB")
    print(f"  PY distribution: {df.group_by('planning_year').len().sort('planning_year').to_dict()}")

    # ── Section A: Alpha scaling (LOO) ──
    print(f"\n  --- Section A: Alpha Scaling (LOO) ---")
    loo_A = []
    for test_py in PYS:
        subset = df.filter(pl.col("planning_year") == test_py)
        if len(subset) == 0:
            continue
        best_alpha, _ = loo_optimal_alpha(df, test_py)

        mcp_arr = subset["mcp_mean"].to_numpy()
        f0_arr = subset["nodal_f0"].to_numpy()
        pred_orig = f0_arr
        pred_adj = f0_arr * best_alpha
        mae_orig = float(np.abs(mcp_arr - pred_orig).mean())
        mae_adj = float(np.abs(mcp_arr - pred_adj).mean())
        bias_orig = float((mcp_arr - pred_orig).mean())
        bias_adj = float((mcp_arr - pred_adj).mean())

        # Direction accuracy
        valid_dir = (mcp_arr != 0) & (pred_orig != 0)
        dir_orig = float(np.mean(np.sign(mcp_arr[valid_dir]) == np.sign(pred_orig[valid_dir]))) * 100
        valid_dir2 = (mcp_arr != 0) & (pred_adj != 0)
        dir_adj = float(np.mean(np.sign(mcp_arr[valid_dir2]) == np.sign(pred_adj[valid_dir2]))) * 100

        loo_A.append({
            "py": test_py, "alpha": best_alpha, "beta": 0.0,
            "mae_orig": mae_orig, "mae_adj": mae_adj,
            "bias_orig": bias_orig, "bias_adj": bias_adj,
            "dir_orig": dir_orig, "dir_adj": dir_adj,
            "cov_pct": 100.0,
        })

    print_loo_table(loo_A, f"  A. Alpha-Only LOO ({q})")
    summary_rows_A.append({"quarter": q, "rows": loo_A})

    # ── Section B + C: Prior-year residual correction ──
    # Build prior-year residual: for each (path, PY), get residual from PY-1
    df = df.with_columns(
        (pl.col("mcp_mean") - pl.col("nodal_f0")).alias("residual")
    )
    prior = df.select([
        pl.col("planning_year").alias("prior_py"),
        "path", "class_type",
        pl.col("residual").alias("prior_residual"),
    ])
    df = df.with_columns(
        (pl.col("planning_year") - 1).alias("prior_py")
    )
    df = df.join(
        prior,
        left_on=["prior_py", "path", "class_type"],
        right_on=["prior_py", "path", "class_type"],
        how="left",
    ).drop("prior_py")

    n_with_prior = int(df["prior_residual"].is_not_null().sum())
    print(f"\n  Prior-year residual coverage: {n_with_prior:,} / {n_total:,} ({n_with_prior/n_total*100:.1f}%)")

    # Section B: Residual-only (alpha=1.0, optimize beta)
    print(f"\n  --- Section B: Residual-Only (alpha=1.0, LOO beta) ---")
    loo_B = []
    for test_py in PYS:
        subset = df.filter(
            (pl.col("planning_year") == test_py) & pl.col("prior_residual").is_not_null()
        )
        if len(subset) == 0:
            continue

        # LOO: train on all other PYs with prior_residual
        train = df.filter(
            (pl.col("planning_year") != test_py) & pl.col("prior_residual").is_not_null()
        )
        mcp_tr = train["mcp_mean"].to_numpy()
        f0_tr = train["nodal_f0"].to_numpy()
        pr_tr = train["prior_residual"].to_numpy()
        best_b, best_mae_tr = 0.0, np.inf
        for b in BETA_GRID:
            pred = f0_tr + pr_tr * b
            mae = compute_mae(mcp_tr, pred)
            if mae < best_mae_tr:
                best_mae_tr = mae
                best_b = b

        # Evaluate on test PY
        mcp_arr = subset["mcp_mean"].to_numpy()
        f0_arr = subset["nodal_f0"].to_numpy()
        pr_arr = subset["prior_residual"].to_numpy()
        pred_orig = f0_arr
        pred_adj = f0_arr + pr_arr * best_b
        mae_orig = float(np.abs(mcp_arr - pred_orig).mean())
        mae_adj = float(np.abs(mcp_arr - pred_adj).mean())
        bias_orig = float((mcp_arr - pred_orig).mean())
        bias_adj = float((mcp_arr - pred_adj).mean())

        valid_dir = (mcp_arr != 0) & (pred_orig != 0)
        dir_orig = float(np.mean(np.sign(mcp_arr[valid_dir]) == np.sign(pred_orig[valid_dir]))) * 100
        valid_dir2 = (mcp_arr != 0) & (pred_adj != 0)
        dir_adj = float(np.mean(np.sign(mcp_arr[valid_dir2]) == np.sign(pred_adj[valid_dir2]))) * 100

        loo_B.append({
            "py": test_py, "alpha": 1.0, "beta": best_b,
            "mae_orig": mae_orig, "mae_adj": mae_adj,
            "bias_orig": bias_orig, "bias_adj": bias_adj,
            "dir_orig": dir_orig, "dir_adj": dir_adj,
            "cov_pct": len(subset) / int(df.filter(pl.col("planning_year") == test_py).height) * 100,
        })

    if loo_B:
        print_loo_table(loo_B, f"  B. Residual-Only LOO ({q})")
    else:
        print(f"  B. No data with prior residuals for {q}")
    summary_rows_B.append({"quarter": q, "rows": loo_B})

    # Section C: Combined (optimize alpha + beta jointly)
    print(f"\n  --- Section C: Combined Alpha + Beta (LOO) ---")
    loo_C = []
    for test_py in PYS:
        subset = df.filter(
            (pl.col("planning_year") == test_py) & pl.col("prior_residual").is_not_null()
        )
        if len(subset) == 0:
            continue

        best_a, best_b, _ = loo_optimal_alpha_beta(df, test_py)

        mcp_arr = subset["mcp_mean"].to_numpy()
        f0_arr = subset["nodal_f0"].to_numpy()
        pr_arr = subset["prior_residual"].to_numpy()
        pred_orig = f0_arr
        pred_adj = f0_arr * best_a + pr_arr * best_b
        mae_orig = float(np.abs(mcp_arr - pred_orig).mean())
        mae_adj = float(np.abs(mcp_arr - pred_adj).mean())
        bias_orig = float((mcp_arr - pred_orig).mean())
        bias_adj = float((mcp_arr - pred_adj).mean())

        valid_dir = (mcp_arr != 0) & (pred_orig != 0)
        dir_orig = float(np.mean(np.sign(mcp_arr[valid_dir]) == np.sign(pred_orig[valid_dir]))) * 100
        valid_dir2 = (mcp_arr != 0) & (pred_adj != 0)
        dir_adj = float(np.mean(np.sign(mcp_arr[valid_dir2]) == np.sign(pred_adj[valid_dir2]))) * 100

        loo_C.append({
            "py": test_py, "alpha": best_a, "beta": best_b,
            "mae_orig": mae_orig, "mae_adj": mae_adj,
            "bias_orig": bias_orig, "bias_adj": bias_adj,
            "dir_orig": dir_orig, "dir_adj": dir_adj,
            "cov_pct": len(subset) / int(df.filter(pl.col("planning_year") == test_py).height) * 100,
        })

    if loo_C:
        print_loo_table(loo_C, f"  C. Combined Alpha+Beta LOO ({q})")
    else:
        print(f"  C. No data with prior residuals for {q}")
    summary_rows_C.append({"quarter": q, "rows": loo_C})

    # ── Section D: Head-to-head on residual subset ──
    # Compare: orig nodal_f0 vs alpha-only vs residual-only vs combined
    # on the SAME rows (those with prior_residual)
    print(f"\n  --- Section D: Head-to-Head on Prior-Residual Subset ({q}) ---")
    sub = df.filter(
        (pl.col("planning_year") >= 2021)  # need PY-1 to have data
        & pl.col("prior_residual").is_not_null()
    )
    if len(sub) > 0:
        n_sub = len(sub)
        mcp_s = sub["mcp_mean"]
        f0_s = sub["nodal_f0"]

        # Get median alpha and beta from LOO results
        if loo_A:
            med_alpha = float(np.median([r["alpha"] for r in loo_A]))
        else:
            med_alpha = 1.45
        if loo_C:
            med_alpha_c = float(np.median([r["alpha"] for r in loo_C]))
            med_beta_c = float(np.median([r["beta"] for r in loo_C]))
        else:
            med_alpha_c, med_beta_c = 1.45, 0.15
        if loo_B:
            med_beta_b = float(np.median([r["beta"] for r in loo_B]))
        else:
            med_beta_b = 0.15

        pr_s = sub["prior_residual"]

        results_hh = []
        results_hh.append(eval_series(mcp_s, f0_s, "nodal_f0 (orig)", n_sub))
        results_hh.append(eval_series(
            mcp_s, f0_s * med_alpha, f"alpha={med_alpha:.2f}", n_sub
        ))
        results_hh.append(eval_series(
            mcp_s, f0_s + pr_s * med_beta_b, f"resid beta={med_beta_b:.2f}", n_sub
        ))
        results_hh.append(eval_series(
            mcp_s,
            f0_s * med_alpha_c + pr_s * med_beta_c,
            f"a={med_alpha_c:.2f} b={med_beta_c:.2f}",
            n_sub,
        ))
        # Also test H baseline on same subset
        h_s = sub["mtm_1st_mean"]
        results_hh.append(eval_series(mcp_s, h_s, "H (reference)", n_sub))

        print_main_table(results_hh, f"  D. Head-to-Head on {n_sub:,} rows with prior residual ({q})")
    else:
        print("  D. No rows with prior residual")

    del df
    gc.collect()

# ── Cross-Quarter Summary ──
print(f"\n{'='*100}")
print("  CROSS-QUARTER SUMMARY")
print(f"{'='*100}")

for label, summary in [("A. Alpha-Only", summary_rows_A),
                         ("B. Residual-Only", summary_rows_B),
                         ("C. Combined", summary_rows_C)]:
    print(f"\n  --- {label} ---")
    print(f"  {'Quarter':<6} | {'Avg MAE orig':>12} | {'Avg MAE adj':>12} | {'Avg Impr%':>9} | {'Med Alpha':>9} | {'Med Beta':>9}")
    print(f"  {'-'*6}-+-{'-'*12}-+-{'-'*12}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}")
    for item in summary:
        q = item["quarter"]
        rows = item["rows"]
        if not rows:
            print(f"  {q:<6} | {'n/a':>12}")
            continue
        avg_orig = np.mean([r["mae_orig"] for r in rows])
        avg_adj = np.mean([r["mae_adj"] for r in rows])
        avg_impr = (1 - avg_adj / avg_orig) * 100
        med_a = np.median([r["alpha"] for r in rows])
        med_b = np.median([r["beta"] for r in rows])
        print(f"  {q:<6} | {avg_orig:>12.1f} | {avg_adj:>12.1f} | {avg_impr:>+8.1f}% | {med_a:>9.2f} | {med_b:>9.2f}")

print(f"\nDone. Memory: {mem_mb():.0f} MB")
