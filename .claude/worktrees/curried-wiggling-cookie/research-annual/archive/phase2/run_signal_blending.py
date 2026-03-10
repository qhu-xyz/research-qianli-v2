"""Phase 2b: Signal Blending — can combining available baselines beat α-scaled nodal_f0?

Tests linear combinations of available signals via Ridge regression with LOO by PY.

Tiers by signal availability (rows must have nodal_f0 + mcp):
  T1: nodal_f0 + H                         (~99% coverage)
  T2: nodal_f0 + f0_path + H               (~45-55%)
  T3: nodal_f0 + f0_path + f1 + H          (~25-42%)
  T4: all core signals                      (~10-12%)

Also tests:
  - Prior-year residual as additional feature
  - Inverse-MAE weighted average (no fitting)

Data: aq{1,2,3,4}_all_baselines.parquet (already on disk, no Ray needed).
"""

import gc
import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baseline_utils import mem_mb

# ── Config ──────────────────────────────────────────────────────────────
DATA_DIR = "/opt/temp/qianli/annual_research/crossproduct_work"
QUARTERS = ["aq1", "aq2", "aq3", "aq4"]
PYS = [2020, 2021, 2022, 2023, 2024, 2025]
ALPHA_GRID = [1.0, 1.10, 1.20, 1.30, 1.40, 1.45, 1.50, 1.55, 1.60]

# Core signals available in all quarters
CORE_SIGNALS = ["nodal_f0", "mtm_1st_mean", "f0_path_corr", "f1_path",
                "prior_r1_path", "prior_r2_path", "prior_r3_path"]

# Quarter-specific signals
Q_SIGNALS = {"aq2": "q2_path", "aq3": "q3_path", "aq4": "q4_path"}


# ── Helpers ─────────────────────────────────────────────────────────────
def ridge_fit(X: np.ndarray, y: np.ndarray, lam: float = 1.0) -> np.ndarray:
    """Fit ridge regression: w = (X^T X + λI)^{-1} X^T y. Returns weights."""
    n_feat = X.shape[1]
    XtX = X.T @ X
    Xty = X.T @ y
    w = np.linalg.solve(XtX + lam * np.eye(n_feat), Xty)
    return w


def ridge_predict(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    return X @ w


def compute_stats(mcp: np.ndarray, pred: np.ndarray) -> dict:
    """Compute MAE, bias, MedAE, p95, Dir% on valid rows."""
    valid = ~np.isnan(pred) & ~np.isnan(mcp)
    n = int(valid.sum())
    if n == 0:
        return {"n": 0, "mae": np.inf, "bias": 0, "medae": 0, "p95": 0, "dir_pct": 0}
    m, p = mcp[valid], pred[valid]
    res = m - p
    ae = np.abs(res)
    dm = (m != 0) & (p != 0)
    dir_pct = float(np.mean(np.sign(m[dm]) == np.sign(p[dm]))) * 100 if dm.sum() > 0 else 0
    return {
        "n": n,
        "mae": float(ae.mean()),
        "bias": float(res.mean()),
        "medae": float(np.median(ae)),
        "p95": float(np.percentile(ae, 95)),
        "dir_pct": dir_pct,
    }


def loo_alpha(df: pl.DataFrame, test_py: int) -> float:
    """Find optimal alpha for nodal_f0 scaling via LOO."""
    train = df.filter(pl.col("planning_year") != test_py)
    mcp_tr = train["mcp_mean"].to_numpy()
    f0_tr = train["nodal_f0"].to_numpy()
    best_a, best_mae = 1.0, np.inf
    for a in ALPHA_GRID:
        mae = float(np.abs(mcp_tr - f0_tr * a).mean())
        if mae < best_mae:
            best_mae = mae
            best_a = a
    return best_a


def build_X(df: pl.DataFrame, signal_cols: list[str]) -> np.ndarray:
    """Build feature matrix from signal columns, adding intercept."""
    cols = [df[c].to_numpy().astype(float) for c in signal_cols]
    X = np.column_stack(cols + [np.ones(len(df))])  # add intercept
    return X


def print_blend_table(rows: list[dict], title: str):
    """Print blending comparison table."""
    print(f"\n  {title}")
    hdr = (
        f"  {'Method':<35} | {'n':>8} | {'Bias':>7} | {'MAE':>6} | {'MedAE':>6} "
        f"| {'p95':>7} | {'Dir%':>5} | {'vs raw':>6}"
    )
    sep = (
        f"  {'-'*35}-+-{'-'*8}-+-{'-'*7}-+-{'-'*6}-+-{'-'*6}"
        f"-+-{'-'*7}-+-{'-'*5}-+-{'-'*6}"
    )
    print(hdr)
    print(sep)
    raw_mae = rows[0]["mae"] if rows else 0
    for r in rows:
        impr = (1 - r["mae"] / raw_mae) * 100 if raw_mae > 0 else 0
        impr_str = f"{impr:>+5.1f}%" if r != rows[0] else "  base"
        print(
            f"  {r['label']:<35} | {r['n']:>8,} | {r['bias']:>+7.0f} | {r['mae']:>6.0f} "
            f"| {r['medae']:>6.0f} | {r['p95']:>7.0f} | {r['dir_pct']:>5.1f} | {impr_str}"
        )


def print_py_table(rows: list[dict], title: str):
    """Print per-PY LOO results."""
    print(f"\n  {title}")
    hdr = (
        f"  {'PY':>4} | {'n':>7} | {'Nodal f0':>8} | {'α-scaled':>8} | {'Blend':>8} "
        f"| {'vs raw':>6} | {'vs α':>6} | {'Blend wts':>40}"
    )
    sep = (
        f"  {'-'*4}-+-{'-'*7}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}"
        f"-+-{'-'*6}-+-{'-'*6}-+-{'-'*40}"
    )
    print(hdr)
    print(sep)
    for r in rows:
        vs_raw = (1 - r["blend_mae"] / r["raw_mae"]) * 100 if r["raw_mae"] > 0 else 0
        vs_alpha = (1 - r["blend_mae"] / r["alpha_mae"]) * 100 if r["alpha_mae"] > 0 else 0
        print(
            f"  {r['py']:>4} | {r['n']:>7,} | {r['raw_mae']:>8.0f} | {r['alpha_mae']:>8.0f} "
            f"| {r['blend_mae']:>8.0f} | {vs_raw:>+5.1f}% | {vs_alpha:>+5.1f}% | {r['wts']}"
        )
    # Averages
    avg_raw = np.mean([r["raw_mae"] for r in rows])
    avg_alpha = np.mean([r["alpha_mae"] for r in rows])
    avg_blend = np.mean([r["blend_mae"] for r in rows])
    vs_raw = (1 - avg_blend / avg_raw) * 100
    vs_alpha = (1 - avg_blend / avg_alpha) * 100
    print(
        f"  {'AVG':>4} | {'':>7} | {avg_raw:>8.0f} | {avg_alpha:>8.0f} "
        f"| {avg_blend:>8.0f} | {vs_raw:>+5.1f}% | {vs_alpha:>+5.1f}%"
    )


# ── Tier definitions ────────────────────────────────────────────────────
def get_tiers(q: str) -> list[tuple[str, list[str]]]:
    """Return list of (tier_name, signal_columns) for a quarter."""
    tiers = [
        ("T1: nodal+H", ["nodal_f0", "mtm_1st_mean"]),
        ("T2: nodal+f0path+H", ["nodal_f0", "f0_path_corr", "mtm_1st_mean"]),
        ("T3: nodal+f0path+f1+H", ["nodal_f0", "f0_path_corr", "f1_path", "mtm_1st_mean"]),
    ]
    # T4: add prior rounds
    t4_cols = ["nodal_f0", "f0_path_corr", "f1_path",
               "prior_r3_path", "prior_r2_path", "prior_r1_path", "mtm_1st_mean"]
    tiers.append(("T4: +priors", t4_cols))

    # T5: add quarterly forward if available
    if q in Q_SIGNALS:
        qc = Q_SIGNALS[q]
        t5_cols = t4_cols + [qc]
        tiers.append((f"T5: +{qc}", t5_cols))

    return tiers


# ── Main ────────────────────────────────────────────────────────────────
print(f"Phase 2b: Signal Blending. Memory: {mem_mb():.0f} MB")
print("=" * 110)

# Lambda grid for ridge
LAMBDA_GRID = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

cross_quarter_summary = []

for q in QUARTERS:
    print(f"\n{'='*110}")
    print(f"  QUARTER: {q.upper()}")
    print(f"{'='*110}")

    # Load all data
    df = pl.read_parquet(f"{DATA_DIR}/{q}_all_baselines.parquet")
    df = df.filter(pl.col("planning_year") >= 2020)
    n_total = len(df)
    print(f"  Loaded {n_total:,} rows.  Memory: {mem_mb():.0f} MB")

    # Add path key for prior-year residual
    df = df.with_columns(
        (pl.col("source_id") + "," + pl.col("sink_id")).alias("path")
    )

    # Build prior-year residual
    df_with_f0 = df.filter(pl.col("nodal_f0").is_not_null())
    df_with_f0 = df_with_f0.with_columns(
        (pl.col("mcp_mean") - pl.col("nodal_f0")).alias("residual")
    )
    prior = df_with_f0.select([
        pl.col("planning_year").alias("prior_py"),
        "path", "class_type",
        pl.col("residual").alias("prior_residual"),
    ])
    df = df.with_columns(
        (pl.col("planning_year") - 1).alias("prior_py")
    )
    df = df.join(
        prior, left_on=["prior_py", "path", "class_type"],
        right_on=["prior_py", "path", "class_type"], how="left",
    ).drop("prior_py")
    del df_with_f0, prior
    gc.collect()

    tiers = get_tiers(q)
    q_summary = {"quarter": q, "tiers": {}}

    # ── Section A: Per-tier head-to-head (overall, no LOO) ──
    # This uses ALL data to show the potential
    print(f"\n  --- Section A: Tier Coverage and Signal Availability ---")
    for tier_name, sig_cols in tiers:
        mask = pl.all_horizontal([pl.col(c).is_not_null() for c in sig_cols])
        mask = mask & pl.col("mcp_mean").is_not_null()
        n_tier = df.filter(mask).height
        print(f"  {tier_name:<30}: {n_tier:>8,} rows ({n_tier/n_total*100:5.1f}%)")

    # ── Section B: LOO Ridge blending per tier ──
    print(f"\n  --- Section B: LOO Ridge Regression per Tier ---")

    for tier_name, sig_cols in tiers:
        # Filter to rows with all signals in this tier + mcp + nodal_f0
        required = sig_cols + ["mcp_mean", "nodal_f0"]
        mask = pl.all_horizontal([pl.col(c).is_not_null() for c in required])
        tier_df = df.filter(mask)
        n_tier = len(tier_df)

        if n_tier < 1000:
            print(f"\n  {tier_name}: Too few rows ({n_tier}), skipping")
            continue

        loo_rows = []
        for test_py in PYS:
            test = tier_df.filter(pl.col("planning_year") == test_py)
            n_test = len(test)
            if n_test < 100:
                continue
            train = tier_df.filter(pl.col("planning_year") != test_py)
            if len(train) < 500:
                continue

            # Build feature matrices
            X_train = build_X(train, sig_cols)
            y_train = train["mcp_mean"].to_numpy()
            X_test = build_X(test, sig_cols)
            y_test = test["mcp_mean"].to_numpy()

            # LOO lambda selection: use a simple split within training
            best_lam, best_val_mae = 1.0, np.inf
            train_pys = [py for py in PYS if py != test_py]
            if len(train_pys) >= 3:
                # Inner CV: leave one train-PY out
                for lam in LAMBDA_GRID:
                    val_maes = []
                    for val_py in train_pys:
                        inner_train = tier_df.filter(
                            (pl.col("planning_year") != test_py) &
                            (pl.col("planning_year") != val_py)
                        )
                        inner_val = tier_df.filter(pl.col("planning_year") == val_py)
                        if len(inner_train) < 500 or len(inner_val) < 100:
                            continue
                        X_it = build_X(inner_train, sig_cols)
                        y_it = inner_train["mcp_mean"].to_numpy()
                        X_iv = build_X(inner_val, sig_cols)
                        y_iv = inner_val["mcp_mean"].to_numpy()
                        w = ridge_fit(X_it, y_it, lam)
                        pred = ridge_predict(X_iv, w)
                        val_maes.append(float(np.abs(y_iv - pred).mean()))
                    if val_maes:
                        avg_val = np.mean(val_maes)
                        if avg_val < best_val_mae:
                            best_val_mae = avg_val
                            best_lam = lam
            else:
                best_lam = 10.0  # safe default with few PYs

            # Fit on full training set with best lambda
            w = ridge_fit(X_train, y_train, best_lam)
            pred_blend = ridge_predict(X_test, w)

            # Baselines for comparison
            f0_test = test["nodal_f0"].to_numpy()
            alpha = loo_alpha(tier_df, test_py)
            pred_alpha = f0_test * alpha

            raw_mae = float(np.abs(y_test - f0_test).mean())
            alpha_mae = float(np.abs(y_test - pred_alpha).mean())
            blend_mae = float(np.abs(y_test - pred_blend).mean())

            # Format weights
            wt_strs = [f"{sig_cols[i][:8]}={w[i]:+.3f}" for i in range(len(sig_cols))]
            wt_str = " ".join(wt_strs) + f" int={w[-1]:+.0f}"

            loo_rows.append({
                "py": test_py, "n": n_test,
                "raw_mae": raw_mae, "alpha_mae": alpha_mae, "blend_mae": blend_mae,
                "wts": wt_str, "alpha": alpha, "lam": best_lam,
            })

        if loo_rows:
            print_py_table(loo_rows, f"{tier_name} ({n_tier:,} rows, λ selected by inner CV)")
            q_summary["tiers"][tier_name] = loo_rows

    # ── Section C: Prior-year residual as additional feature ──
    # Test: nodal_f0 + H + prior_residual (for rows that have all three)
    print(f"\n  --- Section C: Adding Prior-Year Residual to T1 ---")
    t1r_cols = ["nodal_f0", "mtm_1st_mean", "prior_residual"]
    required = t1r_cols + ["mcp_mean"]
    mask = pl.all_horizontal([pl.col(c).is_not_null() for c in required])
    t1r_df = df.filter(mask)
    n_t1r = len(t1r_df)
    print(f"  nodal+H+prior_resid: {n_t1r:,} rows ({n_t1r/n_total*100:.1f}%)")

    if n_t1r > 1000:
        loo_rows_t1r = []
        for test_py in PYS:
            test = t1r_df.filter(pl.col("planning_year") == test_py)
            n_test = len(test)
            if n_test < 100:
                continue
            train = t1r_df.filter(pl.col("planning_year") != test_py)
            if len(train) < 500:
                continue

            X_train = build_X(train, t1r_cols)
            y_train = train["mcp_mean"].to_numpy()
            X_test = build_X(test, t1r_cols)
            y_test = test["mcp_mean"].to_numpy()

            # Simple lambda = 10
            w = ridge_fit(X_train, y_train, 10.0)
            pred_blend = ridge_predict(X_test, w)

            f0_test = test["nodal_f0"].to_numpy()
            alpha = loo_alpha(t1r_df, test_py)

            raw_mae = float(np.abs(y_test - f0_test).mean())
            alpha_mae = float(np.abs(y_test - f0_test * alpha).mean())
            blend_mae = float(np.abs(y_test - pred_blend).mean())

            wt_strs = [f"{t1r_cols[i][:8]}={w[i]:+.3f}" for i in range(len(t1r_cols))]
            wt_str = " ".join(wt_strs) + f" int={w[-1]:+.0f}"

            loo_rows_t1r.append({
                "py": test_py, "n": n_test,
                "raw_mae": raw_mae, "alpha_mae": alpha_mae, "blend_mae": blend_mae,
                "wts": wt_str, "alpha": alpha, "lam": 10.0,
            })

        if loo_rows_t1r:
            print_py_table(loo_rows_t1r, f"T1+resid: nodal+H+prior_residual ({n_t1r:,} rows)")

    # ── Section D: Inverse-MAE weighted average (no fitting) ──
    print(f"\n  --- Section D: Inverse-MAE Weighted Average ---")
    # On T2 rows: weight each signal by 1/MAE from training
    t2_cols = ["nodal_f0", "f0_path_corr", "mtm_1st_mean"]
    required = t2_cols + ["mcp_mean"]
    mask = pl.all_horizontal([pl.col(c).is_not_null() for c in required])
    t2_df = df.filter(mask)
    n_t2 = len(t2_df)

    if n_t2 > 1000:
        inv_rows = []
        for test_py in PYS:
            test = t2_df.filter(pl.col("planning_year") == test_py)
            n_test = len(test)
            if n_test < 100:
                continue
            train = t2_df.filter(pl.col("planning_year") != test_py)

            y_test = test["mcp_mean"].to_numpy()
            f0_test = test["nodal_f0"].to_numpy()

            # Compute MAE of each signal on training set
            mcp_tr = train["mcp_mean"].to_numpy()
            maes = {}
            for c in t2_cols:
                arr = train[c].to_numpy().astype(float)
                maes[c] = float(np.abs(mcp_tr - arr).mean())

            # Inverse-MAE weights
            inv_total = sum(1/v for v in maes.values())
            weights = {c: (1/maes[c]) / inv_total for c in t2_cols}

            # Predict
            pred = np.zeros(n_test)
            for c in t2_cols:
                pred += weights[c] * test[c].to_numpy().astype(float)

            blend_mae = float(np.abs(y_test - pred).mean())
            raw_mae = float(np.abs(y_test - f0_test).mean())
            alpha = loo_alpha(t2_df, test_py)
            alpha_mae = float(np.abs(y_test - f0_test * alpha).mean())

            wt_str = " ".join(f"{c[:8]}={weights[c]:.3f}" for c in t2_cols)

            inv_rows.append({
                "py": test_py, "n": n_test,
                "raw_mae": raw_mae, "alpha_mae": alpha_mae, "blend_mae": blend_mae,
                "wts": wt_str, "alpha": alpha, "lam": 0,
            })

        if inv_rows:
            print_py_table(inv_rows, f"Inv-MAE weighted (T2: nodal+f0path+H, {n_t2:,} rows)")

    # ── Section E: Head-to-head summary on fully-matched rows ──
    # Compare all approaches on the SAME rows (all core signals present)
    print(f"\n  --- Section E: Head-to-Head on Fully Matched Rows ---")
    all_sigs = CORE_SIGNALS.copy()
    if q in Q_SIGNALS:
        all_sigs.append(Q_SIGNALS[q])
    required = all_sigs + ["mcp_mean", "prior_residual"]
    mask_full = pl.all_horizontal([pl.col(c).is_not_null() for c in required])
    full_df = df.filter(mask_full)
    n_full = len(full_df)

    if n_full < 1000:
        # Fallback: without prior_residual
        required = all_sigs + ["mcp_mean"]
        mask_full = pl.all_horizontal([pl.col(c).is_not_null() for c in required])
        full_df = df.filter(mask_full)
        n_full = len(full_df)
        has_resid = False
    else:
        has_resid = True

    print(f"  Fully matched: {n_full:,} rows ({n_full/n_total*100:.1f}%)")

    if n_full > 500:
        hh_results = []
        # Evaluate each method across all PYs combined
        mcp_full = full_df["mcp_mean"].to_numpy()
        f0_full = full_df["nodal_f0"].to_numpy()
        h_full = full_df["mtm_1st_mean"].to_numpy().astype(float)

        # 1. Raw nodal_f0
        s = compute_stats(mcp_full, f0_full)
        s["label"] = "Nodal f0 (raw)"
        hh_results.append(s)

        # 2. Alpha-scaled (median alpha from LOO)
        alphas = []
        for test_py in PYS:
            if full_df.filter(pl.col("planning_year") == test_py).height > 0:
                alphas.append(loo_alpha(full_df, test_py))
        med_alpha = float(np.median(alphas)) if alphas else 1.55
        s = compute_stats(mcp_full, f0_full * med_alpha)
        s["label"] = f"α={med_alpha:.2f} nodal_f0"
        hh_results.append(s)

        # 3. H baseline
        s = compute_stats(mcp_full, h_full)
        s["label"] = "H (DA congestion)"
        hh_results.append(s)

        # 4. Each individual signal
        for sig in all_sigs:
            if sig in ("nodal_f0", "mtm_1st_mean"):
                continue  # already shown
            arr = full_df[sig].to_numpy().astype(float)
            s = compute_stats(mcp_full, arr)
            s["label"] = sig
            hh_results.append(s)

        # 5. LOO Ridge blend (all signals) — aggregate across PYs
        blend_preds = np.full(n_full, np.nan)
        py_col = full_df["planning_year"].to_numpy()
        for test_py in PYS:
            test_mask = py_col == test_py
            if test_mask.sum() < 50:
                continue
            train_mask = ~test_mask
            X_train = build_X(full_df.filter(pl.col("planning_year") != test_py), all_sigs)
            y_train = full_df.filter(pl.col("planning_year") != test_py)["mcp_mean"].to_numpy()
            X_test = build_X(full_df.filter(pl.col("planning_year") == test_py), all_sigs)
            w = ridge_fit(X_train, y_train, 10.0)
            blend_preds[test_mask] = ridge_predict(X_test, w)

        s = compute_stats(mcp_full, blend_preds)
        s["label"] = "Ridge blend (all signals)"
        hh_results.append(s)

        # 6. If prior_residual available, add it as feature
        if has_resid:
            blend_preds2 = np.full(n_full, np.nan)
            sigs_plus = all_sigs + ["prior_residual"]
            for test_py in PYS:
                test_mask = py_col == test_py
                if test_mask.sum() < 50:
                    continue
                X_train = build_X(full_df.filter(pl.col("planning_year") != test_py), sigs_plus)
                y_train = full_df.filter(pl.col("planning_year") != test_py)["mcp_mean"].to_numpy()
                X_test = build_X(full_df.filter(pl.col("planning_year") == test_py), sigs_plus)
                w = ridge_fit(X_train, y_train, 10.0)
                blend_preds2[test_mask] = ridge_predict(X_test, w)

            s = compute_stats(mcp_full, blend_preds2)
            s["label"] = "Ridge blend + prior_resid"
            hh_results.append(s)

        # 7. Inverse-MAE weighted
        mcp_tr_all = mcp_full  # Use all data for weights (slight cheat, but for overview)
        inv_preds = np.zeros(n_full)
        inv_total_w = 0
        for sig in all_sigs:
            arr = full_df[sig].to_numpy().astype(float)
            mae_s = float(np.abs(mcp_full - arr).mean())
            if mae_s > 0:
                w_s = 1.0 / mae_s
                inv_preds += w_s * arr
                inv_total_w += w_s
        inv_preds /= inv_total_w
        s = compute_stats(mcp_full, inv_preds)
        s["label"] = "Inv-MAE weighted avg"
        hh_results.append(s)

        print_blend_table(hh_results, f"Head-to-Head on {n_full:,} fully matched rows ({q})")

    # Store for cross-quarter summary
    cross_quarter_summary.append(q_summary)

    del df
    gc.collect()

# ── Cross-Quarter Summary ──
print(f"\n{'='*110}")
print("  CROSS-QUARTER SUMMARY: Best Approach per Tier")
print(f"{'='*110}")

for q_data in cross_quarter_summary:
    q = q_data["quarter"]
    tiers = q_data["tiers"]
    print(f"\n  {q.upper()}")
    for tier_name, rows in tiers.items():
        if not rows:
            continue
        avg_raw = np.mean([r["raw_mae"] for r in rows])
        avg_alpha = np.mean([r["alpha_mae"] for r in rows])
        avg_blend = np.mean([r["blend_mae"] for r in rows])
        vs_raw = (1 - avg_blend / avg_raw) * 100
        vs_alpha = (1 - avg_blend / avg_alpha) * 100
        n_avg = int(np.mean([r["n"] for r in rows]))
        print(
            f"    {tier_name:<30}: MAE raw={avg_raw:.0f}, α={avg_alpha:.0f}, "
            f"blend={avg_blend:.0f} ({vs_raw:+.1f}% vs raw, {vs_alpha:+.1f}% vs α)"
        )

print(f"\nDone. Memory: {mem_mb():.0f} MB")
