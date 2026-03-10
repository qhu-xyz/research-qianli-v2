"""Phase 2c: Intuitive signal blending — de-biased average, median, best-available cascade.

The problem with raw blending: signals have different biases (f0: +220, H: +425).
Convex combination can't fix that. Ridge "fixes" it by overfitting (scaling to 1.9×).

Solution: de-bias each signal first (subtract LOO mean error), then combine.

Approaches tested:
  1. De-biased simple average (equal weight)
  2. De-biased median (robust to outliers)
  3. De-biased inverse-MAE weighted
  4. Best-available cascade (use highest-quality available signal per row)
  5. α-scaled nodal_f0 (benchmark from Phase 2a)

All use LOO by PY: bias estimated from training PYs, evaluated on held-out PY.
"""

import gc
import numpy as np
import polars as pl
import resource

def mem_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

DATA_DIR = "/opt/temp/qianli/annual_research/crossproduct_work"
PYS = [2020, 2021, 2022, 2023, 2024, 2025]

# Signals in priority order (best standalone MAE to worst)
CORE_SIGNALS = ["nodal_f0", "f0_path_corr", "f1_path",
                "prior_r3_path", "prior_r2_path", "prior_r1_path", "mtm_1st_mean"]
Q_SIGNALS = {"aq2": "q2_path", "aq3": "q3_path", "aq4": "q4_path"}
ALPHA_GRID = np.arange(1.0, 1.65, 0.05)


def compute_stats(mcp, pred):
    valid = ~np.isnan(pred) & ~np.isnan(mcp)
    n = int(valid.sum())
    if n == 0:
        return {"n": 0, "mae": np.inf, "bias": 0, "medae": 0, "p95": 0, "dir_pct": 0}
    m, p = mcp[valid], pred[valid]
    ae = np.abs(m - p)
    dm = (m != 0) & (p != 0)
    dir_pct = float(np.mean(np.sign(m[dm]) == np.sign(p[dm]))) * 100 if dm.sum() > 0 else 0
    return {
        "n": n, "mae": float(ae.mean()), "bias": float((m - p).mean()),
        "medae": float(np.median(ae)), "p95": float(np.percentile(ae, 95)), "dir_pct": dir_pct,
    }


def print_table(rows, title):
    print(f"\n  {title}")
    hdr = (f"  {'Method':<40} | {'n':>7} | {'Bias':>7} | {'MAE':>6} "
           f"| {'MedAE':>6} | {'p95':>7} | {'Dir%':>5} | {'vs α':>6}")
    print(hdr)
    print(f"  {'-'*40}-+-{'-'*7}-+-{'-'*7}-+-{'-'*6}-+-{'-'*6}-+-{'-'*7}-+-{'-'*5}-+-{'-'*6}")
    alpha_mae = next((r["mae"] for r in rows if "α-scaled" in r["label"]), rows[0]["mae"])
    for r in rows:
        vs = (1 - r["mae"] / alpha_mae) * 100 if alpha_mae > 0 else 0
        vs_str = f"{vs:>+5.1f}%" if "α-scaled" not in r["label"] else "  base"
        print(f"  {r['label']:<40} | {r['n']:>7,} | {r['bias']:>+7.0f} | {r['mae']:>6.0f} "
              f"| {r['medae']:>6.0f} | {r['p95']:>7.0f} | {r['dir_pct']:>5.1f} | {vs_str}")


print(f"Phase 2c: Intuitive Blending. Memory: {mem_mb():.0f} MB")
print("=" * 110)

for q in ["aq1", "aq2", "aq3", "aq4"]:
    print(f"\n{'='*110}")
    print(f"  {q.upper()}")
    print(f"{'='*110}")

    df = pl.read_parquet(f"{DATA_DIR}/{q}_all_baselines.parquet")
    df = df.filter(pl.col("planning_year") >= 2020)
    n_total = len(df)

    # Get all signal columns for this quarter
    all_sigs = [s for s in CORE_SIGNALS if s in df.columns]
    if q in Q_SIGNALS and Q_SIGNALS[q] in df.columns:
        all_sigs.append(Q_SIGNALS[q])

    # Convert to numpy for speed
    mcp = df["mcp_mean"].to_numpy()
    py_arr = df["planning_year"].to_numpy()
    sig_arrays = {}
    for s in all_sigs:
        sig_arrays[s] = df[s].to_numpy().astype(float)

    # ── Approach 0: α-scaled nodal_f0 (benchmark) ──
    # ── Approach 1: De-biased simple average ──
    # ── Approach 2: De-biased median ──
    # ── Approach 3: De-biased inv-MAE weighted ──
    # ── Approach 4: Best-available cascade ──

    # For each tier of signal availability, run LOO
    # Define tiers by which signals are present per row
    f0 = sig_arrays["nodal_f0"]
    has_f0 = ~np.isnan(f0)

    # ── Run on ALL rows with nodal_f0 ──
    # For each row, collect available de-biased signals; average/median them
    mask_all = has_f0 & ~np.isnan(mcp)
    idx_all = np.where(mask_all)[0]
    n_eval = len(idx_all)

    # Pre-allocate prediction arrays
    pred_alpha = np.full(n_total, np.nan)
    pred_debias_avg = np.full(n_total, np.nan)
    pred_debias_med = np.full(n_total, np.nan)
    pred_debias_wmae = np.full(n_total, np.nan)
    pred_cascade = np.full(n_total, np.nan)
    pred_debias_avg_noh = np.full(n_total, np.nan)
    pred_debias_med_noh = np.full(n_total, np.nan)

    # Cascade priority: f1 > f0_path > prior_r3 > q_forward > nodal_f0
    cascade_order = ["f1_path", "f0_path_corr", "prior_r3_path"]
    if q in Q_SIGNALS and Q_SIGNALS[q] in sig_arrays:
        cascade_order.append(Q_SIGNALS[q])
    cascade_order.append("nodal_f0")

    for test_py in PYS:
        test_mask = (py_arr == test_py) & mask_all
        train_mask = (py_arr != test_py) & mask_all
        n_test = test_mask.sum()
        if n_test < 100:
            continue

        # 1. α-scaled benchmark
        best_a, best_mae = 1.0, np.inf
        for a in ALPHA_GRID:
            mae = float(np.abs(mcp[train_mask] - f0[train_mask] * a).mean())
            if mae < best_mae:
                best_mae, best_a = mae, a
        pred_alpha[test_mask] = f0[test_mask] * best_a

        # 2-3. De-bias each signal on training set
        biases = {}
        maes = {}
        for s in all_sigs:
            arr = sig_arrays[s]
            valid_tr = train_mask & ~np.isnan(arr)
            if valid_tr.sum() < 500:
                continue
            biases[s] = float((mcp[valid_tr] - arr[valid_tr]).mean())
            maes[s] = float(np.abs(mcp[valid_tr] - (arr[valid_tr] + biases[s])).mean())

        # For each test row, collect de-biased predictions from available signals
        test_idx = np.where(test_mask)[0]
        for i in test_idx:
            available = []
            available_noh = []
            available_wmae = []
            for s in all_sigs:
                if s not in biases:
                    continue
                v = sig_arrays[s][i]
                if np.isnan(v):
                    continue
                corrected = v + biases[s]
                available.append(corrected)
                if s != "mtm_1st_mean":
                    available_noh.append(corrected)
                available_wmae.append((corrected, 1.0 / maes[s] if maes[s] > 0 else 0))

            if available:
                pred_debias_avg[i] = np.mean(available)
                pred_debias_med[i] = np.median(available)
            if available_noh:
                pred_debias_avg_noh[i] = np.mean(available_noh)
                pred_debias_med_noh[i] = np.median(available_noh)
            if available_wmae:
                vals, wts = zip(*available_wmae)
                wt_sum = sum(wts)
                if wt_sum > 0:
                    pred_debias_wmae[i] = sum(v * w for v, w in available_wmae) / wt_sum

        # 4. Best-available cascade (de-biased)
        for i in test_idx:
            for s in cascade_order:
                if s not in biases:
                    continue
                v = sig_arrays[s][i]
                if not np.isnan(v):
                    pred_cascade[i] = v + biases[s]
                    break

    # ── Evaluate on all rows with nodal_f0 ──
    m = mcp[mask_all]
    results = []

    s = compute_stats(m, pred_alpha[mask_all])
    s["label"] = "α-scaled nodal_f0"
    results.append(s)

    s = compute_stats(m, f0[mask_all])
    s["label"] = "Nodal f0 (raw)"
    results.append(s)

    s = compute_stats(m, pred_debias_avg[mask_all])
    s["label"] = "De-biased avg (all signals)"
    results.append(s)

    s = compute_stats(m, pred_debias_avg_noh[mask_all])
    s["label"] = "De-biased avg (excl H)"
    results.append(s)

    s = compute_stats(m, pred_debias_med[mask_all])
    s["label"] = "De-biased median (all signals)"
    results.append(s)

    s = compute_stats(m, pred_debias_med_noh[mask_all])
    s["label"] = "De-biased median (excl H)"
    results.append(s)

    s = compute_stats(m, pred_debias_wmae[mask_all])
    s["label"] = "De-biased inv-MAE weighted"
    results.append(s)

    s = compute_stats(m, pred_cascade[mask_all])
    s["label"] = "Best-available cascade (de-biased)"
    results.append(s)

    print_table(results, f"All rows with nodal_f0 ({n_eval:,} rows)")

    # ── Also evaluate on FULLY MATCHED rows for fair comparison ──
    full_mask = mask_all.copy()
    for s in all_sigs:
        full_mask &= ~np.isnan(sig_arrays[s])
    n_full = full_mask.sum()

    if n_full > 500:
        m_full = mcp[full_mask]
        results_full = []

        s = compute_stats(m_full, pred_alpha[full_mask])
        s["label"] = "α-scaled nodal_f0"
        results_full.append(s)

        s = compute_stats(m_full, f0[full_mask])
        s["label"] = "Nodal f0 (raw)"
        results_full.append(s)

        s = compute_stats(m_full, pred_debias_avg[full_mask])
        s["label"] = "De-biased avg (all signals)"
        results_full.append(s)

        s = compute_stats(m_full, pred_debias_avg_noh[full_mask])
        s["label"] = "De-biased avg (excl H)"
        results_full.append(s)

        s = compute_stats(m_full, pred_debias_med[full_mask])
        s["label"] = "De-biased median (all signals)"
        results_full.append(s)

        s = compute_stats(m_full, pred_debias_med_noh[full_mask])
        s["label"] = "De-biased median (excl H)"
        results_full.append(s)

        s = compute_stats(m_full, pred_debias_wmae[full_mask])
        s["label"] = "De-biased inv-MAE weighted"
        results_full.append(s)

        s = compute_stats(m_full, pred_cascade[full_mask])
        s["label"] = "Best-available cascade (de-biased)"
        results_full.append(s)

        # Also add each individual signal de-biased
        for sig in all_sigs:
            arr = sig_arrays[sig]
            # De-bias using overall LOO (approximate, for display only)
            pred_sig = np.full(n_total, np.nan)
            for test_py in PYS:
                train_m = (py_arr != test_py) & ~np.isnan(arr) & ~np.isnan(mcp)
                test_m = (py_arr == test_py) & full_mask
                if train_m.sum() < 500 or test_m.sum() < 50:
                    continue
                bias = float((mcp[train_m] - arr[train_m]).mean())
                pred_sig[test_m] = arr[test_m] + bias
            s = compute_stats(m_full, pred_sig[full_mask])
            s["label"] = f"  {sig} (de-biased)"
            results_full.append(s)

        print_table(results_full, f"Fully matched ({n_full:,} rows)")

    # ── Per-PY breakdown for best approaches ──
    print(f"\n  Per-PY breakdown (all rows with f0, {n_eval:,} rows):")
    print(f"  {'PY':>4} | {'n':>7} | {'α-scaled':>8} | {'Dbias avg':>9} | {'Dbias med':>9} "
          f"| {'Cascade':>8} | {'Dbias avg-H':>11} | {'Dbias med-H':>11}")
    print(f"  {'-'*4}-+-{'-'*7}-+-{'-'*8}-+-{'-'*9}-+-{'-'*9}"
          f"-+-{'-'*8}-+-{'-'*11}-+-{'-'*11}")
    for test_py in PYS:
        tm = (py_arr == test_py) & mask_all
        nt = tm.sum()
        if nt < 100:
            continue
        mt = mcp[tm]
        a_mae = float(np.abs(mt - pred_alpha[tm]).mean())
        da_mae = float(np.nanmean(np.abs(mt - pred_debias_avg[tm])))
        dm_mae = float(np.nanmean(np.abs(mt - pred_debias_med[tm])))
        c_mae = float(np.nanmean(np.abs(mt - pred_cascade[tm])))
        danh_mae = float(np.nanmean(np.abs(mt - pred_debias_avg_noh[tm])))
        dmnh_mae = float(np.nanmean(np.abs(mt - pred_debias_med_noh[tm])))
        print(f"  {test_py:>4} | {nt:>7,} | {a_mae:>8.0f} | {da_mae:>9.0f} | {dm_mae:>9.0f} "
              f"| {c_mae:>8.0f} | {danh_mae:>11.0f} | {dmnh_mae:>11.0f}")

    del df
    gc.collect()

print(f"\nDone. Memory: {mem_mb():.0f} MB")
