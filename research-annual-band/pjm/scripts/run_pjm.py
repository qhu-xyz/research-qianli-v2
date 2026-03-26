"""PJM Annual: Frozen bid price pipeline for all 4 rounds.

Frozen spec (2026-03-21):
  R1 baseline: w(flow,bin) × mtm_1st_period + (1-w) × recent_1
    recent_1 = March N DA congestion (sink-source) × 12, annual scale
    w estimated per (flow_type, bin) on training, minimizing MAE
  R2-R4 baseline: mtm_1st_period (pure, no blend)
  Calibration: (bin, flow_type, class_type) for all rounds
  Bins: 5 quantile bins on training |baseline|
  CV: expanding window, min 2 training PYs
  Cap: ±50,000 annual $ on bid prices

Pxx definition:
  Pxx = single bid price at which there is an xx% chance of clearing on training data.
  P20 = baseline + quantile(residual, 0.20) → 20% clearing chance.
  Each level is independent. No pairs. No band width.

Usage:
    cd /home/xyz/workspace/pmodel && source .venv/bin/activate
    python /home/xyz/workspace/research-qianli-v2/research-annual-band/pjm/scripts/run_pjm.py
"""

from __future__ import annotations
import sys, time, resource, os
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "miso" / "archive_v1" / "scripts"))
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from run_v9_bands import compute_quantile_boundaries, assign_bins, N_BINS

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# ── Config ──
MIN_BLEND = 100
MIN_BAND = 200
BID_CAP = 50_000  # annual $
W_GRID = np.arange(0, 1.01, 0.05)
BL_LABELS = [f"q{i+1}" for i in range(N_BINS)]
CP_LEVELS = [0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95]
HOLDOUT_PYS = [2023, 2024, 2025]
CLASSES = ["onpeak", "dailyoffpeak", "wkndonpeak"]

# R1 uses blend with recent_1; R2-R4 use pure baseline
BLEND_ROUNDS = {1}


def mem_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def actual_clear_rate(mcp: np.ndarray, bid: np.ndarray) -> float:
    if len(mcp) == 0:
        return float("nan")
    return float(np.mean(mcp <= bid) * 100)


def load_data():
    """Load PJM annual paths + recent_1 revenue feature."""
    df = pl.read_parquet(DATA_DIR / "canonical_annual_paths.parquet")

    # Load recent_1 (pre-computed)
    rev_path = DATA_DIR / "pjm_recent1.parquet"
    if rev_path.exists():
        rev = pl.read_parquet(rev_path)
        df = df.join(
            rev.select(["source_id", "sink_id", "class_type", "planning_year", "round", "recent_1"]),
            on=["source_id", "sink_id", "class_type", "planning_year", "round"],
            how="left",
        )
    else:
        df = df.with_columns(pl.lit(None).cast(pl.Float64).alias("recent_1"))
        print(f"WARNING: {rev_path} not found. R1 blend will fall back to pure baseline.")

    # Filter clean
    df = df.filter(
        pl.col("mcp").is_not_null() & pl.col("mcp").is_not_nan()
        & pl.col("mtm_1st_period").is_not_null() & pl.col("mtm_1st_period").is_not_nan()
    )
    return df


def run_round(rnd, mcp, baseline, rev1, pys, cls_arr):
    """Run pipeline for one round. Returns bids, bin_arr, has_result, bl_used."""
    N = len(mcp)
    use_blend = rnd in BLEND_ROUNDS and not np.all(np.isnan(rev1))

    bl_w = np.clip(baseline, *np.percentile(baseline, [1, 99]))
    if use_blend:
        rv_w = np.clip(rev1, *np.percentile(rev1[np.isfinite(rev1)], [1, 99]))
    flow = np.where(bl_w > 0, "prevail", "counter")

    bids = {lv: np.zeros(N) for lv in CP_LEVELS}
    bin_arr = np.empty(N, dtype="U4")
    has_result = np.zeros(N, dtype=bool)
    bl_used = np.zeros(N)

    for tpy in range(2019, 2026):
        trm = pys < tpy
        tem = pys == tpy
        if trm.sum() == 0 or tem.sum() == 0:
            continue
        if len(np.unique(pys[trm])) < 2:
            continue

        boundaries, _ = compute_quantile_boundaries(
            pl.Series("x", np.abs(bl_w[trm])), N_BINS
        )
        trb = assign_bins(pl.Series("x", np.abs(bl_w[trm])), boundaries, BL_LABELS).to_numpy()
        teb = assign_bins(pl.Series("x", np.abs(bl_w[tem])), boundaries, BL_LABELS).to_numpy()
        trf = np.where(bl_w[trm] > 0, "prevail", "counter")
        tef = np.where(bl_w[tem] > 0, "prevail", "counter")
        trc = cls_arr[trm]
        tec = cls_arr[tem]
        te_idx = np.where(tem)[0]
        bin_arr[te_idx] = teb

        # Baseline
        if use_blend:
            cell_w = {}
            for ft in ["prevail", "counter"]:
                for b in BL_LABELS:
                    ct = (trf == ft) & (trb == b)
                    if ct.sum() < MIN_BLEND:
                        cell_w[(ft, b)] = 1.0
                    else:
                        best_w = 1.0
                        best_mae = np.abs(mcp[trm][ct] - bl_w[trm][ct]).mean()
                        for wc in W_GRID:
                            bl_cand = wc * bl_w[trm][ct] + (1 - wc) * rv_w[trm][ct]
                            mae_c = np.abs(mcp[trm][ct] - bl_cand).mean()
                            if mae_c < best_mae:
                                best_mae = mae_c
                                best_w = wc
                        cell_w[(ft, b)] = best_w
            tra = bl_w[trm].copy()
            tea = bl_w[tem].copy()
            for ft in ["prevail", "counter"]:
                for b in BL_LABELS:
                    w = cell_w[(ft, b)]
                    ct = (trf == ft) & (trb == b)
                    ce = (tef == ft) & (teb == b)
                    tra[ct] = w * bl_w[trm][ct] + (1 - w) * rv_w[trm][ct]
                    tea[ce] = w * bl_w[tem][ce] + (1 - w) * rv_w[tem][ce]
        else:
            tra = bl_w[trm]
            tea = bl_w[tem]

        bl_used[te_idx] = tea
        train_res = mcp[trm] - tra

        # Calibrate: (bin, flow, class) with explicit fallback tracking
        n_fallbacks = 0
        n_cells = 0
        for ft in ["prevail", "counter"]:
            for b in BL_LABELS:
                for c in CLASSES:
                    mk_tr = (trf == ft) & (trb == b) & (trc == c)
                    mk_te = (tef == ft) & (teb == b) & (tec == c)
                    if mk_te.sum() == 0:
                        continue
                    n_cells += 1
                    if mk_tr.sum() < MIN_BAND:
                        mk_tr = (trf == ft) & (trb == b)  # fallback: pool class
                        n_fallbacks += 1
                    if mk_tr.sum() == 0:
                        continue
                    ce_idx = te_idx[mk_te]
                    bl_test = tea[mk_te]
                    for lv in CP_LEVELS:
                        q = float(np.quantile(train_res[mk_tr], lv))
                        bids[lv][ce_idx] = np.clip(bl_test + q, -BID_CAP, BID_CAP)
                    has_result[ce_idx] = True
        if n_fallbacks > 0:
            rate = n_fallbacks / max(n_cells, 1) * 100
            print(f"    PY{tpy}: {n_fallbacks}/{n_cells} cells fell back to (bin,flow) ({rate:.0f}%)")
            if rate > 20:
                print(f"    WARNING: fallback rate {rate:.0f}% exceeds 20% threshold")

    return bids, bin_arr, has_result, bl_used, flow


def audit(rnd, mcp, bids, bin_arr, has_result, bl_used, flow_arr, pys, cls_arr):
    """Print comprehensive audit."""
    res = mcp - bl_used
    hol = has_result & np.isin(pys, HOLDOUT_PYS)

    print(f"\n{'='*130}")
    print(f"  R{rnd}: {hol.sum():,} holdout paths")
    print(f"{'='*130}")

    # Overall
    print(f"\n  Overall holdout:")
    for lv in CP_LEVELS:
        act = actual_clear_rate(mcp[hol], bids[lv][hol])
        print(f"    P{int(lv*100):02d}: {act:.1f}% (target {lv*100:.0f}%)")

    # By (PY, flow, bin)
    print(f"\n  Holdout cells (PY, flow, bin):")
    print(f"  {'PY':>4} {'Flow':>8} {'Bin':>4} {'N':>6} | {'bias':>7} | "
          f"{'P10':>6} {'P20':>6} {'P50':>6} {'P90':>6} {'P95':>6}")
    for py in HOLDOUT_PYS:
        for ft in ["prevail", "counter"]:
            for b in BL_LABELS:
                m = has_result & (pys == py) & (flow_arr == ft) & (bin_arr == b)
                if m.sum() < 50:
                    continue
                mc = mcp[m]
                bias = res[m].mean()
                vals = [f"{actual_clear_rate(mc, bids[lv][m]):>5.1f}%" for lv in [0.10, 0.20, 0.50, 0.90, 0.95]]
                flag = ""
                for lv in [0.10, 0.20, 0.50, 0.90, 0.95]:
                    if abs(actual_clear_rate(mc, bids[lv][m]) - lv * 100) > 10:
                        flag = " ***"
                        break
                print(f"  {py:>4} {ft:>8} {b:>4} {m.sum():>6} | {bias:>+7,.0f} | {' '.join(vals)}{flag}")
        print()

    # Finest grain: q4+q5 holdout
    print(f"  Finest grain: q4+q5 holdout (PY, flow, class, bin):")
    print(f"  {'PY':>4} {'Flow':>8} {'Cls':>8} {'Bn':>3} {'N':>5} | {'bias':>6} | "
          f"{'P10':>6} {'P20':>6} {'P50':>6} {'P90':>6} {'P95':>6}")
    for py in HOLDOUT_PYS:
        for ft in ["prevail", "counter"]:
            for c in CLASSES:
                for b in ["q4", "q5"]:
                    m = has_result & (pys == py) & (flow_arr == ft) & (cls_arr == c) & (bin_arr == b)
                    if m.sum() < 30:
                        continue
                    mc = mcp[m]
                    bias = res[m].mean()
                    vals = [f"{actual_clear_rate(mc, bids[lv][m]):>5.1f}%" for lv in [0.10, 0.20, 0.50, 0.90, 0.95]]
                    flag = ""
                    for lv in [0.10, 0.20, 0.50, 0.90, 0.95]:
                        if abs(actual_clear_rate(mc, bids[lv][m]) - lv * 100) > 10:
                            flag = " ***"
                            break
                    print(f"  {py:>4} {ft:>8} {c[:8]:>8} {b:>3} {m.sum():>5} | "
                          f"{bias:>+5,.0f} | {' '.join(vals)}{flag}")
        print()

    # Worst 8 per level
    for lv in [0.10, 0.20, 0.50, 0.95]:
        tgt = lv * 100
        label = f"P{int(tgt):02d}"
        cells = []
        for py in range(2019, 2026):
            for ft in ["prevail", "counter"]:
                for b in BL_LABELS:
                    m = has_result & (pys == py) & (flow_arr == ft) & (bin_arr == b)
                    if m.sum() < 100:
                        continue
                    act = actual_clear_rate(mcp[m], bids[lv][m])
                    bias = res[m].mean()
                    cells.append((py, ft, b, m.sum(), act, act - tgt, bias))
        cells.sort(key=lambda x: -abs(x[5]))
        print(f"  Worst 5 at {label} (target {tgt:.0f}%):")
        print(f"  {'PY':>4} {'Flow':>8} {'Bin':>4} {'N':>6} | {label:>6} {'miss':>7} | {'bias':>7}")
        for c in cells[:5]:
            py, ft, b, n, act, miss, bias = c
            tag = " DEV" if py < 2023 else ""
            print(f"  {py:>4} {ft:>8} {b:>4} {n:>6} | {act:>5.1f}% {miss:>+5.1f}pp | {bias:>+7,.0f}{tag}")
        print()


def main():
    t0 = time.time()
    print(f"PJM Annual: Frozen bid price pipeline")
    print(f"  R1: blend_r1 + (bin,flow,class)")
    print(f"  R2-R4: pure mtm_1st_period + (bin,flow,class)")
    print(f"  mem={mem_mb():.0f}MB")

    df = load_data()
    print(f"Loaded {df.height:,} paths, mem={mem_mb():.0f}MB")

    for rnd in [1, 2, 3, 4]:
        r = df.filter(pl.col("round") == rnd)

        mcp = r["mcp"].to_numpy().astype(np.float64)
        baseline = r["mtm_1st_period"].to_numpy().astype(np.float64)
        rev1_col = r["recent_1"] if "recent_1" in r.columns else pl.Series("recent_1", [np.nan] * r.height)
        rev1 = rev1_col.to_numpy().astype(np.float64)
        pys = r["planning_year"].to_numpy()
        cls_arr = r["class_type"].to_numpy()

        valid = np.isfinite(mcp) & np.isfinite(baseline)
        if rnd in BLEND_ROUNDS:
            valid &= np.isfinite(rev1)
        mcp = mcp[valid]
        baseline = baseline[valid]
        rev1 = rev1[valid]
        pys = pys[valid]
        cls_arr = cls_arr[valid]

        print(f"\n{'#'*130}")
        print(f"  ROUND {rnd}: {len(mcp):,} paths")
        print(f"{'#'*130}")

        bids, bin_arr, has_result, bl_used, flow_arr = run_round(
            rnd, mcp, baseline, rev1, pys, cls_arr
        )
        audit(rnd, mcp, bids, bin_arr, has_result, bl_used, flow_arr, pys, cls_arr)

    elapsed = time.time() - t0
    print(f"\nDone. elapsed={elapsed:.0f}s, mem={mem_mb():.0f}MB")


if __name__ == "__main__":
    main()
