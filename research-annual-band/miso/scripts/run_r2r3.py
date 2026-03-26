"""MISO R2/R3: Pure MTM baseline + hybrid calibration cells.

Frozen spec (2026-03-20):
  Baseline: mtm_1st_period (= mtm_1st_mean × 3, quarterly). No blend.
  Banding:  single-sided quantile per calibration cell
  Cells:    q1-q3 by (bin, flow_type), q4-q5 by (bin) only. Class dropped.
  Bins:     5 quantile bins on training |baseline|
  CV:       equal-weight expanding window, min 2 training PYs
  Cap:      ±15,000 quarterly $ on bid prices

Pxx definition:
  Pxx = single bid price at which there is an xx% chance of clearing on training data.
  P20 = baseline + quantile(residual, 0.20) → 20% clearing chance (cheap bid).
  P95 = baseline + quantile(residual, 0.95) → 95% clearing chance (expensive bid).
  Each level is independent. No pairs. No band width.

Usage:
    cd /home/xyz/workspace/pmodel && source .venv/bin/activate
    python /home/xyz/workspace/research-qianli-v2/research-annual-band/miso/scripts/run_r2r3.py
"""

from __future__ import annotations
import sys, time, resource
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "archive_v1" / "scripts"))
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from run_v9_bands import compute_quantile_boundaries, assign_bins, N_BINS

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

# ── Config ──
MIN_BAND_CELL = 300
BID_CAP = 15_000
ALL_PYS = list(range(2020, 2026))
HOLDOUT_PYS = [2023, 2024, 2025]
BL_LABELS = [f"q{i+1}" for i in range(N_BINS)]

# Calibration cell definition:
#   q1-q3: (bin, flow_type) — flow split retained
#   q4-q5: (bin) only — flow pooled to reduce prevail/counter divergence at P10-P50
LOW_BINS = {"q1", "q2", "q3"}
HIGH_BINS = {"q4", "q5"}

# Pxx levels to compute
CP_LEVELS = [0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95]
CP_LABELS = [f"p{int(lv*100):02d}" for lv in CP_LEVELS]


def mem_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def actual_clear_rate(mcp: np.ndarray, bid: np.ndarray) -> float:
    if len(mcp) == 0:
        return float("nan")
    return float(np.mean(mcp <= bid) * 100)


def run_pipeline(mcp, baseline, pys, qtrs, cls_arr, test_pys):
    """Run the frozen R2/R3 pipeline. Returns per-path arrays."""
    N = len(mcp)
    flow = np.where(baseline > 0, "prevail", "counter")

    bids = {lv: np.zeros(N) for lv in CP_LEVELS}
    bin_arr = np.empty(N, dtype="U4")
    has_result = np.zeros(N, dtype=bool)

    for qtr in ["aq1", "aq2", "aq3", "aq4"]:
        qm = qtrs == qtr
        for tpy in test_pys:
            trm = qm & (pys < tpy)
            tem = qm & (pys == tpy)
            if trm.sum() == 0 or tem.sum() == 0:
                continue
            if len(np.unique(pys[trm])) < 2:
                continue

            # Step 1: bins from training |baseline|
            boundaries, _ = compute_quantile_boundaries(
                pl.Series("x", np.abs(baseline[trm])), N_BINS
            )
            trb = assign_bins(pl.Series("x", np.abs(baseline[trm])), boundaries, BL_LABELS).to_numpy()
            teb = assign_bins(pl.Series("x", np.abs(baseline[tem])), boundaries, BL_LABELS).to_numpy()
            trf = np.where(baseline[trm] > 0, "prevail", "counter")
            tef = np.where(baseline[tem] > 0, "prevail", "counter")
            te_idx = np.where(tem)[0]
            bin_arr[te_idx] = teb

            # Step 2: residuals
            train_res = mcp[trm] - baseline[trm]

            # Step 3: calibrate per cell (hybrid: q1-q3 by bin+flow, q4-q5 by bin)
            cal_keys = []
            for b in BL_LABELS:
                if b in LOW_BINS:
                    for ft in ["prevail", "counter"]:
                        cal_keys.append((ft, b))
                else:
                    cal_keys.append(("__pool__", b))

            for key in cal_keys:
                if key[0] == "__pool__":
                    b = key[1]
                    mk_tr = trb == b
                    mk_te = teb == b
                else:
                    ft, b = key
                    mk_tr = (trf == ft) & (trb == b)
                    mk_te = (tef == ft) & (teb == b)

                if mk_tr.sum() < MIN_BAND_CELL:
                    mk_tr = trb == (key[1] if len(key) == 2 else key[0])

                if mk_tr.sum() == 0 or mk_te.sum() == 0:
                    continue

                ce_idx = te_idx[mk_te]
                bl_test = baseline[ce_idx]
                res_cell = train_res[mk_tr]

                for lv in CP_LEVELS:
                    q = float(np.quantile(res_cell, lv))
                    bids[lv][ce_idx] = np.clip(bl_test + q, -BID_CAP, BID_CAP)
                has_result[ce_idx] = True

    return bids, bin_arr, has_result, flow


def audit(rnd, mcp, baseline, bids, bin_arr, has_result, flow_arr, pys, qtrs, cls_arr, label=""):
    """Print comprehensive audit tables at finest grain."""
    res = mcp - baseline
    hol = has_result & np.isin(pys, HOLDOUT_PYS)

    print(f"\n{'='*130}")
    print(f"  {label}: {hol.sum():,} holdout paths")
    print(f"{'='*130}")

    # Overall
    print(f"\n  Overall holdout:")
    print(f"  {'Level':>5} {'Target':>7} | {'Actual':>7} {'Miss':>7}")
    for lv in CP_LEVELS:
        tgt = lv * 100
        act = actual_clear_rate(mcp[hol], bids[lv][hol])
        print(f"  P{int(tgt):02d}  {tgt:>6.0f}% | {act:>6.1f}% {act-tgt:>+5.1f}pp")

    # By (PY, flow) — P10/P20/P50/P90/P95
    print(f"\n  By (PY, flow):")
    print(f"  {'PY':>4} {'Flow':>8} {'N':>7} | {'bias':>7} {'MAE':>6} | "
          f"{'P10':>6} {'P20':>6} {'P50':>6} {'P90':>6} {'P95':>6}")
    for py in sorted(np.unique(pys)):
        for ft in ["prevail", "counter"]:
            m = has_result & (pys == py) & (flow_arr == ft)
            if m.sum() < 100:
                continue
            mc = mcp[m]
            r_sub = res[m]
            bias = r_sub.mean()
            mae = np.abs(r_sub).mean()
            vals = [f"{actual_clear_rate(mc, bids[lv][m]):>5.1f}%" for lv in [0.10, 0.20, 0.50, 0.90, 0.95]]
            tag = " DEV" if py < 2023 else ""
            print(f"  {py:>4} {ft:>8} {m.sum():>7,} | {bias:>+7,.0f} {mae:>6,.0f} | {' '.join(vals)}{tag}")

    # Every holdout cell at (PY, flow, bin) — P10/P20/P50/P90/P95
    print(f"\n  Holdout cells (PY, flow, bin) — P10/P20/P50/P90/P95:")
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
                vals = []
                any_flag = False
                for lv in [0.10, 0.20, 0.50, 0.90, 0.95]:
                    act = actual_clear_rate(mc, bids[lv][m])
                    tgt = lv * 100
                    if abs(act - tgt) > 10:
                        any_flag = True
                    vals.append(f"{act:>5.1f}%")
                flag = " ***" if any_flag else ""
                print(f"  {py:>4} {ft:>8} {b:>4} {m.sum():>6} | {bias:>+7,.0f} | {' '.join(vals)}{flag}")
        print()

    # Worst 10 cells at P20
    print(f"  Worst 10 P20 cells (target 20%, by |miss|):")
    print(f"  {'PY':>4} {'Flow':>8} {'Bin':>4} {'N':>6} | {'P20':>6} {'miss':>7} | {'bias':>7}")
    cells = []
    for py in ALL_PYS:
        for ft in ["prevail", "counter"]:
            for b in BL_LABELS:
                m = has_result & (pys == py) & (flow_arr == ft) & (bin_arr == b)
                if m.sum() < 100:
                    continue
                p20 = actual_clear_rate(mcp[m], bids[0.20][m])
                bias = res[m].mean()
                cells.append((py, ft, b, m.sum(), p20, p20 - 20, bias))
    cells.sort(key=lambda x: -abs(x[5]))
    for c in cells[:10]:
        py, ft, b, n, p20, miss, bias = c
        tag = " DEV" if py < 2023 else ""
        print(f"  {py:>4} {ft:>8} {b:>4} {n:>6} | {p20:>5.1f}% {miss:>+5.1f}pp | {bias:>+7,.0f}{tag}")

    # Finest grain: (PY, qtr, flow, class, bin) — q4+q5 only to keep output manageable
    print(f"\n  Finest grain: (PY, qtr, flow, class, bin) — q4+q5 holdout:")
    print(f"  {'PY':>4} {'Qtr':>4} {'Flow':>8} {'Cls':>5} {'Bin':>4} {'N':>5} | {'bias':>6} | "
          f"{'P10':>6} {'P20':>6} {'P50':>6} {'P90':>6} {'P95':>6}")
    for py in HOLDOUT_PYS:
        for qtr in ["aq1", "aq2", "aq3", "aq4"]:
            for ft in ["prevail", "counter"]:
                for cls in ["onpeak", "offpeak"]:
                    for b in ["q4", "q5"]:
                        m = (has_result & (pys == py) & (qtrs == qtr) &
                             (flow_arr == ft) & (cls_arr == cls) & (bin_arr == b))
                        if m.sum() < 30:
                            continue
                        mc = mcp[m]
                        bias = res[m].mean()
                        vals = []
                        any_flag = False
                        for lv in [0.10, 0.20, 0.50, 0.90, 0.95]:
                            act = actual_clear_rate(mc, bids[lv][m])
                            if abs(act - lv * 100) > 10:
                                any_flag = True
                            vals.append(f"{act:>5.1f}%")
                        flag = " ***" if any_flag else ""
                        print(f"  {py:>4} {qtr:>4} {ft:>8} {cls[:5]:>5} {b:>4} {m.sum():>5} | "
                              f"{bias:>+5,.0f} | {' '.join(vals)}{flag}")


def main():
    t0 = time.time()
    print(f"MISO R2/R3: Pure MTM baseline + hybrid calibration")
    print(f"  Calibration: q1-q3 by (bin,flow), q4-q5 by (bin). Class dropped.")
    print(f"  mem={mem_mb():.0f}MB")

    df = pl.read_parquet(DATA_DIR / "canonical_annual_paths.parquet")
    print(f"Loaded {df.height:,} paths, mem={mem_mb():.0f}MB")

    for rnd in [2, 3]:
        r = df.filter(pl.col("round") == rnd)
        mcp = r["mcp"].to_numpy()
        baseline = r["mtm_1st_period"].to_numpy()
        pys_arr = r["planning_year"].to_numpy()
        qtrs_arr = r["period_type"].to_numpy()
        cls_arr = r["class_type"].to_numpy()

        print(f"\n{'#'*130}")
        print(f"  ROUND {rnd}: {len(mcp):,} paths")
        print(f"{'#'*130}")

        # Baseline diagnostics
        res_raw = mcp - baseline
        print(f"\n  Baseline: MAE={np.abs(res_raw).mean():,.0f}, bias={res_raw.mean():+,.0f}")

        bids, bin_arr, has_result, flow_arr = run_pipeline(
            mcp, baseline, pys_arr, qtrs_arr, cls_arr, ALL_PYS
        )
        audit(rnd, mcp, baseline, bids, bin_arr, has_result, flow_arr,
              pys_arr, qtrs_arr, cls_arr, label=f"R{rnd}")

    elapsed = time.time() - t0
    print(f"\nDone. elapsed={elapsed:.0f}s, mem={mem_mb():.0f}MB")


if __name__ == "__main__":
    main()
