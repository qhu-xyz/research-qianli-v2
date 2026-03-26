"""MISO R1 V2: Blended baseline + per-(flow, bin, class) asymmetric bands.

Frozen spec (2026-03-20):
  Baseline: w(flow, bin) × f0 + (1-w) × 1(rev), w estimated on train, min cell 100
  Banding:  asymmetric signed quantile pairs per (flow_type, bin, class_type)
  Bins:     5 quantile bins on training |raw f0|, frozen throughout pipeline
  Width:    equal-weight expanding window
  Cap:      ±15,000 quarterly $ on band edges (governance overlay)

Terminology note:
  The labels P10/P30/P50/... in this script are legacy two-sided band-width labels.
  They are not one-sided clearing probabilities. For example, legacy P50 means the
  upper-edge buy clearing target is 75%, not 50%.

Usage:
    cd /home/xyz/workspace/pmodel && source .venv/bin/activate
    python /home/xyz/workspace/research-qianli-v2/research-annual-band/miso/scripts/run_r1_v2.py
"""

from __future__ import annotations
import sys, os, time, resource, gc, json
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "archive_v1" / "scripts"))

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from run_v9_bands import compute_quantile_boundaries, assign_bins, COVERAGE_LEVELS, COVERAGE_LABELS, N_BINS

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

# ── Config ──
W_GRID = [1.0, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5]
MIN_BLEND_CELL = 100
MIN_BAND_CELL = 300
BAND_CAP = 15_000  # quarterly $ governance cap on band edges
ALL_PYS = list(range(2020, 2026))
DEV_PYS = list(range(2020, 2023))
HOLDOUT_PYS = [2023, 2024, 2025]


def mem_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def buy_clear(mcp, upper):
    if len(mcp) == 0:
        return float("nan")
    return float(np.mean(mcp <= upper) * 100)


# Legacy upper-edge buy clearing targets implied by the symmetric residual pairs.
BUY_TARGETS = {f"p{int(l*100)}": (1+l)/2*100 for l in COVERAGE_LEVELS}


def find_best_w(mcp_t, f0_t, rev_t):
    best_w, best_mae = 1.0, np.abs(mcp_t - f0_t).mean()
    for w in W_GRID:
        mae = np.abs(mcp_t - (w * f0_t + (1-w) * rev_t)).mean()
        if mae < best_mae:
            best_mae, best_w = mae, w
    return best_w


def load_data():
    """Load R1 paths with nodal_f0 and 1(rev)."""
    paths = pl.read_parquet(DATA_DIR / "canonical_annual_paths.parquet").filter(pl.col("round") == 1)

    # Load nodal_f0 from old baselines
    old_r1_dir = Path("/opt/temp/qianli/annual_research/crossproduct_work")
    old_dfs = []
    for q in ["aq1", "aq2", "aq3", "aq4"]:
        f = old_r1_dir / f"{q}_all_baselines.parquet"
        if f.exists():
            df = pl.read_parquet(f).select(["source_id", "sink_id", "class_type", "planning_year", "nodal_f0"])
            df = df.with_columns(pl.lit(q).alias("period_type"))
            old_dfs.append(df)
    old_r1 = pl.concat(old_dfs, how="diagonal").unique(
        subset=["source_id", "sink_id", "class_type", "planning_year", "period_type"]
    )

    # Load 1(rev) from Option B
    rev = pl.read_parquet(DATA_DIR / "r1_1rev_option_b.parquet")

    r1 = (
        paths
        .join(old_r1.select(["source_id", "sink_id", "class_type", "planning_year", "period_type", "nodal_f0"]),
              on=["source_id", "sink_id", "class_type", "planning_year", "period_type"], how="left")
        .join(rev.select(["source_id", "sink_id", "class_type", "planning_year", "period_type", "1_rev"]),
              on=["source_id", "sink_id", "class_type", "planning_year", "period_type"], how="left")
        .filter(pl.col("nodal_f0").is_not_null() & pl.col("1_rev").is_not_null() & pl.col("mcp").is_not_null())
    )
    return r1


def run_pipeline(r1, test_pys, label=""):
    """Run the frozen V2 pipeline on specified test PYs. Returns per-path results."""
    bl_f0_raw = (r1["nodal_f0"] * 3).to_numpy()
    bl_rev_raw = r1["1_rev"].to_numpy()
    mcp = r1["mcp"].to_numpy()
    pys = r1["planning_year"].to_numpy()
    qtrs = r1["period_type"].to_numpy()
    classes = r1["class_type"].to_numpy()

    # Winsorize at P1/P99
    bl_f0 = np.clip(bl_f0_raw, *np.percentile(bl_f0_raw, [1, 99]))
    bl_rev = np.clip(bl_rev_raw, *np.percentile(bl_rev_raw, [1, 99]))

    # Collect results per test path
    all_rows = []

    for qtr in ["aq1", "aq2", "aq3", "aq4"]:
        qm = qtrs == qtr
        for tpy in test_pys:
            trm = qm & (pys < tpy)
            tem = qm & (pys == tpy)
            if trm.sum() == 0 or tem.sum() == 0:
                continue
            if len(np.unique(pys[trm])) < 2:
                continue

            # Step 1: Fixed bins from training |raw f0|
            boundaries, bl_labels = compute_quantile_boundaries(
                pl.Series("x", np.abs(bl_f0[trm])), N_BINS
            )
            trb = assign_bins(pl.Series("x", np.abs(bl_f0[trm])), boundaries, bl_labels).to_numpy()
            teb = assign_bins(pl.Series("x", np.abs(bl_f0[tem])), boundaries, bl_labels).to_numpy()
            trf = np.where(bl_f0[trm] > 0, "prevail", "counter")
            tef = np.where(bl_f0[tem] > 0, "prevail", "counter")

            # Step 2: Blend weights per (flow, bin) on train
            cell_w = {}
            for ft in ["prevail", "counter"]:
                for b in bl_labels:
                    ct = (trf == ft) & (trb == b)
                    if ct.sum() >= MIN_BLEND_CELL:
                        cell_w[(ft, b)] = find_best_w(mcp[trm][ct], bl_f0[trm][ct], bl_rev[trm][ct])
                    else:
                        cell_w[(ft, b)] = 1.0

            # Step 3: Adjusted baseline
            tra = bl_f0[trm].copy()
            tea = bl_f0[tem].copy()
            for ft in ["prevail", "counter"]:
                for b in bl_labels:
                    w = cell_w[(ft, b)]
                    ct = (trf == ft) & (trb == b)
                    ce = (tef == ft) & (teb == b)
                    tra[ct] = w * bl_f0[trm][ct] + (1-w) * bl_rev[trm][ct]
                    tea[ce] = w * bl_f0[tem][ce] + (1-w) * bl_rev[tem][ce]

            # Step 4: Calibrate bands per (flow, bin, class)
            train_res = mcp[trm] - tra
            bp = {}
            for b in bl_labels:
                bp[b] = {}
                for ft in ["prevail", "counter"]:
                    bp[b][ft] = {}
                    for cls in ["onpeak", "offpeak"]:
                        mk = (trb == b) & (trf == ft) & (classes[trm] == cls)
                        if mk.sum() < MIN_BAND_CELL:
                            mk = (trb == b) & (trf == ft)
                        res = train_res[mk]
                        pairs = {}
                        for lv, cl in zip(COVERAGE_LEVELS, COVERAGE_LABELS):
                            lo = float(np.quantile(res, (1-lv)/2))
                            hi = float(np.quantile(res, (1+lv)/2))
                            pairs[cl] = (lo, hi)
                        bp[b][ft][cls] = pairs

            # Step 5: Apply bands + governance cap
            test_mcp = mcp[tem]
            test_cls = classes[tem]
            for i in range(tem.sum()):
                b, ft, cls = teb[i], tef[i], test_cls[i]
                bl_adj = tea[i]
                m = test_mcp[i]
                pairs = bp[b][ft].get(cls, bp[b][ft].get("onpeak"))
                row = {
                    "qtr": qtr, "py": tpy, "flow": ft, "bin": b, "cls": cls,
                    "mcp": float(m), "bl_adj": float(bl_adj), "bl_raw": float(bl_f0[tem][i]),
                    "residual": float(m - bl_adj),
                    "w": cell_w[(ft, b)],
                }
                for cl in COVERAGE_LABELS:
                    lo, hi = pairs[cl]
                    raw_upper = bl_adj + hi
                    raw_lower = bl_adj + lo
                    # Governance cap
                    capped_upper = min(raw_upper, BAND_CAP)
                    capped_lower = max(raw_lower, -BAND_CAP)
                    row[f"upper_{cl}"] = float(capped_upper)
                    row[f"lower_{cl}"] = float(capped_lower)
                    row[f"width_{cl}"] = float(capped_upper - capped_lower)
                    row[f"upper_raw_{cl}"] = float(raw_upper)
                    row[f"lower_raw_{cl}"] = float(raw_lower)
                all_rows.append(row)

    return all_rows


def audit(rows, label=""):
    """Run comprehensive audit on pipeline results."""
    import pandas as pd
    df = pd.DataFrame(rows)
    n = len(df)
    print(f"\n{'='*120}")
    print(f"  {label}: {n:,} test paths")
    print(f"{'='*120}")

    mcp = df["mcp"].values

    # ── TABLE 1: Overall coverage + width (capped) ──
    print(f"\n  TABLE 1: Overall coverage + width (with ±{BAND_CAP:,} cap)")
    print(f"  {'Level':>6} {'Target':>7} | {'Cov':>7} {'Width':>7} | {'Cov(raw)':>9} {'Width(raw)':>10} | {'Cap effect':>11}")
    for cl in COVERAGE_LABELS:
        t = BUY_TARGETS[cl]
        cov = buy_clear(mcp, df[f"upper_{cl}"].values)
        wid = np.median(df[f"width_{cl}"].values)
        cov_raw = buy_clear(mcp, df[f"upper_raw_{cl}"].values)
        wid_raw = np.median(df[f"upper_raw_{cl}"].values - df[f"lower_raw_{cl}"].values)
        cap_eff = cov - cov_raw
        print(f"  {cl:>6} {t:>6.1f}% | {cov:>6.1f}% {wid:>7,.0f} | {cov_raw:>8.1f}% {wid_raw:>10,.0f} | {cap_eff:>+10.1f}pp")

    # ── TABLE 2: P50 + P95 by (PY, flow_type) ──
    print(f"\n  TABLE 2: P50 + P95 by (PY, flow_type)")
    print(f"  {'PY':>6} {'Flow':>8} {'N':>7} | {'P50':>6} {'W50':>6} | {'P95':>6} {'W95':>7}")
    for py in sorted(df["py"].unique()):
        for ft in ["prevail", "counter"]:
            sub = df[(df["py"]==py)&(df["flow"]==ft)]
            n_sub = len(sub)
            if n_sub < 100: continue
            m = sub["mcp"].values
            p50 = buy_clear(m, sub["upper_p50"].values)
            w50 = np.median(sub["width_p50"].values)
            p95 = buy_clear(m, sub["upper_p95"].values)
            w95 = np.median(sub["width_p95"].values)
            print(f"  {py:>6} {ft:>8} {n_sub:>7,} | {p50:>5.1f}% {w50:>6,.0f} | {p95:>5.1f}% {w95:>7,.0f}")

    # ── TABLE 3: Alerts at finest grain ──
    print(f"\n  TABLE 3: Alert cells (P50 miss > 10pp or P95 miss > 5pp)")
    print(f"  {'Qtr':>4} {'PY':>4} {'Fl':>4} {'Bn':>3} {'Cl':>3} {'N':>5} | "
          f"{'P50':>6} {'miss':>6} {'W50':>6} | {'P95':>6} {'miss':>6} {'W95':>7} | {'bias':>7} {'mode':>10}")

    for name, g in df.groupby(["qtr", "py", "flow", "bin", "cls"]):
        qtr, py, ft, b, cls = name
        n_g = len(g)
        if n_g < 20: continue
        m = g["mcp"].values
        res = g["residual"].values
        bias = res.mean()
        mae = np.abs(res).mean()

        p50 = buy_clear(m, g["upper_p50"].values)
        w50 = np.median(g["width_p50"].values)
        p95 = buy_clear(m, g["upper_p95"].values)
        w95 = np.median(g["width_p95"].values)

        m50 = p50 - 75.0
        m95 = p95 - 97.5

        mode = "bad center" if abs(bias) > 0.5 * mae else "bad width"

        if abs(m50) > 10 or abs(m95) > 5:
            print(f"  {qtr:>4} {py:>4} {ft[:4]:>4} {b:>3} {cls[:3]:>3} {n_g:>5} | "
                  f"{p50:>5.1f}% {m50:>+5.1f} {w50:>6,.0f} | "
                  f"{p95:>5.1f}% {m95:>+5.1f} {w95:>7,.0f} | "
                  f"{bias:>+7,.0f} {mode:>10}")

    # ── TABLE 4: Governance cap impact ──
    capped_paths = 0
    for cl in COVERAGE_LABELS:
        capped = ((df[f"upper_raw_{cl}"] > BAND_CAP) | (df[f"lower_raw_{cl}"] < -BAND_CAP)).sum()
        if capped > 0:
            capped_paths += capped
    print(f"\n  TABLE 4: Governance cap impact")
    print(f"  Paths with any band edge capped at ±{BAND_CAP:,}: ~{capped_paths:,} (across all levels)")

    # Count cap impact at P95
    capped_p95 = ((df["upper_raw_p95"] > BAND_CAP) | (df["lower_raw_p95"] < -BAND_CAP)).sum()
    print(f"  Paths with P95 edge capped: {capped_p95:,} ({capped_p95/n*100:.1f}%)")


def main():
    t0 = time.time()
    print(f"MISO R1 V2: Blended baseline + per-(flow, bin, class) bands")
    print(f"mem={mem_mb():.0f}MB")

    r1 = load_data()
    print(f"Loaded {r1.height:,} paths, mem={mem_mb():.0f}MB")

    # Run on ALL PYs
    print(f"\n--- Running on ALL PYs ({ALL_PYS}) ---")
    all_rows = run_pipeline(r1, ALL_PYS)
    audit(all_rows, "ALL PYs (dev + holdout)")

    # Run on holdout only
    print(f"\n--- Running on HOLDOUT PYs ({HOLDOUT_PYS}) ---")
    holdout_rows = [r for r in all_rows if r["py"] in HOLDOUT_PYS]
    audit(holdout_rows, "HOLDOUT (PY2023-2025)")

    elapsed = time.time() - t0
    print(f"\nDone. elapsed={elapsed:.0f}s, mem={mem_mb():.0f}MB")


if __name__ == "__main__":
    main()
