"""Phase 1.5 — Model-universe coverage for PY=2024-06, R2, onpeak.

Loads density, applies right-tail threshold, maps through bridge to branches.
Reports how many branches survive the universe filter and how much mapped GT SP they capture.

Run: cd /home/xyz/workspace/pmodel && source .venv/bin/activate && python /home/xyz/workspace/research-qianli-v2/research-annual-signal-pjm/scripts/universe_coverage.py
"""
from __future__ import annotations

import polars as pl

# ── Config ─────────────────────────────────────────────────────────────
PY = "2024-06"
ROUND = 2
CLASS_TYPE = "onpeak"

SPICE_BASE = "/opt/data/xyz-dataset/spice_data/pjm"
DA_BASE = "/opt/data/xyz-dataset/modeling_data/pjm/PJM_DA_SHADOW_PRICE.parquet"

# Density root for PY 2024-06 R2 = legacy
DENSITY_ROOT = f"{SPICE_BASE}/PJM_SPICE_DENSITY_DISTRIBUTION.parquet/network_model=miso/spice_version=v6/auction_type=annual"
BRIDGE_ROOT = f"{SPICE_BASE}/PJM_SPICE_CONSTRAINT_INFO.parquet/network_model=miso/spice_version=v6/auction_type=annual"

# Use same right-tail bins as MISO (80, 90, 100, 110 equivalent)
# PJM density bins go from -300 to +300. Right tail = high positive bins.
RIGHT_TAIL_BINS = ["80", "90", "100", "110"]

CONVENTION_THRESHOLD = 10

# Settlement months for PY 2024-06
PY_MONTHS = [
    (2024, m) for m in range(6, 13)
] + [
    (2025, m) for m in range(1, 6)
]

QUARTER_MONTHS = {
    "aq1": [(2024, 6), (2024, 7), (2024, 8)],
    "aq2": [(2024, 9), (2024, 10), (2024, 11)],
    "aq3": [(2024, 12), (2025, 1), (2025, 2)],
    "aq4": [(2025, 3), (2025, 4), (2025, 5)],
}


def load_density_cids() -> pl.DataFrame:
    """Load density, filter to R2, compute right-tail max per CID."""
    frames = []
    for year, month in PY_MONTHS:
        path = f"{DENSITY_ROOT}/auction_month={PY}/market_month={year}-{month:02d}/"
        df = pl.read_parquet(path)
        df = df.filter(pl.col("market_round") == ROUND)
        frames.append(df)

    density = pl.concat(frames)
    print(f"  Density loaded: {len(density)} rows, {density['constraint_id'].n_unique()} CIDs")

    # Compute right-tail max per CID across all months and outage dates
    rt_cols = [c for c in RIGHT_TAIL_BINS if c in density.columns]
    if not rt_cols:
        raise ValueError(f"Right-tail bins {RIGHT_TAIL_BINS} not found in density columns")

    cid_rt = density.group_by("constraint_id").agg(
        pl.max_horizontal(*[pl.col(c).max() for c in rt_cols]).alias("right_tail_max")
    )
    return cid_rt


def load_bridge_all_quarters() -> pl.DataFrame:
    """Load bridge for all 4 quarters, deduplicate to unique CID→branch."""
    frames = []
    for q in ["aq1", "aq2", "aq3", "aq4"]:
        path = f"{BRIDGE_ROOT}/auction_month={PY}/market_round={ROUND}/period_type={q}/"
        df = pl.read_parquet(path)
        df = df.filter(
            (pl.col("class_type") == CLASS_TYPE)
            & (pl.col("convention") < CONVENTION_THRESHOLD)
        )
        frames.append(df.select(["constraint_id", "branch_name"]))

    bridge = pl.concat(frames).unique(subset=["constraint_id", "branch_name"])

    # Drop ambiguous (CID → multiple branches)
    cid_counts = bridge.group_by("constraint_id").agg(
        pl.col("branch_name").n_unique().alias("n_branches")
    )
    ambiguous = cid_counts.filter(pl.col("n_branches") > 1)["constraint_id"].to_list()
    bridge = bridge.filter(~pl.col("constraint_id").is_in(ambiguous))
    bridge = bridge.unique(subset=["constraint_id"])
    return bridge


def load_gt_per_quarter() -> dict[str, pl.DataFrame]:
    """Load DA and map through bridge per quarter. Returns branch-level GT per quarter."""
    gt_by_q = {}
    for q, year_months in QUARTER_MONTHS.items():
        # Load DA
        frames = []
        years_needed = sorted(set(y for y, _ in year_months))
        for year in years_needed:
            months = [m for y, m in year_months if y == year]
            df = pl.scan_parquet(f"{DA_BASE}/year={year}/").filter(
                pl.col("month").is_in(months)
            ).collect()
            frames.append(df)
        da = pl.concat(frames).with_columns(
            (pl.col("monitored_facility") + ":" + pl.col("contingency_facility").str.replace("ACTUAL", "BASE")).alias("constraint_id")
        )
        da_agg = da.group_by("constraint_id").agg(
            pl.col("shadow_price").abs().sum().alias("abs_sp")
        )

        # Load bridge for this quarter
        path = f"{BRIDGE_ROOT}/auction_month={PY}/market_round={ROUND}/period_type={q}/"
        bridge = pl.read_parquet(path).filter(
            (pl.col("class_type") == CLASS_TYPE)
            & (pl.col("convention") < CONVENTION_THRESHOLD)
        )
        # Deduplicate
        cid_counts = bridge.group_by("constraint_id").agg(pl.col("branch_name").n_unique().alias("n"))
        ambig = cid_counts.filter(pl.col("n") > 1)["constraint_id"].to_list()
        bridge = bridge.filter(~pl.col("constraint_id").is_in(ambig)).unique(subset=["constraint_id"])

        # Map
        mapped = da_agg.join(bridge.select(["constraint_id", "branch_name"]), on="constraint_id", how="inner")
        branch_gt = mapped.group_by("branch_name").agg(pl.col("abs_sp").sum())
        gt_by_q[q] = branch_gt
    return gt_by_q


def main():
    print("=" * 80)
    print(f"Model-Universe Coverage — PY={PY}, R{ROUND}, {CLASS_TYPE}")
    print("=" * 80)

    # Step 1: Load density and compute right-tail max per CID
    print("\n── Step 1: Density right-tail max ──")
    cid_rt = load_density_cids()
    print(f"  Total CIDs with density: {len(cid_rt)}")
    print(f"  Right-tail max stats:")
    print(f"    min:    {cid_rt['right_tail_max'].min():.6f}")
    print(f"    median: {cid_rt['right_tail_max'].median():.6f}")
    print(f"    mean:   {cid_rt['right_tail_max'].mean():.6f}")
    print(f"    max:    {cid_rt['right_tail_max'].max():.6f}")

    # Step 2: Sweep thresholds to understand sensitivity
    print("\n── Step 2: Universe size vs threshold ──")
    thresholds = [0.0, 0.0001, 0.0003, 0.0005, 0.001, 0.003, 0.005, 0.01]
    for t in thresholds:
        n_active = int((cid_rt["right_tail_max"] >= t).sum())
        pct = n_active / len(cid_rt) * 100
        print(f"  threshold >= {t:.4f}: {n_active:,} CIDs ({pct:.1f}%)")

    # Step 3: Map active CIDs to branches via bridge
    print("\n── Step 3: CID → branch mapping ──")
    bridge = load_bridge_all_quarters()
    print(f"  Bridge unique CIDs: {len(bridge)}")

    # Use MISO-equivalent threshold as starting point
    miso_threshold = 0.0003467728739657263
    active_cids = set(cid_rt.filter(pl.col("right_tail_max") >= miso_threshold)["constraint_id"].to_list())
    n_active = len(active_cids)
    print(f"  Active CIDs (MISO threshold {miso_threshold:.6f}): {n_active}")

    # Map active CIDs through bridge
    active_mapped = bridge.filter(pl.col("constraint_id").is_in(list(active_cids)))
    active_branches = active_mapped["branch_name"].unique()
    n_branches = len(active_branches)
    print(f"  Active branches (at least 1 active CID): {n_branches}")

    # Step 4: Compare with GT branches — how much mapped GT SP is in the universe?
    print("\n── Step 4: Universe vs GT overlap ──")
    gt_by_q = load_gt_per_quarter()
    active_branch_set = set(active_branches.to_list())

    results = []
    for q in ["aq1", "aq2", "aq3", "aq4"]:
        branch_gt = gt_by_q[q]
        total_gt_sp = float(branch_gt["abs_sp"].sum())
        in_universe = branch_gt.filter(pl.col("branch_name").is_in(list(active_branch_set)))
        universe_sp = float(in_universe["abs_sp"].sum()) if len(in_universe) > 0 else 0.0
        outside_sp = total_gt_sp - universe_sp
        n_gt_branches = len(branch_gt)
        n_in = len(in_universe)
        n_out = n_gt_branches - n_in
        coverage = universe_sp / total_gt_sp if total_gt_sp > 0 else 0.0

        results.append({
            "quarter": q,
            "gt_branches": n_gt_branches,
            "in_universe": n_in,
            "outside_universe": n_out,
            "gt_sp": total_gt_sp,
            "universe_sp": universe_sp,
            "outside_sp": outside_sp,
            "coverage_pct": coverage * 100,
        })
        print(f"  {q}: {n_in}/{n_gt_branches} GT branches in universe, "
              f"SP coverage {coverage:.1%} ({universe_sp:,.0f} / {total_gt_sp:,.0f})")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY — Model-Universe Coverage (separate from GT mapping)")
    print("=" * 80)
    summary = pl.DataFrame(results)
    print(summary.to_pandas().to_string(index=False))

    total_gt = sum(r["gt_sp"] for r in results)
    total_univ = sum(r["universe_sp"] for r in results)
    print(f"\nAnnual GT SP in universe: {total_univ:,.0f} / {total_gt:,.0f} = {total_univ/total_gt:.1%}")
    print(f"Universe branches: {n_branches}")
    print(f"Threshold used: {miso_threshold:.10f} (MISO default, needs PJM calibration)")


if __name__ == "__main__":
    main()
