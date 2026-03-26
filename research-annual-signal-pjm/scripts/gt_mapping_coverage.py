"""Phase 1.4 — GT mapping coverage for one slice: PY=2024-06, R2, onpeak.

Maps DA to branches via monitored-line matching through quarterly bridge (aq1-aq4).
This is the correct approach: many DA CIDs (different contingencies) collapse onto
the same branch via the monitored line. The contingency is irrelevant for the branch
mapping (99.9% of monitored lines map to exactly 1 branch in PJM).

Reports mapped_SP vs total_SP per quarter.

Run: cd /home/xyz/workspace/pmodel && source .venv/bin/activate && python /home/xyz/workspace/research-qianli-v2/research-annual-signal-pjm/scripts/gt_mapping_coverage.py
"""
from __future__ import annotations

import re

import polars as pl

# ── Config ─────────────────────────────────────────────────────────────
PY = "2024-06"
ROUND = 2
CLASS_TYPE = "onpeak"

SPICE_BASE = "/opt/data/xyz-dataset/spice_data/pjm"
DA_BASE = "/opt/data/xyz-dataset/modeling_data/pjm/PJM_DA_SHADOW_PRICE.parquet"

QUARTER_MONTHS = {
    "aq1": [(2024, 6), (2024, 7), (2024, 8)],
    "aq2": [(2024, 9), (2024, 10), (2024, 11)],
    "aq3": [(2024, 12), (2025, 1), (2025, 2)],
    "aq4": [(2025, 3), (2025, 4), (2025, 5)],
}

BRIDGE_ROOT = f"{SPICE_BASE}/PJM_SPICE_CONSTRAINT_INFO.parquet/network_model=miso/spice_version=v6/auction_type=annual"
CONVENTION_THRESHOLD = 10


def normalize_ws(s: str) -> str:
    """Collapse whitespace for matching."""
    return re.sub(r"\s+", " ", s).strip()


def load_da_for_months(year_months: list[tuple[int, int]]) -> pl.DataFrame:
    """Load DA, aggregate |SP| by monitored_facility (the mapping key)."""
    frames = []
    for year in sorted(set(y for y, _ in year_months)):
        months_in_year = [m for y, m in year_months if y == year]
        df = (
            pl.scan_parquet(f"{DA_BASE}/year={year}/")
            .filter(pl.col("month").is_in(months_in_year))
            .collect()
        )
        frames.append(df)
    da = pl.concat(frames)
    # Aggregate by monitored_facility — this is the branch mapping key
    return da.group_by("monitored_facility").agg(
        pl.col("shadow_price").abs().sum().alias("abs_sp"),
        pl.col("contingency_facility").n_unique().alias("n_contingencies"),
    )


def build_monitored_to_branch(quarter: str) -> tuple[dict[str, str], int]:
    """Build monitored_line → branch lookup from bridge.

    Returns (lookup_dict, n_ambiguous_monitored_lines).
    Drops the rare cases where a monitored line maps to >1 branch (generic interface names).
    """
    path = f"{BRIDGE_ROOT}/auction_month={PY}/market_round={ROUND}/period_type={quarter}/"
    df = pl.read_parquet(path)
    df = df.filter(
        (pl.col("class_type") == CLASS_TYPE)
        & (pl.col("convention") < CONVENTION_THRESHOLD)
    )
    # Extract monitored part (before ':')
    df = df.with_columns(
        pl.col("constraint_id").str.split(":").list.first().str.strip_chars().alias("monitored")
    )
    # Group by monitored → branch. Keep only unique (1-to-1) mappings.
    mon_br = df.group_by("monitored").agg(
        pl.col("branch_name").n_unique().alias("n_branches"),
        pl.col("branch_name").first().alias("branch_name"),
        pl.col("constraint_id").n_unique().alias("n_cids"),
    )
    n_ambiguous = int((mon_br["n_branches"] > 1).sum())
    unique = mon_br.filter(pl.col("n_branches") == 1)
    lookup = {}
    for row in unique.iter_rows(named=True):
        lookup[normalize_ws(row["monitored"])] = row["branch_name"]
    return lookup, n_ambiguous


def main():
    print("=" * 80)
    print(f"GT Mapping Coverage (monitored-line matching) — PY={PY}, R{ROUND}, {CLASS_TYPE}")
    print("=" * 80)

    results = []

    for quarter, year_months in QUARTER_MONTHS.items():
        print(f"\n── {quarter} ({year_months[0][0]}-{year_months[0][1]:02d} to {year_months[-1][0]}-{year_months[-1][1]:02d}) ──")

        # Load DA aggregated by monitored_facility
        da_agg = load_da_for_months(year_months)
        total_sp = float(da_agg["abs_sp"].sum())
        n_da_lines = len(da_agg)
        n_da_contingencies = int(da_agg["n_contingencies"].sum())

        # Build monitored → branch lookup
        lookup, n_ambiguous = build_monitored_to_branch(quarter)

        # Match DA monitored_facility → branch
        mapped_sp = 0.0
        n_mapped = 0
        branch_sp: dict[str, float] = {}
        unmapped_rows = []
        for row in da_agg.iter_rows(named=True):
            mon_norm = normalize_ws(row["monitored_facility"])
            if mon_norm in lookup:
                n_mapped += 1
                mapped_sp += row["abs_sp"]
                branch = lookup[mon_norm]
                branch_sp[branch] = branch_sp.get(branch, 0.0) + row["abs_sp"]
            else:
                unmapped_rows.append((row["monitored_facility"], row["abs_sp"]))

        n_unmapped = n_da_lines - n_mapped
        unmapped_sp = total_sp - mapped_sp
        n_branches = len(branch_sp)
        recovery = mapped_sp / total_sp if total_sp > 0 else 0.0

        print(f"  DA monitored lines:  {n_da_lines} (covering {n_da_contingencies} CIDs)")
        print(f"  Bridge monitored:    {len(lookup)} (unique, non-ambiguous)")
        print(f"  Ambiguous (dropped): {n_ambiguous}")
        print(f"  Mapped lines:        {n_mapped}")
        print(f"  Unmapped lines:      {n_unmapped}")
        print(f"  Mapped branches:     {n_branches}")
        print(f"  Total |SP|:          {total_sp:,.0f}")
        print(f"  Mapped |SP|:         {mapped_sp:,.0f}")
        print(f"  Unmapped |SP|:       {unmapped_sp:,.0f}")
        print(f"  Recovery:            {recovery:.1%}")

        if unmapped_rows:
            unmapped_rows.sort(key=lambda x: -x[1])
            print(f"  Top 5 unmapped monitored lines:")
            for mon, sp in unmapped_rows[:5]:
                print(f"    {mon[:70]}: |SP|={sp:,.0f}")

        results.append({
            "quarter": quarter,
            "n_da_lines": n_da_lines,
            "n_mapped": n_mapped,
            "n_unmapped": n_unmapped,
            "n_branches": n_branches,
            "total_sp": total_sp,
            "mapped_sp": mapped_sp,
            "recovery_pct": recovery * 100,
        })

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY — GT Mapping Coverage (monitored-line matching)")
    print("=" * 80)
    summary = pl.DataFrame(results)
    print(summary.to_pandas().to_string(index=False))

    total = summary.select(
        pl.col("total_sp").sum().alias("total_sp"),
        pl.col("mapped_sp").sum().alias("mapped_sp"),
    )
    annual_recovery = float(total["mapped_sp"][0]) / float(total["total_sp"][0])
    print(f"\nAnnual total |SP|:  {float(total['total_sp'][0]):,.0f}")
    print(f"Annual mapped |SP|: {float(total['mapped_sp'][0]):,.0f}")
    print(f"Annual recovery:    {annual_recovery:.1%}")


if __name__ == "__main__":
    main()
