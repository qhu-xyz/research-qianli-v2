"""Sweep GT recovery using monitored-line matching across PYs, R2, onpeak.

Primary mapping: DA.monitored_facility → bridge monitored part → branch_name.
Also adds f0 monthly fallback where monthly bridge exists.

Run: cd /home/xyz/workspace/pmodel && source .venv/bin/activate && python /home/xyz/workspace/research-qianli-v2/research-annual-signal-pjm/scripts/gt_recovery_sweep.py
"""
from __future__ import annotations

import re
from pathlib import Path

import polars as pl

SPICE_BASE = "/opt/data/xyz-dataset/spice_data/pjm"
DA_BASE = "/opt/data/xyz-dataset/modeling_data/pjm/PJM_DA_SHADOW_PRICE.parquet"
BRIDGE_DS = "PJM_SPICE_CONSTRAINT_INFO.parquet"
CONVENTION_THRESHOLD = 10
CLASS_TYPE = "onpeak"
ROUND = 2

MONTHLY_ROOTS = [
    f"{SPICE_BASE}/{BRIDGE_DS}/network_model=miso/spice_version=v6/auction_type=monthly",
    f"{SPICE_BASE}/{BRIDGE_DS}/network_model=pjm/spice_version=v6/auction_type=monthly",
]


def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def resolve_annual_root(py: str) -> str:
    if py == "2025-06":
        return f"{SPICE_BASE}/{BRIDGE_DS}/spice_version=v6/auction_type=annual"
    return f"{SPICE_BASE}/{BRIDGE_DS}/network_model=miso/spice_version=v6/auction_type=annual"


def get_quarter_months(py: str) -> dict[str, list[tuple[int, int]]]:
    year = int(py[:4])
    return {
        "aq1": [(year, 6), (year, 7), (year, 8)],
        "aq2": [(year, 9), (year, 10), (year, 11)],
        "aq3": [(year, 12), (year + 1, 1), (year + 1, 2)],
        "aq4": [(year + 1, 3), (year + 1, 4), (year + 1, 5)],
    }


def load_da(year_months: list[tuple[int, int]]) -> pl.DataFrame:
    """Load DA, aggregate |SP| by monitored_facility."""
    frames = []
    for year in sorted(set(y for y, _ in year_months)):
        months = [m for y, m in year_months if y == year]
        try:
            df = pl.scan_parquet(f"{DA_BASE}/year={year}/").filter(
                pl.col("month").is_in(months)
            ).collect()
            if len(df) > 0:
                frames.append(df)
        except Exception:
            pass
    if not frames:
        return pl.DataFrame({"monitored_facility": [], "abs_sp": []}).cast({"abs_sp": pl.Float64})
    da = pl.concat(frames)
    return da.group_by("monitored_facility").agg(
        pl.col("shadow_price").abs().sum().alias("abs_sp")
    )


def build_mon_lookup(path: str) -> dict[str, str]:
    """Build monitored → branch from a bridge partition. Skip ambiguous."""
    if not Path(path).is_dir():
        return {}
    df = pl.read_parquet(path)
    df = df.filter(
        (pl.col("class_type") == CLASS_TYPE) & (pl.col("convention") < CONVENTION_THRESHOLD)
    )
    df = df.with_columns(
        pl.col("constraint_id").str.split(":").list.first().str.strip_chars().alias("monitored")
    )
    mon_br = df.group_by("monitored").agg(
        pl.col("branch_name").n_unique().alias("n"),
        pl.col("branch_name").first().alias("branch_name"),
    )
    return {
        normalize_ws(r["monitored"]): r["branch_name"]
        for r in mon_br.filter(pl.col("n") == 1).iter_rows(named=True)
    }


def process(py: str, q: str, year_months: list[tuple[int, int]]) -> dict:
    da_agg = load_da(year_months)
    if len(da_agg) == 0:
        return {"py": py, "q": q, "ann_pct": 0.0, "final_pct": 0.0, "total_sp": 0.0}
    total_sp = float(da_agg["abs_sp"].sum())

    # Annual bridge: monitored-line matching
    root = resolve_annual_root(py)
    ann_lookup = build_mon_lookup(f"{root}/auction_month={py}/market_round={ROUND}/period_type={q}/")

    matched_sp = 0.0
    unmatched_mons = set()
    for row in da_agg.iter_rows(named=True):
        mon = normalize_ws(row["monitored_facility"])
        if mon in ann_lookup:
            matched_sp += row["abs_sp"]
        else:
            unmatched_mons.add(row["monitored_facility"])
    ann_sp = matched_sp

    # f0 fallback for still-unmatched
    f0_sp = 0.0
    for year, month in year_months:
        sm = f"{year}-{month:02d}"
        for mroot in MONTHLY_ROOTS:
            f0_lookup = build_mon_lookup(f"{mroot}/auction_month={sm}/market_round=1/period_type=f0/")
            if not f0_lookup:
                continue
            newly_matched = set()
            for mon_raw in unmatched_mons:
                mon = normalize_ws(mon_raw)
                if mon in f0_lookup:
                    sp = float(da_agg.filter(pl.col("monitored_facility") == mon_raw)["abs_sp"].sum())
                    f0_sp += sp
                    newly_matched.add(mon_raw)
            unmatched_mons -= newly_matched

    final_sp = ann_sp + f0_sp
    return {
        "py": py, "q": q,
        "ann_pct": ann_sp / total_sp * 100 if total_sp > 0 else 0,
        "final_pct": final_sp / total_sp * 100 if total_sp > 0 else 0,
        "total_sp": total_sp,
    }


def main():
    pys = ["2019-06", "2020-06", "2021-06", "2023-06", "2024-06", "2025-06"]
    results = []

    for py in pys:
        qm = get_quarter_months(py)
        for q, ym in qm.items():
            r = process(py, q, ym)
            results.append(r)
            f0_delta = f" +f0→{r['final_pct']:.1f}%" if r["final_pct"] > r["ann_pct"] + 0.1 else ""
            print(f"  {py} {q}: {r['ann_pct']:5.1f}%{f0_delta}")

    df = pl.DataFrame(results)

    for label, col in [("Monitored-line (annual bridge)", "ann_pct"), ("+ f0 fallback", "final_pct")]:
        pivot = df.pivot(on="q", index="py", values=col)
        cols = ["py"] + [c for c in ["aq1", "aq2", "aq3", "aq4"] if c in pivot.columns]
        print(f"\n{label}:")
        print(pivot.select(cols).to_pandas().to_string(index=False, float_format="%.1f"))

    # Annual
    ann = df.group_by("py").agg(
        (pl.col("ann_pct") / 100 * pl.col("total_sp")).sum().alias("a_sp"),
        (pl.col("final_pct") / 100 * pl.col("total_sp")).sum().alias("f_sp"),
        pl.col("total_sp").sum().alias("tot"),
    ).with_columns(
        (pl.col("a_sp") / pl.col("tot") * 100).alias("ann"),
        (pl.col("f_sp") / pl.col("tot") * 100).alias("+f0"),
    ).sort("py")
    print("\nAnnual recovery:")
    print(ann.select(["py", "ann", "+f0"]).to_pandas().to_string(index=False, float_format="%.1f"))


if __name__ == "__main__":
    main()
