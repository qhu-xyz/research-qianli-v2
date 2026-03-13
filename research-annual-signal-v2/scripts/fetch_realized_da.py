"""Build realized DA cache for all months 2017-04 through 2026-02.

Usage:
  cd /home/xyz/workspace/pmodel && source .venv/bin/activate
  python /home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/scripts/fetch_realized_da.py

Requires Ray cluster.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Ray setup
os.environ["RAY_ADDRESS"] = "ray://10.8.0.36:10001"


def main():
    from pbase.config.ray import init_ray
    import pmodel
    init_ray(extra_modules=[pmodel])

    import polars as pl
    from pbase.analysis.tools.all_positions import MisoApTools

    project_root = Path(__file__).resolve().parent.parent
    cache_dir = project_root / "data" / "realized_da"
    cache_dir.mkdir(parents=True, exist_ok=True)

    tools = MisoApTools().tools

    # Generate all months from 2017-04 through 2026-02
    months = []
    for year in range(2017, 2027):
        for month in range(1, 13):
            m = f"{year:04d}-{month:02d}"
            if m < "2017-04" or m > "2026-02":
                continue
            months.append(m)

    for month_str in months:
        for peak_type in ["onpeak", "offpeak"]:
            suffix = "_offpeak" if peak_type == "offpeak" else ""
            out_path = cache_dir / f"{month_str}{suffix}.parquet"

            if out_path.exists():
                continue

            year, mon = month_str.split("-")
            st = f"{year}-{mon}-01"
            # et_ex is first day of NEXT month
            next_mon = int(mon) + 1
            next_year = int(year)
            if next_mon > 12:
                next_mon = 1
                next_year += 1
            et_ex = f"{next_year:04d}-{next_mon:02d}-01"

            print(f"Fetching {month_str} {peak_type}...")
            da = tools.get_da_shadow_by_peaktype(
                st=st, et_ex=et_ex, peak_type=peak_type
            )

            if da is None or len(da) == 0:
                print(f"  SKIP {month_str} {peak_type}: no data returned")
                continue

            # Validate raw API output columns
            assert "constraint_id" in da.columns, f"API missing constraint_id, got: {list(da.columns)}"
            assert "shadow_price" in da.columns, f"API missing shadow_price, got: {list(da.columns)}"

            # Convert to polars, aggregate per constraint_id
            # abs(sum(shadow_price)): netting within month+ctype, then abs
            da_pl = pl.from_pandas(da)
            agg = (
                da_pl
                .group_by("constraint_id")
                .agg(pl.col("shadow_price").sum().abs().alias("realized_sp"))
            )
            assert (agg["realized_sp"] >= 0).all(), "realized_sp must be non-negative after abs"
            # Ensure constraint_id is string
            agg = agg.with_columns(pl.col("constraint_id").cast(pl.Utf8))

            agg.write_parquet(str(out_path))
            print(f"  Wrote {out_path} ({len(agg)} cids)")

    import ray
    ray.shutdown()
    print("Done.")


if __name__ == "__main__":
    main()
