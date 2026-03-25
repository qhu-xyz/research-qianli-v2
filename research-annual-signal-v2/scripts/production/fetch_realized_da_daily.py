"""Build daily realized DA cache for all months 2016-04 through 2026-02.

Fetches hourly DA shadow prices via pbase, aggregates to daily level:
  realized_sp = abs(sum(shadow_price)) per (constraint_id, date)

Output: data/realized_da_daily/{YYYY-MM-DD}_{peak_type}.parquet

Requires Ray cluster.

Usage:
    cd /home/xyz/workspace/pmodel && source .venv/bin/activate
    python /home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/scripts/fetch_realized_da_daily.py
"""
from __future__ import annotations

import os
import time
from pathlib import Path

os.environ["RAY_ADDRESS"] = "ray://10.8.0.36:10001"


def main():
    from pbase.config.ray import init_ray
    import pmodel
    init_ray(extra_modules=[pmodel])

    import polars as pl
    from pbase.analysis.tools.all_positions import MisoApTools

    cache_dir = Path(
        os.environ.get("ROOT_QH_TMP_PATH", "/opt/tmp/qianli")
    ) / "realized_da_daily"
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Cache dir: {cache_dir}")

    tools = MisoApTools().tools

    # Generate all months from 2016-04 through 2026-02
    months = []
    for year in range(2016, 2027):
        for month in range(1, 13):
            m = f"{year:04d}-{month:02d}"
            if m < "2016-04" or m > "2026-02":
                continue
            months.append(m)

    t0 = time.time()
    total_files = 0
    skipped = 0

    for month_str in months:
        for peak_type in ["onpeak", "offpeak"]:
            # Check if ALL days for this month are already cached
            # We don't know exact days yet, so fetch if ANY day is missing.
            # Use a sentinel file to mark month+peak as complete.
            sentinel = cache_dir / f".done_{month_str}_{peak_type}"
            if sentinel.exists():
                skipped += 1
                continue

            year_s, mon_s = month_str.split("-")
            st = f"{year_s}-{mon_s}-01"
            next_mon = int(mon_s) + 1
            next_year = int(year_s)
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
                sentinel.touch()
                continue

            assert "constraint_id" in da.columns, f"API missing constraint_id, got: {list(da.columns)}"
            assert "shadow_price" in da.columns, f"API missing shadow_price, got: {list(da.columns)}"

            da_pl = pl.from_pandas(da)

            # Build date column from year/month/day
            assert "year" in da_pl.columns and "month" in da_pl.columns and "day" in da_pl.columns, (
                f"Missing year/month/day columns, got: {da_pl.columns}"
            )
            da_pl = da_pl.with_columns(
                pl.date(pl.col("year"), pl.col("month"), pl.col("day")).alias("trade_date")
            )

            # Aggregate per (constraint_id, trade_date): SIGNED sum(shadow_price)
            # Store signed so abs(sum(signed_daily)) == abs(sum(monthly)) exactly.
            # Abs is applied only at the final aggregation level (month or partial-month).
            daily_agg = (
                da_pl
                .group_by(["constraint_id", "trade_date"])
                .agg(pl.col("shadow_price").sum().alias("signed_sp"))
            )
            daily_agg = daily_agg.with_columns(pl.col("constraint_id").cast(pl.Utf8))

            # Write one file per day
            dates_in_month = daily_agg["trade_date"].unique().sort()
            for d in dates_in_month:
                day_df = daily_agg.filter(pl.col("trade_date") == d).select(
                    ["constraint_id", "signed_sp"]
                )
                date_str = str(d)
                out_path = cache_dir / f"{date_str}_{peak_type}.parquet"
                day_df.write_parquet(str(out_path))
                total_files += 1

            sentinel.touch()
            print(f"  Wrote {len(dates_in_month)} daily files for {month_str} {peak_type}")

    import ray
    ray.shutdown()
    elapsed = time.time() - t0
    print(f"\nDone. {total_files} new files, {skipped} months skipped. {elapsed:.0f}s")

    # Write manifest
    import json
    from datetime import datetime

    all_parquets = sorted(f.name for f in cache_dir.glob("*.parquet"))
    dates_on = sorted(set(f.replace("_onpeak.parquet", "") for f in all_parquets if "_onpeak" in f))
    dates_off = sorted(set(f.replace("_offpeak.parquet", "") for f in all_parquets if "_offpeak" in f))
    manifest = {
        "cache_name": "realized_da_daily",
        "cache_dir": str(cache_dir),
        "build_timestamp": datetime.now().isoformat(),
        "source_command": f"python {Path(__file__).name}",
        "source_api": "MisoApTools.tools.get_da_shadow_by_peaktype",
        "aggregation": "signed sum(shadow_price) per (constraint_id, trade_date, peak_type)",
        "schema": {"constraint_id": "Utf8", "signed_sp": "Float64"},
        "ctypes": ["onpeak", "offpeak"],
        "onpeak_dates": {"count": len(dates_on), "first": dates_on[0] if dates_on else None, "last": dates_on[-1] if dates_on else None},
        "offpeak_dates": {"count": len(dates_off), "first": dates_off[0] if dates_off else None, "last": dates_off[-1] if dates_off else None},
        "total_files": len(all_parquets),
    }
    manifest_path = cache_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest: {manifest_path}")
    print(f"  onpeak: {manifest['onpeak_dates']}")
    print(f"  offpeak: {manifest['offpeak_dates']}")


if __name__ == "__main__":
    main()
