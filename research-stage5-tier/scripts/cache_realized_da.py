#!/usr/bin/env python
"""One-time script: fetch and cache realized DA shadow prices for all needed months.

Months: 2019-06 to 2023-05 (~47 months).
Skips months already cached.

Usage:
    cd /home/xyz/workspace/pmodel && source .venv/bin/activate
    python /home/xyz/workspace/research-qianli-v2/research-stage5-tier/scripts/cache_realized_da.py
"""
from __future__ import annotations

import os
import sys
import resource
from pathlib import Path

# Ensure ml package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import polars as pl

from ml.config import REALIZED_DA_CACHE
from ml.realized_da import fetch_and_cache_month, _cache_path


def mem_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def generate_months(start: str, end: str) -> list[str]:
    """Generate YYYY-MM strings from start to end inclusive."""
    sy, sm = map(int, start.split("-"))
    ey, em = map(int, end.split("-"))
    months = []
    y, m = sy, sm
    while (y, m) <= (ey, em):
        months.append(f"{y:04d}-{m:02d}")
        m += 1
        if m > 12:
            m = 1
            y += 1
    return months


def main() -> None:
    # Initialize Ray
    os.environ["RAY_ADDRESS"] = "ray://10.8.0.36:10001"
    from pbase.config.ray import init_ray
    import pmodel
    init_ray(extra_modules=[pmodel])

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--peak-type", default="onpeak", choices=["onpeak", "offpeak"])
    parser.add_argument("--start", default="2019-06")
    parser.add_argument("--end", default="2025-12")
    args = parser.parse_args()

    peak_type = args.peak_type
    months = generate_months(args.start, args.end)
    cache_dir = REALIZED_DA_CACHE
    print(f"[cache_realized_da] {len(months)} months, peak_type={peak_type}, cache_dir={cache_dir}")
    print(f"[cache_realized_da] mem={mem_mb():.0f} MB")

    cached = 0
    fetched = 0
    for i, month in enumerate(months, 1):
        parquet_path = _cache_path(month, peak_type, cache_dir)
        if parquet_path.exists():
            cached += 1
            print(f"  [{i}/{len(months)}] {month} -- already cached, skipping")
            continue

        print(f"  [{i}/{len(months)}] {month} -- fetching ... ", end="", flush=True)
        out = fetch_and_cache_month(month, peak_type=peak_type, cache_dir=cache_dir)
        df = pl.read_parquet(str(out))
        n_bind = len(df.filter(pl.col("realized_sp") > 0))
        fetched += 1
        print(f"{len(df)} constraints, {n_bind} binding, mem={mem_mb():.0f} MB")

    import ray
    ray.shutdown()

    print(f"\n[cache_realized_da] Done. {fetched} fetched, {cached} already cached.")
    print(f"[cache_realized_da] Final mem={mem_mb():.0f} MB")


if __name__ == "__main__":
    main()
