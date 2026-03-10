#!/usr/bin/env python
"""Cache PJM realized DA shadow prices for all months needed by the ML pipeline.

Must be run before any experiment scripts. Requires Ray connection.

Usage:
    cd /home/xyz/workspace/pmodel && source .venv/bin/activate
    cd /home/xyz/workspace/research-qianli-v2/research-pjm-signal-f0p-0
    python scripts/cache_realized_da.py --peak-types onpeak dailyoffpeak wkndonpeak
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.config import REALIZED_DA_CACHE
from ml.realized_da import fetch_and_cache_month, _cache_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--peak-types", nargs="+",
                        default=["onpeak", "dailyoffpeak", "wkndonpeak"])
    parser.add_argument("--start", default="2019-01",
                        help="First month to cache (YYYY-MM)")
    parser.add_argument("--end", default="2026-02",
                        help="Last month to cache (YYYY-MM)")
    parser.add_argument("--cache-dir", default=REALIZED_DA_CACHE)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Init Ray
    os.environ["RAY_ADDRESS"] = "ray://10.8.0.36:10001"
    from pbase.config.ray import init_ray
    import pmodel
    init_ray(extra_modules=[pmodel])

    import pandas as pd

    # Generate all months in range
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    months = []
    current = start
    while current <= end:
        months.append(current.strftime("%Y-%m"))
        current += pd.DateOffset(months=1)

    print(f"[cache] Caching {len(months)} months × {len(args.peak_types)} peak types")
    print(f"[cache] Cache dir: {args.cache_dir}")

    if args.dry_run:
        # Count existing vs missing
        existing = sum(
            1 for m in months for pt in args.peak_types
            if _cache_path(m, pt, args.cache_dir).exists()
        )
        total = len(months) * len(args.peak_types)
        print(f"[cache] DRY RUN: {existing}/{total} already cached, {total - existing} to fetch")
        return

    t0 = time.time()
    fetched = 0
    skipped = 0
    already_cached = 0

    for pt in args.peak_types:
        for m in months:
            p = _cache_path(m, pt, args.cache_dir)
            if p.exists():
                already_cached += 1
                continue
            try:
                fetch_and_cache_month(m, pt, cache_dir=args.cache_dir)
                fetched += 1
            except Exception as e:
                print(f"[cache] SKIP {m}/{pt}: {e}")
                skipped += 1

    elapsed = time.time() - t0
    print(f"\n[cache] Done in {elapsed:.0f}s: {fetched} fetched, "
          f"{already_cached} cached, {skipped} skipped")


if __name__ == "__main__":
    main()
