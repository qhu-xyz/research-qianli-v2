# v70/cache.py
"""Realized DA cache management for PJM V7.0 signal generation."""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.config import collect_usable_months, delivery_month, has_period_type

REALIZED_DA_CACHE = os.environ.get(
    "PJM_REALIZED_DA_CACHE",
    str(Path(__file__).resolve().parent.parent / "data" / "realized_da"),
)

_MAX_BF_LOOKBACK = 15


def _cache_path(month: str, peak_type: str) -> Path:
    if peak_type == "onpeak":
        return Path(REALIZED_DA_CACHE) / f"{month}.parquet"
    return Path(REALIZED_DA_CACHE) / f"{month}_{peak_type}.parquet"


def _prev_month(m: str) -> str:
    import pandas as pd
    return (pd.Timestamp(m) - pd.DateOffset(months=1)).strftime("%Y-%m")


def _months_before(month: str, n: int) -> list[str]:
    import pandas as pd
    ts = pd.Timestamp(month)
    return [(ts - pd.DateOffset(months=i)).strftime("%Y-%m") for i in range(1, n + 1)]


def required_realized_da_months(
    auction_month: str, ptypes: list[str], ctypes: list[str],
) -> set[tuple[str, str]]:
    needed: set[tuple[str, str]] = set()
    for ptype in ptypes:
        if not has_period_type(auction_month, ptype):
            continue
        train_months = collect_usable_months(auction_month, ptype, n_months=8)
        for tm in train_months:
            dm = delivery_month(tm, ptype)
            for ct in ctypes:
                needed.add((dm, ct))
            bf_cutoff = _prev_month(tm)
            for bf_month in _months_before(bf_cutoff, _MAX_BF_LOOKBACK):
                for ct in ctypes:
                    needed.add((bf_month, ct))
        bf_cutoff = _prev_month(auction_month)
        for bf_month in _months_before(bf_cutoff, _MAX_BF_LOOKBACK):
            for ct in ctypes:
                needed.add((bf_month, ct))
    return needed


def ensure_realized_da_cache(
    auction_month: str, ptypes: list[str], ctypes: list[str],
) -> None:
    needed = required_realized_da_months(auction_month, ptypes, ctypes)
    missing = [(m, pt) for m, pt in sorted(needed) if not _cache_path(m, pt).exists()]

    if not missing:
        print(f"[cache] All {len(needed)} required DA files present")
        return

    print(f"[cache] {len(missing)}/{len(needed)} missing, fetching...")
    from ml.realized_da import fetch_and_cache_month

    fetched, skipped = 0, 0
    for month, peak_type in missing:
        try:
            out = fetch_and_cache_month(month, peak_type, cache_dir=REALIZED_DA_CACHE)
            fetched += 1 if out.exists() else 0
        except Exception as e:
            skipped += 1
            print(f"[cache]   SKIP {month}/{peak_type}: {e}")

    print(f"[cache] Fetched {fetched}, skipped {skipped}")
