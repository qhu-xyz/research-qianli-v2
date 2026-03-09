"""Realized DA cache management for V7.0 signal generation.

Provides preflight check to ensure all required realized DA months
are cached before training begins.

Imports from research-stage5-tier/ml/ for config and DA fetching.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure stage5 ml package is importable
_STAGE5 = Path(__file__).resolve().parent.parent.parent / "research-stage5-tier"
if str(_STAGE5) not in sys.path:
    sys.path.insert(0, str(_STAGE5))

from ml.config import collect_usable_months, delivery_month, has_period_type

REALIZED_DA_CACHE = os.environ.get(
    "REALIZED_DA_CACHE",
    "/opt/temp/qianli/spice_data/miso/realized_da",
)


def _cache_path(month: str, peak_type: str) -> Path:
    if peak_type == "onpeak":
        return Path(REALIZED_DA_CACHE) / f"{month}.parquet"
    return Path(REALIZED_DA_CACHE) / f"{month}_{peak_type}.parquet"


def required_realized_da_months(
    auction_month: str,
    ptypes: list[str],
    ctypes: list[str],
) -> set[tuple[str, str]]:
    """Compute all (month, peak_type) pairs needed for training.

    For each ML ptype, we need realized DA for delivery_month of each
    training month (ground truth labels).
    """
    needed: set[tuple[str, str]] = set()
    for ptype in ptypes:
        if not has_period_type(auction_month, ptype):
            continue
        train_months = collect_usable_months(auction_month, ptype, n_months=8)
        for tm in train_months:
            dm = delivery_month(tm, ptype)
            for ct in ctypes:
                needed.add((dm, ct))
    return needed


def ensure_realized_da_cache(
    auction_month: str,
    ptypes: list[str],
    ctypes: list[str],
) -> None:
    """Check all required realized DA months are cached; fetch missing ones."""
    needed = required_realized_da_months(auction_month, ptypes, ctypes)
    missing = [(m, pt) for m, pt in sorted(needed) if not _cache_path(m, pt).exists()]

    if not missing:
        print(f"[cache] All {len(needed)} required realized DA files present")
        return

    print(f"[cache] {len(missing)} missing realized DA files, fetching...")
    from ml.realized_da import fetch_and_cache_month

    for month, peak_type in missing:
        print(f"[cache]   fetching {month}/{peak_type}...")
        out = fetch_and_cache_month(month, peak_type, cache_dir=REALIZED_DA_CACHE)
        if not out.exists() or out.stat().st_size == 0:
            raise RuntimeError(f"Failed to fetch realized DA for {month}/{peak_type}")
        print(f"[cache]   done: {out}")

    print(f"[cache] All {len(needed)} realized DA files now present")
