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

# Maximum BF lookback window used in inference
_MAX_BF_LOOKBACK = 15


def _cache_path(month: str, peak_type: str) -> Path:
    if peak_type == "onpeak":
        return Path(REALIZED_DA_CACHE) / f"{month}.parquet"
    return Path(REALIZED_DA_CACHE) / f"{month}_{peak_type}.parquet"


def _prev_month(m: str) -> str:
    import pandas as pd
    ts = pd.Timestamp(m)
    return (ts - pd.DateOffset(months=1)).strftime("%Y-%m")


def _months_before(month: str, n: int) -> list[str]:
    """Return the n months strictly before `month`, newest first."""
    import pandas as pd
    ts = pd.Timestamp(month)
    return [
        (ts - pd.DateOffset(months=i)).strftime("%Y-%m")
        for i in range(1, n + 1)
    ]


def required_realized_da_months(
    auction_month: str,
    ptypes: list[str],
    ctypes: list[str],
) -> set[tuple[str, str]]:
    """Compute all (month, peak_type) pairs needed for training + BF features.

    Two sources of requirements:
    1. Training labels: realized DA for delivery_month of each training month.
    2. BF lookback: binding_freq_15 needs 15 months of realized DA before
       prev_month(auction_month). Also need BF months for each training month.
    """
    needed: set[tuple[str, str]] = set()
    for ptype in ptypes:
        if not has_period_type(auction_month, ptype):
            continue
        train_months = collect_usable_months(auction_month, ptype, n_months=8)

        for tm in train_months:
            # Labels: delivery month GT
            dm = delivery_month(tm, ptype)
            for ct in ctypes:
                needed.add((dm, ct))

            # BF features for each training month: months before prev_month(tm)
            bf_cutoff = _prev_month(tm)
            for bf_month in _months_before(bf_cutoff, _MAX_BF_LOOKBACK):
                for ct in ctypes:
                    needed.add((bf_month, ct))

        # BF features for inference month itself
        bf_cutoff = _prev_month(auction_month)
        for bf_month in _months_before(bf_cutoff, _MAX_BF_LOOKBACK):
            for ct in ctypes:
                needed.add((bf_month, ct))

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

    print(f"[cache] {len(missing)}/{len(needed)} missing realized DA files, fetching...")
    from ml.realized_da import fetch_and_cache_month

    fetched, skipped = 0, 0
    for month, peak_type in missing:
        try:
            out = fetch_and_cache_month(month, peak_type, cache_dir=REALIZED_DA_CACHE)
            if out.exists() and out.stat().st_size > 0:
                fetched += 1
            else:
                skipped += 1
                print(f"[cache]   WARNING: empty result for {month}/{peak_type}")
        except Exception as e:
            skipped += 1
            print(f"[cache]   WARNING: failed to fetch {month}/{peak_type}: {e}")

    print(f"[cache] Fetched {fetched}, skipped {skipped} (may be too old for DA data)")

    # Re-check: any RECENT months still missing are fatal (old gaps like 2017-2018 are OK)
    still_missing = [(m, pt) for m, pt in sorted(needed) if not _cache_path(m, pt).exists()]
    recent_missing = [(m, pt) for m, pt in still_missing if m >= "2019-02"]
    if recent_missing:
        raise RuntimeError(
            f"[cache] {len(recent_missing)} recent realized DA files still missing after fetch — "
            f"cannot generate reliable BF features. Missing: "
            f"{', '.join(f'{m}/{pt}' for m, pt in recent_missing[:5])}"
            f"{'...' if len(recent_missing) > 5 else ''}"
        )
