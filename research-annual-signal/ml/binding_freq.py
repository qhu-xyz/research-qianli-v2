"""Monthly binding frequency features for annual constraints.

Maps realized DA shadow prices (keyed by constraint_id) to V6.1 annual
signal (keyed by branch_name) via MISO_SPICE_CONSTRAINT_INFO bridge table,
then computes multi-window binding frequency per branch.

Data flow:
  Realized DA (constraint_id, realized_sp)
    -> Bridge table (constraint_id -> branch_name), partition-filtered
    -> Monthly binding sets: {month: set(branch_name)}
    -> Binding frequency: count(months bound) / window_size
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from ml.config import SPICE_DATA_BASE

# Stage5-tier's realized DA cache — we read from it, don't write to it.
REALIZED_DA_CACHE = Path("/home/xyz/workspace/research-qianli-v2/research-stage5-tier/data/realized_da")

# Default binding frequency windows
BF_WINDOWS = [1, 3, 6, 12, 15, 24, 36, 48]

# In-memory cache for bridge tables: (auction_month, period_type) -> DataFrame
_BRIDGE_CACHE: dict[tuple[str, str], pl.DataFrame] = {}

# In-memory cache for binding sets: (auction_month, period_type) -> dict
_BINDING_SETS_CACHE: dict[tuple[str, str, str], dict[str, set[str]]] = {}


def _load_bridge(auction_month: str, period_type: str) -> pl.DataFrame:
    """Load constraint_id -> branch_name mapping from MISO_SPICE_CONSTRAINT_INFO.

    Partition-filtered to (auction_type='annual', auction_month, period_type,
    class_type='onpeak') to avoid fan-out where one constraint_id maps to
    different branch_names across partitions.

    Returns unique (constraint_id, branch_name) pairs.
    """
    cache_key = (auction_month, period_type)
    if cache_key in _BRIDGE_CACHE:
        return _BRIDGE_CACHE[cache_key]

    info_path = Path(SPICE_DATA_BASE) / "MISO_SPICE_CONSTRAINT_INFO.parquet"
    bridge = (
        pl.scan_parquet(str(info_path))
        .filter(
            (pl.col("auction_type") == "annual")
            & (pl.col("auction_month") == auction_month)
            & (pl.col("period_type") == period_type)
            & (pl.col("class_type") == "onpeak")
        )
        .select(["constraint_id", "branch_name"])
        .collect()
        .unique()
    )
    _BRIDGE_CACHE[cache_key] = bridge
    return bridge


def build_monthly_binding_sets(
    auction_month: str,
    period_type: str,
    cutoff_month: str,
) -> dict[str, set[str]]:
    """Build monthly binding sets: {month: set(branch_name that bound)}.

    Parameters
    ----------
    auction_month : str
        Planning year (e.g., "2024-06"). Used to select bridge table partition.
    period_type : str
        Quarter round (e.g., "aq1"). Used to select bridge table partition.
    cutoff_month : str
        Strict upper bound for months to include (e.g., "2024-04").
        Only months < cutoff_month are used.

    Returns
    -------
    dict mapping month string to set of branch_names that were binding.
    """
    cache_key = (auction_month, period_type, cutoff_month)
    if cache_key in _BINDING_SETS_CACHE:
        return _BINDING_SETS_CACHE[cache_key]

    bridge = _load_bridge(auction_month, period_type)
    binding_sets: dict[str, set[str]] = {}

    for f in sorted(REALIZED_DA_CACHE.glob("*.parquet")):
        if "_offpeak" in f.stem:
            continue
        month = f.stem
        if month >= cutoff_month:
            continue

        da = pl.read_parquet(str(f)).filter(pl.col("realized_sp") > 0)
        if len(da) == 0:
            binding_sets[month] = set()
            continue

        mapped = da.join(bridge, on="constraint_id", how="inner")
        binding_sets[month] = set(mapped["branch_name"].to_list())

    _BINDING_SETS_CACHE[cache_key] = binding_sets
    return binding_sets


def compute_binding_freq(
    branch_names: list[str],
    binding_sets: dict[str, set[str]],
    cutoff_month: str,
    window: int,
) -> np.ndarray:
    """Compute binding frequency for a list of branch_names.

    Parameters
    ----------
    branch_names : list[str]
        Constraint identifiers to compute frequency for.
    binding_sets : dict[str, set[str]]
        Output of build_monthly_binding_sets().
    cutoff_month : str
        Strict upper bound (same as used to build binding_sets).
    window : int
        Number of prior months to look back.

    Returns
    -------
    np.ndarray of shape (len(branch_names),) with values in [0, 1].
    When fewer than `window` months are available, uses all available months
    and divides by actual count. Returns 0 when no data exists (no history
    = no binding = 0 is a valid signal, better than NaN for tree models).
    """
    available = sorted(m for m in binding_sets.keys() if m < cutoff_month)
    lookback = available[-window:] if len(available) >= window else available
    n = len(lookback)
    if n == 0:
        return np.zeros(len(branch_names), dtype=np.float64)

    # Vectorize: build set of branch_names per lookback month, count membership
    freq = np.zeros(len(branch_names), dtype=np.float64)
    for m in lookback:
        s = binding_sets.get(m, set())
        for i, bn in enumerate(branch_names):
            if bn in s:
                freq[i] += 1
    return freq / n


def enrich_with_binding_freq(
    df: pl.DataFrame,
    auction_month: str,
    period_type: str,
    windows: list[int] | None = None,
) -> pl.DataFrame:
    """Add binding frequency columns to a V6.1 DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        V6.1 data with 'branch_name' column.
    auction_month : str
        Planning year (e.g., "2024-06"). Must be in "YYYY-06" format.
    period_type : str
        Quarter round (e.g., "aq1").
    windows : list[int]
        Lookback windows. Default: [1, 3, 6, 12, 24].

    Returns
    -------
    pl.DataFrame with bf_1, bf_3, bf_6, bf_12, bf_24 columns added.
    """
    if windows is None:
        windows = list(BF_WINDOWS)

    # Validate auction_month format
    assert auction_month.endswith("-06"), (
        f"auction_month must be 'YYYY-06' for annual auctions, got '{auction_month}'. "
        f"The annual auction cutoff is derived as YYYY-04 from this value."
    )

    # Cutoff: annual auction submitted ~April of planning year
    py = int(auction_month.split("-")[0])
    cutoff = f"{py}-04"

    binding_sets = build_monthly_binding_sets(auction_month, period_type, cutoff)
    branch_names = df["branch_name"].to_list()

    # How many months of realized DA are available (model can learn data quality)
    available = sorted(m for m in binding_sets.keys() if m < cutoff)
    n_avail = len(available)
    df = df.with_columns(pl.lit(float(n_avail)).alias("bf_months_avail"))

    for w in windows:
        col_name = f"bf_{w}"
        freq = compute_binding_freq(branch_names, binding_sets, cutoff, w)
        df = df.with_columns(pl.Series(col_name, freq))

    return df
