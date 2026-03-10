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

# Floor for realized DA months. The cache contains backfilled data from 2017-04,
# but using pre-2019-06 data HURTS VC@20: early years having BF=0 (NaN-as-missing)
# acts as implicit regularization, forcing feature diversity. See v10e/v12 experiments.
DA_FLOOR_MONTH = "2019-06"

# Default binding frequency windows
BF_WINDOWS = [1, 3, 6, 12, 15, 24, 36, 48]

# In-memory cache for bridge tables: (auction_month, period_type) -> DataFrame
_BRIDGE_CACHE: dict[tuple[str, str], pl.DataFrame] = {}

# In-memory cache for binding sets: (auction_month, period_type, cutoff, floor) -> dict
_BINDING_SETS_CACHE: dict[tuple[str, str, str, str], dict[str, set[str]]] = {}


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
    floor_month: str | None = None,
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
    effective_floor = floor_month if floor_month is not None else DA_FLOOR_MONTH
    cache_key = (auction_month, period_type, cutoff_month, effective_floor)
    if cache_key in _BINDING_SETS_CACHE:
        return _BINDING_SETS_CACHE[cache_key]

    bridge = _load_bridge(auction_month, period_type)
    binding_sets: dict[str, set[str]] = {}

    for f in sorted(REALIZED_DA_CACHE.glob("*.parquet")):
        if "_offpeak" in f.stem or "_partial_" in f.stem:
            continue
        month = f.stem
        if month >= cutoff_month or month < effective_floor:
            continue

        da = pl.read_parquet(str(f)).filter(pl.col("realized_sp") > 0)
        if len(da) == 0:
            binding_sets[month] = set()
            continue

        mapped = da.join(bridge, on="constraint_id", how="inner")
        binding_sets[month] = set(mapped["branch_name"].to_list())

    _BINDING_SETS_CACHE[cache_key] = binding_sets
    return binding_sets


def build_monthly_binding_sets_offpeak(
    auction_month: str,
    period_type: str,
    cutoff_month: str,
    floor_month: str | None = None,
) -> dict[str, set[str]]:
    """Like build_monthly_binding_sets but uses offpeak realized DA files.

    Offpeak binding is a complementary signal: constraints binding in offpeak
    hours may also bind in onpeak, or the offpeak pattern may predict future
    onpeak binding.
    """
    effective_floor = floor_month if floor_month is not None else DA_FLOOR_MONTH
    cache_key = (auction_month, period_type, cutoff_month, effective_floor + "_offpeak")
    if cache_key in _BINDING_SETS_CACHE:
        return _BINDING_SETS_CACHE[cache_key]

    bridge = _load_bridge(auction_month, period_type)
    binding_sets: dict[str, set[str]] = {}

    for f in sorted(REALIZED_DA_CACHE.glob("*_offpeak.parquet")):
        if "_partial_" in f.stem:
            continue
        month = f.stem.replace("_offpeak", "")
        if month >= cutoff_month or month < effective_floor:
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


def compute_binding_freq_decayed(
    branch_names: list[str],
    binding_sets: dict[str, set[str]],
    cutoff_month: str,
    window: int,
    half_life: float = 12.0,
) -> np.ndarray:
    """Compute exponentially-decayed binding frequency.

    Like compute_binding_freq but weights recent months more heavily.
    Uses exp(-ln(2)/half_life * months_ago), so a month `half_life` months
    ago gets half the weight of the most recent month.

    This allows using backfilled data (2017+) without letting stale grid
    topology dominate: old months naturally contribute less.
    """
    available = sorted(m for m in binding_sets.keys() if m < cutoff_month)
    lookback = available[-window:] if len(available) >= window else available
    n = len(lookback)
    if n == 0:
        return np.zeros(len(branch_names), dtype=np.float64)

    decay_rate = np.log(2) / half_life
    freq = np.zeros(len(branch_names), dtype=np.float64)
    total_weight = 0.0

    for idx, m in enumerate(lookback):
        months_ago = n - idx  # most recent = 1, oldest = n
        weight = np.exp(-decay_rate * months_ago)
        total_weight += weight
        s = binding_sets.get(m, set())
        for i, bn in enumerate(branch_names):
            if bn in s:
                freq[i] += weight

    return freq / total_weight if total_weight > 0 else freq


def enrich_with_binding_freq(
    df: pl.DataFrame,
    auction_month: str,
    period_type: str,
    windows: list[int] | None = None,
    floor_month: str | None = None,
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
        Lookback windows. Default: BF_WINDOWS.
    floor_month : str | None
        Override DA_FLOOR_MONTH. Pass "2017-04" to use backfill data.

    Returns
    -------
    pl.DataFrame with bf_N columns added.
    """
    if windows is None:
        windows = list(BF_WINDOWS)

    assert auction_month.endswith("-06"), (
        f"auction_month must be 'YYYY-06' for annual auctions, got '{auction_month}'. "
        f"The annual auction cutoff is derived as YYYY-04 from this value."
    )

    py = int(auction_month.split("-")[0])
    cutoff = f"{py}-04"

    binding_sets = build_monthly_binding_sets(auction_month, period_type, cutoff, floor_month=floor_month)
    branch_names = df["branch_name"].to_list()

    available = sorted(m for m in binding_sets.keys() if m < cutoff)
    n_avail = len(available)
    df = df.with_columns(pl.lit(float(n_avail)).alias("bf_months_avail"))

    for w in windows:
        col_name = f"bf_{w}"
        freq = compute_binding_freq(branch_names, binding_sets, cutoff, w)
        df = df.with_columns(pl.Series(col_name, freq))

    return df


def enrich_with_binding_freq_decayed(
    df: pl.DataFrame,
    auction_month: str,
    period_type: str,
    windows: list[int] | None = None,
    half_life: float = 12.0,
    floor_month: str | None = None,
) -> pl.DataFrame:
    """Add exponentially-decayed binding frequency columns.

    Like enrich_with_binding_freq but uses time-decayed counts.
    Columns named bfd_{window} to distinguish from raw bf_{window}.
    Also adds raw bf columns for comparison.
    """
    if windows is None:
        windows = list(BF_WINDOWS)

    assert auction_month.endswith("-06")

    py = int(auction_month.split("-")[0])
    cutoff = f"{py}-04"

    # Use backfill floor for decayed features
    effective_floor = floor_month if floor_month is not None else "2017-04"
    binding_sets = build_monthly_binding_sets(auction_month, period_type, cutoff, floor_month=effective_floor)
    branch_names = df["branch_name"].to_list()

    available = sorted(m for m in binding_sets.keys() if m < cutoff)
    n_avail = len(available)
    df = df.with_columns(pl.lit(float(n_avail)).alias("bf_months_avail"))

    for w in windows:
        # Decayed version
        freq_d = compute_binding_freq_decayed(branch_names, binding_sets, cutoff, w, half_life=half_life)
        df = df.with_columns(pl.Series(f"bfd_{w}", freq_d))
        # Raw version (for comparison)
        freq_r = compute_binding_freq(branch_names, binding_sets, cutoff, w)
        df = df.with_columns(pl.Series(f"bf_{w}", freq_r))

    return df


def build_partial_binding_map(
    auction_month: str,
    period_type: str,
    n_days: int = 12,
    peak_type: str = "onpeak",
) -> dict[str, float]:
    """Load partial-month DA cache and map to branch_name via bridge table.

    Returns {branch_name: days_binding / days_total} for the April partial
    month corresponding to the given planning year.

    Uses max(days_binding) when multiple constraint_ids map to the same
    branch_name to avoid fan-out inflation.
    """
    py = int(auction_month.split("-")[0])
    partial_month = f"{py:04d}-04"

    suffix = "" if peak_type == "onpeak" else f"_{peak_type}"
    partial_file = REALIZED_DA_CACHE / f"{partial_month}_partial_d{n_days}{suffix}.parquet"

    if not partial_file.exists():
        return {}

    da = pl.read_parquet(str(partial_file))
    if len(da) == 0:
        return {}

    da = da.filter(pl.col("days_binding") > 0)
    if len(da) == 0:
        return {}

    bridge = _load_bridge(auction_month, period_type)
    mapped = da.join(bridge, on="constraint_id", how="inner")
    if len(mapped) == 0:
        return {}

    agg = mapped.group_by("branch_name").agg([
        pl.col("days_binding").max(),
        pl.col("days_total").first(),
    ])

    return dict(zip(
        agg["branch_name"].to_list(),
        (agg["days_binding"] / agg["days_total"]).to_list(),
    ))


def compute_partial_bf(
    branch_names: list[str],
    auction_month: str,
    period_type: str,
    n_days: int = 12,
    peak_type: str = "onpeak",
) -> np.ndarray:
    """Compute partial-month BF for first n_days of April (single year).

    Returns array of shape (len(branch_names),) with values in [0, 1].
    """
    binding_map = build_partial_binding_map(
        auction_month, period_type, n_days, peak_type,
    )
    return np.array(
        [binding_map.get(bn, 0.0) for bn in branch_names],
        dtype=np.float64,
    )


def compute_multi_year_partial_bf(
    branch_names: list[str],
    auction_month: str,
    period_type: str,
    n_days: int = 12,
    peak_type: str = "onpeak",
    floor_year: int = 2020,
) -> np.ndarray:
    """Compute multi-year April partial BF: fraction of prior Aprils where branch bound.

    For eval PY YYYY-06, uses April partials from floor_year through YYYY-1.
    This aggregates across years to get much better coverage than single-year partial.

    Returns array of shape (len(branch_names),) with values in [0, 1].
    """
    py = int(auction_month.split("-")[0])
    years = list(range(floor_year, py))  # prior years only

    if not years:
        return np.zeros(len(branch_names), dtype=np.float64)

    # Count how many Aprils each branch bound in
    counts = np.zeros(len(branch_names), dtype=np.float64)
    n_years_with_data = 0

    bn_index = {bn: i for i, bn in enumerate(branch_names)}

    for y in years:
        am = f"{y:04d}-06"
        binding_map = build_partial_binding_map(am, period_type, n_days, peak_type)
        if not binding_map:
            continue
        n_years_with_data += 1
        for bn, val in binding_map.items():
            idx = bn_index.get(bn)
            if idx is not None and val > 0:
                counts[idx] += 1

    if n_years_with_data == 0:
        return np.zeros(len(branch_names), dtype=np.float64)

    return counts / n_years_with_data


def enrich_with_partial_bf(
    df: pl.DataFrame,
    auction_month: str,
    period_type: str,
    n_days: int = 12,
    include_offpeak: bool = True,
    floor_year: int = 2020,
) -> pl.DataFrame:
    """Add partial-month BF columns.

    Adds:
      bf_partial     — single-year current April partial (onpeak)
      bf_april       — multi-year April BF (onpeak)
      bfo_partial    — single-year current April partial (offpeak)
      bfo_april      — multi-year April BF (offpeak)
    """
    branch_names = df["branch_name"].to_list()

    # Onpeak
    freq = compute_partial_bf(branch_names, auction_month, period_type, n_days, "onpeak")
    df = df.with_columns(pl.Series("bf_partial", freq))

    freq_my = compute_multi_year_partial_bf(
        branch_names, auction_month, period_type, n_days, "onpeak", floor_year,
    )
    df = df.with_columns(pl.Series("bf_april", freq_my))

    if include_offpeak:
        freq_off = compute_partial_bf(branch_names, auction_month, period_type, n_days, "offpeak")
        df = df.with_columns(pl.Series("bfo_partial", freq_off))

        freq_off_my = compute_multi_year_partial_bf(
            branch_names, auction_month, period_type, n_days, "offpeak", floor_year,
        )
        df = df.with_columns(pl.Series("bfo_april", freq_off_my))

    return df


def enrich_with_offpeak_bf(
    df: pl.DataFrame,
    auction_month: str,
    period_type: str,
    windows: list[int] | None = None,
    floor_month: str | None = None,
) -> pl.DataFrame:
    """Add offpeak binding frequency columns (bfo_N).

    Offpeak binding is complementary to onpeak: constraints that bind
    in offpeak hours provide additional signal about structural congestion.
    """
    if windows is None:
        windows = [6, 12, 24]

    assert auction_month.endswith("-06")

    py = int(auction_month.split("-")[0])
    cutoff = f"{py}-04"

    binding_sets = build_monthly_binding_sets_offpeak(
        auction_month, period_type, cutoff, floor_month=floor_month,
    )
    branch_names = df["branch_name"].to_list()

    for w in windows:
        freq = compute_binding_freq(branch_names, binding_sets, cutoff, w)
        df = df.with_columns(pl.Series(f"bfo_{w}", freq))

    return df
