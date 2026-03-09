"""Ground truth: realized DA constraint shadow prices.

Fetches realized DA shadow prices for a target quarter via
MisoApTools.tools.get_da_shadow_by_peaktype() and maps them to V6.1
constraint universe via MISO_SPICE_CONSTRAINT_INFO bridge table.

Mapping chain:
  DA shadow constraint_id (numeric MISO IDs)
  -> MISO_SPICE_CONSTRAINT_INFO constraint_id -> branch_name
  -> V6.1 branch_name

REQUIRES RAY. Call init_ray() before using any function here.
Results are cached to parquet to avoid repeated Ray calls.
"""
from __future__ import annotations

import resource
from pathlib import Path

import pandas as pd
import polars as pl

from ml.config import get_market_months, SPICE_DATA_BASE


# Use absolute path to avoid cwd issues
CACHE_DIR = Path(__file__).resolve().parent.parent / "cache" / "ground_truth"

# Cached partition-filtered bridge tables: (auction_month, period_type) -> DataFrame
_CID_TO_BRANCH_CACHE: dict[tuple[str, str], pl.DataFrame] = {}


def mem_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _load_cid_to_branch(
    auction_month: str,
    period_type: str,
) -> pl.DataFrame:
    """Load constraint_id -> branch_name mapping filtered to target partition.

    Filters MISO_SPICE_CONSTRAINT_INFO to (auction_type='annual', auction_month,
    period_type, class_type='onpeak') to avoid cross-partition fan-out where
    a constraint_id maps to different branch_names in different partitions.

    Returns unique (constraint_id, branch_name) pairs for the target partition.
    """
    cache_key = (auction_month, period_type)
    if cache_key in _CID_TO_BRANCH_CACHE:
        return _CID_TO_BRANCH_CACHE[cache_key]
    info_path = Path(SPICE_DATA_BASE) / "MISO_SPICE_CONSTRAINT_INFO.parquet"
    info = (
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
    _CID_TO_BRANCH_CACHE[cache_key] = info
    print(f"[ground_truth] loaded constraint_info mapping ({auction_month}/{period_type}): {len(info)} entries")
    return info


def fetch_realized_da_quarter(
    planning_year: str,
    aq_round: str,
    peak_type: str = "onpeak",
) -> pl.DataFrame:
    """Fetch realized DA shadow prices for a quarter and map to branch_name.

    Returns DataFrame with columns: branch_name, realized_shadow_price.
    """
    from pbase.analysis.tools.all_positions import MisoApTools

    market_months = get_market_months(planning_year, aq_round)
    aptools = MisoApTools()

    all_da = []
    for mm in market_months:
        year, month = mm.split("-")
        st = pd.Timestamp(f"{year}-{month}-01", tz="US/Central")
        et = st + pd.offsets.MonthBegin(1)
        print(f"[ground_truth] Fetching DA shadow: {st.date()} to {et.date()}, mem={mem_mb():.0f} MB")

        da_shadow = aptools.tools.get_da_shadow_by_peaktype(
            st=st, et_ex=et, peak_type=peak_type,
        )
        if da_shadow is not None and len(da_shadow) > 0:
            all_da.append(da_shadow)

    if not all_da:
        return pl.DataFrame({"branch_name": [], "realized_shadow_price": []})

    da_pd = pd.concat(all_da)
    da_pl = pl.from_pandas(da_pd.reset_index())

    # Step 1: Aggregate sum(abs(shadow_price)) per DA constraint_id
    per_cid = da_pl.group_by("constraint_id").agg(
        pl.col("shadow_price").abs().sum().alias("realized_shadow_price")
    ).filter(pl.col("realized_shadow_price") > 0)
    per_cid = per_cid.with_columns(pl.col("constraint_id").cast(pl.Utf8))

    # Step 2: Map DA constraint_id -> branch_name via partition-filtered constraint_info
    cid_to_branch = _load_cid_to_branch(planning_year, aq_round)
    mapped = per_cid.join(cid_to_branch, on="constraint_id", how="left")
    mapped = mapped.filter(pl.col("branch_name").is_not_null())

    # Step 3: Aggregate per branch_name (multiple DA constraint_ids can map to same branch)
    per_branch = mapped.group_by("branch_name").agg(
        pl.col("realized_shadow_price").sum()
    )

    n_unmapped = per_cid.join(cid_to_branch, on="constraint_id", how="anti").height
    if n_unmapped > 0:
        print(f"[ground_truth] {n_unmapped}/{len(per_cid)} DA constraints unmapped")

    return per_branch


def get_ground_truth(
    planning_year: str,
    aq_round: str,
    v61_df: pl.DataFrame,
    cache: bool = True,
) -> pl.DataFrame:
    """Get realized DA shadow prices aligned with V6.1 constraint universe.

    Parameters
    ----------
    planning_year, aq_round : str
        Target quarter.
    v61_df : pl.DataFrame
        V6.1 data for this quarter (defines constraint universe).
    cache : bool
        If True, cache results to parquet.

    Returns
    -------
    pl.DataFrame
        V6.1 data with 'realized_shadow_price' column added.
        Constraints that didn't bind get realized_shadow_price = 0.0.
    """
    cache_path = CACHE_DIR / f"{planning_year}_{aq_round}.parquet"

    if cache and cache_path.exists():
        realized = pl.read_parquet(str(cache_path))
        print(f"[ground_truth] loaded from cache: {cache_path} ({len(realized)} binding)")
    else:
        realized = fetch_realized_da_quarter(planning_year, aq_round)
        if cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            realized.write_parquet(str(cache_path))
            print(f"[ground_truth] cached to {cache_path}")

    # Drop existing realized_shadow_price if present (from prior join)
    if "realized_shadow_price" in v61_df.columns:
        v61_df = v61_df.drop("realized_shadow_price")

    # Join: V6.1 constraint universe LEFT JOIN realized DA on branch_name
    result = v61_df.join(realized, on="branch_name", how="left")
    result = result.with_columns(
        pl.col("realized_shadow_price").fill_null(0.0)
    )

    n_binding = len(result.filter(pl.col("realized_shadow_price") > 0))
    print(f"[ground_truth] {planning_year}/{aq_round}: {n_binding}/{len(result)} constraints binding")

    return result
