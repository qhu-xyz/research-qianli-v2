"""Data loading for annual LTR ranking pipeline.

Loads V6.1 annual signal and enriches with spice6 density features
aggregated across the 3 market months in each quarter.

Data sources:
- V6.1 signal: parquet files at V61_SIGNAL_BASE/{year}/{aq}/onpeak
- Spice6 density: MISO_SPICE_DENSITY_DISTRIBUTION.parquet (has exceedance prob columns)
- Constraint limits: MISO_SPICE_CONSTRAINT_LIMIT.parquet
"""
from __future__ import annotations

import resource
from pathlib import Path

import polars as pl

from ml.config import (
    V61_SIGNAL_BASE,
    SPICE_DATA_BASE,
    get_market_months,
)


def mem_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def load_v61_group(planning_year: str, aq_round: str) -> pl.DataFrame:
    """Load V6.1 annual signal for one (planning_year, aq_round).

    Returns V6.1 data with query_group column added.
    """
    path = Path(V61_SIGNAL_BASE) / planning_year / aq_round / "onpeak"
    if not path.exists():
        raise FileNotFoundError(f"V6.1 data not found: {path}")
    df = pl.read_parquet(str(path))
    if "__index_level_0__" in df.columns:
        df = df.drop("__index_level_0__")

    query_group = f"{planning_year}/{aq_round}"
    df = df.with_columns(pl.lit(query_group).alias("query_group"))
    return df


# Positive exceedance columns we want from the density distribution
_EXCEED_COLS = ["80", "85", "90", "100", "110"]


def load_spice6_density_annual(
    planning_year: str,
    aq_round: str,
) -> pl.DataFrame:
    """Load and aggregate spice6 density features for a quarter (3 months).

    Uses MISO_SPICE_DENSITY_DISTRIBUTION.parquet which has exceedance probability
    columns ("80", "85", "90", "100", "110"). No flow_direction in this table —
    join is on constraint_id only.

    Aggregates mean across all outage_dates and all 3 market_months.
    Returns one row per constraint_id.
    """
    market_months = get_market_months(planning_year, aq_round)
    density_dist_path = Path(SPICE_DATA_BASE) / "MISO_SPICE_DENSITY_DISTRIBUTION.parquet"
    constraint_limit_path = Path(SPICE_DATA_BASE) / "MISO_SPICE_CONSTRAINT_LIMIT.parquet"

    # Load density distribution filtered to annual and this quarter's months
    dist = (
        pl.scan_parquet(str(density_dist_path))
        .filter(
            (pl.col("auction_type") == "annual")
            & (pl.col("auction_month") == planning_year)
            & (pl.col("market_month").is_in(market_months))
        )
        .select(["constraint_id"] + _EXCEED_COLS)
        .collect()
    )

    if len(dist) == 0:
        return pl.DataFrame()

    # Aggregate: mean exceedance prob per constraint across all outage_dates and months
    density = dist.group_by("constraint_id").agg([
        pl.col(c).mean().alias(f"prob_exceed_{c}") for c in _EXCEED_COLS
    ])

    # Load constraint limits
    if constraint_limit_path.exists():
        limits = (
            pl.scan_parquet(str(constraint_limit_path))
            .filter(
                (pl.col("auction_type") == "annual")
                & (pl.col("auction_month") == planning_year)
                & (pl.col("market_month").is_in(market_months))
            )
            .collect()
        )
        if len(limits) > 0:
            limit_agg = limits.group_by("constraint_id").agg(
                pl.col("limit").mean().alias("constraint_limit")
            )
            density = density.join(limit_agg, on="constraint_id", how="left")

    if "constraint_limit" not in density.columns:
        density = density.with_columns(pl.lit(0.0).alias("constraint_limit"))

    return density


_CACHE_DIR = Path(__file__).resolve().parent.parent / "cache" / "enriched"
_BF_CACHE_DIR = Path(__file__).resolve().parent.parent / "cache" / "enriched_bf"


def load_v61_enriched(planning_year: str, aq_round: str) -> pl.DataFrame:
    """Load V6.1 data enriched with spice6 density features.

    Caches to local parquet after first load (avoids re-scanning 18 GB
    density distribution on NFS).
    """
    cache_path = _CACHE_DIR / f"{planning_year}_{aq_round}.parquet"
    if cache_path.exists():
        df = pl.read_parquet(str(cache_path))
        n_matched = len(df.filter(pl.col("prob_exceed_110") > 0))
        print(f"[data_loader] loaded from cache: {cache_path.name} ({n_matched}/{len(df)} spice6)")
        return df

    df = load_v61_group(planning_year, aq_round)

    spice6 = load_spice6_density_annual(planning_year, aq_round)
    if len(spice6) > 0:
        # Join on constraint_id only (density distribution has no flow_direction)
        df = df.join(spice6, on="constraint_id", how="left")
        spice6_cols = [c for c in spice6.columns if c != "constraint_id"]
        df = df.with_columns([pl.col(c).fill_null(0.0) for c in spice6_cols])
        n_matched = len(df.filter(pl.col("prob_exceed_110") > 0))
        print(f"[data_loader] spice6 enrichment: {n_matched}/{len(df)} matched")
    else:
        print(f"[data_loader] WARNING: no spice6 data for {planning_year}/{aq_round}")
        for col in ["prob_exceed_110", "prob_exceed_100", "prob_exceed_90",
                     "prob_exceed_85", "prob_exceed_80", "constraint_limit"]:
            df = df.with_columns(pl.lit(0.0).alias(col))

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.write_parquet(str(cache_path))
    print(f"[data_loader] cached to {cache_path.name}")
    return df


def load_v61_enriched_bf(planning_year: str, aq_round: str) -> pl.DataFrame:
    """Load V6.1 data enriched with spice6 density AND binding frequency features.

    Caches separately from non-BF enriched data.
    """
    cache_path = _BF_CACHE_DIR / f"{planning_year}_{aq_round}.parquet"
    if cache_path.exists():
        df = pl.read_parquet(str(cache_path))
        print(f"[data_loader] loaded from BF cache: {cache_path.name}")
        return df

    # Start from existing enriched data (has spice6 features)
    df = load_v61_enriched(planning_year, aq_round)

    # Add binding frequency features
    from ml.binding_freq import enrich_with_binding_freq

    df = enrich_with_binding_freq(df, planning_year, aq_round)
    n_bf = int((df["bf_12"] > 0).sum())
    print(f"[data_loader] bf enrichment: {n_bf}/{len(df)} with bf_12 > 0")

    _BF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.write_parquet(str(cache_path))
    print(f"[data_loader] BF cached to {cache_path.name}")
    return df


def load_multiple_groups(groups: list[str]) -> pl.DataFrame:
    """Load and concatenate multiple (planning_year/aq_round) groups."""
    dfs = []
    for group_id in groups:
        planning_year, aq_round = group_id.split("/")
        try:
            df = load_v61_enriched(planning_year, aq_round)
            dfs.append(df)
            print(f"[data_loader] loaded {group_id}: {len(df)} rows, mem={mem_mb():.0f} MB")
        except FileNotFoundError:
            print(f"[data_loader] WARNING: skipping {group_id} (not found)")

    if not dfs:
        raise ValueError(f"No data found for groups: {groups}")
    return pl.concat(dfs, how="diagonal")
