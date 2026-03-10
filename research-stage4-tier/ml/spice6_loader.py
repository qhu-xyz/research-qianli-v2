"""Load spice6 density features for a single auction month.

Aggregates exceedance probabilities across outage dates (mean) and joins
constraint_limit. Returns one row per (constraint_id, flow_direction).
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from ml.config import SPICE6_DENSITY_BASE


def load_spice6_density(
    auction_month: str,
    period_type: str = "f0",
) -> pl.DataFrame:
    """Load and aggregate spice6 density features for one month.

    Parameters
    ----------
    auction_month : str
        Month in YYYY-MM format.
    period_type : str
        "f0" uses market_round=1.

    Returns
    -------
    pl.DataFrame
        Columns: constraint_id, flow_direction, prob_exceed_110, prob_exceed_100,
        prob_exceed_90, prob_exceed_85, prob_exceed_80, constraint_limit.
    """
    market_round = "1" if period_type == "f0" else "1"
    base = (
        Path(SPICE6_DENSITY_BASE)
        / f"auction_month={auction_month}"
        / f"market_month={auction_month}"
        / f"market_round={market_round}"
    )

    if not base.exists():
        return pl.DataFrame()

    # Load score_df from all outage dates
    score_dfs = []
    limit_dfs = []
    for od_dir in sorted(base.iterdir()):
        if not od_dir.name.startswith("outage_date="):
            continue
        score_path = od_dir / "score_df.parquet"
        limit_path = od_dir / "limit.parquet"
        if score_path.exists():
            score_dfs.append(pl.read_parquet(score_path))
        if limit_path.exists():
            limit_dfs.append(pl.read_parquet(limit_path))

    if not score_dfs:
        return pl.DataFrame()

    all_scores = pl.concat(score_dfs)

    # Aggregate across outage dates: mean per (constraint_id, flow_direction)
    density = all_scores.group_by(["constraint_id", "flow_direction"]).agg([
        pl.col("110").mean().alias("prob_exceed_110"),
        pl.col("100").mean().alias("prob_exceed_100"),
        pl.col("90").mean().alias("prob_exceed_90"),
        pl.col("85").mean().alias("prob_exceed_85"),
        pl.col("80").mean().alias("prob_exceed_80"),
    ])

    # Aggregate constraint_limit across outage dates
    if limit_dfs:
        all_limits = pl.concat(limit_dfs)
        limits = all_limits.group_by("constraint_id").agg(
            pl.col("limit").mean().alias("constraint_limit")
        )
        density = density.join(limits, on="constraint_id", how="left")
    else:
        density = density.with_columns(pl.lit(0.0).alias("constraint_limit"))

    return density
