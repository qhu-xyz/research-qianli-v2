"""Load spice6 density features for a single auction month.

Aggregates exceedance probabilities across outage dates (mean) and joins
constraint_limit. Returns one row per (constraint_id, flow_direction).
"""
from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from ml.config import SPICE6_DENSITY_BASE, delivery_month as _delivery_month

logger = logging.getLogger(__name__)


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
        Period type (f0, f1, etc.). Determines market_month = delivery_month.

    Returns
    -------
    pl.DataFrame
        Columns: constraint_id, flow_direction, prob_exceed_110, prob_exceed_100,
        prob_exceed_90, prob_exceed_85, prob_exceed_80, constraint_limit.
    """
    market_month = _delivery_month(auction_month, period_type)
    logger.info("spice6 density: auction=%s ptype=%s market_month=%s", auction_month, period_type, market_month)
    market_round = "1"
    base = (
        Path(SPICE6_DENSITY_BASE)
        / f"auction_month={auction_month}"
        / f"market_month={market_month}"
        / f"market_round={market_round}"
    )

    if not base.exists():
        return pl.DataFrame()

    # Load score data from all outage dates.
    # Two schemas exist:
    #   - score_df.parquet (through 2025-12): columns 110, 105, 100, ..., 60
    #   - score.parquet (from 2026-01+): single "score" column (≈ prob_exceed_110, Spearman=0.9994)
    score_dfs = []
    limit_dfs = []
    use_legacy_schema = None  # will detect from first file found

    for od_dir in sorted(base.iterdir()):
        if not od_dir.name.startswith("outage_date="):
            continue
        score_path = od_dir / "score_df.parquet"
        if score_path.exists():
            if use_legacy_schema is None:
                use_legacy_schema = True
        else:
            score_path = od_dir / "score.parquet"
            if use_legacy_schema is None:
                use_legacy_schema = False
        limit_path = od_dir / "limit.parquet"
        if score_path.exists():
            score_dfs.append(pl.read_parquet(score_path))
        if limit_path.exists():
            limit_dfs.append(pl.read_parquet(limit_path))

    if not score_dfs:
        return pl.DataFrame()

    all_scores = pl.concat(score_dfs)

    # Aggregate across outage dates: mean per (constraint_id, flow_direction)
    if use_legacy_schema:
        # Legacy: per-threshold exceedance columns
        density = all_scores.group_by(["constraint_id", "flow_direction"]).agg([
            pl.col("110").mean().alias("prob_exceed_110"),
            pl.col("100").mean().alias("prob_exceed_100"),
            pl.col("90").mean().alias("prob_exceed_90"),
            pl.col("85").mean().alias("prob_exceed_85"),
            pl.col("80").mean().alias("prob_exceed_80"),
        ])
    else:
        # New schema: single score ≈ prob_exceed_110 (Spearman=0.9994)
        density = all_scores.group_by(["constraint_id", "flow_direction"]).agg([
            pl.col("score").mean().alias("prob_exceed_110"),
        ])
        # Fill missing threshold columns with 0 (not used by v10e-lag1 features)
        for col in ["prob_exceed_100", "prob_exceed_90", "prob_exceed_85", "prob_exceed_80"]:
            density = density.with_columns(pl.lit(0.0).alias(col))

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
