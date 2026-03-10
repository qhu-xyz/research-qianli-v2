# ml/spice6_loader.py
"""Load spice6 density features for PJM.

Adapted from MISO spice6_loader.py. Key PJM differences:
  - Base path points to PJM density directory
  - PJM always uses new schema (score.parquet with single 'score' column)
  - No legacy score_df.parquet files exist for PJM
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

    Returns
    -------
    pl.DataFrame
        Columns: constraint_id, flow_direction, prob_exceed_110, prob_exceed_100,
        prob_exceed_90, prob_exceed_85, prob_exceed_80, constraint_limit.
    """
    market_month = _delivery_month(auction_month, period_type)
    logger.info("spice6 density: auction=%s ptype=%s market_month=%s",
                auction_month, period_type, market_month)
    market_round = "1"
    base = (
        Path(SPICE6_DENSITY_BASE)
        / f"auction_month={auction_month}"
        / f"market_month={market_month}"
        / f"market_round={market_round}"
    )

    if not base.exists():
        return pl.DataFrame()

    score_dfs = []
    limit_dfs = []

    for od_dir in sorted(base.iterdir()):
        if not od_dir.name.startswith("outage_date="):
            continue
        score_path = od_dir / "score.parquet"
        limit_path = od_dir / "limit.parquet"
        if score_path.exists():
            score_dfs.append(pl.read_parquet(score_path))
        if limit_path.exists():
            limit_dfs.append(pl.read_parquet(limit_path))

    if not score_dfs:
        return pl.DataFrame()

    all_scores = pl.concat(score_dfs)

    density = all_scores.group_by(["constraint_id", "flow_direction"]).agg([
        pl.col("score").mean().alias("prob_exceed_110"),
    ])
    for col in ["prob_exceed_100", "prob_exceed_90", "prob_exceed_85", "prob_exceed_80"]:
        density = density.with_columns(pl.lit(0.0).alias(col))

    if limit_dfs:
        all_limits = pl.concat(limit_dfs)
        limits = all_limits.group_by("constraint_id").agg(
            pl.col("limit").mean().alias("constraint_limit")
        )
        density = density.join(limits, on="constraint_id", how="left")
    else:
        density = density.with_columns(pl.lit(0.0).alias("constraint_limit"))

    return density
