# ml/spice6_loader.py
"""Load spice6 features for PJM.

Data pipeline clarification (2026-03-11):
  - V6.2B signal already contains ori_mean (= density score = prob_exceed_110)
    and mix_mean. The old load_spice6_density() was redundant — it re-loaded
    the same density score that V6.2B already had embedded.
  - constraint_limit (from limit.parquet) is NOT in V6.2B, so we still load it.
  - ml_pred provides binding_probability, predicted_shadow_price, hist_da,
    prob_exceed_100 — none of these are in V6.2B.

PJM density has a single 'score' column (prob of exceeding 110% line rating).
There are no prob_exceed_80/85/90/100 columns in density — those were
hardcoded to 0.0 in the old loader, which caused a column collision with
ml_pred's prob_exceed_100 (creating a prob_exceed_100_right column).
"""
from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from ml.config import SPICE6_DENSITY_BASE, delivery_month as _delivery_month

logger = logging.getLogger(__name__)


def load_constraint_limits(
    auction_month: str,
    period_type: str = "f0",
) -> pl.DataFrame:
    """Load constraint_limit from spice6 density limit.parquet files.

    Returns
    -------
    pl.DataFrame
        Columns: constraint_id, constraint_limit.
    """
    market_month = _delivery_month(auction_month, period_type)
    market_round = "1"
    base = (
        Path(SPICE6_DENSITY_BASE)
        / f"auction_month={auction_month}"
        / f"market_month={market_month}"
        / f"market_round={market_round}"
    )

    if not base.exists():
        return pl.DataFrame()

    limit_dfs = []
    for od_dir in sorted(base.iterdir()):
        if not od_dir.name.startswith("outage_date="):
            continue
        limit_path = od_dir / "limit.parquet"
        if limit_path.exists():
            limit_dfs.append(pl.read_parquet(limit_path))

    if not limit_dfs:
        return pl.DataFrame()

    all_limits = pl.concat(limit_dfs)
    limits = all_limits.group_by("constraint_id").agg(
        pl.col("limit").mean().alias("constraint_limit")
    )
    return limits


def load_spice6_mlpred(
    auction_month: str,
    period_type: str = "f0",
    class_type: str = "onpeak",
) -> pl.DataFrame:
    """Load spice6 ml_pred features for one month.

    Returns DataFrame with columns:
        constraint_id, flow_direction, binding_probability,
        predicted_shadow_price, hist_da, prob_exceed_100
    """
    from ml.config import SPICE6_MLPRED_BASE

    market_month = _delivery_month(auction_month, period_type)
    path = (
        Path(SPICE6_MLPRED_BASE)
        / f"auction_month={auction_month}"
        / f"market_month={market_month}"
        / f"class_type={class_type}"
        / "final_results.parquet"
    )

    if not path.exists():
        logger.warning("ml_pred not found: %s", path)
        return pl.DataFrame()

    # Safe columns only (per CLAUDE.md — never use actual_*, error, abs_error)
    safe_cols = [
        "constraint_id", "flow_direction",
        "binding_probability", "predicted_shadow_price",
        "hist_da", "prob_exceed_100",
    ]
    df = pl.read_parquet(str(path))
    keep = [c for c in safe_cols if c in df.columns]
    df = df.select(keep)

    return df
