"""CID-level deviation profile features from raw density exceedance data.

Rebuilds V4.4-style flow deviation signal from our density distribution parquet.
The raw density bins are exceedance probabilities: P(flow >= X% of limit).

Architecture: "anchor CID + runner-up + concentration"
  - Pick ONE anchor CID per branch (highest dev score) and carry its full profile
  - Separately expose runner-up CID's score and the gap
  - Count how many CIDs have meaningful signal
  - Never mix bins from different CIDs (avoids synthetic Frankenstein profiles)
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import polars as pl

from ml.markets.miso.config import DENSITY_PATH, get_market_months
from ml.markets.miso.bridge import map_cids_to_branches

logger = logging.getLogger(__name__)

# Key bins for raw features — sparse grid, let tree model split
KEY_POS_BINS = ["40", "50", "60", "70", "80", "90", "100"]
KEY_NEG_BINS = ["-40", "-50", "-60", "-70", "-80", "-90", "-100"]

# Full positive tail for deviation distance computation
FULL_POS_BINS = [
    "40", "45", "50", "55", "60", "65", "70", "75",
    "80", "85", "90", "95", "100",
]
FULL_NEG_BINS = [
    "-100", "-95", "-90", "-85", "-80", "-75", "-70", "-65",
    "-60", "-55", "-50", "-45", "-40",
]
ALL_BINS = list(set(KEY_POS_BINS + KEY_NEG_BINS + FULL_POS_BINS + FULL_NEG_BINS))


def load_raw_density_cid_level(
    planning_year: str,
    aq_quarter: str,
    market_round: int,
) -> pl.DataFrame:
    """Load raw density at CID x outage_date level for one (PY, quarter, round)."""
    market_months = get_market_months(planning_year, aq_quarter)
    frames = []
    for mm in market_months:
        path = (
            f"{DENSITY_PATH}/spice_version=v6/auction_type=annual"
            f"/auction_month={planning_year}/market_month={mm}/market_round={market_round}/"
        )
        if not Path(path).exists():
            logger.warning("Density partition missing: %s", path)
            continue
        df = pl.read_parquet(path).select(
            ["constraint_id", "outage_date"] + ALL_BINS
        )
        frames.append(df)

    if not frames:
        raise ValueError(f"No density data for {planning_year}/{aq_quarter}")
    return pl.concat(frames, how="diagonal")


def compute_cid_scores(raw: pl.DataFrame) -> pl.DataFrame:
    """Compute per-CID: mean exceedance at key bins + deviation score for anchor selection.

    Returns one row per CID with:
      - exc_{bin}: mean exceedance at each key bin (positive and negative)
      - exc_max_{bin}: max-across-outages exceedance (worst scenario spike)
      - dev_pos: positive tail deviation distance (base=2, log1p)
      - dev_neg: negative tail deviation distance (base=2, log1p)
      - dev_anchor: max(dev_pos, dev_neg) — used for anchor CID selection
    """
    agg_exprs = []

    # Mean, max, and fraction-stressed at key positive bins
    for b in KEY_POS_BINS:
        agg_exprs.append(pl.col(b).mean().alias(f"exc_{b}"))
        agg_exprs.append(pl.col(b).max().alias(f"exc_max_{b}"))
        # Fraction of scenarios with meaningful exceedance (>1%)
        # This is DIFFERENT from mean: AUST_TAYS has exc_80_mean=0.08 (50th pct)
        # but frac_80=0.73 (top tier) — most scenarios show stress, not just a few
        agg_exprs.append((pl.col(b) > 0.01).mean().alias(f"frac_{b}"))

    # Mean, max, and fraction-stressed at key negative bins
    for b in KEY_NEG_BINS:
        safe = b.replace("-", "n")
        agg_exprs.append(pl.col(b).mean().alias(f"exc_{safe}"))
        agg_exprs.append(pl.col(b).max().alias(f"exc_max_{safe}"))
        agg_exprs.append((pl.col(b) > 0.01).mean().alias(f"frac_{safe}"))

    # Full tail means for deviation distance
    for b in FULL_POS_BINS:
        agg_exprs.append(pl.col(b).mean().alias(f"_p_{b}"))
    for b in FULL_NEG_BINS:
        safe = b.replace("-", "n")
        agg_exprs.append(pl.col(b).mean().alias(f"_n_{safe}"))

    profiles = raw.group_by("constraint_id").agg(agg_exprs)

    # Deviation distance with base=2 (gentler than 7, still tail-emphasizing)
    n = len(FULL_POS_BINS)
    w = np.array([2.0 ** i for i in range(n)])

    pos_expr = pl.lit(0.0)
    for i, b in enumerate(FULL_POS_BINS):
        pos_expr = pos_expr + pl.col(f"_p_{b}") * w[i]

    neg_expr = pl.lit(0.0)
    for i, b in enumerate(FULL_NEG_BINS):
        safe = b.replace("-", "n")
        neg_expr = neg_expr + pl.col(f"_n_{safe}") * w[i]

    profiles = profiles.with_columns(
        pos_expr.log1p().alias("dev_pos"),
        neg_expr.log1p().alias("dev_neg"),
        pl.max_horizontal(pos_expr.log1p(), neg_expr.log1p()).alias("dev_anchor"),
    )

    # Drop intermediates
    drop = [c for c in profiles.columns if c.startswith("_p_") or c.startswith("_n_")]
    return profiles.drop(drop)


def aggregate_to_branch(
    cid_scores: pl.DataFrame,
    planning_year: str,
    aq_quarter: str,
    market_round: int,
) -> pl.DataFrame:
    """Dual-anchor aggregation: separate positive-tail and negative-tail anchors.

    Each direction gets its own anchor CID — the CID with the highest dev score
    in that direction. The anchor's full raw bin profile is carried through.
    This avoids the problem of a negative-tail CID winning the anchor while
    a different CID has the real positive-tail signal.

    Output features (prefixed p_ and n_ for direction):
      - p_exc_{bin}, p_exc_max_{bin}: positive anchor CID's raw bins
      - n_exc_{bin}, n_exc_max_{bin}: negative anchor CID's raw bins
      - p_dev, n_dev: anchor CID deviation scores
      - top2_p_dev, top2_n_dev: runner-up scores
      - dev_gap_p, dev_gap_n: anchor - runner-up
      - n_cids_hot_p, n_cids_hot_n: count with meaningful signal per direction
      - n_cids_total
    """
    mapped, _ = map_cids_to_branches(
        cid_df=cid_scores,
        auction_type="annual",
        auction_month=planning_year,
        period_type=aq_quarter,
        market_round=market_round,
    )

    if len(mapped) == 0:
        logger.warning("No CIDs mapped to branches for %s/%s", planning_year, aq_quarter)
        return pl.DataFrame({"branch_name": pl.Series([], dtype=pl.Utf8)})

    # Positive-tail anchor: CID with highest dev_pos per branch
    pos_ranked = mapped.sort("dev_pos", descending=True)
    pos_bin_cols = [c for c in cid_scores.columns
                    if (c.startswith("exc_") or c.startswith("frac_"))
                    and "n" not in c.split("_")[-1]
                    and c != "constraint_id"]
    pos_anchor = pos_ranked.group_by("branch_name").first().select(
        ["branch_name", "dev_pos"] + pos_bin_cols
    )
    # Rename to p_ prefix
    rename_map = {"dev_pos": "p_dev"}
    for c in pos_bin_cols:
        rename_map[c] = f"p_{c}"
    pos_anchor = pos_anchor.rename(rename_map)

    # Positive runner-up + concentration
    pos_stats = pos_ranked.group_by("branch_name").agg(
        pl.col("dev_pos").sort(descending=True).slice(1, 1).first().alias("top2_p_dev"),
        (pl.col("dev_pos") > 1.0).sum().cast(pl.UInt32).alias("n_cids_hot_p"),
    ).with_columns(pl.col("top2_p_dev").fill_null(0.0))

    # Negative-tail anchor: CID with highest dev_neg per branch
    neg_ranked = mapped.sort("dev_neg", descending=True)
    neg_bin_cols = [c for c in cid_scores.columns
                    if c.startswith("exc_") and "n" in c.split("_")[-1]
                    and c != "constraint_id"]
    # Also include exc_max_n* cols
    neg_bin_cols = [c for c in cid_scores.columns
                    if ("exc_n" in c or "exc_max_n" in c or "frac_n" in c)
                    and c != "constraint_id"]
    neg_anchor = neg_ranked.group_by("branch_name").first().select(
        ["branch_name", "dev_neg"] + neg_bin_cols
    )
    rename_map_n = {"dev_neg": "n_dev"}
    for c in neg_bin_cols:
        rename_map_n[c] = f"n_{c}"
    neg_anchor = neg_anchor.rename(rename_map_n)

    # Negative runner-up + concentration
    neg_stats = neg_ranked.group_by("branch_name").agg(
        pl.col("dev_neg").sort(descending=True).slice(1, 1).first().alias("top2_n_dev"),
        (pl.col("dev_neg") > 1.0).sum().cast(pl.UInt32).alias("n_cids_hot_n"),
    ).with_columns(pl.col("top2_n_dev").fill_null(0.0))

    # Total CID count
    total_counts = mapped.group_by("branch_name").agg(
        pl.len().cast(pl.UInt32).alias("n_cids_total"),
    )

    # Join everything
    result = pos_anchor.join(pos_stats, on="branch_name", how="left")
    result = result.join(neg_anchor, on="branch_name", how="left")
    result = result.join(neg_stats, on="branch_name", how="left")
    result = result.join(total_counts, on="branch_name", how="left")

    # Gaps and best-direction
    result = result.with_columns(
        (pl.col("p_dev") - pl.col("top2_p_dev")).alias("dev_gap_p"),
        (pl.col("n_dev") - pl.col("top2_n_dev")).alias("dev_gap_n"),
        pl.max_horizontal("p_dev", "n_dev").alias("dev_best"),
    )

    return result


def build_deviation_features(
    planning_year: str,
    aq_quarter: str,
    market_round: int,
) -> pl.DataFrame:
    """Full pipeline: raw density -> CID scores -> anchor CID branch features."""
    raw = load_raw_density_cid_level(planning_year, aq_quarter, market_round=market_round)
    logger.info(
        "%s/%s: %d raw rows, %d CIDs",
        planning_year, aq_quarter, len(raw), raw["constraint_id"].n_unique(),
    )

    cid_scores = compute_cid_scores(raw)
    branch_dev = aggregate_to_branch(cid_scores, planning_year, aq_quarter, market_round=market_round)

    logger.info(
        "%s/%s: %d branches with %d features",
        planning_year, aq_quarter, len(branch_dev), len(branch_dev.columns) - 1,
    )
    return branch_dev
