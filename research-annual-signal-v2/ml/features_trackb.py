"""Phase 4b Track B features: constraint propagation, recency, density shape.

These features are computed for dormant branches only (bf_combined_12 == 0).
They supplement the existing 12 density bin features from Phase 3.

Constraint propagation: for each dormant branch's CIDs, compute how actively
those CIDs bind on OTHER branches. "Cold branch, hot constraint" signal.

Recency: how recently the branch last bound, historical SP from older windows.

Shape: distribution statistics from density bins (entropy, skewness, etc.)
Note: bins are NOT calibrated probabilities — they are raw simulation scores.
"""
from __future__ import annotations

import logging

import numpy as np
import polars as pl

from ml.data_loader import load_cid_mapping

logger = logging.getLogger(__name__)


def compute_constraint_propagation(
    planning_year: str,
    aq_quarter: str,
    monthly_binding: pl.DataFrame,
    dormant_branches: list[str],
    cutoff_month: str,
    market_round: int,
) -> pl.DataFrame:
    """Compute constraint propagation features for dormant branches.

    For each dormant branch, look up its CIDs, find all OTHER branches
    containing those CIDs, and aggregate their binding activity.

    Args:
        planning_year: eval PY
        aq_quarter: eval quarter
        monthly_binding: monthly binding table from compute_history_features
            (columns: month, branch_name, combined_bound, combined_sp, ...)
        dormant_branches: list of dormant branch names
        cutoff_month: BF cutoff month (only use months <= this)

    Returns:
        DataFrame with columns: branch_name, max_cid_bf_other, mean_cid_bf_other,
        sum_cid_sp_other, max_cid_sp_other, n_active_cids_other, active_cid_ratio_other
    """
    # Load branch↔CID mapping
    cid_map = load_cid_mapping(planning_year, aq_quarter, market_round=market_round)

    # Get CIDs for dormant branches
    dormant_cids = cid_map.filter(pl.col("branch_name").is_in(dormant_branches))

    if len(dormant_cids) == 0:
        return _empty_constraint_propagation(dormant_branches)

    # Get CIDs for ALL branches (for cross-reference)
    all_cid_branch = cid_map.select(["constraint_id", "branch_name"])

    # Monthly binding within window
    binding_in_window = monthly_binding.filter(pl.col("month") <= cutoff_month)

    if len(binding_in_window) == 0:
        return _empty_constraint_propagation(dormant_branches)

    # Per-branch binding stats (BF and total SP)
    branch_stats = binding_in_window.group_by("branch_name").agg(
        (pl.col("combined_bound").sum() / pl.col("combined_bound").len()).alias("branch_bf"),
        pl.col("combined_sp").sum().alias("branch_total_sp"),
    )

    # Join CID mapping to branch stats: for each CID, get stats of ALL branches it appears on
    cid_branch_stats = all_cid_branch.join(branch_stats, on="branch_name", how="left")
    cid_branch_stats = cid_branch_stats.with_columns(
        pl.col("branch_bf").fill_null(0.0),
        pl.col("branch_total_sp").fill_null(0.0),
    )

    # For each dormant branch's CIDs, get OTHER branches' stats
    dormant_cid_list = dormant_cids.select("constraint_id").unique()
    other_stats = (
        cid_branch_stats
        .join(dormant_cid_list, on="constraint_id", how="inner")
        # Exclude dormant branches themselves
        .filter(~pl.col("branch_name").is_in(dormant_branches))
    )

    if len(other_stats) == 0:
        return _empty_constraint_propagation(dormant_branches)

    # Per-CID: aggregate other-branch stats
    per_cid = other_stats.group_by("constraint_id").agg(
        pl.col("branch_bf").max().alias("cid_max_bf_other"),
        pl.col("branch_bf").mean().alias("cid_mean_bf_other"),
        pl.col("branch_total_sp").sum().alias("cid_sum_sp_other"),
        pl.col("branch_total_sp").max().alias("cid_max_sp_other"),
        (pl.col("branch_bf") > 0).any().cast(pl.Int64).alias("cid_has_active_other"),
    )

    # Join back to dormant CIDs and aggregate to branch level
    dormant_with_cid_stats = dormant_cids.join(per_cid, on="constraint_id", how="left")
    dormant_with_cid_stats = dormant_with_cid_stats.with_columns(
        pl.col("cid_max_bf_other").fill_null(0.0),
        pl.col("cid_mean_bf_other").fill_null(0.0),
        pl.col("cid_sum_sp_other").fill_null(0.0),
        pl.col("cid_max_sp_other").fill_null(0.0),
        pl.col("cid_has_active_other").fill_null(0),
    )

    # Branch-level aggregation
    result = dormant_with_cid_stats.group_by("branch_name").agg(
        pl.col("cid_max_bf_other").max().alias("max_cid_bf_other"),
        pl.col("cid_mean_bf_other").mean().alias("mean_cid_bf_other"),
        pl.col("cid_sum_sp_other").sum().alias("sum_cid_sp_other"),
        pl.col("cid_max_sp_other").max().alias("max_cid_sp_other"),
        pl.col("cid_has_active_other").sum().cast(pl.Int64).alias("n_active_cids_other"),
    )

    # Add count_cids for ratio
    cid_counts = dormant_cids.group_by("branch_name").agg(pl.len().alias("_n_cids"))
    result = result.join(cid_counts, on="branch_name", how="left")
    result = result.with_columns(
        (pl.col("n_active_cids_other") / pl.col("_n_cids").cast(pl.Float64)).alias("active_cid_ratio_other")
    ).drop("_n_cids")

    # Ensure all dormant branches present
    all_dormant = pl.DataFrame({"branch_name": dormant_branches})
    result = all_dormant.join(result, on="branch_name", how="left")
    for col in result.columns:
        if col != "branch_name":
            result = result.with_columns(pl.col(col).fill_null(0.0))

    return result


def _empty_constraint_propagation(dormant_branches: list[str]) -> pl.DataFrame:
    return pl.DataFrame({
        "branch_name": dormant_branches,
        "max_cid_bf_other": [0.0] * len(dormant_branches),
        "mean_cid_bf_other": [0.0] * len(dormant_branches),
        "sum_cid_sp_other": [0.0] * len(dormant_branches),
        "max_cid_sp_other": [0.0] * len(dormant_branches),
        "n_active_cids_other": [0.0] * len(dormant_branches),
        "active_cid_ratio_other": [0.0] * len(dormant_branches),
    })


def compute_recency_features(
    monthly_binding: pl.DataFrame,
    dormant_branches: list[str],
    cutoff_month: str,
) -> pl.DataFrame:
    """Compute recency features for dormant branches.

    months_since_last_bind: months from cutoff to most recent binding
    historical_max_sp: peak single-month SP from any period
    historical_sp_12_24m: total SP in months 13-24 before cutoff
    historical_sp_24_36m: total SP in months 25-36 before cutoff
    n_historical_binding_months: count of months where branch had SP > 0
    """
    dormant_binding = monthly_binding.filter(
        (pl.col("branch_name").is_in(dormant_branches))
        & (pl.col("month") <= cutoff_month)
    )

    if len(dormant_binding) == 0:
        return _empty_recency(dormant_branches)

    cutoff_y = int(cutoff_month[:4])
    cutoff_m = int(cutoff_month[5:7])

    def _months_diff(month_str: str) -> int:
        y, m = int(month_str[:4]), int(month_str[5:7])
        return (cutoff_y - y) * 12 + (cutoff_m - m)

    dormant_binding = dormant_binding.with_columns(
        pl.col("month").map_elements(_months_diff, return_dtype=pl.Int64).alias("months_ago")
    )

    # Per-branch aggregation
    result = dormant_binding.group_by("branch_name").agg(
        pl.col("months_ago").filter(pl.col("combined_bound")).min().alias("months_since_last_bind"),
        pl.col("combined_sp").max().alias("historical_max_sp"),
        pl.col("combined_bound").sum().alias("n_historical_binding_months"),
    )

    # Historical SP windows
    sp_12_24 = (
        dormant_binding
        .filter((pl.col("months_ago") >= 12) & (pl.col("months_ago") < 24))
        .group_by("branch_name")
        .agg(pl.col("combined_sp").sum().alias("historical_sp_12_24m"))
    )
    sp_24_36 = (
        dormant_binding
        .filter((pl.col("months_ago") >= 24) & (pl.col("months_ago") < 36))
        .group_by("branch_name")
        .agg(pl.col("combined_sp").sum().alias("historical_sp_24_36m"))
    )

    result = result.join(sp_12_24, on="branch_name", how="left")
    result = result.join(sp_24_36, on="branch_name", how="left")

    # Ensure all dormant branches present
    all_dormant = pl.DataFrame({"branch_name": dormant_branches})
    result = all_dormant.join(result, on="branch_name", how="left")

    result = result.with_columns(
        pl.col("months_since_last_bind").fill_null(999),
        pl.col("historical_max_sp").fill_null(0.0),
        pl.col("n_historical_binding_months").fill_null(0),
        pl.col("historical_sp_12_24m").fill_null(0.0),
        pl.col("historical_sp_24_36m").fill_null(0.0),
    )

    return result


def _empty_recency(dormant_branches: list[str]) -> pl.DataFrame:
    return pl.DataFrame({
        "branch_name": dormant_branches,
        "months_since_last_bind": [999] * len(dormant_branches),
        "historical_max_sp": [0.0] * len(dormant_branches),
        "n_historical_binding_months": [0] * len(dormant_branches),
        "historical_sp_12_24m": [0.0] * len(dormant_branches),
        "historical_sp_24_36m": [0.0] * len(dormant_branches),
    })


def compute_density_shape(
    collapsed: pl.DataFrame,
    dormant_branches: list[str],
) -> pl.DataFrame:
    """Compute density shape features for dormant branches.

    Uses the existing bin_*_cid_max columns from the collapsed model table.
    """
    dormant = collapsed.filter(pl.col("branch_name").is_in(dormant_branches))

    bin_max_cols = sorted([c for c in dormant.columns if c.endswith("_cid_max") and c.startswith("bin_")])

    if len(dormant) == 0 or len(bin_max_cols) == 0:
        return _empty_shape(dormant_branches)

    results = []
    for row in dormant.iter_rows(named=True):
        vals = np.array([row[c] for c in bin_max_cols], dtype=np.float64)

        # Tail sums
        tail_100_cols = [c for c in bin_max_cols if _bin_num(c) >= 100]
        tail_110_cols = [c for c in bin_max_cols if _bin_num(c) >= 110]
        tail_sum_100 = sum(row[c] for c in tail_100_cols)
        tail_sum_110 = sum(row[c] for c in tail_110_cols)

        # Shape stats
        total = vals.sum()
        if total > 0:
            probs = vals / total
            entropy = -float(np.sum(probs[probs > 0] * np.log(probs[probs > 0])))
        else:
            entropy = 0.0

        mean_v = vals.mean()
        std_v = vals.std()
        cv = float(std_v / mean_v) if mean_v > 0 else 0.0

        if len(vals) >= 3 and std_v > 0:
            skew = float(np.mean(((vals - mean_v) / std_v) ** 3))
            kurt = float(np.mean(((vals - mean_v) / std_v) ** 4)) - 3.0
        else:
            skew = 0.0
            kurt = 0.0

        results.append({
            "branch_name": row["branch_name"],
            "tail_sum_ge_100": tail_sum_100,
            "tail_sum_ge_110": tail_sum_110,
            "density_entropy": entropy,
            "density_skewness": skew,
            "density_kurtosis": kurt,
            "density_cv": cv,
        })

    return pl.DataFrame(results)


def _bin_num(col_name: str) -> float:
    """Extract numeric bin value from column name like 'bin_100_cid_max'."""
    parts = col_name.replace("bin_", "").replace("_cid_max", "")
    try:
        return float(parts)
    except ValueError:
        return 0.0


def _empty_shape(dormant_branches: list[str]) -> pl.DataFrame:
    return pl.DataFrame({
        "branch_name": dormant_branches,
        "tail_sum_ge_100": [0.0] * len(dormant_branches),
        "tail_sum_ge_110": [0.0] * len(dormant_branches),
        "density_entropy": [0.0] * len(dormant_branches),
        "density_skewness": [0.0] * len(dormant_branches),
        "density_kurtosis": [0.0] * len(dormant_branches),
        "density_cv": [0.0] * len(dormant_branches),
    })


# ── Feature list constants ──────────────────────────────────────────────

CONSTRAINT_PROP_FEATURES = [
    "max_cid_bf_other", "mean_cid_bf_other",
    "sum_cid_sp_other", "max_cid_sp_other",
    "n_active_cids_other", "active_cid_ratio_other",
]

RECENCY_FEATURES = [
    "months_since_last_bind", "historical_max_sp",
    "n_historical_binding_months",
    "historical_sp_12_24m", "historical_sp_24_36m",
]

SHAPE_FEATURES = [
    "tail_sum_ge_100", "tail_sum_ge_110",
    "density_entropy", "density_skewness",
    "density_kurtosis", "density_cv",
]

# Phase 3 features (12) + new features (17) = 29 total
PHASE3_FEATURES = [
    "count_cids", "bin_80_cid_max", "bin_70_cid_max", "bin_90_cid_max",
    "count_active_cids", "bin_60_cid_max", "bin_100_cid_max", "bin_110_cid_max",
    "bin_-50_cid_max", "bin_120_cid_max", "bin_-100_cid_max", "bin_150_cid_max",
]

PHASE4B_TRACK_B_FEATURES = (
    PHASE3_FEATURES
    + CONSTRAINT_PROP_FEATURES
    + RECENCY_FEATURES
    + SHAPE_FEATURES
)
