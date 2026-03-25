"""Phase 6: Class-specific model table builder.

Builds separate model tables for onpeak and offpeak, using class-specific
targets, BF, cohorts, and cross-class features. Reuses shared infrastructure
from ml/ (data_loader, ground_truth, history_features, etc.).

Key differences from combined pipeline (ml/features.py):
  - Target: onpeak_sp or offpeak_sp (not combined)
  - BF: bf_12 or bfo_12 (not bf_combined_12)
  - Cohort: dormant = class-specific BF == 0 (not bf_combined_12 == 0)
  - Cross-class features exposed (other class's BF available as feature)
  - has_hist_da stays COMBINED (not class-specific)
"""
from __future__ import annotations

import logging

import polars as pl

from ml.markets.miso.config import (
    CLASS_BF_COL, CLASS_TARGET_COL, CLASS_NB_FLAG_COL, CROSS_CLASS_BF_COL,
)
from ml.markets.miso.data_loader import load_collapsed
from ml.markets.miso.ground_truth import build_ground_truth
from ml.markets.miso.history_features import compute_history_features
from ml.markets.miso.nb_detection import compute_nb_flags

logger = logging.getLogger(__name__)


def build_class_model_table(
    planning_year: str,
    aq_quarter: str,
    class_type: str,
    market_round: int,
) -> pl.DataFrame:
    """Build a class-specific model table for one (PY, aq, class_type, round).

    Args:
        planning_year: e.g. "2024-06"
        aq_quarter: e.g. "aq1"
        class_type: "onpeak" or "offpeak"
        market_round: auction round (1, 2, or 3 for MISO)

    Returns:
        Branch-level DataFrame with class-specific target, BF, cohort,
        cross-class features, and all shared density/history columns.
    """
    bf_col = CLASS_BF_COL[class_type]
    target_col = CLASS_TARGET_COL[class_type]
    nb_flag_col = CLASS_NB_FLAG_COL[class_type]
    cross_bf_col = CROSS_CLASS_BF_COL[class_type]

    # Step 1: Branch universe (density + limits + metadata features)
    collapsed = load_collapsed(planning_year, aq_quarter, market_round=market_round)
    branches = collapsed["branch_name"].to_list()

    # Step 2: Ground truth (combined — we extract class-specific target below)
    gt_df, gt_diag = build_ground_truth(planning_year, aq_quarter, market_round=market_round)

    # Step 3: History features (exposes bf_12, bfo_12, bf_combined_12, da_rank_value)
    hist_df, monthly_binding = compute_history_features(
        eval_py=planning_year,
        aq_quarter=aq_quarter,
        universe_branches=branches,
        market_round=market_round,
    )

    # Step 4: NB flags (exposes nb_onpeak_12, nb_offpeak_12)
    nb_df = compute_nb_flags(
        universe_branches=branches,
        planning_year=planning_year,
        aq_quarter=aq_quarter,
        gt_df=gt_df,
        monthly_binding_table=monthly_binding,
        market_round=market_round,
    )

    # Assemble
    table = collapsed

    # LEFT JOIN GT
    gt_cols = ["branch_name", "realized_shadow_price", "label_tier", "onpeak_sp", "offpeak_sp"]
    table = table.join(gt_df.select(gt_cols), on="branch_name", how="left")
    table = table.with_columns(
        pl.col("realized_shadow_price").fill_null(0.0),
        pl.col("label_tier").fill_null(0).cast(pl.Int64),
        pl.col("onpeak_sp").fill_null(0.0),
        pl.col("offpeak_sp").fill_null(0.0),
    )

    # LEFT JOIN history features
    table = table.join(hist_df, on="branch_name", how="left")
    bf_cols = [
        "bf_6", "bf_12", "bf_15", "bf_24", "bfo_6", "bfo_12", "bfo_24",
        "bf_combined_6", "bf_combined_12", "bf_combined_24",
        "bfsp_12", "bfsp_off_12",
        "spda_6", "spda_24", "spda_off_6", "spda_off_24",
        "bf_trend_6_12", "bf_accel_6_12", "bfo_trend_6_12", "bfo_accel_6_12",
        "recency_months_since", "recency_off_months_since",
        "recent_max_sp", "recent_max_sp_off",
    ]
    for col in bf_cols:
        if col in table.columns:
            table = table.with_columns(pl.col(col).fill_null(0.0))

    if "da_rank_value" in table.columns:
        n_positive_hist = int(table.filter(
            pl.col("has_hist_da").fill_null(False)
        ).height)
        sentinel = float(n_positive_hist + 1) if n_positive_hist > 0 else 1.0
        table = table.with_columns(pl.col("da_rank_value").fill_null(sentinel))

    if "has_hist_da" in table.columns:
        table = table.with_columns(pl.col("has_hist_da").fill_null(False))

    # LEFT JOIN NB flags
    table = table.join(nb_df, on="branch_name", how="left")
    for col in ["is_nb_6", "is_nb_12", "is_nb_24",
                 "nb_onpeak_6", "nb_onpeak_12", "nb_onpeak_24",
                 "nb_offpeak_6", "nb_offpeak_12", "nb_offpeak_24"]:
        if col in table.columns:
            table = table.with_columns(pl.col(col).fill_null(False))

    # ── Class-specific overrides ────────────────────────────────────────

    # Override realized_shadow_price with class-specific target
    table = table.with_columns(
        pl.col(target_col).alias("realized_shadow_price")
    )

    # Recompute label_tier from class-specific SP
    table = _recompute_class_tiers(table)

    # Class-specific cohort: dormant = class BF == 0 (not bf_combined_12)
    # has_hist_da stays COMBINED — a branch with only offpeak DA still enters onpeak NB pool
    table = table.with_columns(
        pl.when(pl.col(bf_col) > 0)
        .then(pl.lit("established"))
        .when(pl.col("has_hist_da"))
        .then(pl.lit("history_dormant"))
        .otherwise(pl.lit("history_zero"))
        .alias("cohort")
    )

    # Override is_nb_N with class-specific NB flags (only alias what exists)
    nb_prefix = "nb_onpeak" if class_type == "onpeak" else "nb_offpeak"
    nb_aliases = []
    for window in [6, 12, 24]:
        src = f"{nb_prefix}_{window}"
        if src in table.columns:
            nb_aliases.append(pl.col(src).alias(f"is_nb_{window}"))
    if nb_aliases:
        table = table.with_columns(nb_aliases)

    # Add cross-class BF as explicit feature column
    table = table.with_columns(
        pl.col(cross_bf_col).alias("cross_class_bf")
    )

    # Cross-class strength BF (12mo)
    cross_bfsp_col = "bfsp_off_12" if class_type == "onpeak" else "bfsp_12"
    if cross_bfsp_col in table.columns:
        table = table.with_columns(pl.col(cross_bfsp_col).alias("cross_class_bfsp"))

    # Cross-class windowed SPDA (24mo)
    cross_spda_col = "spda_off_24" if class_type == "onpeak" else "spda_24"
    if cross_spda_col in table.columns:
        table = table.with_columns(pl.col(cross_spda_col).alias("cross_spda_24"))

    # ── Class-specific shadow_price_da + da_rank_value ──────────────────
    # The shared history_features computes these from combined_sp.
    # We recompute from class-specific SP using the monthly binding table.
    sp_col = "onpeak_sp" if class_type == "onpeak" else "offpeak_sp"
    class_spda = _compute_class_shadow_price_da(monthly_binding, branches, sp_col)
    # Drop combined shadow_price_da and da_rank_value, replace with class-specific
    table = table.drop(["shadow_price_da", "da_rank_value"])
    table = table.join(class_spda, on="branch_name", how="left")
    table = table.with_columns(
        pl.col("shadow_price_da").fill_null(0.0),
        pl.col("da_rank_value").fill_null(
            float(len(branches) + 1)  # sentinel for zero-history
        ),
    )

    # Cross-class da_rank_value and shadow_price_da
    cross_sp_col = "offpeak_sp" if class_type == "onpeak" else "onpeak_sp"
    cross_spda_df = _compute_class_shadow_price_da(monthly_binding, branches, cross_sp_col)
    cross_spda_df = cross_spda_df.rename({
        "shadow_price_da": "cross_shadow_price_da",
        "da_rank_value": "cross_da_rank_value",
    })
    table = table.join(cross_spda_df, on="branch_name", how="left")
    table = table.with_columns(
        pl.col("cross_shadow_price_da").fill_null(0.0),
        pl.col("cross_da_rank_value").fill_null(float(len(branches) + 1)),
    )

    # Class-specific total_da_sp_quarter: true cross-universe denominator.
    # Uses ALL DA SP for this class (including unmapped cids), not just branch-mapped SP.
    da_sp_key = f"{class_type}_total_da_sp"
    class_total_da_sp = gt_diag.get(da_sp_key, 0.0)
    if class_total_da_sp <= 0:
        class_total_da_sp = float(table["realized_shadow_price"].sum())
        logger.warning("  %s not in GT diagnostics, falling back to branch sum", da_sp_key)
    table = table.with_columns(
        pl.lit(class_total_da_sp).alias("total_da_sp_quarter")
    )

    # Add PY, quarter, class_type columns
    table = table.with_columns(
        pl.lit(planning_year).alias("planning_year"),
        pl.lit(aq_quarter).alias("aq_quarter"),
        pl.lit(class_type).alias("class_type"),
    )

    assert table["branch_name"].n_unique() == len(table), "Duplicate branch_names"

    n_estab = table.filter(pl.col("cohort") == "established").height
    n_dorm = table.filter(pl.col("cohort") == "history_dormant").height
    n_zero = table.filter(pl.col("cohort") == "history_zero").height
    n_bind = table.filter(pl.col("realized_shadow_price") > 0).height

    logger.info(
        "Class model table %s/%s/%s: %d branches, %d binding, "
        "%d established, %d dormant, %d zero-history",
        planning_year, aq_quarter, class_type,
        len(table), n_bind, n_estab, n_dorm, n_zero,
    )

    return table


def _compute_class_shadow_price_da(
    monthly_binding: pl.DataFrame,
    universe_branches: list[str],
    sp_col: str,
) -> pl.DataFrame:
    """Compute class-specific shadow_price_da and da_rank_value from monthly binding.

    shadow_price_da = cumulative class-specific SP per branch (branch-level).
    da_rank_value = dense rank descending of shadow_price_da among positive branches.

    Args:
        monthly_binding: from compute_history_features, has onpeak_sp/offpeak_sp per month
        universe_branches: all branches in the universe
        sp_col: "onpeak_sp" or "offpeak_sp"
    """
    # Cumulative class-specific SP per branch
    cumulative = monthly_binding.group_by("branch_name").agg(
        pl.col(sp_col).sum().alias("shadow_price_da")
    )

    # Ensure all universe branches present
    all_branches = pl.DataFrame({"branch_name": universe_branches})
    result = all_branches.join(cumulative, on="branch_name", how="left")
    result = result.with_columns(pl.col("shadow_price_da").fill_null(0.0))

    # Dense rank descending for positive-SP branches
    positive = result.filter(pl.col("shadow_price_da") > 0)
    n_positive = len(positive)

    if n_positive > 0:
        positive = positive.with_columns(
            pl.col("shadow_price_da")
            .rank(method="dense", descending=True)
            .cast(pl.Float64)
            .alias("da_rank_value")
        )
        sentinel = float(int(positive["da_rank_value"].max()) + 1)
        result = result.join(
            positive.select(["branch_name", "da_rank_value"]),
            on="branch_name",
            how="left",
        ).with_columns(
            pl.col("da_rank_value").fill_null(sentinel)
        )
    else:
        result = result.with_columns(pl.lit(1.0).alias("da_rank_value"))

    return result


def _recompute_class_tiers(table: pl.DataFrame) -> pl.DataFrame:
    """Recompute label_tier from class-specific realized_shadow_price.

    Within each group, positives get tertile tiers 1/2/3, non-positives get 0.
    """
    import numpy as np

    sp = table["realized_shadow_price"].to_numpy().astype(np.float64)
    tiers = np.zeros(len(sp), dtype=np.int64)
    pos_mask = sp > 0
    n_pos = pos_mask.sum()

    if n_pos > 0:
        pos_sp = sp[pos_mask]
        ranks = np.empty(n_pos, dtype=np.int32)
        ranks[pos_sp.argsort().argsort()] = np.arange(n_pos)
        t1 = n_pos // 3
        t2 = 2 * n_pos // 3
        tier_vals = np.where(ranks < t1, 1, np.where(ranks < t2, 2, 3))
        tiers[pos_mask] = tier_vals

    return table.with_columns(pl.Series("label_tier", tiers))


def build_class_model_table_all(
    groups: list[str],
    class_type: str,
    market_round: int,
) -> pl.DataFrame:
    """Build class-specific model tables for multiple groups.

    Args:
        groups: list of "PY/aq" strings
        class_type: "onpeak" or "offpeak"
        market_round: auction round (1, 2, or 3 for MISO)
    """
    frames = []
    for group in groups:
        py, aq = group.split("/")
        frames.append(build_class_model_table(py, aq, class_type, market_round=market_round))
    return pl.concat(frames, how="diagonal")
