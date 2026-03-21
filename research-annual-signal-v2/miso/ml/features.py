"""Model table assembly — joins all sources into ONE table per (PY, quarter).

This module owns the schema contract. All downstream modules (train, evaluate)
receive this single DataFrame.

Key responsibility: creates the full branch universe from data_loader, then
LEFT JOINs GT (positive-binding only). Branches not in GT get zero targets
(realized_shadow_price=0.0, label_tier=0). This is the ONLY place where
non-binding branches get their zero targets.
"""
from __future__ import annotations

import logging

import polars as pl

from ml.data_loader import load_collapsed
from ml.ground_truth import build_ground_truth
from ml.history_features import compute_history_features
from ml.nb_detection import compute_nb_flags

logger = logging.getLogger(__name__)


def build_model_table(planning_year: str, aq_quarter: str) -> pl.DataFrame:
    """Build the full model table for one (PY, quarter).

    Pipeline:
      1. Load branch universe from data_loader (density + limits features)
      2. LEFT JOIN ground_truth — fill missing with zeros
      3. LEFT JOIN history_features — fill missing BF with 0, da_rank with sentinel
      4. LEFT JOIN nb_detection — fill missing NB flags with False
      5. Assign cohorts
      6. Attach total_da_sp_quarter
      7. Add planning_year and aq_quarter columns
    """
    # Step 1: Branch universe (density + limits + metadata features)
    collapsed = load_collapsed(planning_year, aq_quarter)
    branches = collapsed["branch_name"].to_list()

    # Step 2: Ground truth
    gt_df, gt_diag = build_ground_truth(planning_year, aq_quarter)

    # Step 3: History features
    hist_df, monthly_binding = compute_history_features(
        eval_py=planning_year,
        aq_quarter=aq_quarter,
        universe_branches=branches,
    )

    # Step 4: NB flags
    nb_df = compute_nb_flags(
        universe_branches=branches,
        planning_year=planning_year,
        aq_quarter=aq_quarter,
        gt_df=gt_df,
        monthly_binding_table=monthly_binding,
    )

    # Assemble: start with collapsed (density + limits + metadata)
    table = collapsed

    # LEFT JOIN GT — only positive-binding branches have GT rows
    gt_cols = ["branch_name", "realized_shadow_price", "label_tier", "onpeak_sp", "offpeak_sp"]
    table = table.join(gt_df.select(gt_cols), on="branch_name", how="left")

    # Zero-fill for non-binding branches
    table = table.with_columns(
        pl.col("realized_shadow_price").fill_null(0.0),
        pl.col("label_tier").fill_null(0).cast(pl.Int64),
        pl.col("onpeak_sp").fill_null(0.0),
        pl.col("offpeak_sp").fill_null(0.0),
    )

    # LEFT JOIN history features
    table = table.join(hist_df, on="branch_name", how="left")

    # Fill missing history features
    bf_cols = ["bf_6", "bf_12", "bf_15", "bfo_6", "bfo_12", "bf_combined_6", "bf_combined_12"]
    for col in bf_cols:
        if col in table.columns:
            table = table.with_columns(pl.col(col).fill_null(0.0))

    # da_rank_value sentinel for branches without history
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
    for col in ["is_nb_6", "is_nb_12", "is_nb_24", "nb_onpeak_12", "nb_offpeak_12"]:
        if col in table.columns:
            table = table.with_columns(pl.col(col).fill_null(False))

    # Step 5: Cohort assignment
    table = table.with_columns(
        pl.when(pl.col("bf_combined_12") > 0)
        .then(pl.lit("established"))
        .when(pl.col("has_hist_da"))
        .then(pl.lit("history_dormant"))
        .otherwise(pl.lit("history_zero"))
        .alias("cohort")
    )

    # Step 6: total_da_sp_quarter from GT diagnostics
    table = table.with_columns(
        pl.lit(gt_diag["total_da_sp"]).alias("total_da_sp_quarter")
    )

    # Step 7: Add PY and quarter columns
    table = table.with_columns(
        pl.lit(planning_year).alias("planning_year"),
        pl.lit(aq_quarter).alias("aq_quarter"),
    )

    assert table["branch_name"].n_unique() == len(table), "Duplicate branch_names in model table"

    logger.info(
        "Model table %s/%s: %d branches, %d binding (label>0), %d established, "
        "%d dormant, %d zero-history",
        planning_year, aq_quarter, len(table),
        table.filter(pl.col("label_tier") > 0).height,
        table.filter(pl.col("cohort") == "established").height,
        table.filter(pl.col("cohort") == "history_dormant").height,
        table.filter(pl.col("cohort") == "history_zero").height,
    )

    return table


def build_model_table_all(groups: list[str]) -> pl.DataFrame:
    """Build model tables for multiple groups and concatenate.

    Args:
        groups: list of "PY/aq" strings, e.g. ["2024-06/aq1", "2024-06/aq2"]
    """
    frames = []
    for group in groups:
        py, aq = group.split("/")
        frames.append(build_model_table(py, aq))
    return pl.concat(frames, how="diagonal")
