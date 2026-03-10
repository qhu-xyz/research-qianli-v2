"""Data loading for LTR ranking pipeline.

Loads V6.2B signal data and enriches with spice6 density features.
"""
from __future__ import annotations

import resource
from pathlib import Path

import polars as pl

from ml.config import V62B_SIGNAL_BASE
from ml.spice6_loader import load_spice6_density


def mem_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def load_v62b_month(
    auction_month: str,
    period_type: str = "f0",
    class_type: str = "onpeak",
) -> pl.DataFrame:
    """Load V6.2B signal data enriched with spice6 density features.

    Parameters
    ----------
    auction_month : str
        Auction month in YYYY-MM format.
    period_type : str
        Period type (f0, f1, etc.).
    class_type : str
        onpeak or offpeak.

    Returns
    -------
    pl.DataFrame
        V6.2B data enriched with spice6 density features:
        prob_exceed_110, prob_exceed_100, prob_exceed_90, prob_exceed_85,
        prob_exceed_80, constraint_limit.
    """
    path = Path(V62B_SIGNAL_BASE) / auction_month / period_type / class_type
    if not path.exists():
        raise FileNotFoundError(f"V6.2B data not found: {path}")
    df = pl.read_parquet(str(path))
    if "__index_level_0__" in df.columns:
        df = df.drop("__index_level_0__")

    # Enrich with spice6 density features
    spice6 = load_spice6_density(auction_month, period_type)
    if len(spice6) > 0:
        df = df.join(
            spice6,
            on=["constraint_id", "flow_direction"],
            how="left",
        )
        # Fill missing spice6 features with 0 for constraints not in spice6
        spice6_cols = [c for c in spice6.columns if c not in ("constraint_id", "flow_direction")]
        df = df.with_columns([pl.col(c).fill_null(0.0) for c in spice6_cols])
        n_matched = len(df.filter(pl.col("prob_exceed_110") > 0))
        print(f"[data_loader] spice6 enrichment: {n_matched}/{len(df)} matched")
    else:
        print(f"[data_loader] WARNING: no spice6 data for {auction_month}")
        for col in ["prob_exceed_110", "prob_exceed_100", "prob_exceed_90",
                     "prob_exceed_85", "prob_exceed_80", "constraint_limit"]:
            df = df.with_columns(pl.lit(0.0).alias(col))

    # Compute engineered features
    df = _add_engineered_features(df)

    return df


def _add_engineered_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add derived features from raw columns."""
    limit_safe = pl.when(pl.col("constraint_limit") > 0).then(pl.col("constraint_limit")).otherwise(1.0)
    ori_safe = pl.when(pl.col("ori_mean").abs() > 1e-6).then(pl.col("ori_mean")).otherwise(1.0)
    mix_safe = pl.when(pl.col("mix_mean").abs() > 1e-6).then(pl.col("mix_mean")).otherwise(1.0)
    branch_safe = pl.when(pl.col("mean_branch_max").abs() > 1e-6).then(pl.col("mean_branch_max")).otherwise(1.0)

    # --- Tier 1: Utilization ratios (4) ---
    df = df.with_columns([
        (pl.col("ori_mean") / limit_safe).alias("flow_utilization"),
        (pl.col("mix_mean") / limit_safe).alias("mix_utilization"),
        (pl.col("mean_branch_max") / limit_safe).alias("branch_utilization"),
        pl.max_horizontal("prob_exceed_110", "prob_exceed_100", "prob_exceed_90",
                          "prob_exceed_85", "prob_exceed_80").alias("prob_exceed_max"),
    ])

    # --- Tier 2: Flow ratios & differences (6) ---
    df = df.with_columns([
        (pl.col("ori_mean") / mix_safe).alias("ori_mix_ratio"),
        (pl.col("mean_branch_max") / ori_safe).alias("branch_ori_ratio"),
        (pl.col("mean_branch_max") / mix_safe).alias("branch_mix_ratio"),
        (pl.col("ori_mean") - pl.col("mix_mean")).alias("ori_mix_diff"),
        (pl.col("mean_branch_max") - pl.col("ori_mean")).alias("branch_ori_diff"),
        (pl.col("mean_branch_max") - pl.col("mix_mean")).alias("branch_mix_diff"),
    ])

    # --- Tier 3: Rank gaps (3) ---
    df = df.with_columns([
        (pl.col("density_mix_rank_value") - pl.col("density_ori_rank_value")).alias("rank_gap_mix_ori"),
        (pl.col("da_rank_value") - pl.col("density_mix_rank_value")).alias("rank_gap_da_mix"),
        (pl.col("da_rank_value") - pl.col("density_ori_rank_value")).alias("rank_gap_da_ori"),
    ])

    # --- Tier 4: Probability spreads (4) ---
    df = df.with_columns([
        (pl.col("prob_exceed_110") - pl.col("prob_exceed_100")).alias("prob_spread_110_100"),
        (pl.col("prob_exceed_100") - pl.col("prob_exceed_90")).alias("prob_spread_100_90"),
        (pl.col("prob_exceed_90") - pl.col("prob_exceed_80")).alias("prob_spread_90_80"),
        pl.mean_horizontal("prob_exceed_110", "prob_exceed_100", "prob_exceed_90",
                           "prob_exceed_85", "prob_exceed_80").alias("prob_exceed_mean"),
    ])

    # --- Tier 5: Probability-weighted flow signals (4) ---
    df = df.with_columns([
        (pl.col("prob_exceed_110") * pl.col("ori_mean")).alias("prob110_x_flow"),
        (pl.col("prob_exceed_110") * pl.col("mean_branch_max")).alias("prob110_x_branch"),
        (pl.col("prob_exceed_max") * pl.col("flow_utilization")).alias("probmax_x_util"),
        (pl.col("prob_exceed_mean") * pl.col("flow_utilization")).alias("probmean_x_util"),
    ])

    # --- Tier 6: DA rank interactions (4) ---
    df = df.with_columns([
        (pl.col("da_rank_value") * pl.col("density_mix_rank_value")).alias("da_x_dmix"),
        (pl.col("da_rank_value") * pl.col("density_ori_rank_value")).alias("da_x_dori"),
        (pl.col("da_rank_value") * pl.col("prob_exceed_110")).alias("da_x_prob110"),
        (pl.col("da_rank_value") * pl.col("flow_utilization")).alias("da_x_util"),
    ])

    # --- Tier 7: Squared terms for non-linearity (5) ---
    df = df.with_columns([
        (pl.col("da_rank_value") ** 2).alias("da_rank_sq"),
        (pl.col("density_mix_rank_value") ** 2).alias("dmix_rank_sq"),
        (pl.col("flow_utilization") ** 2).alias("flow_util_sq"),
        (pl.col("prob_exceed_110") ** 2).alias("prob110_sq"),
        (pl.col("ori_mean") ** 2).alias("ori_mean_sq"),
    ])

    # --- Tier 8: Log transforms (3) ---
    df = df.with_columns([
        (pl.col("constraint_limit") + 1.0).log().alias("log_constraint_limit"),
        (pl.col("ori_mean").abs() + 1.0).log().alias("log_ori_mean"),
        (pl.col("mean_branch_max").abs() + 1.0).log().alias("log_branch_max"),
    ])

    # --- Tier 9: Three-way interactions (4) ---
    df = df.with_columns([
        (pl.col("da_rank_value") * pl.col("prob_exceed_110") * pl.col("flow_utilization")).alias("da_prob110_util"),
        (pl.col("density_mix_rank_value") * pl.col("prob_exceed_110") * pl.col("mean_branch_max")).alias("dmix_prob110_branch"),
        (pl.col("prob_exceed_110") * pl.col("ori_mix_ratio")).alias("prob110_x_ori_mix_ratio"),
        (pl.col("da_rank_value") * pl.col("branch_utilization")).alias("da_x_branch_util"),
    ])

    return df


def load_train_val_test(
    eval_month: str,
    train_months: int = 6,
    val_months: int = 2,
    period_type: str = "f0",
    class_type: str = "onpeak",
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Load train/val/test splits for a single evaluation month.

    For eval_month M with train=6, val=2:
    - Train: months M-8 through M-3 (6 months)
    - Val: months M-2, M-1 (2 months)
    - Test: month M (the target month)

    Each month's data comes from V6.2B with an added 'query_month' column
    for XGBoost query groups.

    Returns
    -------
    tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]
        (train_df, val_df, test_df) with query_month column added.
    """
    import pandas as pd

    eval_ts = pd.Timestamp(eval_month)
    total_lookback = train_months + val_months

    # Generate month strings for train and val
    train_month_strs = []
    for i in range(total_lookback, val_months, -1):
        m = (eval_ts - pd.DateOffset(months=i)).strftime("%Y-%m")
        train_month_strs.append(m)

    val_month_strs = []
    for i in range(val_months, 0, -1):
        m = (eval_ts - pd.DateOffset(months=i)).strftime("%Y-%m")
        val_month_strs.append(m)

    print(f"[data_loader] eval={eval_month} train={train_month_strs} val={val_month_strs}")
    print(f"[data_loader] mem: {mem_mb():.0f} MB")

    def _load_months(month_strs: list[str]) -> pl.DataFrame:
        dfs = []
        for m in month_strs:
            try:
                df = load_v62b_month(m, period_type, class_type)
                df = df.with_columns(pl.lit(m).alias("query_month"))
                dfs.append(df)
            except FileNotFoundError:
                print(f"[data_loader] WARNING: skipping {m} (not found)")
        if not dfs:
            raise ValueError(f"No data found for months: {month_strs}")
        return pl.concat(dfs)

    train_df = _load_months(train_month_strs)
    val_df = _load_months(val_month_strs) if val_month_strs else None
    test_df = load_v62b_month(eval_month, period_type, class_type)
    test_df = test_df.with_columns(pl.lit(eval_month).alias("query_month"))

    val_len = len(val_df) if val_df is not None else 0
    print(f"[data_loader] train={len(train_df)} val={val_len} test={len(test_df)} "
          f"mem: {mem_mb():.0f} MB")

    return train_df, val_df, test_df
