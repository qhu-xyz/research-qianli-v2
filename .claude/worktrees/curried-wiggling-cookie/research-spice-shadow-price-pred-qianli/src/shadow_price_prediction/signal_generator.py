"""
Signal generator for converting ML predictions to ConstraintsSignal format.

Converts the output of the shadow price prediction pipeline (final_results.parquet)
into the ConstraintsSignal parquet format used by the CIA Multistep trading workflow.

Signal format (matching existing SPICE_F0P_V6.7B.R1):
- Index: "{constraint_id}|{-flow_direction}|spice"
- Columns: constraint_id, branch_name, flow_direction, hist_da,
           predicted_shadow_price, binding_probability_scaled,
           prob_rank, hist_shadow_rank, pred_shadow_rank, rank, tier,
           shadow_sign, shadow_price, equipment
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Rank weights for composite ranking (reverse-engineered from existing 6.7B data)
DEFAULT_RANK_WEIGHTS = {
    "prob_rank": 0.4,
    "hist_shadow_rank": 0.3,
    "pred_shadow_rank": 0.3,
}

N_TIERS = 5  # 0-4, matching existing signal


def aggregate_multi_month_results(
    results_list: list[pd.DataFrame],
) -> pd.DataFrame:
    """Aggregate final_results across multiple market months.

    For quarterly/annual period types, the pipeline produces one final_results
    per market month. This function aggregates them into a single DataFrame
    suitable for signal generation.

    Aggregation logic:
    - predicted_shadow_price: sum across months
    - binding_probability_scaled: max across months
    - hist_da: max across months
    - branch_name: first non-null

    Parameters
    ----------
    results_list : list[pd.DataFrame]
        List of final_results DataFrames, one per market month.
        Each should have MultiIndex (constraint_id, flow_direction).

    Returns
    -------
    pd.DataFrame
        Aggregated DataFrame with same schema as a single final_results.
    """
    if len(results_list) == 0:
        return pd.DataFrame()
    if len(results_list) == 1:
        return results_list[0]

    combined = pd.concat(results_list)

    # Reset index for groupby if MultiIndex
    has_multi = isinstance(combined.index, pd.MultiIndex)
    if has_multi:
        combined = combined.reset_index()

    agg_dict = {
        "branch_name": "first",
        "predicted_shadow_price": "sum",
        "binding_probability_scaled": "max",
    }
    if "hist_da" in combined.columns:
        agg_dict["hist_da"] = "max"

    # Pass through any other numeric columns we need
    for col in ["binding_probability"]:
        if col in combined.columns:
            agg_dict[col] = "max"

    aggregated = combined.groupby(["constraint_id", "flow_direction"]).agg(agg_dict)

    return aggregated


def convert_predictions_to_signal(
    final_results: pd.DataFrame,
    rank_weights: dict[str, float] | None = None,
    n_tiers: int = N_TIERS,
) -> pd.DataFrame:
    """Convert ML pipeline final_results to ConstraintsSignal format.

    Parameters
    ----------
    final_results : pd.DataFrame
        Output from ShadowPricePipeline. Expected to have MultiIndex
        (constraint_id, flow_direction) with columns: branch_name, hist_da,
        predicted_shadow_price, binding_probability_scaled.
    rank_weights : dict, optional
        Weights for composite rank. Keys: prob_rank, hist_shadow_rank,
        pred_shadow_rank. Defaults to {0.4, 0.3, 0.3}.
    n_tiers : int
        Number of tiers for signal stratification. Default 5.

    Returns
    -------
    pd.DataFrame
        Signal DataFrame with proper index and columns for ConstraintsSignal.
    """
    if rank_weights is None:
        rank_weights = DEFAULT_RANK_WEIGHTS

    df = final_results.copy()

    # Reset MultiIndex if present
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()

    # Ensure required columns exist
    required_cols = ["constraint_id", "flow_direction", "predicted_shadow_price", "binding_probability_scaled"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Filter to only predicted-binding constraints (non-zero predictions)
    df = df[df["predicted_shadow_price"] > 0].copy()

    if len(df) == 0:
        logger.warning("No binding predictions found in final_results")
        return pd.DataFrame()

    # Ensure hist_da exists (fill with 0 if missing)
    if "hist_da" not in df.columns:
        logger.warning("hist_da column not found, filling with 0")
        df["hist_da"] = 0.0

    # Compute sub-ranks (lower rank = better/higher value)
    df["prob_rank"] = df["binding_probability_scaled"].rank(ascending=False, pct=True, method="average")
    df["hist_shadow_rank"] = df["hist_da"].rank(ascending=False, pct=True, method="average")
    df["pred_shadow_rank"] = df["predicted_shadow_price"].rank(ascending=False, pct=True, method="average")

    # Compute composite rank
    df["rank"] = (
        rank_weights["prob_rank"] * df["prob_rank"]
        + rank_weights["hist_shadow_rank"] * df["hist_shadow_rank"]
        + rank_weights["pred_shadow_rank"] * df["pred_shadow_rank"]
    )
    # Normalize rank to [0, 1]
    df["rank"] = df["rank"].rank(pct=True, method="first")

    # Assign tiers (0 = best, n_tiers-1 = worst)
    df["tier"] = pd.qcut(df["rank"], q=n_tiers, labels=False, duplicates="drop")

    # Compute derived columns
    df["shadow_sign"] = -df["flow_direction"]
    df["shadow_price"] = df["predicted_shadow_price"] * df["shadow_sign"]
    df["equipment"] = df["branch_name"]

    # Build signal index: {constraint_id}|{-flow_direction}|spice
    df["signal_index"] = (
        df["constraint_id"].astype(str) + "|" + (-df["flow_direction"]).astype(str) + "|spice"
    )
    df = df.set_index("signal_index")
    df.index.name = None

    # Select and order output columns
    output_cols = [
        "constraint_id",
        "branch_name",
        "flow_direction",
        "hist_da",
        "predicted_shadow_price",
        "binding_probability_scaled",
        "prob_rank",
        "hist_shadow_rank",
        "pred_shadow_rank",
        "rank",
        "tier",
        "shadow_sign",
        "shadow_price",
        "equipment",
    ]
    existing_cols = [c for c in output_cols if c in df.columns]
    df = df[existing_cols]

    # Sort by rank (best first)
    df = df.sort_values("rank")

    logger.info(f"Generated signal with {len(df)} constraints, tier distribution: {df['tier'].value_counts().sort_index().to_dict()}")

    return df


def save_signal(
    signal_df: pd.DataFrame,
    rto: str,
    signal_name: str,
    period_type: str,
    class_type: str,
    auction_month: pd.Timestamp | str,
    dry_run: bool = False,
) -> str:
    """Save signal DataFrame using ConstraintsSignal.

    Parameters
    ----------
    signal_df : pd.DataFrame
        Signal DataFrame from convert_predictions_to_signal().
    rto : str
        RTO name (e.g., "miso").
    signal_name : str
        Full signal name (e.g., "TEST.TEST.Signal.MISO.SPICE_F0P_V6.7B.R1").
    period_type : str
        Period type (e.g., "f0", "q2").
    class_type : str
        Class type (e.g., "onpeak", "offpeak").
    auction_month : pd.Timestamp or str
        Auction month.
    dry_run : bool
        If True, don't actually save.

    Returns
    -------
    str
        Path where data was saved.
    """
    from pbase.data.dataset.signal.general import ConstraintsSignal

    if isinstance(auction_month, str):
        auction_month = pd.Timestamp(auction_month)

    loader = ConstraintsSignal(
        rto=rto,
        signal_name=signal_name,
        period_type=period_type,
        class_type=class_type,
    )

    path = loader.save_data(
        data=signal_df,
        auction_month=auction_month,
        dry_run=dry_run,
    )

    if not dry_run:
        logger.info(f"Saved signal to {path}: {len(signal_df)} rows")
    else:
        logger.info(f"Dry run - would save {len(signal_df)} rows to {path}")

    return path


def generate_and_save_signal(
    final_results: pd.DataFrame,
    rto: str,
    signal_name: str,
    period_type: str,
    class_type: str,
    auction_month: pd.Timestamp | str,
    rank_weights: dict[str, float] | None = None,
    n_tiers: int = N_TIERS,
    dry_run: bool = False,
) -> tuple[pd.DataFrame, str]:
    """Convert predictions to signal format and save in one step.

    Parameters
    ----------
    final_results : pd.DataFrame
        Output from ShadowPricePipeline.
    rto : str
        RTO name.
    signal_name : str
        Full signal name.
    period_type : str
        Period type.
    class_type : str
        Class type.
    auction_month : pd.Timestamp or str
        Auction month.
    rank_weights : dict, optional
        Weights for composite rank.
    n_tiers : int
        Number of tiers.
    dry_run : bool
        If True, don't actually save.

    Returns
    -------
    tuple[pd.DataFrame, str]
        (signal_df, save_path)
    """
    signal_df = convert_predictions_to_signal(
        final_results,
        rank_weights=rank_weights,
        n_tiers=n_tiers,
    )

    if signal_df.empty:
        logger.warning(f"Empty signal for {auction_month} {period_type} {class_type}, skipping save")
        return signal_df, ""

    path = save_signal(
        signal_df=signal_df,
        rto=rto,
        signal_name=signal_name,
        period_type=period_type,
        class_type=class_type,
        auction_month=auction_month,
        dry_run=dry_run,
    )

    return signal_df, path
