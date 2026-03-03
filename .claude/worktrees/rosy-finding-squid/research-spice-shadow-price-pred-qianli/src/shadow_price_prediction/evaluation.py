"""
Evaluation metrics for shadow price predictions.
"""
import pandas as pd
import numpy as np
from typing import Dict
from sklearn.metrics import (
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    fbeta_score,
    ndcg_score,
)
from scipy.stats import spearmanr, skew as scipy_skew

# Default K values for top-K metrics.  Chosen to span portfolio sizes
# relevant to FTR signal generation (~1300 constraints in 5 tiers).
DEFAULT_K_VALUES = (100, 250, 500, 1000, 2000)

# K values for constraint-level top-K.  Chosen relative to typical binding
# count per month (~200–420 constraints out of ~5,500 total).
CONSTRAINT_K_VALUES = (20, 50, 100, 200, 300, 500, 1000)


def calculate_metrics(results: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate comprehensive metrics for shadow price predictions.

    Parameters:
    -----------
    results : pd.DataFrame
        Results DataFrame with actual and predicted values

    Returns:
    --------
    metrics : dict
        Dictionary of calculated metrics
    """
    y_actual = results['actual_shadow_price'].values
    y_pred = results['predicted_shadow_price'].values
    y_actual_binary = results['actual_binding'].values
    y_pred_binary = results['predicted_binding'].values

    # Regression metrics
    mae = mean_absolute_error(y_actual, y_pred)
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))

    # Calculate MAPE for non-zero actuals
    non_zero_mask = y_actual > 0
    if non_zero_mask.sum() > 0:
        mape = np.mean(np.abs((y_actual[non_zero_mask] - y_pred[non_zero_mask]) / y_actual[non_zero_mask])) * 100
    else:
        mape = 0.0

    # Classification metrics
    precision = precision_score(y_actual_binary, y_pred_binary, zero_division=0)
    recall = recall_score(y_actual_binary, y_pred_binary, zero_division=0)
    f1 = f1_score(y_actual_binary, y_pred_binary)

    # Accuracy
    accuracy = (y_actual_binary == y_pred_binary).sum() / len(y_actual_binary)

    # Confusion matrix components
    tp = ((y_actual_binary == 1) & (y_pred_binary == 1)).sum()
    fp = ((y_actual_binary == 0) & (y_pred_binary == 1)).sum()
    tn = ((y_actual_binary == 0) & (y_pred_binary == 0)).sum()
    fn = ((y_actual_binary == 1) & (y_pred_binary == 0)).sum()

    return {
        # Regression metrics
        'mae': mae,
        'rmse': rmse,
        'mape': mape,

        # Classification metrics
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'accuracy': accuracy,

        # Confusion matrix
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn),

        # Additional stats
        'total_samples': len(results),
        'actual_binding_count': int(y_actual_binary.sum()),
        'predicted_binding_count': int(y_pred_binary.sum()),
        'actual_binding_rate': float(y_actual_binary.sum() / len(y_actual_binary)),
        'predicted_binding_rate': float(y_pred_binary.sum() / len(y_pred_binary))
    }


def print_metrics_report(
    metrics: Dict[str, float],
    title: str = "Metrics Report",
    level: str = "monthly"
) -> None:
    """
    Print a formatted metrics report.

    Parameters:
    -----------
    metrics : dict
        Dictionary of metrics from calculate_metrics()
    title : str
        Title for the report
    level : str
        Level of aggregation ('monthly' or 'outage')
    """
    print("\n" + "=" * 80)
    print(f"[{title}]")
    print("=" * 80)

    print(f"\nTest Period: {level.capitalize()} Level")
    print(f"  Total samples: {metrics['total_samples']:,}")

    print(f"\n[Shadow Price Prediction Performance]")
    print(f"  MAE:  ${metrics['mae']:.2f}")
    print(f"  RMSE: ${metrics['rmse']:.2f}")
    if metrics['mape'] > 0:
        print(f"  MAPE: {metrics['mape']:.2f}%")

    print(f"\n[Classification Performance]")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall:    {metrics['recall']:.3f}")
    print(f"  F1-Score:  {metrics['f1_score']:.3f}")
    print(f"  Accuracy:  {metrics['accuracy']:.3f}")

    print(f"\n[Binding Classification Details]")
    print(f"  Correctly classified: {metrics['true_positives'] + metrics['true_negatives']:,} / "
          f"{metrics['total_samples']:,} ({metrics['accuracy']*100:.2f}%)")

    print(f"\n  True Positives (correctly identified binding): {metrics['true_positives']:,}")
    print(f"  False Positives (incorrectly predicted as binding): {metrics['false_positives']:,}")
    print(f"  True Negatives (correctly identified non-binding): {metrics['true_negatives']:,}")
    print(f"  False Negatives (missed binding constraints): {metrics['false_negatives']:,}")

    print(f"\n[Binding Rate Analysis]")
    print(f"  Actual binding rate: {metrics['actual_binding_rate']*100:.2f}% "
          f"({metrics['actual_binding_count']:,} samples)")
    print(f"  Predicted binding rate: {metrics['predicted_binding_rate']*100:.2f}% "
          f"({metrics['predicted_binding_count']:,} samples)")


def analyze_results(
    results_per_outage: pd.DataFrame,
    final_results: pd.DataFrame,
    verbose: bool = True
) -> Dict:
    """
    Analyze prediction results at both outage and monthly levels.

    Parameters:
    -----------
    results_per_outage : pd.DataFrame
        Per-outage-date results
    final_results : pd.DataFrame
        Monthly aggregated results
    verbose : bool
        Print detailed report

    Returns:
    --------
    analysis : dict
        Dictionary with metrics for both levels
    """
    # Calculate metrics for both levels
    monthly_metrics = calculate_metrics(final_results)

    # For per-outage, we need to aggregate by outage_date first
    outage_metrics = {}
    for outage_date, outage_data in results_per_outage.groupby('outage_date'):
        outage_metrics[outage_date.strftime('%Y-%m-%d')] = calculate_metrics(outage_data)

    # Print reports if verbose
    if verbose:
        # Monthly aggregated report
        print_metrics_report(monthly_metrics, "MONTHLY AGGREGATED RESULTS", "monthly")

        # Per-outage summary
        print("\n" + "=" * 80)
        print("[PER-OUTAGE SUMMARY]")
        print("=" * 80)
        print(f"\nTotal outage dates: {len(outage_metrics)}")

        # Summary statistics across outages
        avg_f1 = np.mean([m['f1_score'] for m in outage_metrics.values()])
        avg_precision = np.mean([m['precision'] for m in outage_metrics.values()])
        avg_recall = np.mean([m['recall'] for m in outage_metrics.values()])
        avg_mae = np.mean([m['mae'] for m in outage_metrics.values()])

        print(f"\nAverage metrics across outage dates:")
        print(f"  Avg F1-Score:  {avg_f1:.3f}")
        print(f"  Avg Precision: {avg_precision:.3f}")
        print(f"  Avg Recall:    {avg_recall:.3f}")
        print(f"  Avg MAE:       ${avg_mae:.2f}")

        # Show per-outage breakdown
        print(f"\nPer-Outage Performance:")
        print(f"  {'Date':<12} {'Samples':>8} {'F1':>6} {'Prec':>6} {'Rec':>6} {'MAE':>10}")
        print(f"  {'-'*12} {'-'*8} {'-'*6} {'-'*6} {'-'*6} {'-'*10}")
        for date, metrics in sorted(outage_metrics.items()):
            print(f"  {date:<12} {metrics['total_samples']:>8,} "
                  f"{metrics['f1_score']:>6.3f} {metrics['precision']:>6.3f} "
                  f"{metrics['recall']:>6.3f} ${metrics['mae']:>9.2f}")

    return {
        'monthly': monthly_metrics,
        'per_outage': outage_metrics
    }


# ---------------------------------------------------------------------------
# Train/Val/Holdout split evaluation
# ---------------------------------------------------------------------------

def _compute_topk_metrics(
    y_true_bin: np.ndarray,
    y_actual_sp: np.ndarray,
    y_proba: np.ndarray,
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
    y_pred_bin: np.ndarray | None = None,
) -> Dict:
    """Compute ranking-based top-K metrics.

    Threshold-free metrics (ranked by predicted probability)
    --------------------------------------------------------
    value_capture_at_k : float
        Fraction of total actual shadow-price value captured in the top-K
        (by predicted probability).
        ``VC@K = Σ actual_sp[top-K] / Σ actual_sp[all]``
        Primary metric for FTR portfolio quality.
    mean_value_at_k : float
        Average actual shadow price in the probability-ranked top-K.
    lift_at_k : float
        ``Precision@K / base_binding_rate``.
    precision_at_k : float
        Fraction of probability-ranked top-K that are actually binding.
    recall_at_k : float
        Fraction of all actual positives in the probability-ranked top-K.

    Threshold-dependent metric (uses binary predictions)
    ----------------------------------------------------
    capture_at_k : float
        Of the K constraints with the highest *actual* shadow prices,
        how many did the model predict as binding (above threshold)?
        ``Capture@K = |{actual top-K by SP} ∩ {predicted binding}| / K``
        Measures whether the pipeline finds the most valuable constraints.
        Only computed when ``y_pred_bin`` is provided.

    Scalar metrics
    --------------
    ndcg : float
        Normalized Discounted Cumulative Gain using actual shadow price
        as graded relevance.
    """
    n = len(y_proba)
    n_positive = int(y_true_bin.sum())
    base_rate = n_positive / n if n > 0 else 0.0
    total_value = float(y_actual_sp.sum())

    # Sort by predicted probability descending (model ranking)
    order = np.argsort(-y_proba)
    sorted_labels = y_true_bin[order]
    sorted_values = y_actual_sp[order]

    # Sort by actual shadow price descending (ground truth ranking)
    actual_order = np.argsort(-y_actual_sp)

    topk: Dict = {}
    for k in k_values:
        if k > n:
            continue
        hits = int(sorted_labels[:k].sum())
        captured_value = float(sorted_values[:k].sum())

        prec_k = hits / k
        rec_k = hits / n_positive if n_positive > 0 else 0.0
        lift_k = prec_k / base_rate if base_rate > 0 else 0.0
        vc_k = captured_value / total_value if total_value > 0 else 0.0
        mean_val_k = captured_value / k

        entry = {
            "precision": round(prec_k, 4),
            "recall": round(rec_k, 4),
            "lift": round(lift_k, 2),
            "value_capture": round(vc_k, 4),
            "mean_value": round(mean_val_k, 2),
        }

        # Capture@K: of the K most valuable actual constraints, how many
        # did the model predict as binding (threshold-dependent)?
        if y_pred_bin is not None:
            actual_top_k_indices = actual_order[:k]
            caught = int(y_pred_bin[actual_top_k_indices].sum())
            entry["capture"] = round(caught / k, 4)

        topk[k] = entry

    # NDCG — sklearn wants (n_samples, n_labels) shaped arrays, so
    # reshape to a single-query ranking problem.
    if total_value > 0:
        ndcg_val = float(
            ndcg_score(
                y_actual_sp.reshape(1, -1),
                y_proba.reshape(1, -1),
            )
        )
    else:
        ndcg_val = float("nan")

    return {"topk": topk, "ndcg": round(ndcg_val, 4)}


def evaluate_split(
    predictor,
    data: pd.DataFrame,
    split_name: str,
    beta: float = 2.0,
    verbose: bool = True,
) -> Dict | None:
    """Evaluate a trained Predictor on a labeled data split (val or holdout).

    The function groups data by (auction_month, market_month), calls
    ``predictor.predict()`` for each group, concatenates the per-outage
    results, then computes classification and regression metrics.

    Parameters
    ----------
    predictor : Predictor
        A fully-trained ``Predictor`` instance.
    data : pd.DataFrame
        Labeled data with the same schema as test data (must include
        ``label``, ``auction_month``, ``market_month``, etc.).
    split_name : str
        Human-readable name for the split (used in printed output).
    beta : float
        Beta value for F-beta score (default 2.0 to match threshold
        optimisation).
    verbose : bool
        If True, print a compact evaluation summary.

    Returns
    -------
    dict or None
        ``None`` if data is empty or prediction fails for every group.
        Otherwise a dict with keys ``split``, ``n_samples``, ``clf``,
        ``reg_tp``, ``reg_all``.
    """
    if data is None or len(data) == 0:
        if verbose:
            print(f"[{split_name.upper()}] — skipped (no data)")
        return None

    # Predictor.predict() expects uniform auction_month/market_month per call,
    # so group and predict each slice independently.
    all_results: list[pd.DataFrame] = []
    for (am, mm), group_df in data.groupby(["auction_month", "market_month"]):
        if len(group_df) == 0:
            continue
        try:
            results_per_outage, _, _ = predictor.predict(group_df, verbose=False)
            all_results.append(results_per_outage)
        except Exception as exc:  # noqa: BLE001
            if verbose:
                print(f"  Warning: eval predict failed for {am}/{mm}: {exc}")

    if not all_results:
        if verbose:
            print(f"[{split_name.upper()}] — skipped (no valid predictions)")
        return None

    combined = pd.concat(all_results, axis=0)

    # --- Classification metrics ---
    y_true_bin = combined["actual_binding"].values
    y_pred_bin = combined["predicted_binding"].values
    y_proba = combined["binding_probability"].values

    n_samples = len(combined)
    n_positive = int(y_true_bin.sum())
    binding_rate = n_positive / n_samples if n_samples > 0 else 0.0
    pred_binding_rate = float(y_pred_bin.sum()) / n_samples if n_samples > 0 else 0.0

    prec = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    rec = recall_score(y_true_bin, y_pred_bin, zero_division=0)
    f1_val = f1_score(y_true_bin, y_pred_bin, zero_division=0)
    fb = fbeta_score(y_true_bin, y_pred_bin, beta=beta, zero_division=0)

    # AUC metrics require both classes present
    if 0 < n_positive < n_samples:
        auc_roc = roc_auc_score(y_true_bin, y_proba)
        avg_prec = average_precision_score(y_true_bin, y_proba)
    else:
        auc_roc = float("nan")
        avg_prec = float("nan")

    # --- Regression on true positives (predicted=binding AND actual=binding) ---
    tp_mask = (y_true_bin == 1) & (y_pred_bin == 1)
    n_tp = int(tp_mask.sum())

    y_actual_sp = combined["actual_shadow_price"].values
    y_pred_sp = combined["predicted_shadow_price"].values

    if n_tp > 0:
        mae_tp = float(mean_absolute_error(y_actual_sp[tp_mask], y_pred_sp[tp_mask]))
        rmse_tp = float(np.sqrt(mean_squared_error(y_actual_sp[tp_mask], y_pred_sp[tp_mask])))
        sp_result = spearmanr(y_actual_sp[tp_mask], y_pred_sp[tp_mask])
        spearman_tp = float(sp_result.statistic) if not np.isnan(sp_result.statistic) else float("nan")
        mean_actual_tp = float(np.mean(y_actual_sp[tp_mask]))
        mean_pred_tp = float(np.mean(y_pred_sp[tp_mask]))
    else:
        mae_tp = rmse_tp = spearman_tp = float("nan")
        mean_actual_tp = mean_pred_tp = float("nan")

    # --- Regression on all samples (end-to-end pipeline quality) ---
    mae_all = float(mean_absolute_error(y_actual_sp, y_pred_sp))
    rmse_all = float(np.sqrt(mean_squared_error(y_actual_sp, y_pred_sp)))

    # --- Top-K ranking metrics (threshold-free, per-outage level) ---
    ranking = _compute_topk_metrics(y_true_bin, y_actual_sp, y_proba, y_pred_bin=y_pred_bin)

    # --- Constraint-level top-K (aggregate across outage dates, then rank) ---
    constraint_ranking: Dict = {}
    constraint_group_cols = []
    for col in ("branch_name", "constraint_id"):
        if col in combined.columns:
            constraint_group_cols.append(col)
            break
    if "flow_direction" in combined.columns:
        constraint_group_cols.append("flow_direction")

    if constraint_group_cols:
        constraint_agg = (
            combined.groupby(constraint_group_cols, observed=True)
            .agg(
                actual_sp_sum=("actual_shadow_price", "sum"),
                pred_sp_sum=("predicted_shadow_price", "sum"),
                binding_prob_mean=("binding_probability", "mean"),
                actual_binding_max=("actual_binding", "max"),
                predicted_binding_max=("predicted_binding", "max"),
                actual_sp_mean=("actual_shadow_price", "mean"),
                pred_sp_mean=("predicted_shadow_price", "mean"),
                n_outages=("actual_shadow_price", "size"),
            )
            .reset_index()
        )

        c_y_true_bin = (constraint_agg["actual_binding_max"] > 0).astype(int).values
        c_y_actual_sp = constraint_agg["actual_sp_sum"].values
        c_y_proba = constraint_agg["binding_prob_mean"].values
        c_y_pred_bin = (constraint_agg["predicted_binding_max"] > 0).astype(int).values

        if len(constraint_agg) > 0:
            constraint_ranking = _compute_topk_metrics(
                c_y_true_bin, c_y_actual_sp, c_y_proba,
                k_values=CONSTRAINT_K_VALUES,
                y_pred_bin=c_y_pred_bin,
            )
            constraint_ranking["n_constraints"] = len(constraint_agg)
            constraint_ranking["n_binding"] = int(c_y_true_bin.sum())

    result = {
        "split": split_name,
        "n_samples": n_samples,
        "clf": {
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1_val),
            "fbeta": float(fb),
            "auc_roc": float(auc_roc),
            "avg_precision": float(avg_prec),
            "n_positive": n_positive,
            "binding_rate": float(binding_rate),
            "pred_binding_rate": float(pred_binding_rate),
        },
        "reg_tp": {
            "n": n_tp,
            "mae": mae_tp,
            "rmse": rmse_tp,
            "spearman": spearman_tp,
            "mean_actual": mean_actual_tp,
            "mean_pred": mean_pred_tp,
        },
        "reg_all": {
            "mae": mae_all,
            "rmse": rmse_all,
        },
        "ranking": ranking,
        "ranking_constraint": constraint_ranking,
    }

    if verbose:
        _print_split_eval(result)

    return result


def _print_topk_table(topk: Dict, label: str = "", ndcg: float = float("nan")) -> None:
    """Print a top-K ranking table."""
    ndcg_str = f"NDCG={ndcg:.3f}" if not np.isnan(ndcg) else "NDCG=n/a"
    print(f"  {label}({ndcg_str}):")
    if topk:
        has_capture = any("capture" in topk[k] for k in topk)
        if has_capture:
            header = "    {:>6s}  {:>8s}  {:>8s}  {:>5s}  {:>10s}  {:>10s}  {:>10s}".format(
                "K", "Prec@K", "Rec@K", "Lift", "ValCap@K", "MeanVal@K", "Capture@K"
            )
        else:
            header = "    {:>6s}  {:>8s}  {:>8s}  {:>5s}  {:>10s}  {:>10s}".format(
                "K", "Prec@K", "Rec@K", "Lift", "ValCap@K", "MeanVal@K"
            )
        print(header)
        for k in sorted(topk.keys()):
            m = topk[k]
            base = "    {:>6,d}  {:>8.1%}  {:>8.1%}  {:>5.1f}x  {:>9.1%}  ${:>9,.0f}".format(
                k, m["precision"], m["recall"], m["lift"],
                m["value_capture"], m["mean_value"],
            )
            if has_capture:
                base += "  {:>9.1%}".format(m.get("capture", 0.0))
            print(base)


def _print_split_eval(metrics: Dict) -> None:
    """Print a compact evaluation summary for one split."""
    split = metrics["split"]
    clf = metrics["clf"]
    rtp = metrics["reg_tp"]
    rall = metrics["reg_all"]
    ranking = metrics.get("ranking", {})
    ranking_cst = metrics.get("ranking_constraint", {})

    print(f"\n[{split.upper()} EVALUATION]")
    print(
        f"  Classification: n={metrics['n_samples']:,}  "
        f"binding_rate={clf['binding_rate']:.1%}"
    )
    print(
        f"    Prec={clf['precision']:.3f}  Recall={clf['recall']:.3f}  "
        f"F1={clf['f1']:.3f}  F2={clf['fbeta']:.3f}  "
        f"AUC-ROC={clf['auc_roc']:.3f}  AUC-PR={clf['avg_precision']:.3f}"
    )

    if rtp["n"] > 0:
        print(
            f"  Regression (on {rtp['n']:,} TPs):\n"
            f"    MAE=${rtp['mae']:.2f}  RMSE=${rtp['rmse']:.2f}  "
            f"Spearman={rtp['spearman']:.3f}"
        )
    else:
        print("  Regression (on TPs): no true positives predicted")

    print(
        f"  Regression (all {metrics['n_samples']:,}):\n"
        f"    MAE=${rall['mae']:.2f}  RMSE=${rall['rmse']:.2f}"
    )

    # Per-outage top-K ranking metrics
    if ranking:
        _print_topk_table(
            ranking.get("topk", {}),
            label="Per-outage ranking ",
            ndcg=ranking.get("ndcg", float("nan")),
        )

    # Constraint-level top-K ranking metrics
    if ranking_cst:
        n_cst = ranking_cst.get("n_constraints", 0)
        n_bind = ranking_cst.get("n_binding", 0)
        print(
            f"  Constraint-level ranking "
            f"(n={n_cst:,}, binding={n_bind:,}, rate={n_bind / n_cst:.1%}):"
        ) if n_cst > 0 else None
        _print_topk_table(
            ranking_cst.get("topk", {}),
            label="Constraint ranking ",
            ndcg=ranking_cst.get("ndcg", float("nan")),
        )


def aggregate_eval_metrics(
    all_eval_metrics: list[tuple[pd.Timestamp, Dict]],
    verbose: bool = True,
) -> Dict:
    """Aggregate per-auction-month evaluation metrics.

    Parameters
    ----------
    all_eval_metrics : list of (auction_month, eval_dict)
        Each eval_dict has optional 'val' and 'holdout' keys mapping to
        the dicts returned by ``evaluate_split``.
    verbose : bool
        Print a summary table.

    Returns
    -------
    dict
        Keys: ``per_auction_month``, ``mean_val``, ``mean_holdout``.
    """
    if not all_eval_metrics:
        return {}

    result: Dict = {"per_auction_month": {}}

    for split in ("val", "holdout"):
        split_metrics: list[Dict] = []
        for am, em in all_eval_metrics:
            if split in em and em[split] is not None:
                split_metrics.append(em[split])
                result["per_auction_month"].setdefault(
                    am.strftime("%Y-%m"), {}
                )[split] = em[split]

        if not split_metrics:
            continue

        # Compute mean across auction months
        mean_m: Dict = {
            "n_months": len(split_metrics),
            "n_samples_total": sum(m["n_samples"] for m in split_metrics),
            "clf": {},
            "reg_tp": {},
            "reg_all": {},
        }
        for key in ("precision", "recall", "f1", "fbeta", "auc_roc", "avg_precision"):
            vals = [m["clf"][key] for m in split_metrics if not np.isnan(m["clf"][key])]
            mean_m["clf"][key] = float(np.mean(vals)) if vals else float("nan")

        for key in ("mae", "rmse", "spearman"):
            vals = [m["reg_tp"][key] for m in split_metrics if not np.isnan(m["reg_tp"][key])]
            mean_m["reg_tp"][key] = float(np.mean(vals)) if vals else float("nan")

        for key in ("mae", "rmse"):
            vals = [m["reg_all"][key] for m in split_metrics if not np.isnan(m["reg_all"][key])]
            mean_m["reg_all"][key] = float(np.mean(vals)) if vals else float("nan")

        # Aggregate ranking metrics (mean NDCG across months)
        ndcg_vals = [
            m["ranking"]["ndcg"]
            for m in split_metrics
            if "ranking" in m and not np.isnan(m["ranking"].get("ndcg", float("nan")))
        ]
        mean_m["ranking"] = {
            "ndcg": float(np.mean(ndcg_vals)) if ndcg_vals else float("nan"),
        }

        # Aggregate constraint-level ranking (mean NDCG across months)
        cst_ndcg_vals = [
            m["ranking_constraint"]["ndcg"]
            for m in split_metrics
            if "ranking_constraint" in m
            and m["ranking_constraint"]
            and not np.isnan(m["ranking_constraint"].get("ndcg", float("nan")))
        ]
        mean_m["ranking_constraint"] = {
            "ndcg": float(np.mean(cst_ndcg_vals)) if cst_ndcg_vals else float("nan"),
        }

        result[f"mean_{split}"] = mean_m

    if verbose and (result.get("mean_val") or result.get("mean_holdout")):
        _print_eval_aggregate(result)

    return result


def _print_eval_aggregate(agg: Dict) -> None:
    """Print aggregated evaluation summary across auction months."""
    print("\n" + "=" * 80)
    print("[EVALUATION SUMMARY — MEAN ACROSS AUCTION MONTHS]")
    print("=" * 80)

    for split in ("val", "holdout"):
        key = f"mean_{split}"
        if key not in agg:
            continue
        m = agg[key]
        clf = m["clf"]
        rtp = m["reg_tp"]
        rall = m["reg_all"]

        print(f"\n  {split.upper()} ({m['n_months']} month(s), {m['n_samples_total']:,} samples total):")
        print(
            f"    Clf:  Prec={clf['precision']:.3f}  Recall={clf['recall']:.3f}  "
            f"F1={clf['f1']:.3f}  F2={clf['fbeta']:.3f}  "
            f"AUC-ROC={clf['auc_roc']:.3f}  AUC-PR={clf['avg_precision']:.3f}"
        )
        if not np.isnan(rtp.get("mae", float("nan"))):
            print(
                f"    Reg (TPs): MAE=${rtp['mae']:.2f}  RMSE=${rtp['rmse']:.2f}  "
                f"Spearman={rtp['spearman']:.3f}"
            )
        else:
            print("    Reg (TPs): no true positives predicted")
        print(f"    Reg (all): MAE=${rall['mae']:.2f}  RMSE=${rall['rmse']:.2f}")

        # Ranking summary (mean NDCG across months)
        ranking = m.get("ranking", {})
        if ranking:
            ndcg_val = ranking.get("ndcg", float("nan"))
            if not np.isnan(ndcg_val):
                print(f"    Per-outage ranking: NDCG={ndcg_val:.3f}")

        # Constraint-level ranking summary
        cst_ranking = m.get("ranking_constraint", {})
        if cst_ranking:
            cst_ndcg = cst_ranking.get("ndcg", float("nan"))
            if not np.isnan(cst_ndcg):
                print(f"    Constraint ranking: NDCG={cst_ndcg:.3f}")


# ---------------------------------------------------------------------------
# score_results_df — comprehensive scoring for baseline benchmarking
# ---------------------------------------------------------------------------

def score_results_df(
    df: pd.DataFrame,
    beta_clf: float = 0.5,
    beta_current: float = 2.0,
    outage_k_values: tuple[int, ...] = DEFAULT_K_VALUES,
    constraint_k_values: tuple[int, ...] = CONSTRAINT_K_VALUES,
) -> Dict:
    """Score a results_per_outage DataFrame with comprehensive metrics.

    This is the single entry-point for the legacy baseline benchmark.
    It expects the standard ``results_per_outage`` schema produced by
    ``Predictor.predict()``::

        actual_shadow_price, predicted_shadow_price,
        actual_binding, predicted_binding,
        binding_probability, branch_name, constraint_id,
        flow_direction, outage_date, model_used

    Parameters
    ----------
    df : pd.DataFrame
        ``results_per_outage`` from one (auction_month, class_type, period_type) run.
    beta_clf : float
        Beta for the legacy F-beta (default 0.5 = precision-weighted).
    beta_current : float
        Beta for the current F-beta (default 2.0 = recall-weighted).
    outage_k_values : tuple of int
        K values for outage-level ValCap/Lift metrics.
    constraint_k_values : tuple of int
        K values for constraint-level ValCap/Lift metrics.

    Returns
    -------
    dict
        Nested dict with keys: ``stage1``, ``stage2``, ``combined``,
        ``ranking_outage``, ``ranking_constraint``, ``meta``.
    """
    if df is None or len(df) == 0:
        return {"error": "empty dataframe"}

    y_true_bin = df["actual_binding"].values.astype(int)
    y_pred_bin = df["predicted_binding"].values.astype(int)
    y_proba = df["binding_probability"].values.astype(float)
    y_actual_sp = df["actual_shadow_price"].values.astype(float)
    y_pred_sp = df["predicted_shadow_price"].values.astype(float)

    n = len(df)
    n_positive = int(y_true_bin.sum())
    binding_rate = n_positive / n if n > 0 else 0.0
    pred_binding_rate = float(y_pred_bin.sum()) / n if n > 0 else 0.0

    # ---- Stage 1: Classifier ----
    prec = float(precision_score(y_true_bin, y_pred_bin, zero_division=0))
    rec = float(recall_score(y_true_bin, y_pred_bin, zero_division=0))
    f1_val = float(f1_score(y_true_bin, y_pred_bin, zero_division=0))
    fb_legacy = float(fbeta_score(y_true_bin, y_pred_bin, beta=beta_clf, zero_division=0))
    fb_current = float(fbeta_score(y_true_bin, y_pred_bin, beta=beta_current, zero_division=0))

    if 0 < n_positive < n:
        auc_roc = float(roc_auc_score(y_true_bin, y_proba))
        avg_prec = float(average_precision_score(y_true_bin, y_proba))
        brier = float(brier_score_loss(y_true_bin, y_proba))
    else:
        auc_roc = avg_prec = brier = float("nan")

    stage1 = {
        "n_samples": n,
        "n_positive": n_positive,
        "binding_rate": round(binding_rate, 4),
        "pred_binding_rate": round(pred_binding_rate, 4),
        "auc_roc": round(auc_roc, 4),
        "avg_precision": round(avg_prec, 4),
        "brier_score": round(brier, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1_val, 4),
        "fbeta_0.5": round(fb_legacy, 4),
        "fbeta_2.0": round(fb_current, 4),
    }

    # ---- Stage 2: Regressor (on true positives) ----
    tp_mask = (y_true_bin == 1) & (y_pred_bin == 1)
    n_tp = int(tp_mask.sum())

    if n_tp > 1:
        mae_tp = float(mean_absolute_error(y_actual_sp[tp_mask], y_pred_sp[tp_mask]))
        rmse_tp = float(np.sqrt(mean_squared_error(y_actual_sp[tp_mask], y_pred_sp[tp_mask])))
        sp_result = spearmanr(y_actual_sp[tp_mask], y_pred_sp[tp_mask])
        spearman_tp = float(sp_result.statistic) if not np.isnan(sp_result.statistic) else float("nan")
        mean_actual_tp = float(np.mean(y_actual_sp[tp_mask]))
        mean_pred_tp = float(np.mean(y_pred_sp[tp_mask]))
        residuals_tp = y_actual_sp[tp_mask] - y_pred_sp[tp_mask]
        residual_std_tp = float(np.std(residuals_tp))
        residual_skew_tp = float(scipy_skew(residuals_tp))
    else:
        mae_tp = rmse_tp = spearman_tp = float("nan")
        mean_actual_tp = mean_pred_tp = float("nan")
        residual_std_tp = residual_skew_tp = float("nan")

    stage2 = {
        "n_tp": n_tp,
        "mae_tp": round(mae_tp, 2) if not np.isnan(mae_tp) else mae_tp,
        "rmse_tp": round(rmse_tp, 2) if not np.isnan(rmse_tp) else rmse_tp,
        "spearman_tp": round(spearman_tp, 4) if not np.isnan(spearman_tp) else spearman_tp,
        "mean_actual_tp": round(mean_actual_tp, 2) if not np.isnan(mean_actual_tp) else mean_actual_tp,
        "mean_pred_tp": round(mean_pred_tp, 2) if not np.isnan(mean_pred_tp) else mean_pred_tp,
        "bias_tp": round(mean_pred_tp - mean_actual_tp, 2) if not (np.isnan(mean_pred_tp) or np.isnan(mean_actual_tp)) else float("nan"),
        "residual_std_tp": round(residual_std_tp, 2) if not np.isnan(residual_std_tp) else residual_std_tp,
        "residual_skew_tp": round(residual_skew_tp, 4) if not np.isnan(residual_skew_tp) else residual_skew_tp,
    }

    # ---- Combined: end-to-end ----
    rmse_all = float(np.sqrt(mean_squared_error(y_actual_sp, y_pred_sp)))
    mae_all = float(mean_absolute_error(y_actual_sp, y_pred_sp))

    combined = {
        "rmse_all": round(rmse_all, 2),
        "mae_all": round(mae_all, 2),
    }

    # ---- Outage-level ranking (top-K) ----
    ranking_outage = _compute_topk_metrics(y_true_bin, y_actual_sp, y_proba, k_values=outage_k_values, y_pred_bin=y_pred_bin)

    # ---- Constraint-level ranking ----
    constraint_group_cols = []
    for col in ("branch_name", "constraint_id"):
        if col in df.columns:
            constraint_group_cols.append(col)
            break
    if "flow_direction" in df.columns:
        constraint_group_cols.append("flow_direction")

    ranking_constraint: Dict = {}
    if constraint_group_cols:
        constraint_agg = (
            df.groupby(constraint_group_cols, observed=True)
            .agg(
                actual_sp_sum=("actual_shadow_price", "sum"),
                pred_sp_sum=("predicted_shadow_price", "sum"),
                binding_prob_mean=("binding_probability", "mean"),
                actual_binding_max=("actual_binding", "max"),
                predicted_binding_max=("predicted_binding", "max"),
                n_outages=("actual_shadow_price", "size"),
            )
            .reset_index()
        )
        c_y_true_bin = (constraint_agg["actual_binding_max"] > 0).astype(int).values
        c_y_actual_sp = constraint_agg["actual_sp_sum"].values
        c_y_proba = constraint_agg["binding_prob_mean"].values
        c_y_pred_bin = (constraint_agg["predicted_binding_max"] > 0).astype(int).values

        if len(constraint_agg) > 0:
            ranking_constraint = _compute_topk_metrics(
                c_y_true_bin, c_y_actual_sp, c_y_proba,
                k_values=constraint_k_values,
                y_pred_bin=c_y_pred_bin,
            )
            ranking_constraint["n_constraints"] = len(constraint_agg)
            ranking_constraint["n_binding"] = int(c_y_true_bin.sum())

    # ---- Model type breakdown ----
    model_breakdown: Dict = {}
    if "model_used" in df.columns:
        for mtype, mdf in df.groupby("model_used", observed=True):
            n_m = len(mdf)
            n_m_pos = int(mdf["actual_binding"].sum())
            model_breakdown[mtype] = {
                "n_samples": n_m,
                "n_positive": n_m_pos,
                "binding_rate": round(n_m_pos / n_m, 4) if n_m > 0 else 0.0,
            }

    meta = {
        "n_outage_dates": int(df["outage_date"].nunique()) if "outage_date" in df.columns else 0,
        "n_constraints": int(df[constraint_group_cols[0]].nunique()) if constraint_group_cols else 0,
    }

    return {
        "stage1": stage1,
        "stage2": stage2,
        "combined": combined,
        "ranking_outage": ranking_outage,
        "ranking_constraint": ranking_constraint,
        "model_breakdown": model_breakdown,
        "meta": meta,
    }


def print_score_report(scores: Dict, label: str = "") -> None:
    """Print a formatted report from ``score_results_df`` output."""
    if "error" in scores:
        print(f"[{label}] Error: {scores['error']}")
        return

    s1 = scores["stage1"]
    s2 = scores["stage2"]
    comb = scores["combined"]
    ro = scores.get("ranking_outage", {})
    rc = scores.get("ranking_constraint", {})

    print(f"\n{'='*70}")
    if label:
        print(f"  {label}")
    print(f"{'='*70}")

    print(f"\n  Stage 1 (Classifier)  n={s1['n_samples']:,}  binding_rate={s1['binding_rate']:.1%}")
    print(f"    AUC-ROC={s1['auc_roc']:.4f}  AUC-PR={s1['avg_precision']:.4f}  Brier={s1['brier_score']:.4f}")
    print(f"    Prec={s1['precision']:.4f}  Recall={s1['recall']:.4f}  F1={s1['f1']:.4f}")
    print(f"    F0.5={s1['fbeta_0.5']:.4f}  F2.0={s1['fbeta_2.0']:.4f}")
    print(f"    pred_binding_rate={s1['pred_binding_rate']:.1%}")

    print(f"\n  Stage 2 (Regressor on {s2['n_tp']:,} TPs)")
    if s2["n_tp"] > 0:
        print(f"    MAE=${s2['mae_tp']:.2f}  RMSE=${s2['rmse_tp']:.2f}  Spearman={s2['spearman_tp']:.4f}")
        print(f"    mean_actual=${s2['mean_actual_tp']:.2f}  mean_pred=${s2['mean_pred_tp']:.2f}  bias=${s2['bias_tp']:.2f}")
        print(f"    residual_std=${s2['residual_std_tp']:.2f}  residual_skew={s2['residual_skew_tp']:.4f}")
    else:
        print("    (no true positives)")

    print(f"\n  Combined (all {s1['n_samples']:,})")
    print(f"    RMSE=${comb['rmse_all']:.2f}  MAE=${comb['mae_all']:.2f}")

    if ro:
        _print_topk_table(ro.get("topk", {}), label="Outage ranking ", ndcg=ro.get("ndcg", float("nan")))

    if rc:
        n_cst = rc.get("n_constraints", 0)
        n_bind = rc.get("n_binding", 0)
        if n_cst > 0:
            print(f"  Constraint ranking (n={n_cst:,}, binding={n_bind:,}, rate={n_bind/n_cst:.1%}):")
        _print_topk_table(rc.get("topk", {}), label="Constraint ranking ", ndcg=rc.get("ndcg", float("nan")))
