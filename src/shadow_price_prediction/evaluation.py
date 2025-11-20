"""
Evaluation metrics for shadow price predictions.
"""
import pandas as pd
import numpy as np
from typing import Dict
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    f1_score
)


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
