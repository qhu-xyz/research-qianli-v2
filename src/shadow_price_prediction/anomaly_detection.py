"""
Flow anomaly detection for never-binding branches.
"""
import pandas as pd
import numpy as np
from typing import Dict, Set, Tuple

from .config import PredictionConfig


class AnomalyDetector:
    """Detects flow anomalies in never-binding branches."""

    def __init__(self, config: PredictionConfig):
        self.config = config
        self.never_binding_branches: Set[str] = set()
        self.flow_stats: Dict[str, Dict[str, float]] = {}

    def characterize_never_binding_branches(
        self,
        train_data: pd.DataFrame,
        verbose: bool = True
    ) -> None:
        """
        Identify never-binding branches and compute flow statistics.

        Parameters:
        -----------
        train_data : pd.DataFrame
            Training data
        verbose : bool
            Print progress messages
        """
        if not self.config.anomaly_detection.enabled:
            if verbose:
                print("\n[Anomaly Detection Disabled]")
            return

        if verbose:
            print(f"\n[Characterizing Never-Binding Branches (Flow Feature {self.config.anomaly_detection.flow_feature})]")
            print("-" * 80)

        # Group by branch
        train_data_by_branch = {
            branch_name: branch_data
            for branch_name, branch_data in train_data.groupby('branch_name')
        }

        if verbose:
            print(f"Analyzing {len(train_data_by_branch):,} branches for never-binding patterns...")

        for branch_name, branch_data in train_data_by_branch.items():
            # Check if branch NEVER binds in training data
            n_binding = (branch_data['label'] > 0).sum()

            if n_binding == 0 and len(branch_data) >= self.config.anomaly_detection.min_samples_for_stats:
                self.never_binding_branches.add(branch_name)

                # Compute flow statistics for specified feature
                flow_feature = self.config.anomaly_detection.flow_feature
                if flow_feature in branch_data.columns:
                    values = branch_data[flow_feature].values

                    self.flow_stats[branch_name] = {
                        'min': np.min(values),
                        'max': np.max(values),
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'P50': np.percentile(values, 50),
                        'P95': np.percentile(values, 95),
                        'P99': np.percentile(values, 99),
                        'Q1': np.percentile(values, 25),
                        'Q3': np.percentile(values, 75),
                        'IQR': np.percentile(values, 75) - np.percentile(values, 25)
                    }

        if verbose:
            print(f"\n✓ Never-Binding Branches Identified: {len(self.never_binding_branches):,}")
            print(f"  Branches with flow statistics (feature {flow_feature}): {len(self.flow_stats):,}")

            # Show examples
            if len(self.flow_stats) > 0:
                print(f"\n  Example Flow Statistics (first 5 branches):")
                for i, (branch, stats) in enumerate(list(self.flow_stats.items())[:5]):
                    print(f"    {branch[:45]:45s}: mean={stats['mean']:.4f}, "
                          f"P95={stats['P95']:.4f}, P99={stats['P99']:.4f}, "
                          f"IQR={stats['IQR']:.4f}")

                # Show distribution statistics
                all_means = [stats['mean'] for stats in self.flow_stats.values()]
                all_p99s = [stats['P99'] for stats in self.flow_stats.values()]
                all_iqrs = [stats['IQR'] for stats in self.flow_stats.values()]

                print(f"\n  Distribution Summary (across all never-binding branches):")
                print(f"    Mean flow {flow_feature}: min={np.min(all_means):.4f}, "
                      f"median={np.median(all_means):.4f}, max={np.max(all_means):.4f}")
                print(f"    P99 values: min={np.min(all_p99s):.4f}, "
                      f"median={np.median(all_p99s):.4f}, max={np.max(all_p99s):.4f}")
                print(f"    IQR values: min={np.min(all_iqrs):.4f}, "
                      f"median={np.median(all_iqrs):.4f}, max={np.max(all_iqrs):.4f}")

    def detect_flow_anomaly(
        self,
        test_sample: pd.Series,
        branch_name: str
    ) -> Tuple[bool, float, str]:
        """
        Detect flow anomaly for never-binding branch.

        Parameters:
        -----------
        test_sample : pd.Series
            Test sample with flow features
        branch_name : str
            Name of the branch

        Returns:
        --------
        is_anomaly : bool
            True if flow is anomalous (predict binding)
        confidence : float
            Confidence score (0-1) of anomaly detection
        reason : str
            Human-readable explanation
        """
        flow_feature = self.config.anomaly_detection.flow_feature

        if branch_name not in self.flow_stats:
            return False, 0.0, "No flow statistics available"

        if flow_feature not in test_sample:
            return False, 0.0, f"Feature {flow_feature} not found in test sample"

        stats = self.flow_stats[branch_name]
        test_flow = test_sample[flow_feature]

        # Anomaly threshold: P99 + k * IQR
        k_multiplier = self.config.anomaly_detection.k_multiplier
        anomaly_threshold = stats['P99'] + k_multiplier * stats['IQR']

        # Check if test flow exceeds threshold
        is_anomaly = test_flow > anomaly_threshold

        if is_anomaly:
            # Calculate confidence based on how far beyond threshold
            excess = test_flow - anomaly_threshold
            max_excess = stats['max'] - anomaly_threshold

            if max_excess > 0:
                confidence = min(1.0, 0.5 + 0.5 * (excess / max_excess))
            else:
                confidence = 0.75  # Default moderate confidence

            reason = (
                f"Flow {flow_feature} = {test_flow:.4f} exceeds "
                f"P99 + {k_multiplier}*IQR = {anomaly_threshold:.4f}"
            )
        else:
            confidence = 0.0
            reason = (
                f"Flow {flow_feature} = {test_flow:.4f} within normal range "
                f"(threshold = {anomaly_threshold:.4f})"
            )

        return is_anomaly, confidence, reason
