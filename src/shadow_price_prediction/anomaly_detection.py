"""
Flow anomaly detection for never-binding branches.
"""

import numpy as np
import pandas as pd

from .config import PredictionConfig


class AnomalyDetector:
    """Detects flow anomalies in never-binding branches."""

    def __init__(self, config: PredictionConfig):
        self.config = config
        # Per-horizon storage for never-binding branches and flow stats
        self.never_binding_branches: dict[str, set[str]] = {g.name: set() for g in config.horizon_groups}
        self.flow_stats: dict[str, dict[str, dict[str, float]]] = {g.name: {} for g in config.horizon_groups}

    def characterize_never_binding_branches(
        self, train_data: pd.DataFrame, horizon_group: str, verbose: bool = True
    ) -> None:
        """
        Identify never-binding branches and compute flow statistics for a specific horizon group.

        Parameters:
        -----------
        train_data : pd.DataFrame
            Training data (should be pre-filtered to the specific horizon group)
        horizon_group : str
            Horizon group name ('f0', 'f1', 'medium', 'long')
        verbose : bool
            Print progress messages
        """
        if not self.config.anomaly_detection.enabled:
            if verbose:
                print("\n[Anomaly Detection Disabled]")
            return

        if verbose:
            print(
                f"\n[Characterizing Never-Binding Branches for {horizon_group.upper()} (Flow Feature {self.config.anomaly_detection.flow_feature})]"
            )
            print("-" * 80)

        # Group by branch
        train_data_by_branch = {
            branch_name: branch_data for branch_name, branch_data in train_data.groupby("branch_name")
        }

        if verbose:
            print(f"Analyzing {len(train_data_by_branch):,} branches for never-binding patterns...")

        for branch_name, branch_data in train_data_by_branch.items():
            # Check if branch NEVER binds in training data
            n_binding = (branch_data["label"] > 0).sum()

            if n_binding == 0 and len(branch_data) >= self.config.anomaly_detection.min_samples_for_stats:
                self.never_binding_branches[horizon_group].add(branch_name)

                # Compute flow statistics for specified feature
                flow_feature = self.config.anomaly_detection.flow_feature
                if flow_feature in branch_data.columns:
                    values = branch_data[flow_feature].values

                    self.flow_stats[horizon_group][branch_name] = {
                        "min": np.min(values),
                        "max": np.max(values),
                        "mean": np.mean(values),
                        "std": np.std(values),
                        "P50": np.percentile(values, 50),
                        "P95": np.percentile(values, 95),
                        "P99": np.percentile(values, 99),
                        "Q1": np.percentile(values, 25),
                        "Q3": np.percentile(values, 75),
                        "IQR": np.percentile(values, 75) - np.percentile(values, 25),
                    }

        if verbose:
            print(
                f"\n✓ Never-Binding Branches Identified ({horizon_group}): {len(self.never_binding_branches[horizon_group]):,}"
            )
            print(f"  Branches with flow statistics (feature {flow_feature}): {len(self.flow_stats[horizon_group]):,}")

            # Show examples
            if len(self.flow_stats[horizon_group]) > 0:
                print("\n  Example Flow Statistics (first 5 branches):")
                for _i, (branch, stats) in enumerate(list(self.flow_stats[horizon_group].items())[:5]):
                    print(
                        f"    {branch[:45]:45s}: mean={stats['mean']:.4f}, "
                        f"P95={stats['P95']:.4f}, P99={stats['P99']:.4f}, "
                        f"IQR={stats['IQR']:.4f}"
                    )

                # Show distribution statistics
                all_means = [stats["mean"] for stats in self.flow_stats[horizon_group].values()]
                all_p99s = [stats["P99"] for stats in self.flow_stats[horizon_group].values()]
                all_iqrs = [stats["IQR"] for stats in self.flow_stats[horizon_group].values()]

                print(f"\n  Distribution Summary (across all never-binding branches in {horizon_group}):")
                print(
                    f"    Mean flow {flow_feature}: min={np.min(all_means):.4f}, "
                    f"median={np.median(all_means):.4f}, max={np.max(all_means):.4f}"
                )
                print(
                    f"    P99 values: min={np.min(all_p99s):.4f}, "
                    f"median={np.median(all_p99s):.4f}, max={np.max(all_p99s):.4f}"
                )
                print(
                    f"    IQR values: min={np.min(all_iqrs):.4f}, "
                    f"median={np.median(all_iqrs):.4f}, max={np.max(all_iqrs):.4f}"
                )

    def detect_flow_anomaly(
        self, test_sample: pd.Series, branch_name: str, horizon_group: str
    ) -> tuple[bool, float, str]:
        """
        Detect flow anomaly for never-binding branch at a specific horizon.

        Parameters:
        -----------
        test_sample : pd.Series
            Test sample with flow features
        branch_name : str
            Name of the branch
        horizon_group : str
            Horizon group name ('f0', 'f1', 'medium', 'long')

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

        if branch_name not in self.flow_stats.get(horizon_group, {}):
            return False, 0.0, f"No flow statistics available for {horizon_group}"

        if flow_feature not in test_sample:
            return False, 0.0, f"Feature {flow_feature} not found in test sample"

        stats = self.flow_stats[horizon_group][branch_name]
        test_flow = test_sample[flow_feature]

        # Anomaly threshold: P99 + k * IQR
        k_multiplier = self.config.anomaly_detection.k_multiplier
        anomaly_threshold = stats["P99"] + k_multiplier * stats["IQR"]

        # Check if test flow exceeds threshold
        is_anomaly = test_flow > anomaly_threshold

        if is_anomaly:
            # Calculate confidence based on how far beyond threshold
            excess = test_flow - anomaly_threshold
            max_excess = stats["max"] - anomaly_threshold

            if max_excess > 0:
                confidence = min(1.0, 0.5 + 0.5 * (excess / max_excess))
            else:
                confidence = 0.75  # Default moderate confidence

            reason = f"Flow {flow_feature} = {test_flow:.4f} exceeds P99 + {k_multiplier}*IQR = {anomaly_threshold:.4f}"
        else:
            confidence = 0.0
            reason = f"Flow {flow_feature} = {test_flow:.4f} within normal range (threshold = {anomaly_threshold:.4f})"

        return is_anomaly, confidence, reason
