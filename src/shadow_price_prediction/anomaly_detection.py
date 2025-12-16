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
        # Stats: Horizon -> Branch -> Feature -> StatName -> Value
        self.flow_stats: dict[str, dict[str, dict[str, dict[str, float]]]] = {g.name: {} for g in config.horizon_groups}

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

        flow_features = list(
            set(self.config.anomaly_detection.detection_features + self.config.anomaly_detection.probability_features)
        )

        if verbose:
            print(f"\n[Characterizing Never-Binding Branches for {horizon_group.upper()}]")
            print(f"  Detection Features: {self.config.anomaly_detection.detection_features}")
            print(f"  Probability Features: {self.config.anomaly_detection.probability_features}")
            print("-" * 80)

        # Group by branch
        train_data_by_branch = {
            branch_name: branch_data for branch_name, branch_data in train_data.groupby("branch_name")
        }

        if verbose:
            print(f"Analyzing {len(train_data_by_branch):,} branches for never-binding patterns...")

        for branch_name, branch_data in train_data_by_branch.items():
            # Check if branch NEVER binds in training data
            n_binding = (branch_data["label"] > 0).mean()

            if (
                n_binding < self.config.training.min_branch_positive_ratio
                and len(branch_data) >= self.config.anomaly_detection.min_samples_for_stats
            ):
                self.never_binding_branches[horizon_group].add(branch_name)
                self.flow_stats[horizon_group][branch_name] = {}

                # Compute flow statistics for ALL configured features (Detection + Probability)
                for feature in flow_features:
                    if feature in branch_data.columns:
                        values = branch_data[feature].values

                        # Basic robustness check: if values are all identical (variance 0),
                        # we still compute stats but IQR will be 0.

                        self.flow_stats[horizon_group][branch_name][feature] = {
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
            count = len(self.never_binding_branches[horizon_group])
            print(f"\n✓ Never-Binding Branches Identified ({horizon_group}): {count:,}")

            # Show examples (for first feature)
            if count > 0 and len(flow_features) > 0:
                first_feat = flow_features[0]
                print(f"\n  Example Flow Statistics for '{first_feat}' (first 5 branches):")

                examples = list(self.flow_stats[horizon_group].items())[:5]
                for branch, feat_stats in examples:
                    if first_feat in feat_stats:
                        stats = feat_stats[first_feat]
                        print(
                            f"    {branch[:45]:45s}: mean={stats['mean']:.4f}, "
                            f"P95={stats['P95']:.4f}, IQR={stats['IQR']:.4f}"
                        )

                # Show distribution summary
                all_iqrs = []
                for feat_stats in self.flow_stats[horizon_group].values():
                    if first_feat in feat_stats:
                        all_iqrs.append(feat_stats[first_feat]["IQR"])

                if all_iqrs:
                    print(f"\n  Distribution of IQR for '{first_feat}':")
                    print(
                        f"    min={np.min(all_iqrs):.4f}, median={np.median(all_iqrs):.4f}, max={np.max(all_iqrs):.4f}"
                    )

    def detect_flow_anomaly(
        self, test_sample: pd.Series, branch_name: str, horizon_group: str
    ) -> tuple[bool, float, str]:
        """
        Detect flow anomaly using Two-Phase Logic (Detection -> Probability).

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
            True if Detection Score > Detection Threshold
        confidence : float
            Probability of binding (0-1), calculated only if is_anomaly=True
        reason : str
            Human-readable explanation
        """
        if branch_name not in self.flow_stats.get(horizon_group, {}):
            return False, 0.0, f"No flow statistics available for {horizon_group}"

        branch_stats = self.flow_stats[horizon_group][branch_name]
        k_multiplier = self.config.anomaly_detection.k_multiplier
        fraction = self.config.anomaly_detection.iqr_range_fraction

        # --- PHASE 1: DETECTION ---
        # "Is this an anomaly?"
        det_features = self.config.anomaly_detection.detection_features
        det_raw_weights_map = self.config.anomaly_detection.detection_weights
        det_threshold = self.config.anomaly_detection.detection_threshold

        det_raw_weights = {f: det_raw_weights_map.get(f, 1.0) for f in det_features}
        max_det_weight = max(det_raw_weights.values()) if det_raw_weights else 1.0
        # Scale factor so max_weight * 1.0_IQR = detection_threshold
        det_scale_factor = det_threshold / max_det_weight if max_det_weight > 1e-9 else 1.0

        det_score = 0.0
        det_details = []

        for feature in det_features:
            if feature not in test_sample or feature not in branch_stats:
                continue

            stats = branch_stats[feature]
            val = test_sample[feature]
            weight = det_raw_weights[feature] * det_scale_factor

            # Dynamic effective IQR: max of actual IQR and percentage of Range
            data_range = stats["max"] - stats["min"]
            dynamic_floor = data_range * fraction
            effective_iqr = max(stats["IQR"], dynamic_floor)
            # Prevent div by zero if feature is absolutely constant (range=0)
            effective_iqr = max(effective_iqr, 1e-9)

            feature_threshold = stats["P99"] + k_multiplier * effective_iqr

            # Component score
            score = max(0.0, (val - feature_threshold) / effective_iqr)

            det_score += score * weight
            if score > 0:
                det_details.append(f"{feature} (s={score:.1f})")

        is_anomaly = det_score > det_threshold

        if not is_anomaly:
            return False, 0.0, f"Normal Flow (Det Score={det_score:.2f} <= {det_threshold})"

        # --- PHASE 2: PROBABILITY ---
        # "How likely is this to bind?"
        prob_features = self.config.anomaly_detection.probability_features
        prob_raw_weights_map = self.config.anomaly_detection.probability_weights

        alpha = self.config.anomaly_detection.sigmoid_alpha
        beta = self.config.anomaly_detection.sigmoid_beta

        prob_raw_weights = {f: prob_raw_weights_map.get(f, 1.0) for f in prob_features}
        max_prob_weight = max(prob_raw_weights.values()) if prob_raw_weights else 1.0
        # Scale factor so max_weight * 1.0_IQR = beta (prob = 0.5)
        prob_scale_factor = beta / max_prob_weight if max_prob_weight > 1e-9 else 1.0

        prob_score = 0.0
        prob_details = []

        for feature in prob_features:
            if feature not in test_sample or feature not in branch_stats:
                continue

            stats = branch_stats[feature]
            val = test_sample[feature]
            weight = prob_raw_weights[feature] * prob_scale_factor

            # Dynamic effective IQR: max of actual IQR and percentage of Range
            data_range = stats["max"] - stats["min"]
            dynamic_floor = data_range * fraction
            effective_iqr = max(stats["IQR"], dynamic_floor)
            effective_iqr = max(effective_iqr, 1e-9)

            feature_threshold = stats["P99"] + k_multiplier * effective_iqr

            score = max(0.0, (val - feature_threshold) / effective_iqr)

            prob_score += score * weight
            if score > 0:
                prob_details.append(f"{feature}={val:.3f}(s={score:.1f},w={weight:.2f})")

        try:
            probability = 1.0 / (1.0 + np.exp(-alpha * (prob_score - beta)))
        except OverflowError:
            probability = 1.0 if prob_score > beta else 0.0

        reason = f"Anomaly Detected! (Det S={det_score:.2f} > {det_threshold}). Prob S={prob_score:.2f} -> P={probability:.2f}. Contributors: {', '.join(prob_details)}"

        return True, probability, reason
