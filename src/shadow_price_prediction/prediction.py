"""
Prediction pipeline for shadow prices.
"""

import numpy as np
import pandas as pd

from .anomaly_detection import AnomalyDetector
from .config import PredictionConfig
from .models import ShadowPriceModels, predict_ensemble


class Predictor:
    """Handles prediction logic for shadow prices."""

    def __init__(self, config: PredictionConfig, models: ShadowPriceModels, anomaly_detector: AnomalyDetector):
        self.config = config
        self.models = models
        self.anomaly_detector = anomaly_detector

    def predict(self, test_data: pd.DataFrame, verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Make predictions on test data.

        Parameters:
        -----------
        test_data : pd.DataFrame
            Test data
        verbose : bool
            Print progress messages

        Returns:
        --------
        results_per_outage : pd.DataFrame
            Per-outage-date predictions
        final_results : pd.DataFrame
            Monthly aggregated predictions
        """
        if verbose:
            print("\n[Making Predictions with Branch-Specific Models + Flow Anomaly Detection]")
            print("=" * 80)

        # Prepare test features
        X_test = test_data[self.config.features.all_features].copy()
        y_test = test_data["label"].copy()

        # Calculate forecast horizon for weight selection
        # Assumes test_data has auction_month and market_month columns
        auction_month = test_data["auction_month"].iloc[0]
        market_month = test_data["market_month"].iloc[0]
        forecast_horizon = (market_month.year - auction_month.year) * 12 + (market_month.month - auction_month.month)

        if verbose:
            print(f"\nForecast Horizon: {forecast_horizon} months")
            print(f"  Auction Month: {auction_month.strftime('%Y-%m')}")
            print(f"  Market Month: {market_month.strftime('%Y-%m')}")

        # Initialize prediction arrays
        y_pred_binary = np.zeros(len(X_test), dtype=int)
        y_pred_proba = np.zeros(len(X_test))
        y_pred_proba_scaled = np.zeros(len(X_test))
        y_pred_threshold = np.zeros(len(X_test))
        y_pred_shadow_price = np.zeros(len(X_test))
        model_used = [""] * len(X_test)

        # Get unique branches in test set
        unique_branches_in_test = test_data["branch_name"].unique()

        if verbose:
            print(f"\nProcessing {len(unique_branches_in_test):,} unique branches...")
            print(f"Total test samples: {len(X_test):,}")
            print("\nFeature filtering:")
            print(
                f"  Classification uses: {self.config.features.step1_features} "
                f"({len(self.config.features.step1_features)} features)"
            )
            print(
                f"  Regression uses: {self.config.features.step2_features} "
                f"({len(self.config.features.step2_features)} features)"
            )

        # Track statistics
        used_branch_clf_count = 0
        used_default_clf_count = 0
        used_branch_reg_count = 0
        used_default_reg_count = 0
        no_reg_model_count = 0

        anomaly_detection_stats = {
            "branches_checked": 0,
            "samples_checked": 0,
            "anomalies_detected": 0,
            "anomaly_binding_predictions": 0,
            "anomaly_examples": [],
        }

        branches_processed = 0

        # Loop over each unique branch in the test set
        for branch_name in unique_branches_in_test:
            # Get all test samples for this branch using boolean mask
            branch_mask = test_data["branch_name"] == branch_name
            branch_indices = np.where(branch_mask)[0]  # Get positional indices

            X_branch_all = X_test.iloc[branch_indices]
            n_samples_in_branch = len(X_branch_all)

            # Determine horizon group for this branch
            # All samples in test_data have the same forecast_horizon (calculated earlier)
            # Map to horizon group
            if forecast_horizon == 0:
                horizon_group = "f0"
            elif forecast_horizon == 1:
                horizon_group = "f1"
            elif 2 <= forecast_horizon <= 3:
                horizon_group = "medium"
            else:
                horizon_group = "long"

            # Skip if branch has no samples
            if n_samples_in_branch == 0:
                if verbose:
                    print(f"  ⚠️  Branch {branch_name}: Empty branch, skipping...")
                branches_processed += 1
                continue

            # ====================================================================
            # Flow Anomaly Detection for Never-Binding Branches
            # ====================================================================
            if (
                self.config.anomaly_detection.enabled
                and branch_name in self.anomaly_detector.never_binding_branches.get(horizon_group, set())
                and branch_name in self.anomaly_detector.flow_stats.get(horizon_group, {})
            ):
                anomaly_detection_stats["branches_checked"] += 1
                anomaly_detection_stats["samples_checked"] += n_samples_in_branch

                # Check each sample for anomaly
                for i, idx in enumerate(branch_indices):
                    test_sample_all = X_branch_all.iloc[i]

                    is_anomaly, confidence, reason = self.anomaly_detector.detect_flow_anomaly(
                        test_sample_all, branch_name, horizon_group
                    )

                    if is_anomaly:
                        # Predict binding due to anomaly
                        y_pred_binary[idx] = 1
                        y_pred_proba[idx] = 0.5 + 0.5 * confidence
                        y_pred_proba_scaled[idx] = y_pred_proba[idx]  # Scaled prob = prob since threshold is 0.5
                        y_pred_threshold[idx] = 0.5  # Default threshold for anomaly

                        # Use default regressor ensemble for shadow price (for this horizon)
                        # Use default regressor ensemble for shadow price (for this horizon)
                        # We pass a dummy branch name to force default model
                        reg_ensemble_anom, reg_scaler_anom = self.models.get_regressor_ensemble(
                            "NON_EXISTENT_BRANCH", forecast_horizon
                        )

                        if len(reg_ensemble_anom) > 0:
                            # Scale features if scaler is available
                            if reg_scaler_anom:
                                # Need to scale all features first
                                # Create DataFrame for single sample
                                sample_df = test_sample_all.to_frame().T
                                feature_cols = self.config.features.all_features
                                sample_df[feature_cols] = reg_scaler_anom.transform(sample_df[feature_cols])
                                # Clip to prevent explosion from near-constant features
                                test_sample_reg = sample_df[self.config.features.step2_features]
                            else:
                                test_sample_reg = test_sample_all[self.config.features.step2_features].to_frame().T

                            # Get horizon-specific weights for default regressor
                            reg_weights_anom = self.models.get_ensemble_weights_for_horizon(
                                forecast_horizon, "regressor", is_branch=False
                            )

                            y_pred_shadow_price[idx] = predict_ensemble(
                                reg_ensemble_anom,
                                test_sample_reg,
                                predict_proba=False,
                                weight_overrides=reg_weights_anom,
                            )[0]

                        # Track anomaly
                        anomaly_detection_stats["anomalies_detected"] += 1
                        anomaly_detection_stats["anomaly_binding_predictions"] += 1

                        # Store first 10 examples
                        if len(anomaly_detection_stats["anomaly_examples"]) < 10:
                            anomaly_detection_stats["anomaly_examples"].append(
                                {
                                    "branch_name": branch_name,
                                    "flow_100": test_sample_all["prob_exceed_100"],
                                    "confidence": confidence,
                                    "reason": reason,
                                }
                            )

                        model_used[idx] = f"Anomaly: {branch_name[:30]}"
                    else:
                        # Predict non-binding
                        y_pred_binary[idx] = 0
                        y_pred_proba[idx] = 0.0
                        y_pred_proba_scaled[idx] = 0.0
                        y_pred_threshold[idx] = 0.5  # Default threshold
                        y_pred_shadow_price[idx] = 0.0
                        model_used[idx] = f"Never-Binding: {branch_name[:30]}"

                # Skip standard classifier for this branch
                branches_processed += 1
                continue

            # ====================================================================
            # Stage 1: Classification (Binding Detection)
            # ====================================================================

            # Get appropriate classifier ensemble, threshold, and scaler for this horizon
            clf_ensemble_to_use, threshold_to_use, clf_scaler = self.models.get_classifier_ensemble(
                branch_name, forecast_horizon
            )

            # Scale features if scaler is available
            if clf_scaler:
                # We need to scale ALL features then select step1 features
                # Create a copy of all features for this branch
                X_branch_scaled_all = X_branch_all.copy()
                feature_cols = self.config.features.all_features
                X_branch_scaled_all[feature_cols] = clf_scaler.transform(X_branch_scaled_all[feature_cols])
                # Clip to prevent explosion from near-constant features

                # Extract classification features from SCALED data
                X_branch_clf = X_branch_scaled_all[self.config.features.step1_features]
            else:
                # Fallback (should not happen if model exists)
                X_branch_clf = X_branch_all[self.config.features.step1_features]

            # Determine if we are using a branch-specific model (for logging/weights)
            has_branch_model = False
            if forecast_horizon == 0:
                has_branch_model = branch_name in self.models.clf_ensembles_f0
            elif forecast_horizon == 1:
                has_branch_model = branch_name in self.models.clf_ensembles_f1
            elif 2 <= forecast_horizon <= 3:
                has_branch_model = branch_name in self.models.clf_ensembles_medium
            else:
                has_branch_model = branch_name in self.models.clf_ensembles_long

            if has_branch_model:
                used_branch_clf_count += n_samples_in_branch
                model_name = f"Branch: {branch_name}"
            else:
                used_default_clf_count += n_samples_in_branch
                model_name = "Default"

            # Get horizon-specific weights for classifier
            clf_weights = self.models.get_ensemble_weights_for_horizon(forecast_horizon, "classifier", has_branch_model)

            # Safety check: Skip if no samples to predict
            if len(X_branch_clf) == 0:
                if verbose:
                    print(f"  ⚠️  Branch {branch_name}: No samples to predict, skipping...")
                branches_processed += 1
                continue

            # Get probabilities using ensemble with horizon-specific weights
            y_pred_proba_branch = predict_ensemble(
                clf_ensemble_to_use, X_branch_clf, predict_proba=True, weight_overrides=clf_weights
            )
            y_pred_binary_branch = (y_pred_proba_branch >= threshold_to_use).astype(int)

            # Calculate scaled binding probability
            # Map [0, threshold] -> [0, 0.5] and [threshold, 1] -> [0.5, 1]
            # f(p) = 0.5 * (p / t) if p < t
            # f(p) = 0.5 + 0.5 * (p - t) / (1 - t) if p >= t

            # Avoid division by zero if threshold is 0 or 1 (unlikely but safe)
            t = np.clip(threshold_to_use, 1e-6, 1.0 - 1e-6)

            y_pred_proba_scaled_branch = np.where(
                y_pred_proba_branch < t,
                0.5 * (y_pred_proba_branch / t),
                0.5 + 0.5 * (y_pred_proba_branch - t) / (1.0 - t),
            )

            # Store predictions
            y_pred_binary[branch_indices] = y_pred_binary_branch
            y_pred_proba[branch_indices] = y_pred_proba_branch
            y_pred_proba_scaled[branch_indices] = y_pred_proba_scaled_branch
            y_pred_threshold[branch_indices] = threshold_to_use

            # ====================================================================
            # Stage 2: Regression (Shadow Price Prediction)
            # ====================================================================
            # Find binding samples
            binding_mask_in_branch = y_pred_binary_branch == 1

            if binding_mask_in_branch.sum() > 0:
                binding_indices_in_branch = branch_indices[binding_mask_in_branch]
                n_binding_in_branch = len(binding_indices_in_branch)

                # Get appropriate regressor ensemble and scaler for this horizon
                reg_ensemble_to_use, reg_scaler = self.models.get_regressor_ensemble(branch_name, forecast_horizon)

                # Scale features if scaler is available
                if reg_scaler:
                    # We need to scale ALL features then select step2 features
                    X_binding_all_raw = X_branch_all.iloc[binding_mask_in_branch].copy()
                    feature_cols = self.config.features.all_features
                    X_binding_all_raw[feature_cols] = reg_scaler.transform(X_binding_all_raw[feature_cols])

                    # Extract regression features from SCALED data
                    X_binding_in_branch = X_binding_all_raw[self.config.features.step2_features]
                else:
                    # Extract regression features from RAW data (fallback)
                    X_binding_in_branch = X_branch_all.iloc[binding_mask_in_branch][self.config.features.step2_features]

                # Determine if we are using a branch-specific model
                has_branch_reg_model = False
                if forecast_horizon == 0:
                    has_branch_reg_model = branch_name in self.models.reg_ensembles_f0
                elif forecast_horizon == 1:
                    has_branch_reg_model = branch_name in self.models.reg_ensembles_f1
                elif 2 <= forecast_horizon <= 3:
                    has_branch_reg_model = branch_name in self.models.reg_ensembles_medium
                else:
                    has_branch_reg_model = branch_name in self.models.reg_ensembles_long

                if has_branch_reg_model:
                    used_branch_reg_count += n_binding_in_branch
                else:
                    used_default_reg_count += n_binding_in_branch

                # Get horizon-specific weights for regressor
                reg_weights = self.models.get_ensemble_weights_for_horizon(
                    forecast_horizon, "regressor", has_branch_reg_model
                )

                # Predict shadow prices
                if reg_ensemble_to_use is not None:
                    y_pred_shadow_price_branch = predict_ensemble(
                        reg_ensemble_to_use, X_binding_in_branch, predict_proba=False, weight_overrides=reg_weights
                    )
                    y_pred_shadow_price[binding_indices_in_branch] = y_pred_shadow_price_branch

            # Store model name
            for idx in branch_indices:
                model_used[idx] = model_name

            branches_processed += 1

            # Progress indicator
            if verbose and branches_processed % 1000 == 0:
                print(f"  Progress: {branches_processed:,} / {len(unique_branches_in_test):,} branches processed...")

        if verbose:
            print("\n✓ Predictions Complete")
            self._print_stats(
                X_test,
                used_branch_clf_count,
                used_default_clf_count,
                used_branch_reg_count,
                used_default_reg_count,
                no_reg_model_count,
                unique_branches_in_test,
                anomaly_detection_stats,
            )

        # Create results DataFrames
        results_per_outage, final_results = self._create_results_dataframes(
            test_data,
            y_pred_binary,
            y_pred_proba,
            y_pred_proba_scaled,
            y_pred_threshold,
            y_pred_shadow_price,
            model_used,
            verbose,
        )

        return results_per_outage, final_results

    def _print_stats(
        self,
        X_test,
        used_branch_clf,
        used_default_clf,
        used_branch_reg,
        used_default_reg,
        no_reg,
        unique_branches,
        anomaly_stats,
    ):
        """Print prediction statistics."""
        print("\n[Model Usage Statistics]")
        print("  Classification:")
        print(f"    Branch-specific models: {used_branch_clf:,} samples ({used_branch_clf / len(X_test) * 100:.2f}%)")
        print(f"    Default fallback model: {used_default_clf:,} samples ({used_default_clf / len(X_test) * 100:.2f}%)")
        print("  Regression (for predicted binding):")
        print(f"    Branch-specific models: {used_branch_reg:,} samples")
        print(f"    Default fallback model: {used_default_reg:,} samples")
        print(f"    No model available: {no_reg:,} samples")

        print("\n  Dynamic Thresholds Applied:")
        # Statistics are horizon-dependent, skipping detailed breakdown to avoid confusion

        if self.config.anomaly_detection.enabled:
            print("\n[Flow Anomaly Detection Statistics]")
            print(f"  Never-binding branches checked: {anomaly_stats['branches_checked']:,}")
            print(f"  Test samples checked: {anomaly_stats['samples_checked']:,}")
            print(
                f"  Anomalies detected: {anomaly_stats['anomalies_detected']:,} "
                f"({anomaly_stats['anomalies_detected'] / max(1, anomaly_stats['samples_checked']) * 100:.2f}%)"
            )
            print(f"  Binding predictions from anomalies: {anomaly_stats['anomaly_binding_predictions']:,}")

            if len(anomaly_stats["anomaly_examples"]) > 0:
                print("\n  Top Anomaly Examples:")
                for i, example in enumerate(anomaly_stats["anomaly_examples"][:5], 1):
                    print(
                        f"    {i}. {example['branch_name'][:40]:40s}: "
                        f"flow_100={example['flow_100']:.4f}, conf={example['confidence']:.2f}"
                    )
                    print(f"       {example['reason']}")

    def _create_results_dataframes(
        self,
        test_data: pd.DataFrame,
        y_pred_binary: np.ndarray,
        y_pred_proba: np.ndarray,
        y_pred_proba_scaled: np.ndarray,
        y_pred_threshold: np.ndarray,
        y_pred_shadow_price: np.ndarray,
        model_used: list[str],
        verbose: bool,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Create results DataFrames at outage level and monthly aggregated level."""
        if verbose:
            print("\n" + "=" * 80)
            print("[AGGREGATING PREDICTIONS ACROSS ALL OUTAGE DATES]")
            print("=" * 80)

        # Create per-outage results
        results_per_outage = test_data[
            self.config.features.all_features + ["label", "branch_name", "outage_date", "auction_month", "market_month"]
        ].copy()
        results_per_outage["predicted_shadow_price"] = y_pred_shadow_price
        results_per_outage["binding_probability"] = y_pred_proba
        results_per_outage["binding_probability_scaled"] = y_pred_proba_scaled
        results_per_outage["threshold"] = y_pred_threshold
        results_per_outage["predicted_binding"] = y_pred_binary
        results_per_outage["model_used"] = model_used
        results_per_outage["actual_binding"] = (
            (test_data["label"] > self.config.training.label_threshold).astype(int).values
        )
        results_per_outage = results_per_outage.rename(columns={"label": "actual_shadow_price"})

        if verbose:
            print("\nBefore aggregation:")
            print(f"  Total samples (constraint × outage_date): {len(results_per_outage):,}")
            print(f"  Unique constraints: {results_per_outage.index.get_level_values(0).nunique():,}")
            print(f"  Unique outage dates: {results_per_outage['outage_date'].nunique()}")

        # Aggregate by constraint_id (sum across all outage dates)
        final_results = (
            results_per_outage.groupby(level=[0, 1])
            .agg(
                {
                    "branch_name": "first",
                    **{feat: "mean" for feat in self.config.features.all_features},
                    "actual_shadow_price": "sum",
                    "predicted_shadow_price": "sum",
                    "binding_probability": "mean",
                    "binding_probability_scaled": "mean",
                    "threshold": "first",
                    "predicted_binding": "sum",
                    "actual_binding": "max",
                    "auction_month": "first",
                    "market_month": "first",
                    "model_used": "first",
                }
            )
            .rename(columns={"predicted_binding": "predicted_binding_count"})
        )

        final_results["predicted_binding"] = final_results["predicted_binding_count"] >= 1

        # Calculate errors
        final_results["error"] = final_results["predicted_shadow_price"] - final_results["actual_shadow_price"]
        final_results["abs_error"] = np.abs(final_results["error"])

        if verbose:
            print("\nAfter aggregation:")
            print(f"  Total unique constraints: {len(final_results):,}")
            print(
                f"  Aggregation: SUM of shadow prices across {results_per_outage['outage_date'].nunique()} outage dates"
            )
            print("  Binding definition: Constraint binds if it binds in ANY outage date")

        return results_per_outage, final_results
