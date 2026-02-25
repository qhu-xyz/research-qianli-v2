"""
Prediction pipeline for shadow prices.
"""

from typing import Any

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

    def transform_features(self, data: pd.DataFrame, scaled_col_suffix: str | None = None) -> pd.DataFrame:
        """
        Transform features using the trained scalers.

        Parameters:
        -----------
        data : pd.DataFrame
            Data to transform. Must contain 'forecast_horizon' and 'branch_name'.
        scaled_col_suffix : str, optional
            If provided, scaled features will be added as new columns with this suffix.

        Returns:
        --------
        data_scaled : pd.DataFrame
            Data with scaled features.
        """
        return self.models.transform_features(data, scaled_col_suffix)

    def predict(self, test_data: pd.DataFrame, verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
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
        metrics : dict
            Evaluation metrics
        """
        if verbose:
            print("\n[Making Predictions with Branch-Specific Models + Flow Anomaly Detection]")
            print("=" * 80)

        # Prepare test features (filter to columns present in test_data)
        available_features = [f for f in self.config.features.all_features if f in test_data.columns]
        X_test = test_data[available_features].copy()
        # Add missing feature columns as 0 (e.g., seasonal features for short histories)
        for f in self.config.features.all_features:
            if f not in X_test.columns:
                X_test[f] = 0.0

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

        anomaly_detection_stats: dict[str, Any] = {
            "branches_checked": 0,
            "samples_checked": 0,
            "anomalies_detected": 0,
            "anomaly_binding_predictions": 0,
            "anomaly_examples": [],
        }

        branches_processed = 0

        # Loop over each unique branch in the test set
        # Loop over each unique branch in the test set
        # Get unique (branch_name, flow_direction) pairs
        if "flow_direction" in test_data.columns:
            unique_pairs = test_data[["branch_name", "flow_direction"]].drop_duplicates().values
        else:
            # Fallback if flow_direction is in index (should have been reset before calling predict?)
            # But predict receives test_data which might have index
            if "flow_direction" in test_data.index.names:
                unique_pairs = test_data.reset_index()[["branch_name", "flow_direction"]].drop_duplicates().values
            else:
                # Fallback if flow_direction is missing completely (should not happen)
                unique_pairs = [(b, 0) for b in unique_branches_in_test]  # Dummy flow

        for branch_name, flow_direction in unique_pairs:
            # Get all test samples for this branch using boolean mask
            if "flow_direction" in test_data.columns:
                branch_mask = (test_data["branch_name"] == branch_name) & (
                    test_data["flow_direction"] == flow_direction
                )
            elif "flow_direction" in test_data.index.names:
                branch_mask = (test_data["branch_name"] == branch_name) & (
                    test_data.index.get_level_values("flow_direction") == flow_direction
                )
            else:
                branch_mask = test_data["branch_name"] == branch_name

            branch_indices = np.where(branch_mask)[0]  # Get positional indices

            X_branch_all = X_test.iloc[branch_indices]

            # --- Pre-Processing: Force Unbind Rule ---
            # Exclude samples that satisfy the unbind rule from further processing
            # They will remain 0 (default initialization) and be marked "Force Unbind"
            test_unbind_rule = self.config.training.test_unbind_rule
            if test_unbind_rule is not None:
                feat_name, threshold = test_unbind_rule
                if feat_name in X_branch_all.columns:
                    mask_unbind_local = (X_branch_all[feat_name] < threshold).values

                    if np.any(mask_unbind_local):
                        # Mark excluded samples
                        unbind_indices_global = branch_indices[mask_unbind_local]
                        for idx in unbind_indices_global:
                            model_used[idx] = "Force Unbind"

                        # Keep only valid samples
                        mask_valid_local = ~mask_unbind_local
                        X_branch_all = X_branch_all[mask_valid_local]
                        branch_indices = branch_indices[mask_valid_local]

            n_samples_in_branch = len(X_branch_all)

            # Determine horizon group for this branch
            # All samples in test_data have the same forecast_horizon (calculated earlier)
            # Map to horizon group
            # Determine horizon group for this branch
            # All samples in test_data have the same forecast_horizon (calculated earlier)
            # Map to horizon group
            horizon_group = "f0"  # Default
            for g in self.config.horizon_groups:
                if g.min_horizon <= forecast_horizon <= g.max_horizon:
                    horizon_group = g.name
                    break

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
                        y_pred_proba[idx] = confidence
                        y_pred_proba_scaled[idx] = y_pred_proba[idx]  # Scaled prob = prob since threshold is 0.5
                        y_pred_threshold[idx] = 0.5  # Default threshold for anomaly

                        # Use default regressor ensemble for shadow price (for this horizon)
                        # We pass a dummy branch name and flow direction to force default model
                        reg_ensemble_anom, reg_scaler_anom = self.models.get_regressor_ensemble(
                            "NON_EXISTENT_BRANCH", 0, forecast_horizon
                        )

                        if len(reg_ensemble_anom) > 0:
                            # Get expected features for default regressor
                            feats = self.models.reg_default_features.get(horizon_group)
                            if not feats:
                                feats = [f[0] for f in self.config.features.step2_features]

                            # Scale features if scaler is available
                            if reg_scaler_anom:
                                # Need to scale all features first
                                # Create DataFrame for single sample
                                sample_df = test_sample_all.to_frame().T
                                feature_cols = self.config.features.all_features
                                sample_df[feature_cols] = reg_scaler_anom.transform(sample_df[feature_cols])

                                # Use valid features
                                valid_feats = [f for f in feats if f in sample_df.columns]
                                test_sample_reg = sample_df[valid_feats]
                            else:
                                valid_feats = [f for f in feats if f in test_sample_all.index]
                                test_sample_reg = test_sample_all[valid_feats].to_frame().T

                            # Get horizon-specific weights for default regressor
                            reg_weights_anom = self.models.get_ensemble_weights_for_horizon(
                                forecast_horizon, "regressor", is_branch=False
                            )

                            # Predict shadow price with inverse log-transform
                            raw_pred = predict_ensemble(
                                reg_ensemble_anom,
                                test_sample_reg,
                                predict_proba=False,
                                weight_overrides=reg_weights_anom,
                            )[0]

                            y_pred_shadow_price[idx] = np.maximum(0, np.expm1(np.clip(raw_pred, None, 12)))

                            # Apply heuristic boost for anomalies (Recommendation #1)
                            # Anomalies are "surprise" events, so we boost the prediction
                            y_pred_shadow_price[idx] = max(y_pred_shadow_price[idx] * 2.0, 20.0)

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
                branch_name, flow_direction, forecast_horizon
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
                step1_names = [f[0] for f in self.config.features.step1_features]
                X_branch_clf = X_branch_scaled_all[step1_names]
            else:
                # Fallback (should not happen if model exists)
                step1_names = [f[0] for f in self.config.features.step1_features]
                X_branch_clf = X_branch_all[step1_names]

            # Use selected features if available
            selected_features = None
            # Check for branch specific selected features
            if (branch_name, flow_direction) in self.models.clf_branch_features.get(horizon_group, {}):
                selected_features = self.models.clf_branch_features[horizon_group][(branch_name, flow_direction)]
            # Check for default selected features
            elif horizon_group in self.models.clf_default_features:
                selected_features = self.models.clf_default_features[horizon_group]

            if selected_features:
                # Ensure we only pick features that exist (they should, but safe check)
                valid_feats = [f for f in selected_features if f in X_branch_clf.columns]
                X_branch_clf = X_branch_clf[valid_feats]

            # Determine if we are using a branch-specific model (for logging/weights)
            # Note: get_classifier_ensemble returns branch specific if available, else default
            # We can check if branch_name is in the ensemble dict to know for sure,
            # but simpler to check if returned ensemble is default
            # Actually, let's just count based on result
            # Or check if (branch_name, flow_direction) is in keys
            # But we don't have easy access to keys here without re-checking logic
            # Let's just assume if clf_scaler is not None and not default scaler...
            # A bit complex. Let's just rely on the fact that we got a model.

            # Get horizon-specific weights
            # Note: weight overrides are handled inside predict_ensemble if passed,
            # but here we need to pass them.
            # We need to know if it's a branch model or default to pass correct 'is_branch' flag
            # Re-check logic:
            is_branch_model = False
            # Check if we have a specific model for this branch/flow
            # Check if we have a specific model for this branch/flow
            if (branch_name, flow_direction) in self.models.clf_ensembles.get(horizon_group, {}):
                is_branch_model = True

            if is_branch_model:
                used_branch_clf_count += n_samples_in_branch
            else:
                used_default_clf_count += n_samples_in_branch

            clf_weights = self.models.get_ensemble_weights_for_horizon(
                forecast_horizon, "classifier", is_branch=is_branch_model
            )

            # Safety check: Skip if no samples to predict
            if len(X_branch_clf) == 0:
                if verbose:
                    print(f"  ⚠️  Branch {branch_name}: No samples to predict, skipping...")
                branches_processed += 1
                continue

            # Get probabilities using ensemble with horizon-specific weights
            y_proba_branch = predict_ensemble(
                clf_ensemble_to_use, X_branch_clf, predict_proba=True, weight_overrides=clf_weights
            )

            # --- Fallback & Blending Logic ---
            if not is_branch_model:
                # Default Model Validity Check
                if not self.models.clf_default_valid.get(horizon_group, True):
                    y_proba_branch = np.zeros_like(y_proba_branch)

            elif is_branch_model:
                # Branch Model Fallback Check (Blending)
                if self.models.clf_branch_fallback.get(horizon_group, {}).get((branch_name, flow_direction), False):
                    # Calculate Probability from Default Model for Blending
                    prob_default = np.zeros_like(y_proba_branch)

                    if self.models.clf_default_valid.get(horizon_group, True):
                        # Use default ensemble
                        default_ens = self.models.clf_default_ensembles.get(horizon_group, [])
                        if default_ens:
                            # Prepare features for default model
                            default_feats = self.models.clf_default_features.get(horizon_group, [])
                            # Use X_branch_scaled_all if available (from scaler block), else X_branch_all
                            # We need to re-extract because 'X_branch_clf' is already filtered for branch model
                            if "X_branch_scaled_all" in locals():
                                X_source = X_branch_scaled_all
                            else:
                                X_source = X_branch_all

                            valid_default_feats = [f for f in default_feats if f in X_source.columns]
                            X_default_clf = X_source[valid_default_feats]

                            # Get default weights
                            default_weights = self.models.get_ensemble_weights_for_horizon(
                                forecast_horizon, "classifier", is_branch=False
                            )

                            prob_default = predict_ensemble(
                                default_ens, X_default_clf, predict_proba=True, weight_overrides=default_weights
                            )

                    # Blend: w * Branch + (1-w) * Default
                    w_fallback = self.config.feature_selection.fallback_weight
                    y_proba_branch = w_fallback * y_proba_branch + (1 - w_fallback) * prob_default

            # Store predictions
            y_pred_proba[branch_indices] = y_proba_branch
            y_pred_threshold[branch_indices] = threshold_to_use

            # Apply threshold
            y_pred_binary_branch = (y_proba_branch >= threshold_to_use).astype(int)
            y_pred_binary[branch_indices] = y_pred_binary_branch

            # Scale probability for output (0.5 = threshold)
            # If prob < threshold, map [0, threshold] -> [0.5, 0.5]
            # If prob >= threshold, map [threshold, 1] -> [0.5, 1]
            # Avoid division by zero
            safe_threshold = np.clip(threshold_to_use, 0.001, 0.999)

            # Vectorized scaling
            mask_below = y_proba_branch < safe_threshold
            mask_above = ~mask_below

            y_pred_proba_scaled_branch = np.zeros_like(y_proba_branch)
            y_pred_proba_scaled_branch[mask_below] = 0.5 * (y_proba_branch[mask_below] / safe_threshold)
            y_pred_proba_scaled_branch[mask_above] = 0.5 + 0.5 * (
                (y_proba_branch[mask_above] - safe_threshold) / (1.0 - safe_threshold)
            )
            y_pred_proba_scaled[branch_indices] = y_pred_proba_scaled_branch

            # Log model usage
            model_name = f"{'Branch' if is_branch_model else 'Default'} CLF"
            for idx in branch_indices:
                model_used[idx] = model_name

            # ====================================================================
            # Stage 2: Regression (Shadow Price Estimation)
            # ====================================================================

            # Only predict shadow price for binding constraints
            binding_mask_local = y_pred_binary_branch == 1

            if np.any(binding_mask_local):
                # Get appropriate regressor ensemble
                reg_ensemble_to_use, reg_scaler = self.models.get_regressor_ensemble(
                    branch_name, flow_direction, forecast_horizon
                )

                if len(reg_ensemble_to_use) == 0:
                    # No regression model available (even default)
                    no_reg_model_count += binding_mask_local.sum()
                    # Shadow price remains 0
                else:
                    # Prepare regression features
                    # Use the SAME scaler if it exists (it should match the one from classifier if branch-specific)
                    # Or reg_scaler might be different if we fell back to default regressor but used branch classifier?
                    # get_regressor_ensemble handles logic.

                    # Check if branch model
                    is_branch_reg_model = False
                    if (branch_name, flow_direction) in self.models.reg_ensembles.get(horizon_group, {}):
                        is_branch_reg_model = True

                    # Determine correct features for the model
                    if is_branch_reg_model:
                        # Use features stored during training for this branch
                        feats = self.models.reg_branch_features.get(horizon_group, {}).get(
                            (branch_name, flow_direction)
                        )
                        if not feats:
                            # Fallback if not found (shouldn't happen if trained)
                            feats = [f[0] for f in self.config.features.step2_features]
                    else:
                        # Use default model features
                        feats = self.models.reg_default_features.get(horizon_group)
                        if not feats:
                            feats = [f[0] for f in self.config.features.step2_features]

                    if reg_scaler:
                        # If we already scaled for classification and it's the SAME scaler, we could reuse
                        # But simpler to just transform again to be safe/robust
                        X_branch_scaled_reg_all = X_branch_all.copy()
                        X_branch_scaled_reg_all[feature_cols] = reg_scaler.transform(
                            X_branch_scaled_reg_all[feature_cols]
                        )
                        # Extract regression features from SCALED
                        # Ensure features exist in dataframe
                        valid_feats = [f for f in feats if f in X_branch_scaled_reg_all.columns]
                        X_branch_reg = X_branch_scaled_reg_all[valid_feats]
                    else:
                        valid_feats = [f for f in feats if f in X_branch_all.columns]
                        X_branch_reg = X_branch_all[valid_feats]

                    # Filter for binding samples
                    # We need to subset X_branch_reg using binding_mask_local
                    X_branch_reg_binding = X_branch_reg[binding_mask_local]

                    if is_branch_reg_model:
                        used_branch_reg_count += binding_mask_local.sum()
                    else:
                        used_default_reg_count += binding_mask_local.sum()

                    reg_weights = self.models.get_ensemble_weights_for_horizon(
                        forecast_horizon, "regressor", is_branch=is_branch_reg_model
                    )

                    # Predict shadow price (log-transformed)
                    raw_pred = predict_ensemble(
                        reg_ensemble_to_use,
                        X_branch_reg_binding,
                        predict_proba=False,
                        weight_overrides=reg_weights,
                    )

                    # Inverse transform (expm1) and clip to [0, expm1(12) ≈ $162K]
                    shadow_price_pred = np.maximum(0, np.expm1(np.clip(raw_pred, None, 12)))

                    # --- Fallback & Blending Logic (Regression) ---
                    if not is_branch_reg_model:
                        # Default Model Validity Check
                        if not self.models.reg_default_valid.get(horizon_group, True):
                            shadow_price_pred = np.zeros_like(shadow_price_pred)

                    elif is_branch_reg_model:
                        # Branch Model Fallback Check (Blending)
                        if self.models.reg_branch_fallback.get(horizon_group, {}).get(
                            (branch_name, flow_direction), False
                        ):
                            val_default = np.zeros_like(shadow_price_pred)

                            if self.models.reg_default_valid.get(horizon_group, True):
                                default_reg_ens = self.models.reg_default_ensembles.get(horizon_group, [])
                                if default_reg_ens:
                                    # Prepare features for default model
                                    # We need original scaled data for default features? No, we need to scale using default scaler.
                                    # X_branch_all contains unscaled features for this branch subset
                                    X_branch_binding_raw = X_branch_all.iloc[np.where(binding_mask_local)[0]]

                                    scaler_def = self.models.scalers_default_reg.get(horizon_group)
                                    if scaler_def:
                                        X_def_scaled = X_branch_binding_raw.copy()
                                        # Need to scale all features
                                        X_def_scaled[self.config.features.all_features] = scaler_def.transform(
                                            X_def_scaled[self.config.features.all_features]
                                        )

                                        default_feats = self.models.reg_default_features.get(horizon_group, [])
                                        # Intersect with available columns (which should be all scaled features)
                                        valid_feats = [f for f in default_feats if f in X_def_scaled.columns]
                                        X_def_reg = X_def_scaled[valid_feats]

                                        default_weights = self.models.get_ensemble_weights_for_horizon(
                                            forecast_horizon, "regressor", is_branch=False
                                        )

                                        raw_pred_def = predict_ensemble(
                                            default_reg_ens,
                                            X_def_reg,
                                            predict_proba=False,
                                            weight_overrides=default_weights,
                                        )
                                        val_default = np.maximum(0, np.expm1(np.clip(raw_pred_def, None, 12)))

                            # Blend in Linear Space
                            w_fallback = self.config.feature_selection.fallback_weight
                            shadow_price_pred = w_fallback * shadow_price_pred + (1 - w_fallback) * val_default

                    # Map back to full array
                    # binding_indices_local are indices within X_branch_all
                    # We need to map to global indices
                    binding_indices_global = branch_indices[binding_mask_local]
                    y_pred_shadow_price[binding_indices_global] = shadow_price_pred

                    # Update model usage log
                    reg_model_name = f" + {'Branch' if is_branch_reg_model else 'Default'} REG"
                    for idx in binding_indices_global:
                        model_used[idx] += reg_model_name

            branches_processed += 1

            # Progress indicator
            if verbose and branches_processed % 1000 == 0:
                print(f"  Progress: {branches_processed:,} / {len(unique_pairs):,} branch-flow pairs processed...")

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

        # # Calculate Metrics only if labels are available
        # if results_per_outage["actual_shadow_price"].notna().any():
        #     metrics = analyze_results(results_per_outage, final_results, verbose=verbose)
        #     if verbose:
        #         print("\n[Evaluation Metrics]")
        #         for k, v in metrics.items():
        #             print(f"  {k}: {v:.4f}")
        # else:
        #     if verbose:
        #         print("\n[Evaluation Metrics]")
        #         print("  Skipping metrics calculation (no labels available)")
        #     metrics = {}
        metrics = {}

        return results_per_outage, final_results, metrics

    def _print_stats(
        self,
        X_test,  # noqa: N803
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

    def aggregate_probabilities(self, series):
        p = 3
        return np.power((series**p).sum() / len(series), 1 / p)

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
        features = [f for f in set(self.config.features.all_features + ["hist_da"]) if f in test_data.columns]
        all_cols = features + [
            "constraint_id",
            "flow_direction",
            "label",
            "branch_name",
            "outage_date",
            "auction_month",
            "market_month",
        ]
        results_per_outage = test_data[all_cols].copy()
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
            results_per_outage.sort_values("binding_probability", ascending=False)
            .groupby(["constraint_id", "flow_direction"])
            .agg(
                {
                    "branch_name": "first",
                    **{feat: "mean" for feat in features},
                    "actual_shadow_price": "sum",
                    "predicted_shadow_price": "sum",
                    "binding_probability": "max",
                    "binding_probability_scaled": self.aggregate_probabilities,
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
