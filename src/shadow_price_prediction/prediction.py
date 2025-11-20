"""
Prediction pipeline for shadow prices.
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict

from .config import PredictionConfig
from .models import ShadowPriceModels, predict_ensemble
from .anomaly_detection import AnomalyDetector


class Predictor:
    """Handles prediction logic for shadow prices."""

    def __init__(
        self,
        config: PredictionConfig,
        models: ShadowPriceModels,
        anomaly_detector: AnomalyDetector
    ):
        self.config = config
        self.models = models
        self.anomaly_detector = anomaly_detector

    def predict(
        self,
        test_data: pd.DataFrame,
        verbose: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
        y_test = test_data['label'].copy()

        # Initialize prediction arrays
        y_pred_binary = np.zeros(len(X_test), dtype=int)
        y_pred_proba = np.zeros(len(X_test))
        y_pred_proba_scaled = np.zeros(len(X_test))
        y_pred_threshold = np.zeros(len(X_test))
        y_pred_shadow_price = np.zeros(len(X_test))
        model_used = [''] * len(X_test)

        # Get unique branches in test set
        unique_branches_in_test = test_data['branch_name'].unique()

        if verbose:
            print(f"\nProcessing {len(unique_branches_in_test):,} unique branches...")
            print(f"Total test samples: {len(X_test):,}")
            print(f"\nFeature filtering:")
            print(f"  Classification uses: {self.config.features.step1_features} "
                  f"({len(self.config.features.step1_features)} features)")
            print(f"  Regression uses: {self.config.features.step2_features} "
                  f"({len(self.config.features.step2_features)} features)")

        # Track statistics
        used_branch_clf_count = 0
        used_default_clf_count = 0
        used_branch_reg_count = 0
        used_default_reg_count = 0
        no_reg_model_count = 0

        anomaly_detection_stats = {
            'branches_checked': 0,
            'samples_checked': 0,
            'anomalies_detected': 0,
            'anomaly_binding_predictions': 0,
            'anomaly_examples': []
        }

        branches_processed = 0

        for branch_name in unique_branches_in_test:
            # Get all test samples for this branch
            branch_mask = test_data['branch_name'] == branch_name
            branch_indices = np.where(branch_mask)[0]

            # Get features (all 5 features)
            X_branch_all = X_test.iloc[branch_indices]
            n_samples_in_branch = len(X_branch_all)

            # ====================================================================
            # Flow Anomaly Detection for Never-Binding Branches
            # ====================================================================
            if (self.config.anomaly_detection.enabled and
                branch_name in self.anomaly_detector.never_binding_branches and
                branch_name in self.anomaly_detector.flow_stats):

                anomaly_detection_stats['branches_checked'] += 1
                anomaly_detection_stats['samples_checked'] += n_samples_in_branch

                # Check each sample for anomaly
                for i, idx in enumerate(branch_indices):
                    test_sample_all = X_branch_all.iloc[i]

                    is_anomaly, confidence, reason = self.anomaly_detector.detect_flow_anomaly(
                        test_sample_all, branch_name
                    )

                    if is_anomaly:
                        # Predict binding due to anomaly
                        y_pred_binary[idx] = 1
                        y_pred_proba[idx] = 0.5 + 0.5 * confidence
                        y_pred_proba_scaled[idx] = y_pred_proba[idx] # Scaled prob = prob since threshold is 0.5
                        y_pred_threshold[idx] = 0.5  # Default threshold for anomaly

                        # Use default regressor ensemble for shadow price
                        if len(self.models.reg_default_ensemble) > 0:
                            test_sample_reg = test_sample_all[self.config.features.step2_features].to_frame().T
                            y_pred_shadow_price[idx] = predict_ensemble(
                                self.models.reg_default_ensemble, test_sample_reg, predict_proba=False
                            )[0]

                        # Track anomaly
                        anomaly_detection_stats['anomalies_detected'] += 1
                        anomaly_detection_stats['anomaly_binding_predictions'] += 1

                        # Store first 10 examples
                        if len(anomaly_detection_stats['anomaly_examples']) < 10:
                            anomaly_detection_stats['anomaly_examples'].append({
                                'branch_name': branch_name,
                                'flow_100': test_sample_all['100'],
                                'confidence': confidence,
                                'reason': reason
                            })

                        model_used[idx] = f"Anomaly: {branch_name[:30]}"
                    else:
                        # Predict non-binding
                        y_pred_binary[idx] = 0
                        y_pred_proba[idx] = 0.0
                        y_pred_proba_scaled[idx] = 0.0
                        y_pred_threshold[idx] = 0.5 # Default threshold
                        y_pred_shadow_price[idx] = 0.0
                        model_used[idx] = f"Never-Binding: {branch_name[:30]}"

                # Skip standard classifier for this branch
                branches_processed += 1
                continue

            # ====================================================================
            # Stage 1: Classification (Binding Detection)
            # ====================================================================
            # Extract only classification features
            X_branch_clf = X_branch_all[self.config.features.step1_features]

            if branch_name in self.models.clf_ensembles:
                # Use branch-specific classifier ensemble
                clf_ensemble_to_use = self.models.clf_ensembles[branch_name]
                used_branch_clf_count += n_samples_in_branch
                model_name = f"Branch: {branch_name}"
                threshold_to_use = self.models.optimal_thresholds.get(
                    branch_name, self.models.optimal_threshold_default
                )
            else:
                # Use default classifier ensemble
                clf_ensemble_to_use = self.models.clf_default_ensemble
                used_default_clf_count += n_samples_in_branch
                model_name = "Default"
                threshold_to_use = self.models.optimal_threshold_default

            # Get probabilities using ensemble
            y_pred_proba_branch = predict_ensemble(clf_ensemble_to_use, X_branch_clf, predict_proba=True)
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
                0.5 + 0.5 * (y_pred_proba_branch - t) / (1.0 - t)
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

                # Extract regression features for binding samples
                X_binding_in_branch = X_branch_all.iloc[binding_mask_in_branch][self.config.features.step2_features]

                if branch_name in self.models.reg_ensembles:
                    # Use branch-specific regressor ensemble
                    reg_ensemble_to_use = self.models.reg_ensembles[branch_name]
                    used_branch_reg_count += len(binding_indices_in_branch)
                elif len(self.models.reg_default_ensemble) > 0:
                    # Use default regressor ensemble
                    reg_ensemble_to_use = self.models.reg_default_ensemble
                    used_default_reg_count += len(binding_indices_in_branch)
                else:
                    # No regressor available
                    reg_ensemble_to_use = None
                    no_reg_model_count += len(binding_indices_in_branch)

                # Batch prediction for regression using ensemble
                if reg_ensemble_to_use is not None:
                    y_pred_shadow_price_branch = predict_ensemble(
                        reg_ensemble_to_use, X_binding_in_branch, predict_proba=False
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
            print(f"\n✓ Predictions Complete")
            self._print_stats(
                X_test, used_branch_clf_count, used_default_clf_count,
                used_branch_reg_count, used_default_reg_count,
                no_reg_model_count, unique_branches_in_test,
                anomaly_detection_stats
            )

        # Create results DataFrames
        results_per_outage, final_results = self._create_results_dataframes(
            test_data, y_pred_binary, y_pred_proba, y_pred_proba_scaled, y_pred_threshold, y_pred_shadow_price, model_used, verbose
        )

        return results_per_outage, final_results

    def _print_stats(
        self, X_test, used_branch_clf, used_default_clf,
        used_branch_reg, used_default_reg, no_reg,
        unique_branches, anomaly_stats
    ):
        """Print prediction statistics."""
        print(f"\n[Model Usage Statistics]")
        print(f"  Classification:")
        print(f"    Branch-specific models: {used_branch_clf:,} samples ({used_branch_clf/len(X_test)*100:.2f}%)")
        print(f"    Default fallback model: {used_default_clf:,} samples ({used_default_clf/len(X_test)*100:.2f}%)")
        print(f"  Regression (for predicted binding):")
        print(f"    Branch-specific models: {used_branch_reg:,} samples")
        print(f"    Default fallback model: {used_default_reg:,} samples")
        print(f"    No model available: {no_reg:,} samples")

        print(f"\n  Dynamic Thresholds Applied:")
        print(f"    Default threshold: {self.models.optimal_threshold_default:.3f}")
        print(f"    Branch-specific thresholds used: {sum(1 for b in unique_branches if b in self.models.optimal_thresholds):,}")
        print(f"    Branches using default threshold: {sum(1 for b in unique_branches if b not in self.models.optimal_thresholds):,}")

        if self.config.anomaly_detection.enabled:
            print(f"\n[Flow Anomaly Detection Statistics]")
            print(f"  Never-binding branches checked: {anomaly_stats['branches_checked']:,}")
            print(f"  Test samples checked: {anomaly_stats['samples_checked']:,}")
            print(f"  Anomalies detected: {anomaly_stats['anomalies_detected']:,} "
                  f"({anomaly_stats['anomalies_detected']/max(1, anomaly_stats['samples_checked'])*100:.2f}%)")
            print(f"  Binding predictions from anomalies: {anomaly_stats['anomaly_binding_predictions']:,}")

            if len(anomaly_stats['anomaly_examples']) > 0:
                print(f"\n  Top Anomaly Examples:")
                for i, example in enumerate(anomaly_stats['anomaly_examples'][:5], 1):
                    print(f"    {i}. {example['branch_name'][:40]:40s}: "
                          f"flow_100={example['flow_100']:.4f}, conf={example['confidence']:.2f}")
                    print(f"       {example['reason']}")

    def _create_results_dataframes(
        self,
        test_data: pd.DataFrame,
        y_pred_binary: np.ndarray,
        y_pred_proba: np.ndarray,
        y_pred_proba_scaled: np.ndarray,
        y_pred_threshold: np.ndarray,
        y_pred_shadow_price: np.ndarray,
        model_used: List[str],
        verbose: bool
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create results DataFrames at outage level and monthly aggregated level."""
        if verbose:
            print("\n" + "=" * 80)
            print("[AGGREGATING PREDICTIONS ACROSS ALL OUTAGE DATES]")
            print("=" * 80)

        # Create per-outage results
        results_per_outage = test_data[
            self.config.features.all_features + ['label', 'branch_name', 'outage_date', 'auction_month', 'market_month']
        ].copy()
        results_per_outage['predicted_shadow_price'] = y_pred_shadow_price
        results_per_outage['binding_probability'] = y_pred_proba
        results_per_outage['binding_probability_scaled'] = y_pred_proba_scaled
        results_per_outage['threshold'] = y_pred_threshold
        results_per_outage['predicted_binding'] = y_pred_binary
        results_per_outage['model_used'] = model_used
        results_per_outage['actual_binding'] = (test_data['label'] > 0).astype(int).values
        results_per_outage = results_per_outage.rename(columns={'label': 'actual_shadow_price'})

        if verbose:
            print(f"\nBefore aggregation:")
            print(f"  Total samples (constraint × outage_date): {len(results_per_outage):,}")
            print(f"  Unique constraints: {results_per_outage.index.get_level_values(0).nunique():,}")
            print(f"  Unique outage dates: {results_per_outage['outage_date'].nunique()}")

        # Aggregate by constraint_id (sum across all outage dates)
        final_results = results_per_outage.groupby(level=[0, 1]).agg({
            'branch_name': 'first',
            **{feat: 'mean' for feat in self.config.features.all_features},
            'actual_shadow_price': 'sum',
            'predicted_shadow_price': 'sum',
            'binding_probability': 'mean',
            'binding_probability_scaled': 'mean',
            'threshold': 'first',
            'predicted_binding': 'sum',
            'actual_binding': 'max',
            'auction_month': 'first',
            'market_month': 'first',
            'model_used': 'first'
        }).rename(columns={'predicted_binding': 'predicted_binding_count'})

        final_results['predicted_binding'] = final_results['predicted_binding_count'] >= 1

        # Calculate errors
        final_results['error'] = final_results['predicted_shadow_price'] - final_results['actual_shadow_price']
        final_results['abs_error'] = np.abs(final_results['error'])

        if verbose:
            print(f"\nAfter aggregation:")
            print(f"  Total unique constraints: {len(final_results):,}")
            print(f"  Aggregation: SUM of shadow prices across {results_per_outage['outage_date'].nunique()} outage dates")
            print(f"  Binding definition: Constraint binds if it binds in ANY outage date")

        return results_per_outage, final_results
