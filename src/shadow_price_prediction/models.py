"""
Model training for classification and regression with ensemble support.
"""

from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .config import ModelSpec, PredictionConfig


class StackingModel:
    """
    Implements Stacking Ensemble (3.B).
    Trains base models and a meta-model on their predictions.
    """

    def __init__(
        self,
        base_models: list[Any],
        meta_model: Any = None,
        use_proba: bool = True,
        cv: int = 3,
        split_by_groups: bool = False,
    ):
        self.base_models = base_models
        self.meta_model = meta_model if meta_model is not None else LogisticRegression()
        self.use_proba = use_proba
        self.cv = cv
        self.split_by_groups = split_by_groups
        self.is_fitted = False

    def fit(self, X, y, groups=None):  # noqa: N803
        # 1. Train base models
        # To avoid overfitting, we should ideally use cross-val predictions for the meta-model training
        # But for simplicity in this custom implementation, we'll use the standard approach:
        # Fit base models on full data, but for meta-model training, we need out-of-fold predictions.

        meta_features = []

        # Determine CV strategy
        cv = self.cv
        if self.split_by_groups and groups is not None:
            # Use GroupKFold if groups are provided and requested
            cv = GroupKFold(n_splits=self.cv)

        for model in self.base_models:
            # Get out-of-fold predictions
            if self.use_proba and hasattr(model, "predict_proba"):
                preds = cross_val_predict(model, X, y, cv=cv, groups=groups, method="predict_proba")[:, 1]
            else:
                preds = cross_val_predict(model, X, y, cv=cv, groups=groups, method="predict")

            meta_features.append(preds.reshape(-1, 1))

            # Fit the model on full data for final inference
            model.fit(X, y)

        meta_X = np.hstack(meta_features)

        # 2. Train meta-model
        self.meta_model.fit(meta_X, y)
        self.is_fitted = True
        return self

    def predict(self, X):  # noqa: N803
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        meta_features = []
        for model in self.base_models:
            if self.use_proba and hasattr(model, "predict_proba"):
                preds = model.predict_proba(X)[:, 1]
            else:
                preds = model.predict(X)
            meta_features.append(preds.reshape(-1, 1))

        meta_X = np.hstack(meta_features)
        return self.meta_model.predict(meta_X)

    def predict_proba(self, X):  # noqa: N803
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        meta_features = []
        for model in self.base_models:
            if self.use_proba and hasattr(model, "predict_proba"):
                preds = model.predict_proba(X)[:, 1]
            else:
                preds = model.predict(X)
            meta_features.append(preds.reshape(-1, 1))

        meta_X = np.hstack(meta_features)
        return self.meta_model.predict_proba(meta_X)


def create_model(model_spec: ModelSpec, is_classifier: bool = True, scale_pos_weight: float | None = None) -> Any:
    """
    Factory function to create a model instance based on ModelSpec.

    Parameters:
    -----------
    model_spec : ModelSpec
        Model specification containing model class and configuration
    is_classifier : bool
        Whether creating a classifier (True) or regressor (False)
    scale_pos_weight : float, optional
        Weight to balance positive class (for XGBoost classifiers)

    Returns:
    --------
    model : sklearn/xgboost model instance
        Instantiated model ready for training
    """
    model_class = model_spec.model_class
    config = model_spec.config

    # Get parameters from config
    params = config.to_dict()

    # 2.A: Handle Class Imbalance for XGBoost
    if is_classifier and "XGBClassifier" in str(model_class) and scale_pos_weight is not None:
        # Only apply if not explicitly set in config
        if "scale_pos_weight" not in params or params["scale_pos_weight"] is None:
            params["scale_pos_weight"] = scale_pos_weight

    # Instantiate the model directly with its parameters
    return model_class(**params)


def train_ensemble(
    model_specs: list[ModelSpec],
    X: pd.DataFrame,  # noqa: N803
    y: pd.Series,
    is_classifier: bool = True,
    use_stacking: bool = False,
    groups: np.ndarray = None,
) -> list[tuple[Any, float]]:
    """
    Train an ensemble of models and return list of (model, weight) tuples.

    Parameters:
    -----------
    model_specs : List[ModelSpec]
        List of model specifications
    X : pd.DataFrame
        Features
    y : pd.Series
        Target
    is_classifier : bool
        Whether training classifiers or regressors
    use_stacking : bool
        If True, use Stacking instead of weighted averaging (Not fully implemented in this function return type,
        but prepared for future extension. For now, we stick to weighted averaging but improve individual models)
    groups : np.ndarray, optional
        Group labels for cross-validation (e.g., auction_month)

    Returns:
    --------
    ensemble : List[Tuple[model, weight]]
        Trained models with their weights
    """
    ensemble = []

    # Calculate scale_pos_weight for this batch
    scale_pos_weight = 1.0
    if is_classifier:
        n_pos = y.sum()
        n_neg = len(y) - n_pos
        if n_pos > 0:
            scale_pos_weight = n_neg / n_pos

    for spec in model_specs:
        model = create_model(spec, is_classifier=is_classifier, scale_pos_weight=scale_pos_weight)

        # Pass groups to fit if it's a StackingModel
        if isinstance(model, StackingModel):
            model.fit(X, y, groups=groups)
        else:
            model.fit(X, y)

        ensemble.append((model, spec.weight))

    return ensemble


def predict_ensemble(
    ensemble: list[tuple[Any, float]],
    X: pd.DataFrame,  # noqa: N803
    predict_proba: bool = False,
    weight_overrides: list[float] | None = None,
) -> np.ndarray:
    """
    Make predictions using an ensemble of models with weighted averaging.

    Parameters:
    -----------
    ensemble : List[Tuple[model, weight]]
        Trained models with their weights
    X : pd.DataFrame
        Features for prediction
    predict_proba : bool
        If True, return probabilities (for classifiers)
    weight_overrides : List[float], optional
        If provided, use these weights instead of the ensemble's default weights.
        Must have the same length as the ensemble.

    Returns:
    --------
    predictions : np.ndarray
        Weighted averaged predictions
    """
    if len(ensemble) == 0:
        # Return zeros if no models are available
        return np.zeros(len(X))

    # Use override weights if provided, otherwise use ensemble weights
    if weight_overrides is not None:
        if len(weight_overrides) != len(ensemble):
            # Fallback to ensemble weights if mismatch
            # This can happen if a model failed to train
            weights = [weight for _, weight in ensemble]
        else:
            weights = weight_overrides
    else:
        weights = [weight for _, weight in ensemble]

    predictions = None
    for i, (model, _) in enumerate(ensemble):
        if predict_proba:
            # For classifiers, get probability of positive class
            pred = model.predict_proba(X)[:, 1]
        else:
            # For regressors or class predictions
            pred = model.predict(X)

        if predictions is None:
            predictions = weights[i] * pred
        else:
            predictions += weights[i] * pred

    return predictions


def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: np.ndarray = None,
    beta: float = 0.5,
    scaling_factor: float = 1.0,
) -> tuple[float, float]:
    """
    Find optimal classification threshold that maximizes F-beta score.

    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_proba : array-like
        Predicted probabilities for positive class
    thresholds : array-like, optional
        Threshold values to search
    beta : float
        Beta parameter for F-beta score

    Returns:
    --------
    optimal_threshold : float
        Threshold that maximizes F-beta score
    max_fbeta : float
        Maximum F-beta score achieved
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    fbeta_scores = []
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        fbeta = fbeta_score(y_true, y_pred, average="weighted", beta=beta)
        fbeta_scores.append(fbeta)

    optimal_idx = np.argmax(fbeta_scores)
    optimal_threshold = thresholds[optimal_idx]
    max_fbeta = fbeta_scores[optimal_idx]

    # Apply scaling factor (heuristic to avoid overfitting)
    optimal_threshold = scaling_factor * optimal_threshold + (1 - scaling_factor) * 0.5

    return optimal_threshold, max_fbeta


class ShadowPriceModels:
    """Trains and stores ensemble of classification and regression models."""

    def __init__(self, config: PredictionConfig):
        self.config = config

        # Scalers by Horizon Group
        # Scalers by Horizon Group
        # Default model scalers: Dict[horizon_group, StandardScaler]
        self.scalers_default: dict[str, StandardScaler] = {}
        # Branch model scalers: Dict[horizon_group, Dict[branch_name, StandardScaler]]
        self.scalers_branch: dict[str, dict[str, StandardScaler]] = {"f0": {}, "f1": {}, "medium": {}, "long": {}}

        # Ensembles by Horizon Group: Dict[branch_name, List[Tuple[model, weight]]]

        # f0 (horizon = 0)
        self.clf_ensembles_f0: dict[str, list[tuple[Any, float]]] = {}
        self.reg_ensembles_f0: dict[str, list[tuple[Any, float]]] = {}
        self.optimal_thresholds_f0: dict[str, float] = {}
        self.clf_default_ensemble_f0: list[tuple[Any, float]] = []
        self.reg_default_ensemble_f0: list[tuple[Any, float]] = []
        self.optimal_threshold_default_f0: float | None = None

        # f1 (horizon = 1)
        self.clf_ensembles_f1: dict[str, list[tuple[Any, float]]] = {}
        self.reg_ensembles_f1: dict[str, list[tuple[Any, float]]] = {}
        self.optimal_thresholds_f1: dict[str, float] = {}
        self.clf_default_ensemble_f1: list[tuple[Any, float]] = []
        self.reg_default_ensemble_f1: list[tuple[Any, float]] = []
        self.optimal_threshold_default_f1: float | None = None

        # Medium Term (f2, f3: 2 <= horizon <= 3)
        self.clf_ensembles_medium: dict[str, list[tuple[Any, float]]] = {}
        self.reg_ensembles_medium: dict[str, list[tuple[Any, float]]] = {}
        self.optimal_thresholds_medium: dict[str, float] = {}
        self.clf_default_ensemble_medium: list[tuple[Any, float]] = []
        self.reg_default_ensemble_medium: list[tuple[Any, float]] = []
        self.optimal_threshold_default_medium: float | None = None

        # Long Term (Quarterly: horizon > 3)
        self.clf_ensembles_long: dict[str, list[tuple[Any, float]]] = {}
        self.reg_ensembles_long: dict[str, list[tuple[Any, float]]] = {}
        self.optimal_thresholds_long: dict[str, float] = {}
        self.clf_default_ensemble_long: list[tuple[Any, float]] = []
        self.reg_default_ensemble_long: list[tuple[Any, float]] = []
        self.optimal_threshold_default_long: float | None = None

        # Legacy/Generic accessors (mapped to f0 for backward compatibility if needed)
        self.clf_ensembles = self.clf_ensembles_f0
        self.reg_ensembles = self.reg_ensembles_f0
        self.optimal_thresholds = self.optimal_thresholds_f0

    def get_ensemble_weights_for_horizon(
        self, horizon: int, model_type: str = "classifier", is_branch: bool = False
    ) -> list[float]:
        """
        Get ensemble weights based on forecast horizon.

        This delegates to the EnsembleConfig method to retrieve
        horizon-stratified weights.

        Parameters:
        -----------
        horizon : int
            Forecast horizon in months
        model_type : str
            'classifier' or 'regressor'
        is_branch : bool
            Whether this is for branch-specific models

        Returns:
        --------
        weights : List[float]
            Normalized ensemble weights
        """
        return self.config.models.get_ensemble_weights_for_horizon(horizon, model_type, is_branch)

    def _get_required_groups(self, test_periods: list[tuple[pd.Timestamp, pd.Timestamp]] | None = None) -> set[str]:
        """Determine which horizon groups are required based on test periods."""
        required_groups = set()

        if not test_periods:
            # If no test periods, assume all are needed
            return {"f0", "f1", "medium", "long"}

        for auction_month, market_month in test_periods:
            # Calculate horizon
            horizon = (market_month.year - auction_month.year) * 12 + (market_month.month - auction_month.month)

            if horizon == 0:
                required_groups.add("f0")
            elif horizon == 1:
                required_groups.add("f1")
            elif 2 <= horizon <= 3:
                required_groups.add("medium")
            elif horizon > 3:
                required_groups.add("long")

        return required_groups

    def _train_single_branch_classifier(
        self, branch_name: str, branch_data: pd.DataFrame, thresholds: np.ndarray
    ) -> tuple[str, list[tuple[Any, float]] | None, float, Any | None]:
        """
        Train a single branch classifier.
        Returns: (branch_name, ensemble, optimal_threshold, scaler)
        """
        try:
            # Skip branches with too few samples
            if len(branch_data) < self.config.training.min_samples_for_branch_model:
                return branch_name, None, 0.5, None

            # Skip branches with only one class (all 0s or all 1s)
            y_branch_raw = (branch_data["label"] > self.config.training.label_threshold).astype(int)
            if len(np.unique(y_branch_raw)) < 2:
                return branch_name, None, 0.5, None

            # Prepare branch data with BRANCH-SPECIFIC SCALING
            feature_cols = self.config.features.all_features
            scaler_branch = MinMaxScaler()
            branch_data_scaled = branch_data.copy()
            branch_data_scaled[feature_cols] = scaler_branch.fit_transform(branch_data[feature_cols])

            X_branch = branch_data_scaled[self.config.features.step1_features].copy()
            y_branch_binary = (branch_data_scaled["label"] > self.config.training.label_threshold).astype(int)
            groups_branch = (
                branch_data_scaled["auction_month"].values if "auction_month" in branch_data_scaled.columns else None
            )

            # Train branch-specific ensemble
            clf_ensemble = train_ensemble(
                self.config.models.branch_classifiers,
                X_branch,
                y_branch_binary,
                is_classifier=True,
                groups=groups_branch,
            )

            # Optimize threshold using ensemble predictions
            y_proba_train_branch = predict_ensemble(clf_ensemble, X_branch, predict_proba=True)
            optimal_threshold_branch, _ = find_optimal_threshold(
                y_branch_binary,
                y_proba_train_branch,
                thresholds,
                self.config.threshold.threshold_beta,
                self.config.threshold.threshold_scaling_factor,
            )

            return branch_name, clf_ensemble, optimal_threshold_branch, scaler_branch

        except Exception as e:
            print(f"    ⚠️ Failed to train classifier for branch {branch_name}: {e}")
            return branch_name, None, 0.5, None

    def _train_single_branch_regressor(
        self, branch_name: str, branch_data: pd.DataFrame
    ) -> tuple[str, list[tuple[Any, float]] | None, Any | None]:
        """
        Train a single branch regressor.
        Returns: (branch_name, ensemble, scaler)
        """
        try:
            # Skip branches with too few samples
            if len(branch_data) < self.config.training.min_samples_for_branch_model:
                return branch_name, None, None

            # Prepare branch data with BRANCH-SPECIFIC SCALING
            feature_cols = self.config.features.all_features
            scaler_branch = MinMaxScaler()
            branch_data_scaled = branch_data.copy()
            branch_data_scaled[feature_cols] = scaler_branch.fit_transform(branch_data[feature_cols])

            # Filter for binding constraints (label > 0)
            binding_mask = branch_data_scaled["label"] > 0
            X_branch_reg = branch_data_scaled.loc[binding_mask, self.config.features.step2_features].copy()
            y_branch_reg = np.log1p(branch_data_scaled.loc[binding_mask, "label"].copy())
            groups_branch_reg = (
                branch_data_scaled.loc[binding_mask, "auction_month"].values
                if "auction_month" in branch_data_scaled.columns
                else None
            )

            if len(X_branch_reg) < self.config.training.min_binding_samples_for_regression:
                return branch_name, None, None

            # Train branch-specific ensemble
            reg_ensemble = train_ensemble(
                self.config.models.branch_regressors,
                X_branch_reg,
                y_branch_reg,
                is_classifier=False,
                groups=groups_branch_reg,
            )

            return branch_name, reg_ensemble, scaler_branch

        except Exception as e:
            print(f"    ⚠️ Failed to train regressor for branch {branch_name}: {e}")
            return branch_name, None, None

    def train_classifiers(
        self,
        train_data: pd.DataFrame,
        test_branches: set[str],
        test_periods: list[tuple[pd.Timestamp, pd.Timestamp]] | None = None,
        verbose: bool = True,
    ) -> None:
        """
        Train branch-specific classifiers and default fallback classifier for each horizon group.

        Parameters:
        -----------
        train_data : pd.DataFrame
            Training data with features and labels
        test_branches : set
            Set of branch names in test set
        test_periods : list of tuples, optional
            List of (auction_month, market_month) tuples to determine required horizon groups
        verbose : bool
            Print progress messages
        """
        if verbose:
            print("\n[Training Classification Models + Threshold Optimization]")
            print("-" * 80)

        # Define horizon groups and their associated attributes
        # (min_horizon, max_horizon, branch_ensembles_dict, default_ensemble_attr_name, default_threshold_attr_name, branch_thresholds_dict)
        horizon_group_configs = {
            "f0": (
                0,
                0,
                self.clf_ensembles_f0,
                "clf_default_ensemble_f0",
                "optimal_threshold_default_f0",
                self.optimal_thresholds_f0,
            ),
            "f1": (
                1,
                1,
                self.clf_ensembles_f1,
                "clf_default_ensemble_f1",
                "optimal_threshold_default_f1",
                self.optimal_thresholds_f1,
            ),
            "medium": (
                2,
                3,
                self.clf_ensembles_medium,
                "clf_default_ensemble_medium",
                "optimal_threshold_default_medium",
                self.optimal_thresholds_medium,
            ),
            "long": (
                4,
                999,
                self.clf_ensembles_long,
                "clf_default_ensemble_long",
                "optimal_threshold_default_long",
                self.optimal_thresholds_long,
            ),
        }

        # Determine required groups
        required_groups = self._get_required_groups(test_periods)

        if verbose:
            skipped_groups = set(horizon_group_configs.keys()) - required_groups
            if skipped_groups:
                print(f"  Skipping groups not in test periods: {', '.join(sorted(skipped_groups))}")

        for group_name, (
            min_h,
            max_h,
            branch_ensembles,
            default_ensemble_attr,
            default_threshold_attr,
            branch_thresholds,
        ) in horizon_group_configs.items():
            if group_name not in required_groups:
                continue

            if verbose:
                print(f"\nProcessing {group_name.upper()} TERM models...")

            # Filter data for this horizon group
            group_data = train_data[
                (train_data["forecast_horizon"] >= min_h) & (train_data["forecast_horizon"] <= max_h)
            ]

            if len(group_data) == 0:
                if verbose:
                    print(f"  ⚠️ No training data for {group_name} term. Skipping.")
                continue

            # 1. Handle Default Model Scaling & Training
            feature_cols = self.config.features.all_features
            scaler_default = MinMaxScaler()

            # Create scaled copy for default model
            group_data_scaled_default = group_data.copy()
            group_data_scaled_default[feature_cols] = scaler_default.fit_transform(group_data[feature_cols])
            self.scalers_default[group_name] = scaler_default

            # Train default classifier on SCALED data
            if verbose:
                print(f"  Training default fallback classifier ({group_name})...")

            X_train_all = group_data_scaled_default[self.config.features.step1_features].copy()
            y_train_all = group_data_scaled_default["label"].copy()
            y_train_binary_all = (y_train_all > self.config.training.label_threshold).astype(int)

            # Extract groups (auction_month) for CV
            groups_all = group_data["auction_month"].values if "auction_month" in group_data.columns else None

            # Train default classifier ensemble
            default_ensemble = train_ensemble(
                self.config.models.default_classifiers,
                X_train_all,
                y_train_binary_all,
                is_classifier=True,
                groups=groups_all,
            )
            setattr(self, default_ensemble_attr, default_ensemble)

            if verbose:
                n_models = len(default_ensemble)
                # model_names = [spec.model_class.__name__ for spec in self.config.models.default_classifiers]
                print(f"    ✓ Default ensemble trained ({n_models} models)")
                print(f"    Total samples: {len(X_train_all):,}")

            # Optimize threshold for default classifier ensemble
            if verbose:
                print(f"  Optimizing threshold for default ensemble ({group_name})...")

            y_proba_train_all = predict_ensemble(default_ensemble, X_train_all, predict_proba=True)
            thresholds = np.linspace(
                self.config.threshold.threshold_range_start,
                self.config.threshold.threshold_range_end,
                self.config.threshold.threshold_range_steps,
            )
            optimal_threshold, max_f1 = find_optimal_threshold(
                y_train_binary_all,
                y_proba_train_all,
                thresholds,
                self.config.threshold.threshold_beta,
                self.config.threshold.threshold_scaling_factor,
            )
            setattr(self, default_threshold_attr, optimal_threshold)

            if verbose:
                print(f"    ✓ Optimal threshold: {optimal_threshold:.3f} (F-beta={max_f1:.3f})")

            # Train branch-specific classifiers
            if verbose:
                print(f"  Training branch-specific classifiers ({group_name})...")

            # Group RAW training data by branch
            train_data_by_branch = {
                branch_name: branch_data
                for branch_name, branch_data in group_data.groupby("branch_name")
                if branch_name in test_branches  # Only process relevant branches
            }

            # Parallel Training
            results = Parallel(n_jobs=-1, backend="threading")(
                delayed(self._train_single_branch_classifier)(branch_name, branch_data, thresholds)
                for branch_name, branch_data in train_data_by_branch.items()
            )

            trained_count = 0
            skipped_count = 0

            for branch_name, ensemble, threshold, scaler in results:
                if ensemble is not None:
                    branch_ensembles[branch_name] = ensemble
                    branch_thresholds[branch_name] = threshold
                    self.scalers_branch[group_name][branch_name] = scaler
                    trained_count += 1
                else:
                    skipped_count += 1

            if verbose:
                print(f"    ✓ Trained {trained_count} branch models (skipped {skipped_count})")

        if verbose:
            print("\n✓ Classification Training Complete")

    def train_regressors(
        self,
        train_data: pd.DataFrame,
        test_branches: set[str],
        test_periods: list[tuple[pd.Timestamp, pd.Timestamp]] | None = None,
        verbose: bool = True,
    ) -> None:
        """
        Train branch-specific regressors and default fallback regressor for each horizon group.

        Parameters:
        -----------
        train_data : pd.DataFrame
            Training data with features and labels
        test_branches : set
            Set of branch names in test set
        test_periods : list of tuples, optional
            List of (auction_month, market_month) tuples to determine required horizon groups
        verbose : bool
            Print progress messages
        """
        if verbose:
            print("\n[Training Regression Models]")
            print("-" * 80)

        # Define horizon groups and their associated attributes
        # (min_horizon, max_horizon, branch_ensembles_dict, default_ensemble_attr_name)
        horizon_group_configs = {
            "f0": (0, 0, self.reg_ensembles_f0, "reg_default_ensemble_f0"),
            "f1": (1, 1, self.reg_ensembles_f1, "reg_default_ensemble_f1"),
            "medium": (2, 3, self.reg_ensembles_medium, "reg_default_ensemble_medium"),
            "long": (4, 999, self.reg_ensembles_long, "reg_default_ensemble_long"),
        }

        # Determine required groups
        required_groups = self._get_required_groups(test_periods)

        if verbose:
            skipped_groups = set(horizon_group_configs.keys()) - required_groups
            if skipped_groups:
                print(f"  Skipping groups not in test periods: {', '.join(sorted(skipped_groups))}")

        for group_name, (min_h, max_h, branch_ensembles, default_ensemble_attr) in horizon_group_configs.items():
            if group_name not in required_groups:
                continue
            if verbose:
                print(f"\nProcessing {group_name.upper()} TERM regressors...")

            # Filter data for this horizon group
            group_data = train_data[
                (train_data["forecast_horizon"] >= min_h) & (train_data["forecast_horizon"] <= max_h)
            ]

            if len(group_data) == 0:
                if verbose:
                    print(f"  ⚠️ No training data for {group_name} term. Skipping.")
                continue

            # 1. Handle Default Model Scaling & Training
            scaler_default = self.scalers_default.get(group_name)
            feature_cols = self.config.features.all_features

            scaler_default = self.scalers_default.get(group_name)

            if scaler_default:
                group_data_scaled_default = group_data.copy()
                group_data_scaled_default[feature_cols] = scaler_default.transform(group_data[feature_cols])

                # Train default regressor on all binding data in this group
                if verbose:
                    print(f"  Training default fallback regressor ({group_name})...")

                X_train_all_reg = group_data_scaled_default[self.config.features.step2_features].copy()
                y_train_all_reg = group_data_scaled_default["label"].copy()
            else:
                if verbose:
                    print(f"  ⚠️ No default scaler for {group_name}. Skipping default regressor.")
                # Create empty/dummy data to skip training block gracefully or handle it
                X_train_all_reg = pd.DataFrame()
                y_train_all_reg = pd.Series()
            groups_all_reg = group_data["auction_month"].values if "auction_month" in group_data.columns else None

            binding_mask_all = y_train_all_reg > self.config.training.label_threshold
            X_train_binding_all = X_train_all_reg[binding_mask_all]
            y_train_binding_all = np.log1p(y_train_all_reg[binding_mask_all])

            if groups_all_reg is not None:
                groups_binding_all = groups_all_reg[binding_mask_all]
            else:
                groups_binding_all = None

            if len(X_train_binding_all) > 10:
                # Train default regressor ensemble
                default_ensemble = train_ensemble(
                    self.config.models.default_regressors,
                    X_train_binding_all,
                    y_train_binding_all,
                    is_classifier=False,
                    groups=groups_binding_all,
                )
                setattr(self, default_ensemble_attr, default_ensemble)

                if verbose:
                    n_models = len(default_ensemble)
                    # model_names = [spec.model_class.__name__ for spec in self.config.models.default_regressors]
                    print(f"    ✓ Default ensemble trained ({n_models} models)")
                    print(f"    Binding samples: {len(X_train_binding_all):,}")
            else:
                if verbose:
                    print("    ⚠️  Insufficient binding samples for default regressor")
                # Ensure the default ensemble for this group is empty if not trained
                setattr(self, default_ensemble_attr, [])

            # Train branch-specific regressors
            if verbose:
                print(f"  Training branch-specific regressors ({group_name})...")

            # Group RAW training data by branch
            train_data_by_branch = {
                branch_name: branch_data
                for branch_name, branch_data in group_data.groupby("branch_name")
                if branch_name in test_branches  # Only process relevant branches
            }

            # Parallel Training
            results = Parallel(n_jobs=-1, backend="threading")(
                delayed(self._train_single_branch_regressor)(branch_name, branch_data)
                for branch_name, branch_data in train_data_by_branch.items()
            )

            trained_count = 0
            skipped_count = 0

            for branch_name, ensemble, scaler in results:
                if ensemble is not None:
                    branch_ensembles[branch_name] = ensemble
                    self.scalers_branch[group_name][branch_name] = scaler
                    trained_count += 1
                else:
                    skipped_count += 1

            if verbose:
                print(f"    ✓ Trained {trained_count} branch models (skipped {skipped_count})")

            # This line was problematic as self.reg_default_ensemble is not defined in the new structure
            # It should refer to the current group's default ensemble
            current_default_ensemble = getattr(self, default_ensemble_attr)
            if verbose:
                print(
                    f"  Default regressor for {group_name}: {'Available' if len(current_default_ensemble) > 0 else 'Not available'}"
                )

        if verbose:
            print("\n✓ Regression Training Complete")
            # These summary lines need to be updated to reflect horizon-specific models
            # For now, they are commented out or simplified as the original structure is gone.
            # print(f"  Models trained: {trained_reg_count:,}") # This count is per group now
            # print(f"  Branches skipped: {skipped_reg_count:,}") # This is per group now
            # print(f"  Default regressor: {'Available' if len(self.reg_default_ensemble) > 0 else 'Not available'}") # This is per group now

    def get_classifier_ensemble(
        self, branch_name: str, horizon: int
    ) -> tuple[list[tuple[Any, float]], float, Any | None]:
        """
        Get appropriate classifier ensemble, threshold, and scaler for a branch and horizon.
        """
        # Determine horizon group
        if horizon == 0:
            group = "f0"
            ensembles = self.clf_ensembles_f0
            default_ensemble = self.clf_default_ensemble_f0
            thresholds = self.optimal_thresholds_f0
            default_threshold = self.optimal_threshold_default_f0
        elif horizon == 1:
            group = "f1"
            ensembles = self.clf_ensembles_f1
            default_ensemble = self.clf_default_ensemble_f1
            thresholds = self.optimal_thresholds_f1
            default_threshold = self.optimal_threshold_default_f1
        elif 2 <= horizon <= 3:
            group = "medium"
            ensembles = self.clf_ensembles_medium
            default_ensemble = self.clf_default_ensemble_medium
            thresholds = self.optimal_thresholds_medium
            default_threshold = self.optimal_threshold_default_medium
        else:
            group = "long"
            ensembles = self.clf_ensembles_long
            default_ensemble = self.clf_default_ensemble_long
            thresholds = self.optimal_thresholds_long
            default_threshold = self.optimal_threshold_default_long

        # Get scaler
        scaler = self.scalers_branch[group].get(branch_name)
        if scaler is None:
            scaler = self.scalers_default.get(group)

        # Check if branch-specific model exists
        if branch_name in ensembles:
            return ensembles[branch_name], thresholds.get(branch_name, 0.5), scaler

        # Fallback to default - validate that training was completed
        if default_threshold is None:
            raise ValueError(
                f"Default optimal threshold for horizon group '{group}' was not set during training. "
                f"This indicates that train_classifiers() was not called or failed for this horizon group. "
                f"Please ensure proper model training before prediction."
            )

        return default_ensemble, default_threshold, scaler

    def get_regressor_ensemble(self, branch_name: str, horizon: int) -> tuple[list[tuple[Any, float]], Any | None]:
        """
        Get appropriate regressor ensemble and scaler for a branch and horizon.
        """
        # Determine horizon group
        if horizon == 0:
            group = "f0"
            ensembles = self.reg_ensembles_f0
            default_ensemble = self.reg_default_ensemble_f0
        elif horizon == 1:
            group = "f1"
            ensembles = self.reg_ensembles_f1
            default_ensemble = self.reg_default_ensemble_f1
        elif 2 <= horizon <= 3:
            group = "medium"
            ensembles = self.reg_ensembles_medium
            default_ensemble = self.reg_default_ensemble_medium
        else:
            group = "long"
            ensembles = self.reg_ensembles_long
            default_ensemble = self.reg_default_ensemble_long

        # Get scaler
        scaler = self.scalers_branch[group].get(branch_name)
        if scaler is None:
            scaler = self.scalers_default.get(group)

        # Check if branch-specific model exists
        if branch_name in ensembles:
            return ensembles[branch_name], scaler

        # Fallback to default
        return default_ensemble, scaler
