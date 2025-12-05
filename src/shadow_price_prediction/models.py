"""
Model training for classification and regression with ensemble support.
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
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


def create_model(
    model_spec: ModelSpec,
    is_classifier: bool = True,
    scale_pos_weight: float | None = None,
    monotone_constraints: str | None = None,
) -> Any:
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
    monotone_constraints : str, optional
        Monotonic constraints string for XGBoost (e.g., "(1,0,-1)")

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

    # 2.B: Handle Monotonic Constraints for XGBoost
    if monotone_constraints is not None and ("XGBClassifier" in str(model_class) or "XGBRegressor" in str(model_class)):
        params["monotone_constraints"] = monotone_constraints

    # Instantiate the model directly with its parameters
    return model_class(**params)


def train_ensemble(
    model_specs: list[ModelSpec],
    X: pd.DataFrame,  # noqa: N803
    y: pd.Series,
    is_classifier: bool = True,
    use_stacking: bool = False,
    groups: np.ndarray = None,
    monotone_constraints: str | None = None,
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
        If True, use Stacking instead of weighted averaging
    groups : np.ndarray, optional
        Group labels for cross-validation
    monotone_constraints : str, optional
        Monotonic constraints string for XGBoost

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
        model = create_model(
            spec,
            is_classifier=is_classifier,
            scale_pos_weight=scale_pos_weight,
            monotone_constraints=monotone_constraints,
        )

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
    weight_overrides: dict[str, float] | list[float] | None = None,
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
        elif isinstance(weight_overrides, dict):
            weights = list(weight_overrides.values())
        else:
            weights = list(weight_overrides)
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

    beta : float
        Beta parameter for F-beta score

    Returns:
    --------
    optimal_threshold : float
        Threshold that maximizes F-beta score
    max_fbeta : float
        Maximum F-beta score achieved
    """
    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    p, r = precision[:-1], recall[:-1]
    fbeta = (1 + beta**2) * p * r / (beta**2 * p + r)
    optimal_idx = np.argmax(fbeta)
    prev_idx = max(optimal_idx - 1, 0)
    optimal_threshold = (thresholds[optimal_idx] + thresholds[prev_idx]) / 2
    max_fbeta = fbeta[optimal_idx]

    # Apply scaling factor (heuristic to avoid overfitting)
    optimal_threshold = scaling_factor * optimal_threshold + (1 - scaling_factor) * 0.5

    return optimal_threshold, max_fbeta


class ShadowPriceModels:
    """Trains and stores ensemble of classification and regression models."""

    def __init__(self, config: PredictionConfig):
        self.config = config

        # Scalers by Horizon Group
        # Default model scalers: Dict[horizon_group, StandardScaler]
        self.scalers_default: dict[str, StandardScaler] = {}
        # Branch model scalers: Dict[horizon_group, Dict[(branch_name, flow_direction), StandardScaler]]
        self.scalers_branch: dict[str, dict[tuple[str, int], StandardScaler]] = {}

        # Ensembles by Horizon Group: Dict[branch_name, List[Tuple[model, weight]]]
        self.clf_ensembles: dict[str, dict[tuple[str, int], list[tuple[Any, float]]]] = {}
        self.reg_ensembles: dict[str, dict[tuple[str, int], list[tuple[Any, float]]]] = {}
        self.optimal_thresholds: dict[str, dict[tuple[str, int], float]] = {}

        self.clf_default_ensembles: dict[str, list[tuple[Any, float]]] = {}
        self.reg_default_ensembles: dict[str, list[tuple[Any, float]]] = {}
        self.optimal_threshold_defaults: dict[str, float] = {}

        # Initialize containers for each horizon group
        for group in self.config.horizon_groups:
            self.scalers_branch[group.name] = {}
            self.clf_ensembles[group.name] = {}
            self.reg_ensembles[group.name] = {}
            self.optimal_thresholds[group.name] = {}
            self.clf_default_ensembles[group.name] = []
            self.reg_default_ensembles[group.name] = []
            self.optimal_threshold_defaults[group.name] = 0.5

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
        return self.config.models.get_ensemble_weights_for_horizon(
            horizon, self.config.horizon_groups, model_type, is_branch
        )

    def _get_required_groups(self, test_periods: list[tuple[pd.Timestamp, pd.Timestamp]] | None = None) -> set[str]:
        """Determine which horizon groups are required based on test periods."""
        required_groups = set()

        if not test_periods:
            # If no test periods, assume all are needed
            return {g.name for g in self.config.horizon_groups}

        for auction_month, market_month in test_periods:
            # Calculate horizon
            horizon = (market_month.year - auction_month.year) * 12 + (market_month.month - auction_month.month)

            for group in self.config.horizon_groups:
                if group.min_horizon <= horizon <= group.max_horizon:
                    required_groups.add(group.name)

        return required_groups

    def _train_single_branch_classifier(
        self,
        branch_name: str,
        flow_direction: int,
        branch_data: pd.DataFrame,
        weight_overrides: dict[str, float] | None = None,
    ) -> tuple[str, int, list[tuple[Any, float]] | None, float | None, StandardScaler | None]:
        """
        Train a single branch classifier.
        Returns: (branch_name, flow_direction, ensemble, optimal_threshold, scaler)
        """
        # Skip branches with too few samples
        if len(branch_data) < self.config.training.min_samples_for_branch_model:
            return branch_name, flow_direction, None, 0.5, None

        # Skip branches with only one class (all 0s or all 1s)
        y_branch_raw = (branch_data["label"] > self.config.training.label_threshold).astype(int)
        if len(np.unique(y_branch_raw)) < 2:
            return branch_name, flow_direction, None, 0.5, None

        # Prepare branch data with BRANCH-SPECIFIC SCALING
        feature_cols = self.config.features.all_features
        scaler_branch = MinMaxScaler()
        branch_data_scaled = branch_data.copy()
        branch_data_scaled[feature_cols] = scaler_branch.fit_transform(branch_data[feature_cols])
        # Extract classification features from SCALED data
        # Note: config.features.step1_features is now a list of tuples (name, constraint)
        step1_feature_names = [f[0] for f in self.config.features.step1_features]
        X_branch = branch_data_scaled[step1_feature_names].copy()
        y_branch_binary = (branch_data_scaled["label"] > self.config.training.label_threshold).astype(int)
        groups_branch = (
            branch_data_scaled["auction_month"].values if "auction_month" in branch_data_scaled.columns else None
        )

        # Construct Monotonic Constraints for XGBoost
        # Format: "(1,0,-1)" corresponding to feature columns in X_branch
        constraints = [f[1] for f in self.config.features.step1_features]
        monotone_constraints = "(" + ",".join(map(str, constraints)) + ")"

        # Train branch-specific ensemble
        clf_ensemble = train_ensemble(
            self.config.models.branch_classifiers,
            X_branch,
            y_branch_binary,
            is_classifier=True,
            groups=groups_branch,
            monotone_constraints=monotone_constraints,
        )

        # Optimize threshold using ensemble predictions
        y_proba_train_branch = predict_ensemble(
            clf_ensemble, X_branch, predict_proba=True, weight_overrides=weight_overrides
        )
        optimal_threshold_branch, _ = find_optimal_threshold(
            y_branch_binary,
            y_proba_train_branch,
            self.config.threshold.threshold_beta,
            self.config.threshold.threshold_scaling_factor,
        )

        return branch_name, flow_direction, clf_ensemble, optimal_threshold_branch, scaler_branch

    def _train_single_branch_regressor(
        self, branch_name: str, flow_direction: int, branch_data: pd.DataFrame
    ) -> tuple[str, int, list[tuple[Any, float]] | None, Any | None]:
        """
        Train a single branch regressor.
        Returns: (branch_name, flow_direction, ensemble, scaler)
        """
        # Skip branches with too few samples
        if len(branch_data) < self.config.training.min_samples_for_branch_model:
            return branch_name, flow_direction, None, None

        # Prepare branch data with BRANCH-SPECIFIC SCALING
        feature_cols = self.config.features.all_features
        scaler_branch = MinMaxScaler()
        branch_data_scaled = branch_data.copy()
        branch_data_scaled[feature_cols] = scaler_branch.fit_transform(branch_data[feature_cols])

        # Filter for binding constraints (label > 0)
        # Extract regression features from SCALED data
        # Note: config.features.step2_features is now a list of tuples (name, constraint)
        step2_feature_names = [f[0] for f in self.config.features.step2_features]
        binding_mask = branch_data_scaled["label"] > 0
        X_branch_reg = branch_data_scaled.loc[binding_mask, step2_feature_names].copy()
        y_branch_reg = np.log1p(branch_data_scaled.loc[binding_mask, "label"].copy())
        groups_branch_reg = (
            branch_data_scaled.loc[binding_mask, "auction_month"].values
            if "auction_month" in branch_data_scaled.columns
            else None
        )

        if len(X_branch_reg) < self.config.training.min_binding_samples_for_regression:
            return branch_name, flow_direction, None, None

        # Construct Monotonic Constraints for XGBoost
        constraints = [f[1] for f in self.config.features.step2_features]
        monotone_constraints = "(" + ",".join(map(str, constraints)) + ")"

        # Train branch-specific ensemble
        reg_ensemble = train_ensemble(
            self.config.models.branch_regressors,
            X_branch_reg,
            y_branch_reg,
            is_classifier=False,
            groups=groups_branch_reg,
            monotone_constraints=monotone_constraints,
        )

        return branch_name, flow_direction, reg_ensemble, scaler_branch

    def train_classifiers(
        self,
        train_data: pd.DataFrame,
        test_branches: set[tuple[str, int]],
        test_periods: list[tuple[pd.Timestamp, pd.Timestamp]] | None = None,
        verbose: bool = True,
    ) -> None:
        """
        Train branch-specific classifiers and default fallback classifier for each horizon group.

        Parameters:
        -----------
        train_data : pd.DataFrame
            Training data with features and labels
        test_branches : set[tuple[str, int]]
            Set of (branch_name, flow_direction) tuples in test set
        test_periods : list of tuples, optional
            List of (auction_month, market_month) tuples to determine required horizon groups
        verbose : bool
            Print progress messages
        """
        if verbose:
            print("\n[Training Classification Models + Threshold Optimization]")
            print("-" * 80)

        # Define horizon groups and their associated attributes
        horizon_group_configs = {g.name: (g.min_horizon, g.max_horizon) for g in self.config.horizon_groups}

        # Determine required groups
        required_groups = self._get_required_groups(test_periods)

        if verbose:
            skipped_groups = set(horizon_group_configs.keys()) - required_groups
            if skipped_groups:
                print(f"  Skipping groups not in test periods: {', '.join(sorted(skipped_groups))}")

        for group_name, (min_h, max_h) in horizon_group_configs.items():
            if group_name not in required_groups:
                continue

            if verbose:
                print(f"\nProcessing {group_name.upper()} TERM models...")

            # Get weight overrides for this group
            weight_overrides = self.config.models.clf_weights.get(group_name)

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

            step1_feature_names = [f[0] for f in self.config.features.step1_features]
            X_train_all = group_data_scaled_default[step1_feature_names].copy()
            y_train_all = group_data_scaled_default["label"].copy()
            y_train_binary_all = (y_train_all > self.config.training.label_threshold).astype(int)

            # Extract groups (auction_month) for CV
            groups_all = group_data["auction_month"].values if "auction_month" in group_data.columns else None

            # Construct Monotonic Constraints for XGBoost
            constraints = [f[1] for f in self.config.features.step1_features]
            monotone_constraints = "(" + ",".join(map(str, constraints)) + ")"

            # Train default classifier ensemble
            default_ensemble = train_ensemble(
                self.config.models.default_classifiers,
                X_train_all,
                y_train_binary_all,
                is_classifier=True,
                groups=groups_all,
                monotone_constraints=monotone_constraints,
            )
            self.clf_default_ensembles[group_name] = default_ensemble

            if verbose:
                n_models = len(default_ensemble)
                print(f"    ✓ Default ensemble trained ({n_models} models)")
                print(f"    Total samples: {len(X_train_all):,}")

            # Optimize threshold for default classifier ensemble
            if verbose:
                print(f"  Optimizing threshold for default ensemble ({group_name})...")

            y_proba_train_all = predict_ensemble(
                default_ensemble, X_train_all, predict_proba=True, weight_overrides=weight_overrides
            )
            # Find optimal threshold
            optimal_threshold, max_f1 = find_optimal_threshold(
                y_train_binary_all,
                y_proba_train_all,
                beta=self.config.threshold.threshold_beta,
                scaling_factor=self.config.threshold.threshold_scaling_factor,
            )
            self.optimal_threshold_defaults[group_name] = optimal_threshold

            if verbose:
                print(f"    ✓ Optimal threshold: {optimal_threshold:.3f} (F-beta={max_f1:.3f})")

            # Train branch-specific classifiers
            if verbose:
                print(f"  Training branch-specific classifiers ({group_name})...")

            # Group RAW training data by branch AND flow_direction
            if "flow_direction" in group_data.index.names:
                group_data_reset = group_data.reset_index()
            else:
                group_data_reset = group_data

            train_data_by_branch = {
                (branch_name, flow_direction): branch_data
                for (branch_name, flow_direction), branch_data in group_data_reset.groupby(
                    ["branch_name", "flow_direction"]
                )
                if (branch_name, flow_direction) in test_branches
            }

            # Parallel Training
            param_dict_list = [
                {
                    "branch_name": branch_name,
                    "flow_direction": flow_direction,
                    "branch_data": branch_data,
                    "weight_overrides": self.config.models.reg_weights.get(group_name),
                }
                for (branch_name, flow_direction), branch_data in train_data_by_branch.items()
            ]

            # Sequential Training (as requested to avoid nested parallelism)
            results = []
            for param_dict in param_dict_list:
                result = self._train_single_branch_classifier(**param_dict)
                results.append(result)
            trained_count = 0
            skipped_count = 0

            for branch_name, flow_direction, ensemble, threshold, scaler in results:
                if ensemble is not None and threshold is not None:
                    self.clf_ensembles[group_name][(branch_name, flow_direction)] = ensemble
                    self.optimal_thresholds[group_name][(branch_name, flow_direction)] = threshold
                    self.scalers_branch[group_name][(branch_name, flow_direction)] = scaler
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
        test_branches: set[tuple[str, int]],
        test_periods: list[tuple[pd.Timestamp, pd.Timestamp]] | None = None,
        verbose: bool = True,
    ) -> None:
        """
        Train branch-specific regressors and default fallback regressor for each horizon group.

        Parameters:
        -----------
        train_data : pd.DataFrame
            Training data with features and labels
        test_branches : set[tuple[str, int]]
            Set of (branch_name, flow_direction) tuples in test set
        test_periods : list of tuples, optional
            List of (auction_month, market_month) tuples to determine required horizon groups
        verbose : bool
            Print progress messages
        """
        if verbose:
            print("\n[Training Regression Models]")
            print("-" * 80)

        # Define horizon groups and their associated attributes
        horizon_group_configs = {g.name: (g.min_horizon, g.max_horizon) for g in self.config.horizon_groups}

        # Determine required groups
        required_groups = self._get_required_groups(test_periods)

        if verbose:
            skipped_groups = set(horizon_group_configs.keys()) - required_groups
            if skipped_groups:
                print(f"  Skipping groups not in test periods: {', '.join(sorted(skipped_groups))}")

        for group_name, (min_h, max_h) in horizon_group_configs.items():
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

            # Train default regressor on SCALED data (binding only)
            if verbose:
                print(f"  Training default fallback regressor ({group_name})...")

            step2_feature_names = [f[0] for f in self.config.features.step2_features]
            binding_mask_all = group_data_scaled_default["label"] > 0
            X_train_reg_all = group_data_scaled_default.loc[binding_mask_all, step2_feature_names].copy()
            y_train_reg_all = np.log1p(group_data_scaled_default.loc[binding_mask_all, "label"].copy())
            groups_reg_all = (
                group_data.loc[binding_mask_all, "auction_month"].values
                if "auction_month" in group_data.columns
                else None
            )

            if len(X_train_reg_all) > self.config.training.min_binding_samples_for_regression:
                default_ensemble = train_ensemble(
                    self.config.models.default_regressors,
                    X_train_reg_all,
                    y_train_reg_all,
                    is_classifier=False,
                    groups=groups_reg_all,
                )
                self.reg_default_ensembles[group_name] = default_ensemble

                if verbose:
                    n_models = len(default_ensemble)
                    print(f"    ✓ Default ensemble trained ({n_models} models)")
                    print(f"    Total binding samples: {len(X_train_reg_all):,}")
            else:
                if verbose:
                    print("    ⚠️  Insufficient binding samples for default regressor. Skipping.")
                self.reg_default_ensembles[group_name] = []

            # Train branch-specific regressors
            if verbose:
                print(f"  Training branch-specific regressors ({group_name})...")

            # Group RAW training data by branch AND flow_direction
            if "flow_direction" in group_data.index.names:
                group_data_reset = group_data.reset_index()
            else:
                group_data_reset = group_data

            train_data_by_branch = {
                (branch_name, flow_direction): branch_data
                for (branch_name, flow_direction), branch_data in group_data_reset.groupby(
                    ["branch_name", "flow_direction"]
                )
                if (branch_name, flow_direction) in test_branches
            }

            # Parallel Training
            param_dict_list = [
                {
                    "branch_name": branch_name,
                    "flow_direction": flow_direction,
                    "branch_data": branch_data,
                }
                for (branch_name, flow_direction), branch_data in train_data_by_branch.items()
            ]

            # Sequential Training (as requested to avoid nested parallelism)
            results = []
            for param_dict in param_dict_list:
                result = self._train_single_branch_regressor(**param_dict)
                results.append(result)

            trained_count = 0
            skipped_count = 0

            for branch_name, flow_direction, ensemble, scaler in results:
                if ensemble is not None:
                    self.reg_ensembles[group_name][(branch_name, flow_direction)] = ensemble
                    self.scalers_branch[group_name][(branch_name, flow_direction)] = scaler
                    trained_count += 1
                else:
                    skipped_count += 1

            if verbose:
                print(f"    ✓ Trained {trained_count} branch models (skipped {skipped_count})")

        if verbose:
            print("\n✓ Regression Training Complete")

    def get_classifier_ensemble(
        self, branch_name: str, flow_direction: int, horizon: int
    ) -> tuple[list[tuple[Any, float]], float, Any | None]:
        """
        Get appropriate classifier ensemble and threshold for a branch and horizon.
        Returns: (ensemble, threshold, scaler)
        """
        # Determine horizon group
        # Determine horizon group
        group = None
        for g in self.config.horizon_groups:
            if g.min_horizon <= horizon <= g.max_horizon:
                group = g.name
                break

        if group is None:
            # Fallback if no group covers this horizon (shouldn't happen if config is complete)
            # Return default/empty
            return [], 0.5, None

        ensembles = self.clf_ensembles.get(group, {})
        thresholds = self.optimal_thresholds.get(group, {})
        default_ensemble = self.clf_default_ensembles.get(group, [])
        default_threshold = self.optimal_threshold_defaults.get(group)

        # Get scaler
        scaler = self.scalers_branch[group].get((branch_name, flow_direction))
        if scaler is None:
            scaler = self.scalers_default.get(group)

        # Check if branch-specific model exists
        if (branch_name, flow_direction) in ensembles:
            return ensembles[(branch_name, flow_direction)], thresholds[(branch_name, flow_direction)], scaler

        # Fallback to default
        if default_threshold is None:
            default_threshold = 0.5
        return default_ensemble, default_threshold, scaler

    def get_regressor_ensemble(
        self, branch_name: str, flow_direction: int, horizon: int
    ) -> tuple[list[tuple[Any, float]], Any | None]:
        """
        Get appropriate regressor ensemble for a branch and horizon.
        Returns: (ensemble, scaler)
        """
        # Determine horizon group
        # Determine horizon group
        group = None
        for g in self.config.horizon_groups:
            if g.min_horizon <= horizon <= g.max_horizon:
                group = g.name
                break

        if group is None:
            return [], None

        ensembles = self.reg_ensembles.get(group, {})
        default_ensemble = self.reg_default_ensembles.get(group, [])

        # Get scaler
        scaler = self.scalers_branch[group].get((branch_name, flow_direction))
        if scaler is None:
            scaler = self.scalers_default.get(group)

        # Check if branch-specific model exists
        if (branch_name, flow_direction) in ensembles:
            return ensembles[(branch_name, flow_direction)], scaler

        # Fallback to default
        return default_ensemble, scaler

    def transform_features(self, data: pd.DataFrame, scaled_col_suffix: str | None = None) -> pd.DataFrame:
        """
        Transform features using the trained scalers.

        Parameters:
        -----------
        data : pd.DataFrame
            Data to transform. Must contain 'forecast_horizon' and 'branch_name'.
            'flow_direction' is optional (defaults to 0).
        scaled_col_suffix : str, optional
            If provided, scaled features will be added as new columns with this suffix
            (e.g., 'feature1_scaled'). If None, original columns are overwritten.

        Returns:
        --------
        data_scaled : pd.DataFrame
            Data with scaled features.
        """
        data_scaled = data.copy()
        feature_cols = self.config.features.all_features

        # Determine target columns
        if scaled_col_suffix:
            target_cols = [f"{col}{scaled_col_suffix}" for col in feature_cols]
            # Initialize new columns to ensure they exist for iloc indexing
            for col in target_cols:
                data_scaled[col] = 0.0
        else:
            target_cols = feature_cols

        # Helper to get flow_direction
        if "flow_direction" in data.columns:
            flow_vals = data["flow_direction"].values
        elif "flow_direction" in data.index.names:
            flow_vals = data.index.get_level_values("flow_direction").values
        else:
            flow_vals = np.zeros(len(data), dtype=int)

        # Helper to get branch_name
        if "branch_name" in data.columns:
            branch_vals = data["branch_name"].values
        elif "branch_name" in data.index.names:
            branch_vals = data.index.get_level_values("branch_name").values
        else:
            raise ValueError("branch_name must be in columns or index")

        # Helper to get forecast_horizon
        if "forecast_horizon" in data.columns:
            horizon_vals = data["forecast_horizon"].values
        else:
            raise ValueError("forecast_horizon must be in columns")

        # Create temporary dataframe for grouping (using RangeIndex for positional access)
        group_df = pd.DataFrame({"h": horizon_vals, "b": branch_vals, "f": flow_vals})

        # Pre-calculate column indices for faster iloc assignment
        # Note: We must use data_scaled here because it might have new columns
        col_indices = [data_scaled.columns.get_loc(c) for c in target_cols]
        # We also need indices of source features for transformation
        source_col_indices = [data.columns.get_loc(c) for c in feature_cols]

        for (h, b, f), indices in group_df.groupby(["h", "b", "f"]).groups.items():
            # Determine horizon group
            # Determine horizon group
            h_group = None
            for g in self.config.horizon_groups:
                if g.min_horizon <= h <= g.max_horizon:
                    h_group = g.name
                    break

            if h_group is None:
                continue

            # Get scaler
            scaler = self.scalers_branch[h_group].get((b, f))
            if scaler is None:
                scaler = self.scalers_default.get(h_group)

            if scaler:
                # Transform features for this slice
                # indices are integer positions in group_df, which correspond to data rows
                # Use source_col_indices to get original values
                subset_values = data.iloc[indices, source_col_indices].values
                transformed_values = scaler.transform(subset_values)

                # Assign back using iloc to target columns
                data_scaled.iloc[indices, col_indices] = transformed_values

        return data_scaled
