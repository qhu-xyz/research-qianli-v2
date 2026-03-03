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


def compute_value_weights(y_binary: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Compute per-sample weights that encode class balance AND shadow price magnitude.

    For classifiers: negatives get weight 1.0, positives get weight proportional to
    log1p(shadow_price), scaled so total positive weight = total negative weight
    (preserving the class balance that scale_pos_weight normally provides).

    Parameters
    ----------
    y_binary : array of {0, 1}
        Binary classification labels.
    labels : array of float
        Raw shadow prices (pre-binarization).

    Returns
    -------
    sample_weight : array of float, same length as y_binary
    """
    weights = np.ones(len(y_binary), dtype=np.float64)
    pos_mask = y_binary == 1
    n_pos = pos_mask.sum()
    n_neg = len(y_binary) - n_pos
    if n_pos == 0 or n_neg == 0:
        return weights

    # Within positives, weight by log1p(shadow_price)
    pos_raw = np.log1p(np.maximum(labels[pos_mask], 0))
    pos_mean = pos_raw.mean()
    if pos_mean > 0:
        pos_raw = pos_raw / pos_mean  # normalize so mean = 1.0

    # Scale positive weights so total_pos_weight = total_neg_weight = n_neg
    # This replaces scale_pos_weight (n_neg/n_pos) with value-aware version
    weights[pos_mask] = pos_raw * (n_neg / n_pos)
    return weights


def compute_regression_value_weights(y_reg: np.ndarray) -> np.ndarray:
    """Compute per-sample weights for regressor training.

    Weights binding samples by log1p(shadow_price), normalized so mean weight = 1.

    Parameters
    ----------
    y_reg : array of float
        log1p-transformed shadow prices (all positive / binding samples).
    """
    raw_labels = np.expm1(y_reg)  # back to dollar scale
    weights = np.log1p(np.maximum(raw_labels, 0))
    mean_w = weights.mean()
    if mean_w > 0:
        weights = weights / mean_w
    return weights


class RobustMinMaxScaler:
    """
    MinMaxScaler that handles near-zero constant features by setting them to 0
    instead of aggressively scaling noise to [0, 1].
    """

    def __init__(self, epsilon=1e-9):
        self.scaler = MinMaxScaler()
        self.epsilon = epsilon
        self.zero_cols = []
        self.feature_names_in_ = None

    def fit(self, X, y=None):  # noqa: N803
        if hasattr(X, "columns"):
            self.feature_names_in_ = X.columns.tolist()

        # Identify near-zero columns
        # Check max absolute value
        max_abs = np.max(np.abs(X), axis=0)

        # If input is DataFrame, max_abs is Series
        if hasattr(max_abs, "values"):
            self.zero_cols = max_abs.index[max_abs < self.epsilon].tolist()
        else:
            # If numpy array
            self.zero_cols = np.where(max_abs < self.epsilon)[0]

        self.scaler.fit(X, y)
        return self

    def transform(self, X):
        X_scaled = self.scaler.transform(X)

        # Zero out identified columns
        if isinstance(X, pd.DataFrame):
            # If we have column names, use them
            if self.zero_cols:
                # Check intersection of columns (in case specific subset passed)
                cols_to_zero = [c for c in self.zero_cols if c in X.columns]
                if cols_to_zero:
                    # Create DataFrame from scaled array (which is numpy)
                    # Or modify X_scaled array directly using column indices
                    # Since X_scaled is numpy, we need indices
                    col_indices = [X.columns.get_loc(c) for c in cols_to_zero]
                    X_scaled[:, col_indices] = 0.0

        else:
            # Numpy array
            if len(self.zero_cols) > 0:
                X_scaled[:, self.zero_cols] = 0.0

        return X_scaled

    def fit_transform(self, X, y=None):  # noqa: N803
        return self.fit(X, y).transform(X)


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


def select_features(
    X: pd.DataFrame,  # noqa: N803
    y: pd.Series | np.ndarray,
    features_config: list[tuple[str, int]],
    selection_config,
    verbose: bool = False,
) -> tuple[pd.DataFrame, list[int], list[str], bool]:
    """
    Select features based on correlation/AUC and monotonic constraints.

    Parameters:
    -----------
    X : pd.DataFrame
        Input features
    y : Array-like
        Target variable
    features_config : List[Tuple[str, int]]
        List of (feature_name, constraint) tuples
    selection_config : FeatureSelectionConfig
        Configuration for feature selection

    Returns:
    --------
    X_selected : pd.DataFrame
        Selected features
    selected_constraints : List[int]
        Corresponding monotonic constraints
    selected_names : List[str]
        Names of selected features
    is_fallback : bool
        True if selection failed and fell back to all features (or fallback strategy).
    """
    if not selection_config or len(X) < 10:  # Skip if too few samples
        return X, [f[1] for f in features_config], [f[0] for f in features_config], False

    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score

    def _get_subset(method_name: str) -> tuple[list[str], list[int]]:
        keep_cols_local = []
        keep_constraints_local = []

        for feat_name, constraint in features_config:
            if feat_name not in X.columns:
                if verbose:
                    print(f"    [feat-sel] {feat_name}: SKIP (not in columns)")
                continue

            if constraint == 0:
                keep_cols_local.append(feat_name)
                keep_constraints_local.append(constraint)
                if verbose:
                    print(f"    [feat-sel] {feat_name}: KEEP (constraint=0, unconstrained)")
                continue

            x_feat = X[feat_name].values

            # Check for constant features (std < 1e-9)
            feat_std = np.std(x_feat)
            if feat_std < 1e-9:
                if verbose:
                    print(f"    [feat-sel] {feat_name}: DROP (constant, std={feat_std:.2e})")
                continue

            is_both = method_name == "both"
            check_auc = is_both or method_name == "auc"
            check_spearman = is_both or method_name == "spearman"

            pass_auc = False
            pass_spearman = False
            auc_val = float("nan")
            corr_val = float("nan")

            # --- AUC Check ---
            unique_y = np.unique(y)
            can_calc_auc = len(unique_y) == 2

            if check_auc and can_calc_auc:
                try:
                    auc_val = roc_auc_score(y, x_feat)
                    if constraint == 1:
                        if auc_val > selection_config.auc_threshold:
                            pass_auc = True
                    elif constraint == -1:
                        if auc_val < selection_config.auc_threshold:
                            pass_auc = True
                except ValueError:
                    pass_auc = False
            elif check_auc and not can_calc_auc:
                pass_auc = True  # Treat as N/A -> Pass

            # --- Spearman Check ---
            if check_spearman:
                corr, _ = spearmanr(x_feat, y)
                corr = float(np.asarray(corr).flat[0])
                if np.isnan(corr):
                    corr = 0.0
                corr_val = corr

                if constraint == 1:
                    if corr > selection_config.min_correlation:
                        pass_spearman = True
                elif constraint == -1:
                    if corr < -selection_config.min_correlation:
                        pass_spearman = True

            # --- Final Decision ---
            valid_pass = False
            if is_both:
                if pass_auc and pass_spearman:
                    valid_pass = True
            elif method_name == "auc":
                valid_pass = pass_auc
            elif method_name == "spearman":
                valid_pass = pass_spearman

            if valid_pass:
                keep_cols_local.append(feat_name)
                keep_constraints_local.append(constraint)

            if verbose:
                status = "KEEP" if valid_pass else "DROP"
                print(
                    f"    [feat-sel] {feat_name}: {status}  "
                    f"(constraint={constraint:+d}, AUC={auc_val:.4f} {'✓' if pass_auc else '✗'}, "
                    f"Spearman={corr_val:.4f} {'✓' if pass_spearman else '✗'})"
                )

        return keep_cols_local, keep_constraints_local

    # Stage 1: Try configured method
    keep_cols, keep_constraints = _get_subset(selection_config.method)

    # Stage 2: Fallback to Spearman if 'both' failed to find ANY feature (except constraint 0)
    # Note: If we have constraint 0 features, keep_cols might not be empty, but we might have lost all predictive features.
    # The requirement is "what if no features satisfy". Strict interpretation: Result is empty.
    # But often we have some '0' constraint features? In config they are commented out, but if they exist...
    # Let's count "monotonic features kept".
    monotonic_kept = sum(1 for c in keep_constraints if c != 0)

    if monotonic_kept == 0 and selection_config.method == "both":
        # Fallback to Spearman
        keep_cols_retry, keep_constraints_retry = _get_subset("spearman")
        if sum(1 for c in keep_constraints_retry if c != 0) > 0:
            keep_cols = keep_cols_retry
            keep_constraints = keep_constraints_retry

    # Stage 3: Fallback to All Features if still empty OR only has non-monotonic (0) features
    # User requirement: "if the left features only have (feature_name, 0)... this is also invalid"
    monotonic_final = sum(1 for c in keep_constraints if c != 0)

    is_fallback = False
    if not keep_cols or monotonic_final == 0:
        is_fallback = True
        return X, [f[1] for f in features_config], [f[0] for f in features_config], is_fallback

    return X[keep_cols], keep_constraints, keep_cols, is_fallback


def train_ensemble(
    model_specs: list[ModelSpec],
    X: pd.DataFrame,  # noqa: N803
    y: pd.Series,
    is_classifier: bool = True,
    use_stacking: bool = False,
    groups: np.ndarray = None,
    monotone_constraints: str | None = None,
    sample_weight: np.ndarray | None = None,
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
    sample_weight : np.ndarray, optional
        Per-sample weights for fit(). If provided, scale_pos_weight is set to 1.0
        since class balance is handled by the weights.

    Returns:
    --------
    ensemble : List[Tuple[model, weight]]
        Trained models with their weights
    """
    ensemble = []

    # Calculate scale_pos_weight for this batch
    scale_pos_weight = 1.0
    if is_classifier and sample_weight is None:
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

        if sample_weight is not None:
            model.fit(X, y, sample_weight=sample_weight)
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
    # Fix: NaN entries (where p+r=0) cause argmax to pick a degenerate
    # threshold.  Replace NaN with 0 so argmax finds the true optimum.
    fbeta = np.nan_to_num(fbeta, nan=0.0)
    optimal_idx = np.argmax(fbeta)

    # optimal_threshold = (thresholds[optimal_idx] + thresholds[prev_idx]) / 2
    optimal_threshold = thresholds[optimal_idx]
    max_fbeta = fbeta[optimal_idx]

    # Apply scaling factor (heuristic to avoid overfitting)
    optimal_threshold = scaling_factor * optimal_threshold + (1 - scaling_factor) * 0.5

    return optimal_threshold, max_fbeta


class ShadowPriceModels:
    """Trains and stores ensemble of classification and regression models."""

    def __init__(self, config: PredictionConfig):
        self.config = config

        # Scalers by Horizon Group — separate for classifier and regressor (BUG-2 fix)
        # Default model scalers: Dict[horizon_group, StandardScaler]
        self.scalers_default_clf: dict[str, StandardScaler] = {}
        self.scalers_default_reg: dict[str, StandardScaler] = {}
        # Branch model scalers: Dict[horizon_group, Dict[(branch_name, flow_direction), StandardScaler]]
        self.scalers_branch_clf: dict[str, dict[tuple[str, int], StandardScaler]] = {}
        self.scalers_branch_reg: dict[str, dict[tuple[str, int], StandardScaler]] = {}

        # Ensembles by Horizon Group: Dict[branch_name, List[Tuple[model, weight]]]
        self.clf_ensembles: dict[str, dict[tuple[str, int], list[tuple[Any, float]]]] = {}
        self.reg_ensembles: dict[str, dict[tuple[str, int], list[tuple[Any, float]]]] = {}
        self.optimal_thresholds: dict[str, dict[tuple[str, int], float]] = {}

        self.clf_default_ensembles: dict[str, list[tuple[Any, float]]] = {}
        self.reg_default_ensembles: dict[str, list[tuple[Any, float]]] = {}
        self.optimal_threshold_defaults: dict[str, float] = {}

        # Selected Features Storage
        # Default models: Dict[horizon_group, List[str]]
        self.clf_default_features: dict[str, list[str]] = {}
        self.reg_default_features: dict[str, list[str]] = {}
        # Branch models: Dict[horizon_group, Dict[(branch_name, flow_direction), List[str]]]
        # We store the selected features so we can filter X during prediction
        self.clf_branch_features: dict[str, dict[tuple[str, int], list[str]]] = {}
        self.reg_branch_features: dict[str, dict[tuple[str, int], list[str]]] = {}

        # Fallback Status Storage
        # Default models: True if model is valid (did NOT fall back to empty/all)
        # If is_fallback=True for default, the model is considered INVALID (Predict 0)
        self.clf_default_valid: dict[str, bool] = {}
        self.reg_default_valid: dict[str, bool] = {}

        # Branch models: True if model triggered fallback (is using ALL features but blended)
        # If is_fallback=True, we use Weighted Blend (Branch + Default)
        self.clf_branch_fallback: dict[str, dict[tuple[str, int], bool]] = {}
        self.reg_branch_fallback: dict[str, dict[tuple[str, int], bool]] = {}

        # Initialize containers for each horizon group
        for group in self.config.horizon_groups:
            self.scalers_branch_clf[group.name] = {}
            self.scalers_branch_reg[group.name] = {}
            self.clf_ensembles[group.name] = {}
            self.reg_ensembles[group.name] = {}
            self.optimal_thresholds[group.name] = {}
            self.clf_default_ensembles[group.name] = []

            # Initialize fallback dicts
            self.clf_branch_fallback[group.name] = {}
            self.reg_branch_fallback[group.name] = {}
            self.reg_default_ensembles[group.name] = []
            self.optimal_threshold_defaults[group.name] = 0.5

            self.clf_branch_features[group.name] = {}
            self.reg_branch_features[group.name] = {}

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
        val_branch_data: pd.DataFrame | None = None,
    ) -> tuple[str, int, list[tuple[Any, float]] | None, float | None, StandardScaler | None, list[str] | None, bool]:
        """
        Train a single branch classifier.
        Returns: (branch_name, flow_direction, ensemble, optimal_threshold, scaler, selected_features, is_fallback)
        """
        # Skip branches with too few samples
        if len(branch_data) < self.config.training.min_samples_for_branch_model:
            return branch_name, flow_direction, None, 0.5, None, None, False

        # Skip branches with only one class (all 0s or all 1s)
        y_branch_raw = (branch_data["label"] > self.config.training.label_threshold).astype(int)
        if len(np.unique(y_branch_raw)) < 2:
            return branch_name, flow_direction, None, 0.5, None, None, False

        # Skip branches with rare positive labels (less than 5%)
        # This prevents training models on branches that are almost never binding
        if y_branch_raw.mean() < self.config.training.min_branch_positive_ratio:
            return branch_name, flow_direction, None, 0.5, None, None, False

        # Prepare branch data with BRANCH-SPECIFIC SCALING
        feature_cols = self.config.features.all_features
        scaler_branch = RobustMinMaxScaler()
        branch_data_scaled = branch_data.copy()
        branch_data_scaled[feature_cols] = scaler_branch.fit_transform(branch_data[feature_cols])
        # Extract classification features from SCALED data
        # Note: config.features.step1_features is now a list of tuples (name, constraint)

        # Re-extract just step1 features as candidates
        step1_candidates = self.config.features.step1_features
        step1_names = [f[0] for f in step1_candidates]
        X_branch_candidates = branch_data_scaled[step1_names].copy()

        y_branch_binary = (branch_data_scaled["label"] > self.config.training.label_threshold).astype(int)
        groups_branch = (
            branch_data_scaled["auction_month"].values if "auction_month" in branch_data_scaled.columns else None
        )

        # Feature Selection
        X_branch, constraints, selected_features, is_fallback = select_features(
            X_branch_candidates, y_branch_binary, step1_candidates, self.config.feature_selection
        )

        if is_fallback:
            return branch_name, flow_direction, None, 0.5, None, None, True

        # Construct Monotonic Constraints for XGBoost
        # Format: "(1,0,-1)" corresponding to feature columns in X_branch
        monotone_constraints = "(" + ",".join(map(str, constraints)) + ")"

        # Compute value weights if enabled
        branch_clf_weight = None
        if self.config.training.value_weighted:
            branch_clf_weight = compute_value_weights(
                y_branch_binary.values, branch_data["label"].values
            )

        # Train branch-specific ensemble
        clf_ensemble = train_ensemble(
            self.config.models.branch_classifiers,
            X_branch,
            y_branch_binary,
            is_classifier=True,
            groups=groups_branch,
            monotone_constraints=monotone_constraints,
            sample_weight=branch_clf_weight,
        )

        # Optimize threshold on validation data (BUG-3 fix: avoid in-sample overfitting)
        optimal_threshold_branch = 0.5  # safe default
        if val_branch_data is not None and len(val_branch_data) >= 10:
            y_val = (val_branch_data["label"] > self.config.training.label_threshold).astype(int)
            if len(np.unique(y_val)) >= 2:
                val_scaled = pd.DataFrame(
                    scaler_branch.transform(val_branch_data[feature_cols]),
                    columns=feature_cols,
                    index=val_branch_data.index,
                )
                X_val = val_scaled[selected_features]
                y_proba_val = predict_ensemble(
                    clf_ensemble, X_val, predict_proba=True, weight_overrides=weight_overrides
                )
                optimal_threshold_branch, _ = find_optimal_threshold(
                    y_val,
                    y_proba_val,
                    self.config.threshold.threshold_beta,
                    self.config.threshold.threshold_scaling_factor,
                )

        return (
            branch_name,
            flow_direction,
            clf_ensemble,
            optimal_threshold_branch,
            scaler_branch,
            selected_features,
            is_fallback,
        )

    def _train_single_branch_regressor(
        self, branch_name: str, flow_direction: int, branch_data: pd.DataFrame
    ) -> tuple[str, int, list[tuple[Any, float]] | None, Any | None, list[str] | None, bool]:
        """
        Train a single branch regressor.
        Returns: (branch_name, flow_direction, ensemble, scaler, selected_features, is_fallback)
        """
        # Skip branches with too few samples
        if len(branch_data) < self.config.training.min_samples_for_branch_model:
            return branch_name, flow_direction, None, None, None, False

        # Prepare branch data with BRANCH-SPECIFIC SCALING
        feature_cols = self.config.features.all_features
        scaler_branch = RobustMinMaxScaler()
        branch_data_scaled = branch_data.copy()
        branch_data_scaled[feature_cols] = scaler_branch.fit_transform(branch_data[feature_cols])

        # Candidates
        step2_candidates = self.config.features.step2_features
        step2_names = [f[0] for f in step2_candidates]

        if self.config.training.unified_regressor:
            # Unified regressor: train on ALL samples (binding + non-binding)
            X_branch_candidates = branch_data_scaled[step2_names].copy()
            y_branch_reg = np.log1p(branch_data_scaled["label"].clip(lower=0).copy())
            groups_branch_reg = (
                branch_data_scaled["auction_month"].values
                if "auction_month" in branch_data_scaled.columns
                else None
            )
        else:
            # Filter for binding constraints (label > 0)
            binding_mask = branch_data_scaled["label"] > 0
            if binding_mask.sum() < self.config.training.min_binding_samples_for_regression:
                return branch_name, flow_direction, None, None, None, False

            X_branch_candidates = branch_data_scaled.loc[binding_mask, step2_names].copy()
            y_branch_reg = np.log1p(branch_data_scaled.loc[binding_mask, "label"].copy())
            groups_branch_reg = (
                branch_data_scaled.loc[binding_mask, "auction_month"].values
                if "auction_month" in branch_data_scaled.columns
                else None
            )

        # Feature Selection
        X_branch_reg, constraints, selected_features, is_fallback = select_features(
            X_branch_candidates, y_branch_reg, step2_candidates, self.config.feature_selection
        )

        if is_fallback:
            return branch_name, flow_direction, None, None, None, True

        # Construct Monotonic Constraints for XGBoost
        monotone_constraints = "(" + ",".join(map(str, constraints)) + ")"

        # Compute value weights if enabled (uses separate reg flag)
        reg_sample_weight = None
        if self.config.training.value_weighted_reg:
            reg_sample_weight = compute_regression_value_weights(y_branch_reg.values)

        # Train branch-specific ensemble
        reg_ensemble = train_ensemble(
            self.config.models.branch_regressors,
            X_branch_reg,
            y_branch_reg,
            is_classifier=False,
            groups=groups_branch_reg,
            monotone_constraints=monotone_constraints,
            sample_weight=reg_sample_weight,
        )

        return branch_name, flow_direction, reg_ensemble, scaler_branch, selected_features, is_fallback

    def train_classifiers(
        self,
        train_data: pd.DataFrame,
        test_branches: set[tuple[str, int]],
        test_periods: list[tuple[pd.Timestamp, pd.Timestamp]] | None = None,
        verbose: bool = True,
        val_data: pd.DataFrame | None = None,
    ) -> None:
        """
        Train branch-specific classifiers and default fallback classifier for each horizon group.

        Parameters:
        -----------
        train_data : pd.DataFrame
            Training data with features and labels (fit portion only)
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
            scaler_default = RobustMinMaxScaler()

            # Create scaled copy for default model
            group_data_scaled_default = group_data.copy()
            group_data_scaled_default[feature_cols] = scaler_default.fit_transform(group_data[feature_cols])
            self.scalers_default_clf[group_name] = scaler_default

            # Train default classifier on SCALED data
            if verbose:
                print(f"  Training default fallback classifier ({group_name})...")

            # Note: config.features.step1_features is now a list of tuples (name, constraint)
            X_train_candidates = group_data_scaled_default[[f[0] for f in self.config.features.step1_features]].copy()
            y_train_all = group_data_scaled_default["label"].copy()
            y_train_binary_all = (y_train_all > self.config.training.label_threshold).astype(int)

            # Extract groups (auction_month) for CV
            groups_all = group_data["auction_month"].values if "auction_month" in group_data.columns else None

            # Feature Selection for Default Model
            X_train_all, constraints, selected_features, is_fallback = select_features(
                X_train_candidates,
                y_train_binary_all,
                self.config.features.step1_features,
                self.config.feature_selection,
                verbose=verbose,
            )
            self.clf_default_features[group_name] = selected_features

            if is_fallback:
                if verbose:
                    print("    ⚠️ Default model fallback triggered (no monotonic features). Marking invalid.")
                self.clf_default_valid[group_name] = False
                self.clf_default_ensembles[group_name] = []
                self.optimal_threshold_defaults[group_name] = 0.5
            else:
                self.clf_default_valid[group_name] = True

                # Construct Monotonic Constraints for XGBoost
                monotone_constraints = "(" + ",".join(map(str, constraints)) + ")"

                # Compute value weights if enabled
                clf_sample_weight = None
                if self.config.training.value_weighted:
                    clf_sample_weight = compute_value_weights(
                        y_train_binary_all.values, y_train_all.values
                    )

                # Train default classifier ensemble
                default_ensemble = train_ensemble(
                    self.config.models.default_classifiers,
                    X_train_all,
                    y_train_binary_all,
                    is_classifier=True,
                    groups=groups_all,
                    monotone_constraints=monotone_constraints,
                    sample_weight=clf_sample_weight,
                )
                self.clf_default_ensembles[group_name] = default_ensemble

                if verbose:
                    n_models = len(default_ensemble)
                    print(f"    ✓ Default ensemble trained ({n_models} models)")
                    print(f"    Total samples: {len(X_train_all):,}")
                    print(f"    Selected features: {len(selected_features)}/{len(self.config.features.step1_features)}")

                # Optimize threshold on validation data (BUG-3 fix: avoid in-sample overfitting)
                if verbose:
                    print(f"  Optimizing threshold for default ensemble ({group_name})...")

                optimal_threshold = 0.5  # safe default
                max_f1 = 0.0

                # Use validation data if available
                if val_data is not None:
                    val_group = val_data[
                        (val_data["forecast_horizon"] >= min_h) & (val_data["forecast_horizon"] <= max_h)
                    ]
                    if len(val_group) >= 10:
                        y_val_binary = (val_group["label"] > self.config.training.label_threshold).astype(int)
                        if len(np.unique(y_val_binary)) >= 2:
                            val_scaled = pd.DataFrame(
                                scaler_default.transform(val_group[feature_cols]),
                                columns=feature_cols,
                                index=val_group.index,
                            )
                            X_val = val_scaled[selected_features]
                            y_proba_val = predict_ensemble(
                                default_ensemble, X_val, predict_proba=True, weight_overrides=weight_overrides
                            )
                            optimal_threshold, max_f1 = find_optimal_threshold(
                                y_val_binary,
                                y_proba_val,
                                beta=self.config.threshold.threshold_beta,
                                scaling_factor=self.config.threshold.threshold_scaling_factor,
                            )
                            if verbose:
                                print(f"    Using validation data ({len(val_group)} samples) for threshold")

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

            # Prepare validation data by branch (for threshold optimization)
            val_data_by_branch: dict[tuple[str, int], pd.DataFrame] = {}
            if val_data is not None:
                val_group = val_data[
                    (val_data["forecast_horizon"] >= min_h) & (val_data["forecast_horizon"] <= max_h)
                ]
                if len(val_group) > 0:
                    val_reset = val_group.reset_index() if "flow_direction" in val_group.index.names else val_group
                    for (bn, fd), vbd in val_reset.groupby(["branch_name", "flow_direction"]):
                        val_data_by_branch[(bn, fd)] = vbd

            # Build param dicts (BUG-1 fix: use clf_weights, not reg_weights)
            param_dict_list = [
                {
                    "branch_name": branch_name,
                    "flow_direction": flow_direction,
                    "branch_data": branch_data,
                    "weight_overrides": self.config.models.clf_weights.get(group_name),
                    "val_branch_data": val_data_by_branch.get((branch_name, flow_direction)),
                }
                for (branch_name, flow_direction), branch_data in train_data_by_branch.items()
            ]

            # Sequential Training (as requested to avoid nested parallelism)
            results = []
            for param_dict in param_dict_list:
                # if param_dict['branch_name'] == 'FOS-IRNT       1':
                #     print('aaaaaa')
                result = self._train_single_branch_classifier(**param_dict)
                results.append(result)
            trained_count = 0
            skipped_count = 0

            for branch_name, flow_direction, ensemble, threshold, scaler, selected_feats, is_fallback_branch in results:
                if ensemble is not None and threshold is not None:
                    self.clf_ensembles[group_name][(branch_name, flow_direction)] = ensemble
                    self.optimal_thresholds[group_name][(branch_name, flow_direction)] = threshold
                    self.scalers_branch_clf[group_name][(branch_name, flow_direction)] = scaler
                    self.clf_branch_features[group_name][(branch_name, flow_direction)] = selected_feats
                    self.clf_branch_fallback[group_name][(branch_name, flow_direction)] = is_fallback_branch
                    trained_count += 1
                else:
                    skipped_count += 1

            if verbose:
                print(f"    ✓ Trained {trained_count} branch models (skipped {skipped_count})")

        if verbose:
            print("\n✓ Classification Training Complete")
        self.log_fallback_statistics()

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
            scaler_default = RobustMinMaxScaler()

            # Create scaled copy for default model
            group_data_scaled_default = group_data.copy()
            group_data_scaled_default[feature_cols] = scaler_default.fit_transform(group_data[feature_cols])
            self.scalers_default_reg[group_name] = scaler_default

            # Train default regressor on SCALED data (binding only)
            if verbose:
                print(f"  Training default fallback regressor ({group_name})...")

            step2_candidates = self.config.features.step2_features
            # Deduplicate features (in case step1 and step2 overlap)
            seen = set()
            step2_deduped = []
            for f in step2_candidates:
                if f[0] not in seen:
                    seen.add(f[0])
                    step2_deduped.append(f)
            step2_candidates = step2_deduped
            step2_names = [f[0] for f in step2_candidates]

            if self.config.training.unified_regressor:
                # Unified regressor: train on ALL samples (binding + non-binding)
                # Target: log1p(max(0, shadow_price)) — non-binding samples have target=0
                X_train_candidates = group_data_scaled_default[step2_names].copy()
                y_train_reg_all = np.log1p(group_data_scaled_default["label"].clip(lower=0).copy())
                groups_reg_all = (
                    group_data["auction_month"].values
                    if "auction_month" in group_data.columns
                    else None
                )
            else:
                binding_mask_all = group_data_scaled_default["label"] > 0
                X_train_candidates = group_data_scaled_default.loc[binding_mask_all, step2_names].copy()
                y_train_reg_all = np.log1p(group_data_scaled_default.loc[binding_mask_all, "label"].copy())
                groups_reg_all = (
                    group_data.loc[binding_mask_all, "auction_month"].values
                    if "auction_month" in group_data.columns
                    else None
                )

            # Feature Selection for Default Regressor
            X_train_reg_all, constraints, selected_features_reg, is_fallback_reg = select_features(
                X_train_candidates, y_train_reg_all, step2_candidates, self.config.feature_selection
            )
            self.reg_default_features[group_name] = selected_features_reg

            if is_fallback_reg:
                if verbose:
                    print("    ⚠️ Default regressor fallback triggered. Marking invalid.")
                self.reg_default_valid[group_name] = False
                self.reg_default_ensembles[group_name] = []
            else:
                self.reg_default_valid[group_name] = True

                monotone_constraints = "(" + ",".join(map(str, constraints)) + ")"

                if len(X_train_reg_all) > self.config.training.min_binding_samples_for_regression:
                    # Compute value weights if enabled (uses separate reg flag)
                    default_reg_weight = None
                    if self.config.training.value_weighted_reg:
                        default_reg_weight = compute_regression_value_weights(y_train_reg_all.values)

                    default_ensemble = train_ensemble(
                        self.config.models.default_regressors,
                        X_train_reg_all,
                        y_train_reg_all,
                        is_classifier=False,
                        groups=groups_reg_all,
                        monotone_constraints=monotone_constraints,
                        sample_weight=default_reg_weight,
                    )
                    self.reg_default_ensembles[group_name] = default_ensemble

                    if verbose:
                        n_models = len(default_ensemble)
                        mode = "unified (all)" if self.config.training.unified_regressor else "binding-only"
                        print(f"    ✓ Default ensemble trained ({n_models} models, {mode})")
                        print(f"    Total training samples: {len(X_train_reg_all):,}")
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

            for branch_name, flow_direction, ensemble, scaler, selected_feats, is_fallback_branch in results:
                if ensemble is not None:
                    self.reg_ensembles[group_name][(branch_name, flow_direction)] = ensemble
                    self.scalers_branch_reg[group_name][(branch_name, flow_direction)] = scaler
                    self.reg_branch_features[group_name][(branch_name, flow_direction)] = selected_feats
                    self.reg_branch_fallback[group_name][(branch_name, flow_direction)] = is_fallback_branch
                    trained_count += 1
                else:
                    skipped_count += 1

            if verbose:
                print(f"    ✓ Trained {trained_count} branch models (skipped {skipped_count})")

        if verbose:
            print("\n✓ Regression Training Complete")
        self.log_fallback_statistics()

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
            thr = self.config.threshold.threshold_override if self.config.threshold.threshold_override is not None else 0.5
            return [], thr, None

        ensembles = self.clf_ensembles.get(group, {})
        thresholds = self.optimal_thresholds.get(group, {})
        default_ensemble = self.clf_default_ensembles.get(group, [])
        default_threshold = self.optimal_threshold_defaults.get(group)

        # Get scaler (use classifier-specific scalers)
        scaler = self.scalers_branch_clf[group].get((branch_name, flow_direction))
        if scaler is None:
            scaler = self.scalers_default_clf.get(group)

        # If threshold_override is set, bypass all optimized thresholds
        thr_override = self.config.threshold.threshold_override
        if thr_override is not None:
            if (branch_name, flow_direction) in ensembles:
                return ensembles[(branch_name, flow_direction)], thr_override, scaler
            return default_ensemble, thr_override, scaler

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

        # Get scaler (use regressor-specific scalers)
        scaler = self.scalers_branch_reg[group].get((branch_name, flow_direction))
        if scaler is None:
            scaler = self.scalers_default_reg.get(group)

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

            # Get scaler (use classifier scalers — transform_features is used for prediction input)
            scaler = self.scalers_branch_clf.get(h_group, {}).get((b, f))
            if scaler is None:
                scaler = self.scalers_default_clf.get(h_group)

            if scaler:
                # Transform features for this slice
                # indices are integer positions in group_df, which correspond to data rows
                # Use source_col_indices to get original values
                subset_values = data.iloc[indices, source_col_indices].values
                transformed_values = scaler.transform(subset_values)

                # Assign back using iloc to target columns
                data_scaled.iloc[indices, col_indices] = transformed_values

        return data_scaled

    def log_fallback_statistics(self):
        """Log statistics about model fallback frequency."""
        print("\n" + "=" * 80)
        print("FALLBACK MONITORING STATISTICS")
        print("=" * 80)

        for group in self.config.horizon_groups:
            print(f"\n[{group.name.upper()}]")

            # --- Classification ---
            clf_fallbacks = self.clf_branch_fallback[group.name]
            total_clf_branches = len(clf_fallbacks)
            if total_clf_branches > 0:
                fallback_count = sum(clf_fallbacks.values())
                fallback_rate = (fallback_count / total_clf_branches) * 100
                print(f"  Classification Fallback Rate: {fallback_count}/{total_clf_branches} ({fallback_rate:.2f}%)")
            else:
                print("  Classification Fallback Rate: N/A (0 branches)")

            clf_def_valid = self.clf_default_valid.get(group.name, True)
            if not clf_def_valid:
                print("  ⚠️  Default Classification Model is INVALID (Fallback Triggered)")

            # --- Regression ---
            reg_fallbacks = self.reg_branch_fallback[group.name]
            total_reg_branches = len(reg_fallbacks)
            if total_reg_branches > 0:
                fallback_count = sum(reg_fallbacks.values())
                fallback_rate = (fallback_count / total_reg_branches) * 100
                print(f"  Regression Fallback Rate:     {fallback_count}/{total_reg_branches} ({fallback_rate:.2f}%)")
            else:
                print("  Regression Fallback Rate:     N/A (0 branches)")

            reg_def_valid = self.reg_default_valid.get(group.name, True)
            if not reg_def_valid:
                print("  ⚠️  Default Regression Model is INVALID (Fallback Triggered)")

    # ── Artifact extraction (for registry) ────────────────────────────────

    def extract_threshold_decisions(self) -> dict:
        """Extract all threshold decisions after classifier training.

        Returns
        -------
        dict
            Keyed by horizon_group, each containing 'default' and branch entries.
            Each entry has threshold, f_beta_used, etc.
        """
        result: dict[str, dict] = {}
        for group in self.config.horizon_groups:
            gn = group.name
            group_data: dict[str, Any] = {}

            # Default threshold
            default_thr = self.optimal_threshold_defaults.get(gn, 0.5)
            group_data["default"] = {
                "threshold": round(float(default_thr), 6),
                "f_beta_used": self.config.threshold.threshold_beta,
                "valid": self.clf_default_valid.get(gn, False),
            }

            # Branch thresholds
            branch_thresholds = self.optimal_thresholds.get(gn, {})
            for (branch_name, flow_dir), thr in branch_thresholds.items():
                key = f"{branch_name}_{flow_dir}"
                group_data[key] = {
                    "threshold": round(float(thr), 6),
                    "f_beta_used": self.config.threshold.threshold_beta,
                    "branch_name": branch_name,
                    "flow_direction": flow_dir,
                }

            result[gn] = group_data
        return result

    def extract_feature_importances(self) -> dict:
        """Extract XGBoost feature importances from all trained models.

        Returns
        -------
        dict
            {
              "stage1": {horizon_group: {"default": {"gain": {...}}, ...}},
              "stage2": {horizon_group: {"default": {"gain": {...}}, ...}},
            }
        """
        result: dict[str, dict] = {"stage1": {}, "stage2": {}}

        for group in self.config.horizon_groups:
            gn = group.name

            # Stage 1 — classifier
            s1_group: dict[str, dict] = {}
            default_clf = self.clf_default_ensembles.get(gn, [])
            if default_clf:
                s1_group["default"] = _extract_ensemble_importance(
                    default_clf, self.clf_default_features.get(gn, [])
                )
            for (bn, fd), ens in self.clf_ensembles.get(gn, {}).items():
                feats = self.clf_branch_features.get(gn, {}).get((bn, fd), [])
                s1_group[f"{bn}_{fd}"] = _extract_ensemble_importance(ens, feats)
            result["stage1"][gn] = s1_group

            # Stage 2 — regressor
            s2_group: dict[str, dict] = {}
            default_reg = self.reg_default_ensembles.get(gn, [])
            if default_reg:
                s2_group["default"] = _extract_ensemble_importance(
                    default_reg, self.reg_default_features.get(gn, [])
                )
            for (bn, fd), ens in self.reg_ensembles.get(gn, {}).items():
                feats = self.reg_branch_features.get(gn, {}).get((bn, fd), [])
                s2_group[f"{bn}_{fd}"] = _extract_ensemble_importance(ens, feats)
            result["stage2"][gn] = s2_group

        return result


def _extract_ensemble_importance(
    ensemble: list[tuple[Any, float]], feature_names: list[str]
) -> dict:
    """Extract weighted-average feature importance from an ensemble.

    Supports XGBoost (get_booster().get_score()) and falls back to
    ``feature_importances_`` for sklearn models.
    """
    importance: dict[str, dict[str, float]] = {"gain": {}}
    if not ensemble or not feature_names:
        return importance

    total_weight = 0.0
    for model, weight in ensemble:
        try:
            booster = getattr(model, "get_booster", None)
            if booster is not None:
                scores = booster().get_score(importance_type="gain")
                for fname in feature_names:
                    # XGBoost uses f0, f1, ... or feature names
                    val = scores.get(fname, 0.0)
                    importance["gain"][fname] = importance["gain"].get(fname, 0.0) + val * weight
            elif hasattr(model, "feature_importances_"):
                fi = model.feature_importances_
                for i, fname in enumerate(feature_names):
                    if i < len(fi):
                        importance["gain"][fname] = importance["gain"].get(fname, 0.0) + fi[i] * weight
            total_weight += weight
        except Exception:
            continue

    # Normalize by total weight
    if total_weight > 0:
        for fname in importance["gain"]:
            importance["gain"][fname] = round(importance["gain"][fname] / total_weight, 6)

    return importance
