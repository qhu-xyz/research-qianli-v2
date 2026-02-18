"""
Configuration classes for shadow price prediction pipeline.
"""

from dataclasses import dataclass, field

from xgboost import XGBClassifier, XGBRegressor

from .iso_configs import (
    MISO_HORIZON_GROUPS,
    MISO_ISO_CONFIG,
    HorizonGroupConfig,
    IsoConfig,
)


@dataclass
class FeatureConfig:
    """Feature column configuration."""

    # Classification features (raw + engineered)
    # Format: (feature_name, monotonicity_constraint)
    # 1 = increasing, -1 = decreasing, 0 = no constraint
    step1_features: list[tuple[str, int]] = field(
        default_factory=lambda: [
            ("prob_exceed_110", 1),
            ("prob_exceed_105", 1),
            ("prob_exceed_100", 1),
            ("prob_exceed_95", 1),
            ("prob_exceed_90", 1),
            # ("prob_exceed_85", 1),
            # ("prob_exceed_80", 1),
            # ("prob_below_100", -1),
            ("prob_below_95", -1),
            ("prob_below_90", -1),
            # ("prob_below_85", -1),
            # ("prob_below_80", -1),
            # ("density_mean", 1),
            # ("density_variance", 0),
            ("density_skewness", 1),
            # ("density_kurtosis", 0),
            # ("hist_da", 1),
        ]
    )

    # Regression features (raw + derived diffs + engineered)
    step2_features: list[tuple[str, int]] = field(
        default_factory=lambda: [
            ("prob_exceed_110", 1),
            ("prob_exceed_105", 1),
            ("prob_exceed_100", 1),
            ("prob_exceed_95", 1),
            ("prob_exceed_90", 1),
            # ("prob_exceed_85", 1),
            # ("prob_exceed_80", 1),
            # ("prob_below_100", -1),
            ("prob_below_95", -1),
            ("prob_below_90", -1),
            # ("prob_below_85", -1),
            # ("prob_below_80", -1),
            # ("density_mean", 1),
            # ("density_variance", 0),
            ("density_skewness", 1),
            # ("density_kurtosis", 0),
            # ("hist_da", 1),
        ]
    )

    # All features needed (union of step1 and step2) - computed automatically
    all_features: list[str] = field(default_factory=list, init=False)

    def __post_init__(self):
        """Compute all_features as union of step1 and step2 features."""
        # Extract just the feature names for all_features
        s1_names = [f[0] for f in self.step1_features]
        s2_names = [f[0] for f in self.step2_features]
        self.all_features = list(set(s1_names + s2_names))


# Period Type to Market Month Offset Mapping
# For q-series, it maps to the start month offset relative to auction month
# Logic handled in DataLoader for specific months
PERIOD_MAPPING = {
    "f0": 0,
    "f1": 1,
    "f2": 2,
    "f3": 3,
    # q-series are handled specially as they map to specific calendar months
    # q2: Sep-Nov, q3: Dec-Feb, q4: Mar-May
}


@dataclass
class ModelConfig:
    """Universal model configuration that works with any sklearn/xgboost model.

    All model parameters are passed via the params dictionary.

    Examples:
    ---------
    # XGBoost Classifier
    ModelConfig(params={'n_estimators': 200, 'max_depth': 4, 'learning_rate': 0.1, 'n_jobs': 1})

    # Logistic Regression
    ModelConfig(params={'C': 1.0, 'max_iter': 1000, 'class_weight': 'balanced', 'n_jobs': 1})

    # ElasticNet
    ModelConfig(params={'alpha': 1.0, 'l1_ratio': 0.5, 'max_iter': 1000})
    """

    params: dict = field(default_factory=dict)

    def to_dict(self):
        """Return parameters for model instantiation."""
        return self.params


@dataclass
class ModelSpec:
    """Specification for a single model with its weight in an ensemble.

    Parameters:
    -----------
    model_class : type
        The actual model class (e.g., XGBClassifier, LogisticRegression, ElasticNet)
    config : ModelConfig
        Model configuration containing parameters
    weight : float
        Weight for ensemble aggregation (default: 1.0)

    Examples:
    ---------
    # XGBoost Classifier
    ModelSpec(
        model_class=XGBClassifier,
        config=ModelConfig(params={'n_estimators': 200, 'max_depth': 4}),
        weight=0.5
    )

    # Logistic Regression
    ModelSpec(
        model_class=LogisticRegression,
        config=ModelConfig(params={'C': 1.0, 'max_iter': 1000}),
        weight=0.5
    )
    """

    model_class: type  # The actual model class (e.g., XGBClassifier, LogisticRegression)
    config: ModelConfig  # Model configuration
    weight: float = 1.0  # Weight for ensemble aggregation

    def __post_init__(self):
        """Validate model specification."""
        if self.weight <= 0:
            raise ValueError(f"Model weight must be positive, got {self.weight}")


@dataclass
class EnsembleConfig:
    """Configuration for ensemble of models."""

    # Default classifier ensemble (XGBoost + LogisticRegression)
    default_classifiers: list[ModelSpec] = field(
        default_factory=lambda: [
            ModelSpec(
                model_class=XGBClassifier,
                config=ModelConfig(
                    params={
                        "n_estimators": 200,
                        "min_child_weight": 10,
                        "n_jobs": 1,
                        "max_depth": 4,
                        "learning_rate": 0.1,
                        "verbosity": 0,
                        "eval_metric": "logloss",
                    }
                ),
                weight=1,
            ),
        ]
    )

    # Branch-specific classifier ensemble (XGBoost + LogisticRegression)
    branch_classifiers: list[ModelSpec] = field(
        default_factory=lambda: [
            ModelSpec(
                model_class=XGBClassifier,
                config=ModelConfig(
                    params={
                        "n_estimators": 100,
                        "max_depth": 4,
                        "learning_rate": 0.1,
                        "min_child_weight": 5,
                        "n_jobs": 1,
                        "verbosity": 0,
                        "eval_metric": "logloss",
                    }
                ),
                weight=1,
            ),
        ]
    )

    # Default regressor ensemble (XGBoost + ElasticNet)
    default_regressors: list[ModelSpec] = field(
        default_factory=lambda: [
            ModelSpec(
                model_class=XGBRegressor,
                config=ModelConfig(
                    params={
                        "n_estimators": 200,
                        "min_child_weight": 2,
                        "n_jobs": 1,
                        "max_depth": 4,
                        "learning_rate": 0.1,
                        "verbosity": 0,
                        "objective": "reg:squarederror",
                    }
                ),
                weight=1,
            ),
            # ModelSpec(
            #     model_class=ElasticNet,
            #     config=ModelConfig(params={
            #         'alpha': 1.0,
            #         'l1_ratio': 0.5,
            #         'max_iter': 1000,
            #         'fit_intercept': True,
            #         'random_state': 42
            #     }),
            #     weight=0.5
            # )
        ]
    )

    # Branch-specific regressor ensemble (XGBoost + ElasticNet)
    branch_regressors: list[ModelSpec] = field(
        default_factory=lambda: [
            ModelSpec(
                model_class=XGBRegressor,
                config=ModelConfig(
                    params={
                        "n_estimators": 100,
                        "max_depth": 4,
                        "learning_rate": 0.1,
                        "min_child_weight": 1,
                        "n_jobs": 1,
                        "verbosity": 0,
                        "objective": "reg:squarederror",
                    }
                ),
                weight=1,
            ),
            # ModelSpec(
            #     model_class=ElasticNet,
            #     config=ModelConfig(params={
            #         'alpha': 1.0,
            #         'l1_ratio': 0.5,
            #         'max_iter': 1000,
            #         'fit_intercept': True,
            #         'random_state': 42
            #     }),
            #     weight=0.5
            # )
        ]
    )

    # Horizon-stratified ensemble weights
    # Format: {group_name: [xgboost_weight, linear_model_weight]}
    # Default: Equal weights for all groups
    clf_weights: dict[str, list[float]] = field(default_factory=dict)
    reg_weights: dict[str, list[float]] = field(default_factory=dict)

    # Horizon thresholds for weight selection (Deprecated but kept for compatibility if needed)
    short_term_max_horizon: int = 1

    def __post_init__(self):
        """Normalize weights to sum to 1.0 for each ensemble."""
        # Normalize default classifier weights
        total_weight = sum(spec.weight for spec in self.default_classifiers)
        for spec in self.default_classifiers:
            spec.weight /= total_weight

        # Normalize branch classifier weights
        total_weight = sum(spec.weight for spec in self.branch_classifiers)
        for spec in self.branch_classifiers:
            spec.weight /= total_weight

        # Normalize default regressor weights
        total_weight = sum(spec.weight for spec in self.default_regressors)
        for spec in self.default_regressors:
            spec.weight /= total_weight

        # Normalize branch regressor weights
        total_weight = sum(spec.weight for spec in self.branch_regressors)
        for spec in self.branch_regressors:
            spec.weight /= total_weight

    def get_ensemble_weights_for_horizon(
        self,
        horizon: int,
        horizon_groups: list["HorizonGroupConfig"],
        model_type: str = "classifier",
        is_branch: bool = False,
    ) -> list[float]:
        """
        Get ensemble weights based on forecast horizon.

        Parameters:
        -----------
        horizon : int
            Forecast horizon in months (0 for f0, 1 for f1, etc.)
        horizon_groups : list[HorizonGroupConfig]
            List of configured horizon groups to resolve the correct group
        model_type : str
            'classifier' or 'regressor'
        is_branch : bool
            Whether this is for branch-specific or default models

        Returns:
        --------
        weights : List[float]
            Normalized weights for the ensemble models
        """
        # Determine default weights based on configured models
        if model_type == "classifier":
            model_list = self.branch_classifiers if is_branch else self.default_classifiers
        else:
            model_list = self.branch_regressors if is_branch else self.default_regressors

        n_models = len(model_list)
        default_weights = [1.0 / n_models] * n_models if n_models > 0 else []

        # Find matching group
        group_name = None
        for g in horizon_groups:
            if g.min_horizon <= horizon <= g.max_horizon:
                group_name = g.name
                break

        if group_name is None:
            # Fallback if no group covers this horizon
            return default_weights

        if model_type == "classifier":
            weights = self.clf_weights.get(group_name, default_weights)
        else:
            weights = self.reg_weights.get(group_name, default_weights)

        # Normalize to sum to 1.0
        total = sum(weights)
        if total == 0:
            return default_weights
        return [w / total for w in weights]


@dataclass
class TrainingConfig:
    """Model training configuration."""

    min_samples_for_branch_model: int = 1
    min_binding_samples_for_regression: int = 1
    train_months_lookback: int = 12  # Number of months to look back for training
    label_threshold: float = 0  # Threshold for binary classification label
    min_branch_positive_ratio: float = 0.02  # Minimum ratio of positive samples to train branch model
    # Rule to modify label to 0 if a feature is below a threshold
    # Format: (feature_name, threshold)
    label_modification_rule: tuple[str, float] | None = ("prob_exceed_90", 1e-5)
    # Rule to modify test prediction to 0 (unbind) if a feature is below a threshold
    # Format: (feature_name, threshold)
    test_unbind_rule: tuple[str, float] | None = ("prob_exceed_90", 1e-5)


@dataclass
class ThresholdConfig:
    """Dynamic threshold optimization configuration."""

    threshold_beta: float = 0.5  # F-beta score beta parameter (0.5 favors precision)
    threshold_scaling_factor: float = 1.0  # Scale factor for optimal threshold (heuristic to avoid overfitting)

    # Dynamic Thresholds (4.B)
    # Adjust threshold based on predicted shadow price magnitude
    # threshold = base_threshold - (factor * log1p(predicted_price))


@dataclass
class AnomalyDetectionConfig:
    """Flow anomaly detection configuration."""

    enabled: bool = True
    k_multiplier: float = 3.0  # IQR multiplier for anomaly threshold
    iqr_range_fraction: float = 0.3  # Dynamic floor fraction of (Max-Min) to handle zero-variance
    min_samples_for_stats: int = 10
    # --- Detection Phase ---
    # Features used to trigger the "Is Anomaly" boolean
    detection_features: list[str] = field(default_factory=lambda: ["prob_exceed_100", "prob_exceed_95"])
    detection_weights: dict[str, float] = field(default_factory=lambda: {"prob_exceed_100": 2.0, "prob_exceed_95": 1.0})
    detection_threshold: float = 0.5  # Weighted Score threshold to trigger anomaly

    # --- Probability Phase ---
    # Features used to calculate the specific binding probability (if anomaly is detected)
    probability_features: list[str] = field(
        default_factory=lambda: ["prob_exceed_100", "prob_exceed_95", "density_skewness"]
    )
    probability_weights: dict[str, float] = field(
        default_factory=lambda: {"prob_exceed_100": 2.0, "prob_exceed_95": 1.5, "density_skewness": 0.7}
    )

    # Sigmoid parameters for probability mapping
    # P = 1 / (1 + exp(-alpha * (Score_prob - beta)))
    sigmoid_alpha: float = 1.0
    sigmoid_beta: float = 0.5


@dataclass
class FeatureSelectionConfig:
    """Feature selection based on correlation/AUC with monotonic constraints."""

    method: str = "both"  # 'spearman', 'auc', or 'both'
    # 'both': Feature must satisfy BOTH Spearman correlation AND AUC directionality checks.

    # Weight for branch model fallback prediction (w * Branch + (1-w) * Default)
    # Only used if Branch model falls back to "All Features" due to selection failure.
    fallback_weight: float = 0.5

    # Features with correlation (or AUC-0.5) weaker than this but consistent sign will be kept?
    # Or features with WRONG sign will be dropped if correlation is stronger than this?
    # Logic:
    # If constraint=1 (Expect Positive):
    #   if corr < -threshold: DROP or FLIP? User said "choose only positive ones"
    #   so if corr < 0 (+buffer?), DROP.
    # We will implement a simple consistency check.

    # Threshold for "strong enough to contradict"
    # If we expect positive, but get negative correlation stronger than this magnitude, we drop it.
    # If we get negative correlation but weak (noise), we might keep it (or drop it too?).
    # User said: "choose only positive ones for monotonic 1"
    # So we should probably drop ANYTHING with wrong sign?
    # Let's use a small negative threshold to allow for slight noise around 0.
    min_correlation: float = 0.00  # If constraint=1, must have corr > min_correlation.
    # If constraint=-1, must have corr < -min_correlation.

    # For AUC:
    # Constraint=1 => AUC > 0.5 (or min_auc)
    # Constraint=-1 => AUC < 0.5 (or 1 - min_auc if symmetric? Usually AUC is 0.5 for random)
    # The requirement is likely "better than random" by some margin.
    # Default 0.5 means "better than random".
    auc_threshold: float = 0.5


@dataclass
class LabelingConfig:
    """Configuration for label assignment strategy."""

    context_window_days: int = 7
    decay_end_weight: float = 0.1
    noise_floor: float = 1.0


@dataclass
class PredictionConfig:
    """Overall prediction pipeline configuration."""

    # ISO Configuration
    iso: IsoConfig = field(default_factory=lambda: MISO_ISO_CONFIG)

    # Horizon Groups
    horizon_groups: list[HorizonGroupConfig] = field(default_factory=lambda: MISO_HORIZON_GROUPS)

    # Market parameters
    period_type: str = "f0"
    class_type: str = "onpeak"
    market_round: int = 1

    # Path configuration
    # path configuration
    # paths: DataPathConfig = field(default_factory=DataPathConfig) # Moved to IsoConfig

    # Feature configuration
    features: FeatureConfig = field(default_factory=FeatureConfig)

    # Label assignment configuration
    labeling: LabelingConfig = field(default_factory=LabelingConfig)

    # Model ensemble configuration
    models: EnsembleConfig = field(default_factory=EnsembleConfig)

    # Training configuration
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Threshold optimization
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)

    # Anomaly detection
    anomaly_detection: AnomalyDetectionConfig = field(default_factory=AnomalyDetectionConfig)

    # Feature selection
    feature_selection: FeatureSelectionConfig = field(default_factory=FeatureSelectionConfig)

    # Data loading parameters
    # run_at_day: int = 10 # Moved to IsoConfig
    # outage_freq: str = "3D"  # Moved to IsoConfig
