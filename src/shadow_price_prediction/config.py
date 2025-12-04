"""
Configuration classes for shadow price prediction pipeline.
"""

from dataclasses import dataclass, field

from xgboost import XGBClassifier, XGBRegressor


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
            ("hist_da", 1),
            # ("105_diff", 0),
            # ("100_diff", 0),
            # ("95_diff", 0),
            # ("90_diff", 0),
            # ("85_diff", 0),
            # ("80_diff", 0),
            # ("70_diff", 0),
            # ("60_diff", 0),
            # ("risk_ratio", 1),
            # ("curvature_100", 1),
            # ("interaction_risk_overload", 1),
            # ("interaction_curvature_exceed", 1),
        ]
    )

    # Regression features (raw + derived diffs + engineered)
    step2_features: list[tuple[str, int]] = field(
        default_factory=lambda: [
            # ("curvature_100", 1),
            ("prob_exceed_110", 1),
            ("prob_exceed_105", 1),
            ("prob_exceed_100", 1),
            ("prob_exceed_95", 1),
            ("prob_exceed_90", 1),
            # ("prob_exceed_85", 1),
            # ("prob_exceed_80", 1),
            ("hist_da", 1),
            # ("105_diff", 0),
            # ("100_diff", 0),
            # ("95_diff", 0),
            # ("90_diff", 0),
            # ("85_diff", 0),
            # ("80_diff", 0),
            # ("70_diff", 0),
            # ("60_diff", 0),
            # ("risk_ratio", 1),
            # ("curvature_100", 1),
            # ("interaction_risk_overload", 1),
            # ("interaction_curvature_exceed", 1),
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


# MISO FTR Auction Schedule (Planning Year: Jun - May)
# Maps Auction Month (1-12) to available period types
AUCTION_SCHEDULE = {
    6: ["f0"],
    7: ["f0", "f1", "q2", "q3", "q4"],
    8: ["f0", "f1", "f2", "f3"],
    9: ["f0", "f1", "f2"],
    10: ["f0", "f1", "q3", "q4"],
    11: ["f0", "f1", "f2", "f3"],
    12: ["f0", "f1", "f2"],
    1: ["f0", "f1", "q4"],
    2: ["f0", "f1", "f2", "f3"],
    3: ["f0", "f1", "f2"],
    4: ["f0", "f1"],
    5: ["f0"],
}


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
    # These override the model-level weights based on forecast horizon
    # Format: [xgboost_weight, linear_model_weight]

    # Short-term horizons (0-1 months): f0, f1
    # Both models contribute equally - abundant data, stable patterns
    short_term_clf_weights: list[float] = field(default_factory=lambda: [0.5, 0.5])
    short_term_reg_weights: list[float] = field(default_factory=lambda: [0.5, 0.5])

    # Long-term horizons (4+ months): q2, q3, q4
    # Heavily favor XGBoost - scarce data, need horizon-aware model
    long_term_clf_weights: list[float] = field(default_factory=lambda: [0.7, 0.3])
    long_term_reg_weights: list[float] = field(default_factory=lambda: [0.7, 0.3])

    # Horizon thresholds for weight selection
    short_term_max_horizon: int = 1  # horizon <= 1

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
        self, horizon: int, model_type: str = "classifier", is_branch: bool = False
    ) -> list[float]:
        """
        Get ensemble weights based on forecast horizon.

        Parameters:
        -----------
        horizon : int
            Forecast horizon in months (0 for f0, 1 for f1, etc.)
        model_type : str
            'classifier' or 'regressor'
        is_branch : bool
            Whether this is for branch-specific or default models

        Returns:
        --------
        weights : List[float]
            Normalized weights for the ensemble models
        """
        # Select weights based on horizon
        if horizon <= self.short_term_max_horizon:
            weights = self.short_term_clf_weights if model_type == "classifier" else self.short_term_reg_weights
        else:
            weights = self.long_term_clf_weights if model_type == "classifier" else self.long_term_reg_weights

        # Normalize to sum to 1.0
        total = sum(weights)
        return [w / total for w in weights]


@dataclass
class TrainingConfig:
    """Model training configuration."""

    min_samples_for_branch_model: int = 1
    min_binding_samples_for_regression: int = 1
    train_months_lookback: int = 12  # Number of months to look back for training
    label_threshold: float = 0.0  # Threshold for binary classification label


@dataclass
class ThresholdConfig:
    """Dynamic threshold optimization configuration."""

    threshold_range_start: float = 0.01
    threshold_range_end: float = 0.99
    threshold_range_steps: int = 99
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
    min_samples_for_stats: int = 10
    flow_feature: str = "prob_exceed_100"  # Feature to use for anomaly detection


@dataclass
class DataPathConfig:
    """Data path templates."""

    density_path_template: str = (
        "/opt/temp/tmp/pw_data/spice6/prod_f0p_model_miso/density/"
        "auction_month={auction_month}/market_month={market_month}/"
        "market_round={market_round}/outage_date={outage_date}"
    )
    constraint_path_template: str = (
        "/opt/temp/tmp/pw_data/spice6/prod_f0p_model_miso/constraint_info/"
        "auction_month={auction_month}/market_round={market_round}/"
        "period_type={period_type}/class_type={class_type}"
    )


@dataclass
class PredictionConfig:
    """Overall prediction pipeline configuration."""

    # Market parameters
    period_type: str = "f0"
    class_type: str = "onpeak"
    market_round: int = 1

    # Path configuration
    paths: DataPathConfig = field(default_factory=DataPathConfig)

    # Feature configuration
    features: FeatureConfig = field(default_factory=FeatureConfig)

    # Model ensemble configuration
    models: EnsembleConfig = field(default_factory=EnsembleConfig)

    # Training configuration
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Threshold optimization
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)

    # Anomaly detection
    anomaly_detection: AnomalyDetectionConfig = field(default_factory=AnomalyDetectionConfig)

    # Data loading parameters
    run_at_day: int = 10
    outage_freq: str = "3D"  # Outage date frequency
