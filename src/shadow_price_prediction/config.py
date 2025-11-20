"""
Configuration classes for shadow price prediction pipeline.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Any
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor


@dataclass
class FeatureConfig:
    """Feature column configuration."""
    # Classification features (raw + engineered)
    step1_features: List[str] = field(default_factory=lambda: [
        '110', '105', '100', '95', '90',
        'prob_overload', 'risk_ratio', 'curvature_100',
        'prob_exceed_100', 'prob_exceed_90', 'log_density_100'
    ])

    # Regression features (raw + derived diffs + engineered)
    step2_features: List[str] = field(default_factory=lambda: [
        '110', '105_diff', '100_diff', '95_diff', '90_diff',
        '85_diff', '80_diff', '70_diff', '60_diff',
        'prob_overload', 'risk_ratio', 'curvature_100',
        'prob_exceed_100', 'prob_exceed_90', 'log_density_100'
    ])

    # All features needed (union of step1 and step2) - computed automatically
    all_features: List[str] = field(default_factory=list, init=False)

    def __post_init__(self):
        """Compute all_features as union of step1 and step2 features."""
        self.all_features = list(set(self.step1_features + self.step2_features))


@dataclass
class ModelConfig:
    """Universal model configuration that works with any sklearn/xgboost model.

    All model parameters are passed via the params dictionary.

    Examples:
    ---------
    # XGBoost Classifier
    ModelConfig(params={'n_estimators': 200, 'max_depth': 4, 'learning_rate': 0.1, 'n_jobs': -1})

    # Logistic Regression
    ModelConfig(params={'C': 1.0, 'max_iter': 1000, 'class_weight': 'balanced', 'n_jobs': -1})

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
    default_classifiers: List[ModelSpec] = field(default_factory=lambda: [
        ModelSpec(
            model_class=XGBClassifier,
            config=ModelConfig(params={
                'n_estimators': 200,
                'min_child_weight': 10,
                'n_jobs': -1,
                'max_depth': 4,
                'learning_rate': 0.1,
                'verbosity': 0,
                'eval_metric': 'logloss'
            }),
            weight=0.5
        ),
        ModelSpec(
            model_class=LogisticRegression,
            config=ModelConfig(params={
                'C': 1.0,
                'max_iter': 1000,
                'class_weight': 'balanced',
                'n_jobs': -1,
                'solver': 'lbfgs',
                'random_state': 42
            }),
            weight=0.5
        )
    ])

    # Branch-specific classifier ensemble (XGBoost + LogisticRegression)
    branch_classifiers: List[ModelSpec] = field(default_factory=lambda: [
        ModelSpec(
            model_class=XGBClassifier,
            config=ModelConfig(params={
                'n_estimators': 100,
                'max_depth': 4,
                'learning_rate': 0.1,
                'min_child_weight': 5,
                'n_jobs': 1,
                'verbosity': 0,
                'eval_metric': 'logloss'
            }),
            weight=0.5
        ),
        ModelSpec(
            model_class=LogisticRegression,
            config=ModelConfig(params={
                'C': 1.0,
                'max_iter': 1000,
                'class_weight': 'balanced',
                'n_jobs': 1,
                'solver': 'lbfgs',
                'random_state': 42
            }),
            weight=0.5
        )
    ])

    # Default regressor ensemble (XGBoost + ElasticNet)
    default_regressors: List[ModelSpec] = field(default_factory=lambda: [
        ModelSpec(
            model_class=XGBRegressor,
            config=ModelConfig(params={
                'n_estimators': 200,
                'min_child_weight': 2,
                'n_jobs': -1,
                'max_depth': 4,
                'learning_rate': 0.1,
                'verbosity': 0,
                'objective': 'reg:squarederror'
            }),
            weight=0.5
        ),
        ModelSpec(
            model_class=ElasticNet,
            config=ModelConfig(params={
                'alpha': 1.0,
                'l1_ratio': 0.5,
                'max_iter': 1000,
                'fit_intercept': True,
                'random_state': 42
            }),
            weight=0.5
        )
    ])

    # Branch-specific regressor ensemble (XGBoost + ElasticNet)
    branch_regressors: List[ModelSpec] = field(default_factory=lambda: [
        ModelSpec(
            model_class=XGBRegressor,
            config=ModelConfig(params={
                'n_estimators': 100,
                'max_depth': 4,
                'learning_rate': 0.1,
                'min_child_weight': 1,
                'n_jobs': 1,
                'verbosity': 0,
                'objective': 'reg:squarederror'
            }),
            weight=0.5
        ),
        ModelSpec(
            model_class=ElasticNet,
            config=ModelConfig(params={
                'alpha': 1.0,
                'l1_ratio': 0.5,
                'max_iter': 1000,
                'fit_intercept': True,
                'random_state': 42
            }),
            weight=0.5
        )
    ])

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




@dataclass
class TrainingConfig:
    """Model training configuration."""
    min_samples_for_branch_model: int = 10
    min_binding_samples_for_regression: int = 1
    train_months_lookback: int = 12  # Number of months to look back for training


@dataclass
class ThresholdConfig:
    """Dynamic threshold optimization configuration."""
    threshold_range_start: float = 0.01
    threshold_range_end: float = 0.99
    threshold_range_steps: int = 99
    threshold_beta: float = 2.0  # F-beta score beta parameter (2 favors recall)
    
    # Dynamic Thresholds (4.B)
    # Adjust threshold based on predicted shadow price magnitude
    # threshold = base_threshold - (factor * log1p(predicted_price))


@dataclass
class AnomalyDetectionConfig:
    """Flow anomaly detection configuration."""
    enabled: bool = True
    k_multiplier: float = 3.0  # IQR multiplier for anomaly threshold
    min_samples_for_stats: int = 10
    flow_feature: str = '100'  # Feature to use for anomaly detection


@dataclass
class DataPathConfig:
    """Data path templates."""
    density_path_template: str = (
        '/opt/temp/tmp/pw_data/spice6/prod_f0p_model_miso/density/'
        'auction_month={auction_month}/market_month={market_month}/'
        'market_round={market_round}/outage_date={outage_date}'
    )
    constraint_path_template: str = (
        '/opt/temp/tmp/pw_data/spice6/prod_f0p_model_miso/constraint_info/'
        'auction_month={auction_month}/market_round={market_round}/'
        'period_type={period_type}/class_type={class_type}'
    )


@dataclass
class PredictionConfig:
    """Overall prediction pipeline configuration."""
    # Date parameters - supports both single period and multiple periods
    test_auction_month: Optional[pd.Timestamp] = None
    test_market_month: Optional[pd.Timestamp] = None

    # Multiple test periods: List of (auction_month, market_month) tuples
    # If provided, this takes precedence over single test_auction_month/test_market_month
    test_periods: Optional[List[Tuple[pd.Timestamp, pd.Timestamp]]] = None

    # Market parameters
    period_type: str = 'f0'
    class_type: str = 'onpeak'
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
    outage_freq: str = '3D'  # Outage date frequency

    def __post_init__(self):
        """Validate and normalize test period configuration."""
        if self.test_periods is None:
            # If no test_periods provided, use single period (backward compatibility)
            if self.test_auction_month is None or self.test_market_month is None:
                # Default to 2025-10
                self.test_auction_month = pd.Timestamp('2025-10')
                self.test_market_month = pd.Timestamp('2025-10')

            # Create test_periods list with single period
            self.test_periods = [(self.test_auction_month, self.test_market_month)]
        else:
            # test_periods is provided, validate it
            if not isinstance(self.test_periods, list) or len(self.test_periods) == 0:
                raise ValueError("test_periods must be a non-empty list of (auction_month, market_month) tuples")

            # Update single period fields to first period for backward compatibility
            self.test_auction_month = self.test_periods[0][0]
            self.test_market_month = self.test_periods[0][1]
