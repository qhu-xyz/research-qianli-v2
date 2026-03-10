"""
Shadow Price Prediction Package

A comprehensive package for predicting shadow prices in MISO electricity markets
using ensemble-based two-stage models (classification + regression) with branch-specific
training and flow anomaly detection.

Key Features:
- Parallel Processing: Ray-based parallel training for multiple test periods (near-linear speedup)
- Per-Period Training: Each test period gets its own training data and models
- Two-Stage Hybrid Model: Classification (binding detection) → Regression (shadow price)
- Ensemble Models: Weighted averaging of multiple model types
- Branch-Specific Models: Separate models trained for each transmission branch
- Dynamic Threshold Optimization: F-beta score optimization per branch

Default configuration:
- Classifiers: XGBoost (50%) + LogisticRegression (50%)
- Regressors: XGBoost (50%) + ElasticNet (50%)

Simplified configuration system (v2.0):
- Single ModelConfig class works with any sklearn/xgboost model
- Pass actual model classes (XGBClassifier, LogisticRegression, etc.) instead of strings
- All model parameters are dynamic via ModelConfig(params={...})

Performance:
- Parallel processing enabled by default for multiple test periods
- Use pipeline.run(use_parallel=True) for maximum speed (default)
- Control workers: n_jobs=0 (auto), n_jobs=3 (specific), n_jobs=-1 (all CPUs)
- Use pipeline.run(use_parallel=False) for sequential processing if needed
"""

from sklearn.linear_model import ElasticNet, LinearRegression, LogisticRegression

# Import model classes for user convenience
from xgboost import XGBClassifier, XGBRegressor

from .config import (
    AnomalyDetectionConfig,
    EnsembleConfig,
    FeatureConfig,
    ModelConfig,
    ModelSpec,
    PredictionConfig,
    ThresholdConfig,
    TrainingConfig,
)
from .evaluation import (
    analyze_results,
    calculate_metrics,
    print_metrics_report,
    print_score_report,
    score_results_df,
)
from .iso_configs import DataPathConfig, HorizonGroupConfig, IsoConfig
from .pipeline import ShadowPricePipeline
from .signal_generator import (
    aggregate_multi_month_results,
    convert_predictions_to_signal,
    generate_and_save_signal,
    save_signal,
)
from .registry import GateResult, ModelRegistry, ModelVersion, PromotionResult

__version__ = "2.1.0"

__all__ = [
    # Main pipeline
    "ShadowPricePipeline",
    # Configuration classes
    "PredictionConfig",
    "FeatureConfig",
    "ModelConfig",
    "ModelSpec",
    "EnsembleConfig",
    "TrainingConfig",
    "ThresholdConfig",
    "AnomalyDetectionConfig",
    "DataPathConfig",
    "HorizonGroupConfig",
    "IsoConfig",
    # Model classes (for convenience)
    "XGBClassifier",
    "XGBRegressor",
    "LogisticRegression",
    "LinearRegression",
    "ElasticNet",
    # Signal generation
    "aggregate_multi_month_results",
    "convert_predictions_to_signal",
    "save_signal",
    "generate_and_save_signal",
    # Evaluation functions
    "calculate_metrics",
    "analyze_results",
    "print_metrics_report",
    "score_results_df",
    "print_score_report",
    # Registry
    "ModelRegistry",
    "ModelVersion",
    "GateResult",
    "PromotionResult",
]
