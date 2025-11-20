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

from .config import (
    PredictionConfig,
    FeatureConfig,
    ModelConfig,
    ModelSpec,
    EnsembleConfig,
    TrainingConfig,
    ThresholdConfig,
    AnomalyDetectionConfig,
    DataPathConfig
)
from .pipeline import ShadowPricePipeline
from .evaluation import calculate_metrics, analyze_results, print_metrics_report

from .models import StackingModel
from .tuning import HyperparameterTuner

# Import model classes for user convenience
from xgboost import XGBClassifier, XGBRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor

__version__ = "2.0.0"

__all__ = [
    # Main pipeline
    'ShadowPricePipeline',

    # Configuration classes
    'PredictionConfig',
    'FeatureConfig',
    'ModelConfig',
    'ModelSpec',
    'EnsembleConfig',
    'TrainingConfig',
    'ThresholdConfig',
    'AnomalyDetectionConfig',
    'DataPathConfig',

    # Model classes (for convenience)
    'XGBClassifier',
    'XGBRegressor',
    'LogisticRegression',
    'LinearRegression',
    'ElasticNet',
    'RandomForestClassifier',
    'RandomForestRegressor',
    'GradientBoostingClassifier',
    'GradientBoostingRegressor',
    'StackingModel',
    'HyperparameterTuner',

    # Evaluation functions
    'calculate_metrics',
    'analyze_results',
    'print_metrics_report',
]
