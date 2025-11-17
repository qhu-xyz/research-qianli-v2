"""Configuration management for MISO shadow price prediction.

This module provides centralized configuration for data paths, model parameters,
and training settings.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class DataConfig:
    """Data paths and loading configuration."""

    # Base path to MISO density data
    density_base_path: str = "/opt/temp/tmp/pw_data/spice6/prod_f0p_model_miso/density"

    # Shadow price binding threshold ($/MW)
    binding_threshold: float = 0.5

    # Date range for training data
    train_start_date: str = "2017-06-01"
    train_end_date: str = "2023-12-31"

    # Validation split
    val_start_date: str = "2024-01-01"
    val_end_date: str = "2024-06-30"

    # Test split
    test_start_date: str = "2024-07-01"
    test_end_date: str = "2024-12-31"

    # Shadow price aggregation method for 3-day periods
    shadow_price_aggregation: str = "mean"  # 'mean', 'max', 'p95'

    def __post_init__(self):
        """Validate configuration."""
        self.density_base_path = Path(self.density_base_path)
        if not self.density_base_path.exists():
            raise ValueError(
                f"Density base path does not exist: {self.density_base_path}"
            )


@dataclass
class FeatureConfig:
    """Feature engineering configuration."""

    # Temporal features
    use_temporal_features: bool = True
    use_cyclical_encoding: bool = True

    # Lag features (in hours)
    lag_hours: list[int] = field(default_factory=lambda: [24, 168])  # 1 day, 1 week

    # Rolling window features (in hours)
    rolling_windows: list[int] = field(default_factory=lambda: [24, 168])


@dataclass
class ModelConfig:
    """Model training configuration."""

    # Two-stage model settings
    use_two_stage: bool = True

    # Stage 1: Classification
    classifier_params: dict = field(
        default_factory=lambda: {
            "n_estimators": 500,
            "max_depth": 10,
            "learning_rate": 0.05,
            "num_leaves": 64,
            "min_child_samples": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "class_weight": "balanced",
            "random_state": 42,
            "verbose": -1,
        }
    )

    # Stage 2: Regression
    regressor_params: dict = field(
        default_factory=lambda: {
            "n_estimators": 500,
            "max_depth": 8,
            "learning_rate": 0.05,
            "num_leaves": 64,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
            "verbose": -1,
        }
    )

    # Classification threshold (probability)
    probability_threshold: float = 0.5

    # Hyperparameter optimization
    use_optuna: bool = False
    n_trials: int = 100


@dataclass
class Config:
    """Main configuration container."""

    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    # Output paths
    output_dir: Path = field(
        default_factory=lambda: Path(
            "/home/xyz/workspace/research_spice_shadow_price_pred/results"
        )
    )
    model_dir: Path = field(default_factory=lambda: Path("results/models"))
    prediction_dir: Path = field(default_factory=lambda: Path("results/predictions"))

    def __post_init__(self):
        """Create output directories."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.prediction_dir.mkdir(parents=True, exist_ok=True)


def get_config() -> Config:
    """Get default configuration.

    Returns
    -------
    Config
        Default configuration object
    """
    return Config()
