"""Build complete training dataset from scores and shadow prices.

This module combines score features and shadow price targets into a complete
dataset ready for model training, with proper train/validation/test splits.
"""

from pathlib import Path
from typing import Callable, Optional

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.score_loader import ScoreLoader
from src.data.shadow_price_loader import ShadowPriceLoader
from src.features.temporal import add_temporal_features
from src.utils.config import Config, get_config


class DatasetBuilder:
    """Build complete training dataset.

    Workflow:
    1. Load score features from parquet files
    2. Load and aggregate shadow prices for matching periods
    3. Add temporal features
    4. Create train/validation/test splits
    5. Handle class imbalance
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        get_da_shadow_func: Optional[Callable] = None,
    ):
        """Initialize dataset builder.

        Parameters
        ----------
        config : Config, optional
            Configuration object. If None, uses default config.
        get_da_shadow_func : Callable, optional
            Function to fetch shadow prices: get_da_shadow(st, et, class_type)
        """
        self.config = config or get_config()
        self.get_da_shadow = get_da_shadow_func

        self.score_loader = ScoreLoader(self.config.data.density_base_path)
        self.shadow_loader = ShadowPriceLoader(
            get_da_shadow_func=get_da_shadow_func,
            aggregation_method=self.config.data.shadow_price_aggregation,
            binding_threshold=self.config.data.binding_threshold,
        )

    def load_scores(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Load score features for date range.

        Parameters
        ----------
        start_date : str
            Start date 'YYYY-MM-DD'
        end_date : str
            End date 'YYYY-MM-DD'

        Returns
        -------
        pd.DataFrame
            Score features
        """
        print(f"\n{'='*60}")
        print(f"Loading score features: {start_date} to {end_date}")
        print(f"{'='*60}")

        scores = self.score_loader.load_date_range(start_date, end_date)

        print(f"Loaded {len(scores):,} score records")
        print(f"Unique constraints: {scores['constraint_id'].nunique()}")
        print(f"Date range: {scores['outage_date'].min()} to {scores['outage_date'].max()}")

        return scores

    def load_shadow_prices(
        self, score_df: pd.DataFrame, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Load and aggregate shadow prices.

        Parameters
        ----------
        score_df : pd.DataFrame
            Score features
        start_date : str
            Start date 'YYYY-MM-DD'
        end_date : str
            End date 'YYYY-MM-DD'

        Returns
        -------
        pd.DataFrame
            Score features with shadow prices
        """
        print(f"\n{'='*60}")
        print(f"Loading shadow prices: {start_date} to {end_date}")
        print(f"{'='*60}")

        # Load shadow prices
        shadow_df = self.shadow_loader.load_shadow_prices(
            start_date, end_date, class_type="constraint"
        )

        print(f"Loaded {len(shadow_df):,} shadow price records")

        # Aggregate for score periods
        result = self.shadow_loader.aggregate_for_score_data(score_df, shadow_df)

        # Report statistics
        binding_rate = (
            result["shadow_price_agg"] > self.config.data.binding_threshold
        ).mean()

        print(f"\nShadow Price Statistics:")
        print(f"Total records: {len(result):,}")
        print(
            f"Records with shadow price: {result['shadow_price_agg'].notna().sum():,}"
        )
        print(f"Binding rate: {binding_rate:.2%}")
        print(
            f"Mean shadow price: ${result['shadow_price_agg'].mean():.2f}/MW"
        )
        print(
            f"Max shadow price: ${result['shadow_price_agg'].max():.2f}/MW"
        )

        return result

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add engineered features.

        Parameters
        ----------
        df : pd.DataFrame
            Base dataset

        Returns
        -------
        pd.DataFrame
            Dataset with additional features
        """
        print(f"\n{'='*60}")
        print("Adding engineered features")
        print(f"{'='*60}")

        # Add temporal features if outage_date exists
        if "outage_date" in df.columns and self.config.features.use_temporal_features:
            df = add_temporal_features(
                df,
                timestamp_col="outage_date",
                use_cyclical=self.config.features.use_cyclical_encoding,
            )
            print("✓ Added temporal features")

        # Add constraint-level features
        if "constraint_id" in df.columns:
            # Historical binding rate per constraint
            constraint_stats = df.groupby("constraint_id")["shadow_price_agg"].agg(
                [
                    ("historical_binding_rate", lambda x: (x > self.config.data.binding_threshold).mean()),
                    ("historical_mean_shadow", "mean"),
                    ("historical_max_shadow", "max"),
                ]
            ).reset_index()

            df = df.merge(constraint_stats, on="constraint_id", how="left")
            print("✓ Added constraint-level features")

        print(f"Total features: {len(df.columns)}")

        return df

    def create_target_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create target variables for two-stage model.

        Parameters
        ----------
        df : pd.DataFrame
            Dataset with shadow_price_agg

        Returns
        -------
        pd.DataFrame
            Dataset with target variables:
            - target_binding: Binary (0/1) for binding classification
            - target_shadow_price: Shadow price for regression
        """
        # Binary binding target
        df["target_binding"] = (
            df["shadow_price_agg"] > self.config.data.binding_threshold
        ).astype(int)

        # Regression target (shadow price)
        df["target_shadow_price"] = df["shadow_price_agg"].fillna(0.0)

        return df

    def split_dataset(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split dataset into train/validation/test sets.

        Uses temporal splits to avoid look-ahead bias.

        Parameters
        ----------
        df : pd.DataFrame
            Complete dataset

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            train_df, val_df, test_df
        """
        print(f"\n{'='*60}")
        print("Creating train/validation/test splits")
        print(f"{'='*60}")

        # Ensure outage_date is datetime
        df["outage_date_dt"] = pd.to_datetime(df["outage_date"])

        # Temporal split based on config
        train_mask = (
            df["outage_date_dt"]
            >= pd.to_datetime(self.config.data.train_start_date)
        ) & (df["outage_date_dt"] <= pd.to_datetime(self.config.data.train_end_date))

        val_mask = (
            df["outage_date_dt"] >= pd.to_datetime(self.config.data.val_start_date)
        ) & (df["outage_date_dt"] <= pd.to_datetime(self.config.data.val_end_date))

        test_mask = (
            df["outage_date_dt"] >= pd.to_datetime(self.config.data.test_start_date)
        ) & (df["outage_date_dt"] <= pd.to_datetime(self.config.data.test_end_date))

        train_df = df[train_mask].copy()
        val_df = df[val_mask].copy()
        test_df = df[test_mask].copy()

        # Report split statistics
        print(f"\nTrain Set: {len(train_df):,} records")
        print(
            f"  Date range: {train_df['outage_date'].min()} to {train_df['outage_date'].max()}"
        )
        print(f"  Binding rate: {train_df['target_binding'].mean():.2%}")

        print(f"\nValidation Set: {len(val_df):,} records")
        print(
            f"  Date range: {val_df['outage_date'].min()} to {val_df['outage_date'].max()}"
        )
        print(f"  Binding rate: {val_df['target_binding'].mean():.2%}")

        print(f"\nTest Set: {len(test_df):,} records")
        print(
            f"  Date range: {test_df['outage_date'].min()} to {test_df['outage_date'].max()}"
        )
        print(f"  Binding rate: {test_df['target_binding'].mean():.2%}")

        return train_df, val_df, test_df

    def build_dataset(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Build complete dataset with train/val/test splits.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            train_df, val_df, test_df
        """
        # Determine overall date range from config
        start_date = self.config.data.train_start_date
        end_date = self.config.data.test_end_date

        # Load scores
        scores = self.load_scores(start_date, end_date)

        # Load and aggregate shadow prices
        data = self.load_shadow_prices(scores, start_date, end_date)

        # Add features
        data = self.add_features(data)

        # Create target variables
        data = self.create_target_variables(data)

        # Split dataset
        train_df, val_df, test_df = self.split_dataset(data)

        return train_df, val_df, test_df

    def save_dataset(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        output_dir: Optional[Path] = None,
    ):
        """Save datasets to parquet files.

        Parameters
        ----------
        train_df : pd.DataFrame
            Training dataset
        val_df : pd.DataFrame
            Validation dataset
        test_df : pd.DataFrame
            Test dataset
        output_dir : Path, optional
            Output directory. If None, uses config output_dir.
        """
        output_dir = output_dir or self.config.output_dir / "data" / "processed"
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Saving datasets to {output_dir}")
        print(f"{'='*60}")

        # Save datasets
        train_path = output_dir / "train.parquet"
        val_path = output_dir / "val.parquet"
        test_path = output_dir / "test.parquet"

        train_df.to_parquet(train_path, index=False)
        val_df.to_parquet(val_path, index=False)
        test_df.to_parquet(test_path, index=False)

        print(f"✓ Saved train: {train_path}")
        print(f"✓ Saved validation: {val_path}")
        print(f"✓ Saved test: {test_path}")

        # Save metadata
        metadata = {
            "train": {
                "n_records": len(train_df),
                "binding_rate": float(train_df["target_binding"].mean()),
                "date_range": [
                    str(train_df["outage_date"].min()),
                    str(train_df["outage_date"].max()),
                ],
            },
            "val": {
                "n_records": len(val_df),
                "binding_rate": float(val_df["target_binding"].mean()),
                "date_range": [
                    str(val_df["outage_date"].min()),
                    str(val_df["outage_date"].max()),
                ],
            },
            "test": {
                "n_records": len(test_df),
                "binding_rate": float(test_df["target_binding"].mean()),
                "date_range": [
                    str(test_df["outage_date"].min()),
                    str(test_df["outage_date"].max()),
                ],
            },
            "feature_columns": list(train_df.columns),
            "n_features": len(train_df.columns),
        }

        metadata_path = output_dir / "metadata.json"
        import json

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"✓ Saved metadata: {metadata_path}")

    def get_feature_columns(self, df: pd.DataFrame) -> list[str]:
        """Get list of feature columns (excluding targets and metadata).

        Parameters
        ----------
        df : pd.DataFrame
            Dataset

        Returns
        -------
        list[str]
            Feature column names
        """
        exclude_cols = {
            # Target columns
            "target_binding",
            "target_shadow_price",
            "shadow_price_agg",
            "shadow_price_mean",
            "shadow_price_max",
            "shadow_price_median",
            "shadow_price_p95",
            # Metadata columns
            "outage_date",
            "outage_date_dt",
            "period_start",
            "auction_month",
            "market_month",
            "market_round",
            "timestamp",
            "datetime",
            # Auxiliary columns
            "binding_frequency",
            "binding_status",
            "n_hours",
        }

        feature_cols = [col for col in df.columns if col not in exclude_cols]

        return feature_cols


def build_training_dataset(
    get_da_shadow_func: Callable,
    config: Optional[Config] = None,
    save: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Convenience function to build complete training dataset.

    Parameters
    ----------
    get_da_shadow_func : Callable
        Function to fetch shadow prices
    config : Config, optional
        Configuration. If None, uses default.
    save : bool, default True
        Save datasets to parquet files

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        train_df, val_df, test_df

    Examples
    --------
    >>> from src.data.dataset_builder import build_training_dataset
    >>>
    >>> # Assuming you have get_da_shadow function available
    >>> train_df, val_df, test_df = build_training_dataset(get_da_shadow)
    >>>
    >>> print(f"Training set: {len(train_df)} records")
    >>> print(f"Binding rate: {train_df['target_binding'].mean():.2%}")
    """
    builder = DatasetBuilder(config=config, get_da_shadow_func=get_da_shadow_func)

    # Build dataset
    train_df, val_df, test_df = builder.build_dataset()

    # Save if requested
    if save:
        builder.save_dataset(train_df, val_df, test_df)

    return train_df, val_df, test_df
