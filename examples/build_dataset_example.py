"""Example: Build training dataset for MISO shadow price prediction.

This example demonstrates how to use the data pipeline to:
1. Load score.parquet files from MISO density data
2. Fetch shadow prices using get_da_shadow() API
3. Create complete training dataset with proper train/val/test splits

Before running:
1. Install dependencies: pip install -r requirements.txt
2. Replace mock_get_da_shadow with your actual function
3. Adjust date ranges in config if needed
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd

from src.data.dataset_builder import build_training_dataset
from src.utils.config import Config

# ============================================================================
# STEP 1: Define your get_da_shadow function
# ============================================================================


def get_da_shadow(st: str, et: str, class_type: str) -> pd.DataFrame:
    """Fetch day-ahead shadow prices from your API.

    REPLACE THIS with your actual get_da_shadow function!

    Parameters
    ----------
    st : str
        Start date 'YYYY-MM-DD'
    et : str
        End date 'YYYY-MM-DD'
    class_type : str
        Constraint class type (e.g., 'constraint')

    Returns
    -------
    pd.DataFrame
        Shadow prices with columns:
        - timestamp (datetime): Hourly timestamps
        - constraint_id (str): Constraint identifier
        - shadow_price (float): Shadow price in $/MW
    """
    # EXAMPLE: Replace with your actual implementation
    # from your_module import fetch_shadow_prices
    # return fetch_shadow_prices(st, et, class_type)

    # Mock implementation for demonstration
    print("⚠️  WARNING: Using mock get_da_shadow function!")
    print(f"   Start: {st}, End: {et}, Class: {class_type}")
    print("   Replace with actual function in production!")

    # Create realistic mock data
    dates = pd.date_range(st, et, freq="H")
    constraints = [f"MISO.CONSTRAINT_{i:03d}" for i in range(1, 21)]

    mock_data = []
    for constraint_id in constraints:
        # Each constraint has different binding frequency
        binding_prob = 0.05 + (hash(constraint_id) % 20) * 0.01  # 5-25%

        for timestamp in dates:
            # Simulate binding events
            if pd.np.random.random() < binding_prob:
                # When binding, shadow price follows gamma distribution
                shadow_price = pd.np.random.gamma(shape=2, scale=10)
            else:
                shadow_price = 0.0

            mock_data.append(
                {
                    "timestamp": timestamp,
                    "constraint_id": constraint_id,
                    "shadow_price": shadow_price,
                }
            )

    return pd.DataFrame(mock_data)


# ============================================================================
# STEP 2: Configure data pipeline
# ============================================================================


def get_custom_config() -> Config:
    """Create custom configuration for data pipeline.

    Adjust these settings based on your needs.
    """
    config = Config()

    # Data paths
    config.data.density_base_path = "/opt/temp/tmp/pw_data/spice6/prod_f0p_model_miso/density"

    # Date ranges for splits
    # ADJUST THESE based on your data availability
    config.data.train_start_date = "2023-01-01"
    config.data.train_end_date = "2023-12-31"

    config.data.val_start_date = "2024-01-01"
    config.data.val_end_date = "2024-06-30"

    config.data.test_start_date = "2024-07-01"
    config.data.test_end_date = "2024-09-30"

    # Shadow price settings
    config.data.binding_threshold = 0.5  # $/MW
    config.data.shadow_price_aggregation = "mean"  # 'mean', 'max', 'median', 'p95'

    # Feature engineering
    config.features.use_temporal_features = True
    config.features.use_cyclical_encoding = True

    # Output directory
    config.output_dir = project_root / "results"

    return config


# ============================================================================
# STEP 3: Build dataset
# ============================================================================


def main():
    """Main function to build training dataset."""

    print("=" * 80)
    print("MISO Shadow Price Prediction - Dataset Builder")
    print("=" * 80)

    # Get configuration
    config = get_custom_config()

    print("\n📋 Configuration:")
    print(f"   Density path: {config.data.density_base_path}")
    print(f"   Train period: {config.data.train_start_date} to {config.data.train_end_date}")
    print(f"   Val period:   {config.data.val_start_date} to {config.data.val_end_date}")
    print(f"   Test period:  {config.data.test_start_date} to {config.data.test_end_date}")
    print(f"   Binding threshold: ${config.data.binding_threshold}/MW")
    print(f"   Aggregation method: {config.data.shadow_price_aggregation}")

    # Build dataset
    print("\n" + "=" * 80)
    print("🔨 Building Training Dataset...")
    print("=" * 80)

    try:
        train_df, val_df, test_df = build_training_dataset(
            get_da_shadow_func=get_da_shadow,
            config=config,
            save=True,  # Save to parquet files
        )

        # Display results
        print("\n" + "=" * 80)
        print("✅ Dataset Build Complete!")
        print("=" * 80)

        print("\n📊 Dataset Summary:")
        print(f"\n{'Training Set':-^60}")
        print(f"   Records: {len(train_df):,}")
        print(f"   Constraints: {train_df['constraint_id'].nunique()}")
        print(f"   Date range: {train_df['outage_date'].min()} to {train_df['outage_date'].max()}")
        print(f"   Binding rate: {train_df['target_binding'].mean():.2%}")
        print(f"   Mean shadow price: ${train_df['target_shadow_price'].mean():.2f}/MW")
        print(f"   Max shadow price: ${train_df['target_shadow_price'].max():.2f}/MW")

        print(f"\n{'Validation Set':-^60}")
        print(f"   Records: {len(val_df):,}")
        print(f"   Constraints: {val_df['constraint_id'].nunique()}")
        print(f"   Binding rate: {val_df['target_binding'].mean():.2%}")

        print(f"\n{'Test Set':-^60}")
        print(f"   Records: {len(test_df):,}")
        print(f"   Constraints: {test_df['constraint_id'].nunique()}")
        print(f"   Binding rate: {test_df['target_binding'].mean():.2%}")

        # Show feature columns
        from src.data.dataset_builder import DatasetBuilder

        builder = DatasetBuilder(config=config, get_da_shadow_func=get_da_shadow)
        feature_cols = builder.get_feature_columns(train_df)

        print(f"\n{'Features':-^60}")
        print(f"   Total features: {len(feature_cols)}")
        print("\n   Feature names (first 20):")
        for i, col in enumerate(feature_cols[:20], 1):
            print(f"      {i:2d}. {col}")
        if len(feature_cols) > 20:
            print(f"      ... and {len(feature_cols) - 20} more features")

        # Save location
        print(f"\n{'Saved Files':-^60}")
        output_dir = config.output_dir / "data" / "processed"
        print(f"   Location: {output_dir}/")
        print(f"   - train.parquet ({len(train_df):,} records)")
        print(f"   - val.parquet ({len(val_df):,} records)")
        print(f"   - test.parquet ({len(test_df):,} records)")
        print("   - metadata.json")

        # Class imbalance warning
        binding_rate = train_df["target_binding"].mean()
        print(f"\n{'⚠️  Class Imbalance Warning':-^60}")
        print(f"   Binding events: {binding_rate:.2%} of data")
        print(f"   Non-binding: {1 - binding_rate:.2%} of data")
        print("\n   When training models, remember to:")
        print("   1. Use class weights in classification")
        print("   2. Evaluate with F1 score, PR-AUC (not accuracy!)")
        print("   3. Use two-stage approach (classify → regress)")
        print("   4. Optimize for binding detection recall")

        # Next steps
        print(f"\n{'Next Steps':-^60}")
        print("   1. Review saved datasets:")
        print(
            f"      python -c \"import pandas as pd; df=pd.read_parquet('{output_dir}/train.parquet'); print(df.info())\""
        )
        print("\n   2. Train two-stage model:")
        print("      - Stage 1: Classify binding events (LightGBM with class_weight='balanced')")
        print("      - Stage 2: Predict shadow price magnitude (LightGBM on binding events)")
        print("\n   3. Validate and test:")
        print("      - Use val.parquet for hyperparameter tuning")
        print("      - Use test.parquet for final evaluation")

        return train_df, val_df, test_df

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()

        print("\n🔧 Troubleshooting:")
        print("   1. Check that density_base_path exists and is accessible")
        print("   2. Verify get_da_shadow function is working correctly")
        print("   3. Ensure date ranges have available data")
        print("   4. Review error message above for specific issues")

        return None, None, None


if __name__ == "__main__":
    # Run dataset builder
    train_df, val_df, test_df = main()

    # Optional: Quick data exploration
    if train_df is not None:
        print("\n" + "=" * 80)
        print("📈 Quick Data Exploration")
        print("=" * 80)

        # Target distribution
        print("\nTarget Shadow Price Distribution:")
        print(train_df["target_shadow_price"].describe())

        # Binding by constraint
        print("\nTop 10 Constraints by Binding Rate:")
        binding_by_constraint = (
            train_df.groupby("constraint_id")["target_binding"]
            .agg(["count", "mean", "sum"])
            .rename(columns={"count": "n_periods", "mean": "binding_rate", "sum": "n_binding"})
            .sort_values("binding_rate", ascending=False)
            .head(10)
        )
        print(binding_by_constraint)

        print("\n✅ Dataset build complete!")
