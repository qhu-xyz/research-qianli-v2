"""Build training dataset for MISO shadow price prediction.

This script demonstrates how to:
1. Load score.parquet files as features
2. Fetch and aggregate shadow prices for 3-day periods
3. Create complete training dataset with train/val/test splits
4. Save datasets for model training

Usage:
    python src/build_training_data.py

Requirements:
    - Access to /opt/temp/tmp/pw_data/spice6/prod_f0p_model_miso/density/
    - get_da_shadow(st, et, class_type) function available in environment
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd

from src.data.dataset_builder import DatasetBuilder
from src.utils.config import Config


def mock_get_da_shadow(st: str, et: str, class_type: str) -> pd.DataFrame:
    """Mock function for get_da_shadow for testing.

    Replace this with your actual get_da_shadow function.

    Parameters
    ----------
    st : str
        Start date 'YYYY-MM-DD'
    et : str
        End date 'YYYY-MM-DD'
    class_type : str
        Constraint class type

    Returns
    -------
    pd.DataFrame
        Shadow price data with columns:
        - timestamp (datetime)
        - constraint_id (str)
        - shadow_price (float)
    """
    print("WARNING: Using mock get_da_shadow function!")
    print(f"  Start: {st}, End: {et}, Class: {class_type}")
    print("  Replace with actual get_da_shadow function for real data")

    # Create mock data for demonstration
    dates = pd.date_range(st, et, freq="H")
    constraints = [f"CONSTRAINT_{i:03d}" for i in range(1, 11)]

    mock_data = []
    for constraint_id in constraints:
        for timestamp in dates:
            # Simulate binding events (10% of the time)
            if pd.np.random.random() < 0.10:
                shadow_price = pd.np.random.gamma(2, 10)  # Skewed distribution
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


def main():
    """Main function to build training dataset."""
    print("=" * 80)
    print("MISO Shadow Price Prediction - Training Data Builder")
    print("=" * 80)

    # Step 1: Configure
    print("\n" + "=" * 80)
    print("Step 1: Configuration")
    print("=" * 80)

    config = Config()

    # Customize configuration if needed
    config.data.train_start_date = "2023-01-01"
    config.data.train_end_date = "2023-12-31"
    config.data.val_start_date = "2024-01-01"
    config.data.val_end_date = "2024-06-30"
    config.data.test_start_date = "2024-07-01"
    config.data.test_end_date = "2024-09-30"

    print("\nData Configuration:")
    print(f"  Density path: {config.data.density_base_path}")
    print(f"  Train period: {config.data.train_start_date} to {config.data.train_end_date}")
    print(f"  Val period: {config.data.val_start_date} to {config.data.val_end_date}")
    print(f"  Test period: {config.data.test_start_date} to {config.data.test_end_date}")
    print(f"  Binding threshold: ${config.data.binding_threshold}/MW")
    print(f"  Shadow aggregation: {config.data.shadow_price_aggregation}")

    # Step 2: Initialize builder
    print("\n" + "=" * 80)
    print("Step 2: Initialize Dataset Builder")
    print("=" * 80)

    # IMPORTANT: Replace mock_get_da_shadow with your actual function
    # For example:
    # from your_module import get_da_shadow
    # builder = DatasetBuilder(config=config, get_da_shadow_func=get_da_shadow)

    builder = DatasetBuilder(config=config, get_da_shadow_func=mock_get_da_shadow)

    print("✓ Dataset builder initialized")

    # Step 3: Build dataset
    print("\n" + "=" * 80)
    print("Step 3: Build Complete Dataset")
    print("=" * 80)

    try:
        train_df, val_df, test_df = builder.build_dataset()

        # Step 4: Inspect data
        print("\n" + "=" * 80)
        print("Step 4: Dataset Summary")
        print("=" * 80)

        print(f"\n{'Train Set':-^60}")
        print(f"Shape: {train_df.shape}")
        print(f"Constraints: {train_df['constraint_id'].nunique()}")
        print(f"Binding rate: {train_df['target_binding'].mean():.2%}")
        print("\nTarget distribution:")
        print(train_df["target_shadow_price"].describe())

        print(f"\n{'Validation Set':-^60}")
        print(f"Shape: {val_df.shape}")
        print(f"Constraints: {val_df['constraint_id'].nunique()}")
        print(f"Binding rate: {val_df['target_binding'].mean():.2%}")

        print(f"\n{'Test Set':-^60}")
        print(f"Shape: {test_df.shape}")
        print(f"Constraints: {test_df['constraint_id'].nunique()}")
        print(f"Binding rate: {test_df['target_binding'].mean():.2%}")

        # Step 5: Get feature columns
        feature_cols = builder.get_feature_columns(train_df)
        print(f"\n{'Feature Columns':-^60}")
        print(f"Total features: {len(feature_cols)}")
        print("\nFeature names:")
        for i, col in enumerate(feature_cols, 1):
            print(f"  {i:3d}. {col}")

        # Step 6: Save datasets
        print("\n" + "=" * 80)
        print("Step 5: Save Datasets")
        print("=" * 80)

        builder.save_dataset(train_df, val_df, test_df)

        # Step 7: Summary
        print("\n" + "=" * 80)
        print("✓ Training Data Build Complete!")
        print("=" * 80)

        print("\nNext Steps:")
        print("1. Inspect saved datasets in results/data/processed/")
        print("2. Review metadata.json for dataset statistics")
        print("3. Train two-stage model using train.parquet")
        print("4. Validate on val.parquet, test on test.parquet")

        print("\nClass Imbalance Note:")
        print(f"  Binding rate is {train_df['target_binding'].mean():.2%}")
        print("  Remember to use:")
        print("  - Class weights in classification")
        print("  - F1 score, PR-AUC (not accuracy!)")
        print("  - Two-stage approach (classify → regress)")

        return train_df, val_df, test_df

    except Exception as e:
        print(f"\n❌ Error building dataset: {e}")
        import traceback

        traceback.print_exc()
        return None, None, None


if __name__ == "__main__":
    train_df, val_df, test_df = main()
