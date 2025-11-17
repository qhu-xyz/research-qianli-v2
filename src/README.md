# Source Code - MISO Shadow Price Prediction

Production-ready Python code for loading score features and building training datasets.

## 📁 Directory Structure

```
src/
├── data/                      # Data loading and processing
│   ├── score_loader.py        # Load score.parquet files
│   ├── shadow_price_loader.py # Load and aggregate shadow prices
│   └── dataset_builder.py     # Build complete training dataset
├── features/                  # Feature engineering
│   └── temporal.py            # Temporal features (hour, day, season, etc.)
├── models/                    # Model implementations (TODO)
├── evaluation/                # Evaluation metrics (TODO)
├── utils/                     # Utilities
│   └── config.py              # Configuration management
└── build_training_data.py     # Main script to build training data
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install pandas numpy scikit-learn tqdm pyarrow
```

### 2. Prepare Your `get_da_shadow` Function

The code expects a function with this signature:

```python
def get_da_shadow(st: str, et: str, class_type: str) -> pd.DataFrame:
    """Fetch day-ahead shadow prices.

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
        - timestamp (datetime)
        - constraint_id (str)
        - shadow_price (float)
    """
    # Your implementation here
    pass
```

### 3. Build Training Dataset

```python
from src.data.dataset_builder import build_training_dataset
from your_module import get_da_shadow  # Your shadow price API

# Build complete dataset with train/val/test splits
train_df, val_df, test_df = build_training_dataset(
    get_da_shadow_func=get_da_shadow,
    save=True  # Save to results/data/processed/
)

print(f"Train: {len(train_df)} records, Binding: {train_df['target_binding'].mean():.2%}")
print(f"Val: {len(val_df)} records")
print(f"Test: {len(test_df)} records")
```

### 4. Or Use the Main Script

```bash
# Edit src/build_training_data.py to use your get_da_shadow function
# Then run:
python src/build_training_data.py
```

## 📊 Data Pipeline Flow

```
┌─────────────────────────────────────────────────────┐
│  1. Load Score Features                             │
│     ScoreLoader.load_date_range()                   │
│     → Reads score.parquet files from density/       │
│     → Returns: scores_df                            │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│  2. Load Shadow Prices                              │
│     ShadowPriceLoader.load_shadow_prices()          │
│     → Calls: get_da_shadow(st, et, class_type)      │
│     → Returns: shadow_df (hourly)                   │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│  3. Aggregate Shadow Prices for 3-Day Periods       │
│     ShadowPriceLoader.aggregate_for_score_data()    │
│     → Groups by constraint_id + outage_date         │
│     → Aggregates: mean, max, median, p95            │
│     → Returns: shadow_agg_df                        │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│  4. Join Scores + Shadow Prices                     │
│     scores_df.merge(shadow_agg_df)                  │
│     → On: constraint_id, outage_date                │
│     → Returns: combined_df                          │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│  5. Add Features                                    │
│     add_temporal_features()                         │
│     → Hour, day, month, season                      │
│     → Cyclical encoding (sin/cos)                   │
│     → Peak indicators                               │
│     → Returns: features_df                          │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│  6. Create Targets                                  │
│     → target_binding = (shadow_price > 0.5)         │
│     → target_shadow_price = shadow_price            │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│  7. Split Dataset (Temporal)                        │
│     → Train: 2017-2023                              │
│     → Val: 2024-01 to 2024-06                       │
│     → Test: 2024-07 to 2024-12                      │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│  8. Save to Parquet                                 │
│     → results/data/processed/train.parquet          │
│     → results/data/processed/val.parquet            │
│     → results/data/processed/test.parquet           │
│     → results/data/processed/metadata.json          │
└─────────────────────────────────────────────────────┘
```

## 🔧 Configuration

Edit `src/utils/config.py` to customize:

```python
@dataclass
class DataConfig:
    # Data paths
    density_base_path: str = "/opt/temp/tmp/pw_data/spice6/prod_f0p_model_miso/density"

    # Shadow price settings
    binding_threshold: float = 0.5  # $/MW
    shadow_price_aggregation: str = "mean"  # 'mean', 'max', 'median', 'p95'

    # Date ranges
    train_start_date: str = "2017-06-01"
    train_end_date: str = "2023-12-31"
    val_start_date: str = "2024-01-01"
    val_end_date: str = "2024-06-30"
    test_start_date: str = "2024-07-01"
    test_end_date: str = "2024-12-31"
```

## 📚 Module Documentation

### `data/score_loader.py`

**ScoreLoader**: Load score.parquet files from MISO density data

```python
from src.data.score_loader import ScoreLoader

loader = ScoreLoader('/opt/temp/tmp/pw_data/spice6/prod_f0p_model_miso/density')

# Load single auction month
df = loader.load_auction_month('2024-01')

# Load date range
df = loader.load_date_range('2024-01-01', '2024-03-31')

# Get constraint statistics
stats = loader.get_coverage_stats(df)
print(stats.head())
```

### `data/shadow_price_loader.py`

**ShadowPriceLoader**: Load and aggregate shadow prices

```python
from src.data.shadow_price_loader import ShadowPriceLoader

loader = ShadowPriceLoader(
    get_da_shadow_func=get_da_shadow,
    aggregation_method='mean',  # 'mean', 'max', 'median', 'p95'
    binding_threshold=0.5
)

# Load shadow prices
shadow_df = loader.load_shadow_prices('2024-01-01', '2024-03-31')

# Aggregate for 3-day periods
result = loader.aggregate_for_score_data(score_df, shadow_df)

# Get binding statistics
stats = loader.get_binding_statistics(result)
print(stats)
```

### `data/dataset_builder.py`

**DatasetBuilder**: Build complete training dataset

```python
from src.data.dataset_builder import DatasetBuilder
from src.utils.config import Config

config = Config()
builder = DatasetBuilder(config=config, get_da_shadow_func=get_da_shadow)

# Build complete dataset
train_df, val_df, test_df = builder.build_dataset()

# Save datasets
builder.save_dataset(train_df, val_df, test_df)

# Get feature columns (excludes targets and metadata)
feature_cols = builder.get_feature_columns(train_df)
print(f"Features: {len(feature_cols)}")
```

### `features/temporal.py`

**add_temporal_features**: Add time-based features

```python
from src.features.temporal import add_temporal_features

df = add_temporal_features(
    df,
    timestamp_col='outage_date',
    use_cyclical=True  # Use sin/cos encoding
)

# Added features:
# - year, month, day, day_of_week, day_of_year, hour
# - hour_sin, hour_cos, dow_sin, dow_cos, month_sin, month_cos
# - is_weekend, is_peak, is_super_peak
# - season_winter, season_spring, season_summer, season_fall
```

## ⚠️ Important Notes

### Class Imbalance

**The data has severe class imbalance!**

```
Binding events: ~5-15% of data
Non-binding: ~85-95% of data
```

**When modeling:**
1. Use **two-stage approach** (classify → regress)
2. Use **class weights** in classifier
3. Evaluate with **F1 score, PR-AUC** (NOT accuracy!)
4. Optimize **binding detection recall** (catch profitable events)

### Temporal Ordering

**Always respect temporal ordering!**

- ✅ Train on past, validate/test on future
- ✅ Use TimeSeriesSplit for cross-validation
- ❌ Never shuffle time series data
- ❌ Don't use future information in features

### Data Alignment

**Score files and shadow prices must align:**

- Each score.parquet file → 3-day period starting from `outage_date`
- Shadow prices aggregated for same 3-day periods
- Join on: `constraint_id` + `outage_date`

## 🧪 Testing

```python
# Test score loader
from src.data.score_loader import load_scores

scores = load_scores(
    '/opt/temp/tmp/pw_data/spice6/prod_f0p_model_miso/density',
    '2024-01-01',
    '2024-01-31'
)
print(f"Loaded {len(scores)} score records")
print(f"Constraints: {scores['constraint_id'].nunique()}")

# Test shadow price loader (with mock function)
from src.data.shadow_price_loader import load_and_aggregate_shadow_prices

def mock_get_da_shadow(st, et, class_type):
    # Create mock data for testing
    dates = pd.date_range(st, et, freq='H')
    constraints = ['CON_001', 'CON_002']
    data = []
    for c in constraints:
        for d in dates:
            data.append({
                'timestamp': d,
                'constraint_id': c,
                'shadow_price': np.random.gamma(2, 10) if np.random.random() < 0.1 else 0.0
            })
    return pd.DataFrame(data)

result = load_and_aggregate_shadow_prices(
    scores,
    mock_get_da_shadow,
    '2024-01-01',
    '2024-01-31'
)
print(f"Binding rate: {(result['shadow_price_agg'] > 0.5).mean():.2%}")
```

## 📝 Example Workflow

```python
# Complete example: Load data and build training set

from src.data.dataset_builder import build_training_dataset
from src.utils.config import Config
from your_api import get_da_shadow  # Your actual function

# 1. Customize configuration
config = Config()
config.data.train_start_date = "2020-01-01"
config.data.train_end_date = "2023-12-31"

# 2. Build dataset
train_df, val_df, test_df = build_training_dataset(
    get_da_shadow_func=get_da_shadow,
    config=config,
    save=True
)

# 3. Inspect data
print(f"\nTraining Set:")
print(f"  Shape: {train_df.shape}")
print(f"  Binding rate: {train_df['target_binding'].mean():.2%}")
print(f"  Date range: {train_df['outage_date'].min()} to {train_df['outage_date'].max()}")

# 4. Get features for modeling
from src.data.dataset_builder import DatasetBuilder

builder = DatasetBuilder(config=config, get_da_shadow_func=get_da_shadow)
feature_cols = builder.get_feature_columns(train_df)

X_train = train_df[feature_cols]
y_train_binding = train_df['target_binding']
y_train_shadow = train_df['target_shadow_price']

print(f"\nFeature matrix: {X_train.shape}")
print(f"Target binding distribution: {y_train_binding.value_counts()}")
```

## 🔗 Next Steps

After building the training data:

1. **Train Two-Stage Model**:
   - Stage 1: LightGBM classifier with class weights
   - Stage 2: LightGBM regressor on binding events

2. **Evaluate Properly**:
   - F1 score for binding classification
   - PR-AUC (not ROC-AUC!)
   - MAE on binding events for regression

3. **Production Deployment**:
   - Save trained models with MLflow
   - Monitor performance drift
   - Retrain when binding rate changes

## 📖 Additional Resources

- Implementation plan: `document/miso_shadow_price_implementation_plan.md`
- Research methodology: `document/shadow_price_prediction_research_plan.md`
- Quick start: `document/GETTING_STARTED.md`
