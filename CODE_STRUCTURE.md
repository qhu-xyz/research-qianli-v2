# Code Structure Overview

## 📂 Project Structure

```
research_spice_shadow_price_pred/
├── src/                           # Source code (production-ready)
│   ├── data/                      # Data loading and processing
│   │   ├── score_loader.py        # Load score.parquet files
│   │   ├── shadow_price_loader.py # Load & aggregate shadow prices
│   │   └── dataset_builder.py     # Build complete training dataset
│   ├── features/                  # Feature engineering
│   │   └── temporal.py            # Temporal features
│   ├── models/                    # ML models (TODO)
│   ├── evaluation/                # Metrics and evaluation (TODO)
│   ├── utils/                     # Utilities
│   │   └── config.py              # Configuration management
│   └── build_training_data.py     # Main script
├── examples/                      # Usage examples
│   └── build_dataset_example.py   # Complete example
├── document/                      # Documentation
│   ├── miso_shadow_price_implementation_plan.md  # Implementation plan
│   ├── shadow_price_prediction_research_plan.md  # Research methodology
│   ├── GETTING_STARTED.md         # Quick start guide
│   └── README.md                  # Documentation index
├── results/                       # Output directory
│   └── data/
│       └── processed/
│           ├── train.parquet      # Training data
│           ├── val.parquet        # Validation data
│           ├── test.parquet       # Test data
│           └── metadata.json      # Dataset statistics
├── QUICKSTART.md                  # Quick start guide
├── CODE_STRUCTURE.md              # This file
└── requirements.txt               # Python dependencies
```

## 🔄 Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│  INPUT DATA                                                     │
├─────────────────────────────────────────────────────────────────┤
│  1. Score.parquet files                                         │
│     /opt/temp/.../density/auction_month=*/market_month=*/       │
│                          market_round=*/outage_date=*/          │
│                          score.parquet                          │
│                                                                 │
│  2. Shadow Prices (via API)                                     │
│     get_da_shadow(st, et, class_type)                           │
│     → Returns: timestamp, constraint_id, shadow_price           │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  LOADING LAYER (src/data/)                                      │
├─────────────────────────────────────────────────────────────────┤
│  ScoreLoader.load_date_range()                                  │
│  ├─ Finds all score.parquet files in date range                │
│  ├─ Extracts partition info (auction_month, market_month, etc) │
│  └─ Returns: DataFrame with scores + partition columns          │
│                                                                 │
│  ShadowPriceLoader.load_shadow_prices()                         │
│  ├─ Calls get_da_shadow(st, et, class_type)                    │
│  ├─ Returns: DataFrame with hourly shadow prices                │
│  └─ Columns: timestamp, constraint_id, shadow_price            │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  AGGREGATION LAYER (src/data/)                                  │
├─────────────────────────────────────────────────────────────────┤
│  ShadowPriceLoader.aggregate_for_score_data()                   │
│  ├─ Groups shadow prices by 3-day periods (outage_date)        │
│  ├─ Aggregates: mean, max, median, p95                         │
│  ├─ Calculates: binding_frequency, n_hours                     │
│  └─ Joins with scores on: constraint_id + outage_date          │
│                                                                 │
│  Result: scores + aggregated shadow prices                      │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  FEATURE ENGINEERING LAYER (src/features/)                      │
├─────────────────────────────────────────────────────────────────┤
│  add_temporal_features()                                        │
│  ├─ Extracts: year, month, day, day_of_week, hour              │
│  ├─ Cyclical encoding: hour_sin/cos, dow_sin/cos, month_sin/cos│
│  ├─ Categoricals: is_weekend, is_peak, is_super_peak           │
│  └─ Season indicators: season_winter, season_spring, etc.      │
│                                                                 │
│  Constraint-level features:                                     │
│  ├─ historical_binding_rate                                     │
│  ├─ historical_mean_shadow                                      │
│  └─ historical_max_shadow                                       │
│                                                                 │
│  Result: Complete feature matrix (30+ features)                 │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  TARGET CREATION (src/data/)                                    │
├─────────────────────────────────────────────────────────────────┤
│  Create targets for two-stage model:                            │
│  ├─ target_binding = (shadow_price_agg > 0.5)                  │
│  └─ target_shadow_price = shadow_price_agg                     │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  DATASET SPLITTING (src/data/)                                  │
├─────────────────────────────────────────────────────────────────┤
│  Temporal split (NO shuffling!):                                │
│  ├─ Train: 2017-2023 (configurable)                            │
│  ├─ Val:   2024 H1                                              │
│  └─ Test:  2024 H2                                              │
│                                                                 │
│  Result: train_df, val_df, test_df                             │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  OUTPUT (results/data/processed/)                               │
├─────────────────────────────────────────────────────────────────┤
│  ├─ train.parquet      Training dataset                         │
│  ├─ val.parquet        Validation dataset                       │
│  ├─ test.parquet       Test dataset                             │
│  └─ metadata.json      Dataset statistics                       │
└─────────────────────────────────────────────────────────────────┘
```

## 🔧 Module Dependencies

```
build_training_data.py
    │
    ├─→ config.py (Config, DataConfig, FeatureConfig)
    │
    └─→ dataset_builder.py (DatasetBuilder)
            │
            ├─→ score_loader.py (ScoreLoader)
            │       │
            │       └─→ Reads: score.parquet files
            │
            ├─→ shadow_price_loader.py (ShadowPriceLoader)
            │       │
            │       └─→ Calls: get_da_shadow(st, et, class_type)
            │
            └─→ temporal.py (add_temporal_features)
```

## 🎯 Key Classes and Functions

### ScoreLoader (src/data/score_loader.py)

```python
loader = ScoreLoader(base_path)

# Load single auction month
df = loader.load_auction_month('2024-01')

# Load date range
df = loader.load_date_range('2024-01-01', '2024-03-31')

# Get statistics
stats = loader.get_coverage_stats(df)
```

**Methods**:
- `load_score_file()`: Load single score.parquet
- `load_auction_month()`: Load all scores for auction month
- `load_date_range()`: Load scores for date range
- `get_available_constraints()`: List unique constraints
- `get_coverage_stats()`: Data coverage by constraint

### ShadowPriceLoader (src/data/shadow_price_loader.py)

```python
loader = ShadowPriceLoader(
    get_da_shadow_func=get_da_shadow,
    aggregation_method='mean',
    binding_threshold=0.5
)

# Load shadow prices
shadow_df = loader.load_shadow_prices('2024-01-01', '2024-03-31')

# Aggregate for 3-day periods
result = loader.aggregate_for_score_data(score_df, shadow_df)
```

**Methods**:
- `load_shadow_prices()`: Fetch from API
- `aggregate_for_period()`: Aggregate single 3-day period
- `aggregate_for_score_data()`: Aggregate all periods in score data
- `get_binding_statistics()`: Binding stats by constraint

### DatasetBuilder (src/data/dataset_builder.py)

```python
builder = DatasetBuilder(config=config, get_da_shadow_func=get_da_shadow)

# Build complete dataset
train_df, val_df, test_df = builder.build_dataset()

# Save datasets
builder.save_dataset(train_df, val_df, test_df)

# Get feature columns
feature_cols = builder.get_feature_columns(train_df)
```

**Methods**:
- `load_scores()`: Load score features
- `load_shadow_prices()`: Load and aggregate shadow prices
- `add_features()`: Add temporal and constraint features
- `create_target_variables()`: Create target_binding and target_shadow_price
- `split_dataset()`: Create train/val/test splits
- `build_dataset()`: Complete pipeline
- `save_dataset()`: Save to parquet
- `get_feature_columns()`: Get feature column names

### Temporal Features (src/features/temporal.py)

```python
df = add_temporal_features(
    df,
    timestamp_col='outage_date',
    use_cyclical=True
)
```

**Functions**:
- `add_temporal_features()`: Main function (adds 20+ features)
- `add_lag_features()`: Time series lags
- `add_rolling_features()`: Rolling window stats

### Config (src/utils/config.py)

```python
from src.utils.config import Config

config = Config()

# Customize
config.data.train_start_date = "2023-01-01"
config.data.binding_threshold = 0.5
config.features.use_temporal_features = True
```

**Classes**:
- `DataConfig`: Data paths and loading settings
- `FeatureConfig`: Feature engineering settings
- `ModelConfig`: Model training settings
- `Config`: Main config container

## 📝 Data Schema

### Score Features (from score.parquet)
```
Columns from score.parquet:
- constraint_id
- [score feature columns - depends on your data]
- auction_month (added)
- market_month (added)
- market_round (added)
- outage_date (added)
```

### Shadow Price Aggregations (added by ShadowPriceLoader)
```
- shadow_price_agg        # Primary (mean/max/median/p95)
- shadow_price_mean       # Mean over 3-day period
- shadow_price_max        # Max over 3-day period
- shadow_price_median     # Median
- shadow_price_p95        # 95th percentile
- binding_frequency       # % hours binding
- n_hours                 # Number of hours in period
```

### Temporal Features (added by temporal.py)
```
- year, month, day, day_of_week, day_of_year, hour
- hour_sin, hour_cos      # Cyclical hour
- dow_sin, dow_cos        # Cyclical day of week
- month_sin, month_cos    # Cyclical month
- doy_sin, doy_cos        # Cyclical day of year
- is_weekend              # Boolean
- is_peak                 # MISO on-peak hours
- is_super_peak           # Summer peak hours
- season_winter, season_spring, season_summer, season_fall
- is_winter_morning       # Winter morning peak
- is_winter_evening       # Winter evening peak
- is_spring_wind          # Spring wind peak
```

### Constraint Features (added by DatasetBuilder)
```
- historical_binding_rate  # Historical % binding
- historical_mean_shadow   # Historical mean shadow price
- historical_max_shadow    # Historical max shadow price
```

### Target Variables
```
- target_binding          # Binary (0/1) for binding
- target_shadow_price     # Shadow price ($/MW)
```

## 🚀 Execution Flow

1. **Import and Setup**:
   ```python
   from src.data.dataset_builder import build_training_dataset
   from src.utils.config import Config
   config = Config()
   ```

2. **Load Scores**:
   - Scan directory for score.parquet files
   - Filter by date range
   - Load and concatenate

3. **Load Shadow Prices**:
   - Call `get_da_shadow(st, et, class_type)`
   - Get hourly shadow prices

4. **Aggregate**:
   - Group by constraint_id + outage_date (3-day periods)
   - Calculate mean, max, median, p95, binding_frequency

5. **Join**:
   - Merge scores with aggregated shadow prices
   - On: constraint_id + outage_date

6. **Engineer Features**:
   - Add temporal features (20+)
   - Add constraint-level features (3)

7. **Create Targets**:
   - target_binding: binary classification target
   - target_shadow_price: regression target

8. **Split**:
   - Temporal split (no shuffling!)
   - Train, validation, test

9. **Save**:
   - Write parquet files
   - Write metadata.json

## ⚡ Performance Notes

- **Loading**: Uses `glob` for fast file discovery
- **Progress**: Shows progress bars with `tqdm`
- **Memory**: Loads in batches, concatenates efficiently
- **Parquet**: Fast read/write with PyArrow
- **Indexing**: Optimized joins on constraint_id + date

## 🔐 Type Safety

All code includes:
- Type hints on function signatures
- Return type annotations
- Dataclass for configurations
- NumPy docstring format
