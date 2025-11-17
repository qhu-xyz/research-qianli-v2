# Quick Start Guide - Build Training Data

This guide shows you how to use the code to build your training dataset from score.parquet files and shadow prices.

## 🎯 What You Have

Production-ready Python code to:
1. ✅ Load score.parquet files from `/opt/temp/tmp/pw_data/spice6/prod_f0p_model_miso/density/`
2. ✅ Fetch shadow prices using your `get_da_shadow(st, et, class_type)` function
3. ✅ Aggregate shadow prices for 3-day periods matching score data
4. ✅ Add temporal features (hour, day, month, season, peak indicators)
5. ✅ Create proper train/validation/test splits (respecting temporal ordering)
6. ✅ Handle class imbalance (85-95% of shadow prices are 0!)
7. ✅ Save ready-to-use parquet files for model training

## 🚀 Three Ways to Use the Code

### Option 1: Use the Example Script (Easiest)

```bash
# 1. Edit the example to use your get_da_shadow function
nano examples/build_dataset_example.py

# Replace this:
def get_da_shadow(st: str, et: str, class_type: str) -> pd.DataFrame:
    # Mock implementation
    ...

# With your actual function:
from your_module import get_da_shadow  # Your actual import

# 2. Adjust date ranges if needed (around line 72)
config.data.train_start_date = "2023-01-01"  # Change these
config.data.train_end_date = "2023-12-31"

# 3. Run the script
python examples/build_dataset_example.py
```

**Output**:
```
results/data/processed/
├── train.parquet      # Training data
├── val.parquet        # Validation data
├── test.parquet       # Test data
└── metadata.json      # Dataset statistics
```

### Option 2: Use as Python Module (Most Flexible)

```python
from src.data.dataset_builder import build_training_dataset
from src.utils.config import Config
from your_module import get_da_shadow  # Your shadow price API

# Optional: Customize configuration
config = Config()
config.data.train_start_date = "2023-01-01"
config.data.train_end_date = "2023-12-31"
config.data.val_start_date = "2024-01-01"
config.data.val_end_date = "2024-06-30"
config.data.test_start_date = "2024-07-01"
config.data.test_end_date = "2024-09-30"

# Build dataset
train_df, val_df, test_df = build_training_dataset(
    get_da_shadow_func=get_da_shadow,
    config=config,
    save=True  # Saves to results/data/processed/
)

# Use the data
print(f"Training: {len(train_df)} records")
print(f"Binding rate: {train_df['target_binding'].mean():.2%}")
```

### Option 3: Step-by-Step Control (Most Control)

```python
from src.data.score_loader import ScoreLoader
from src.data.shadow_price_loader import ShadowPriceLoader
from src.features.temporal import add_temporal_features
from your_module import get_da_shadow

# Step 1: Load scores
score_loader = ScoreLoader('/opt/temp/tmp/pw_data/spice6/prod_f0p_model_miso/density')
scores = score_loader.load_date_range('2024-01-01', '2024-03-31')

# Step 2: Load and aggregate shadow prices
shadow_loader = ShadowPriceLoader(
    get_da_shadow_func=get_da_shadow,
    aggregation_method='mean',
    binding_threshold=0.5
)
shadow_df = shadow_loader.load_shadow_prices('2024-01-01', '2024-03-31')
data = shadow_loader.aggregate_for_score_data(scores, shadow_df)

# Step 3: Add features
data = add_temporal_features(data, timestamp_col='outage_date')

# Step 4: Create targets
data['target_binding'] = (data['shadow_price_agg'] > 0.5).astype(int)
data['target_shadow_price'] = data['shadow_price_agg'].fillna(0.0)

# Step 5: Use the data
print(f"Total records: {len(data)}")
print(f"Binding rate: {data['target_binding'].mean():.2%}")
```

## 📊 What Data You Get

### Input: Score.parquet Files

From: `/opt/temp/tmp/pw_data/spice6/prod_f0p_model_miso/density/`

Structure:
```
density/
└── auction_month=2024-01/
    └── market_month=2024-01/
        └── market_round=1/
            └── outage_date=2024-01-01/
                ├── score.parquet    ← Your features
                ├── density.parquet
                └── limit.parquet
```

Each `outage_date` = Start of 3-day period

### Output: Training Dataset

**Target Variables**:
- `target_binding`: Binary (0/1) - Is constraint binding?
- `target_shadow_price`: Float - Shadow price in $/MW

**Features** (30+ columns):
- **Score features**: All columns from score.parquet
- **Temporal features**:
  - `hour`, `day_of_week`, `month`, `year`
  - `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos` (cyclical encoding)
  - `is_weekend`, `is_peak`, `is_super_peak`
  - `season_winter`, `season_spring`, `season_summer`, `season_fall`
- **Constraint features**:
  - `historical_binding_rate`: Historical binding frequency per constraint
  - `historical_mean_shadow`: Historical average shadow price
  - `historical_max_shadow`: Historical maximum shadow price
- **Shadow price aggregations** (for analysis):
  - `shadow_price_mean`, `shadow_price_max`, `shadow_price_median`, `shadow_price_p95`
  - `binding_frequency`: % of hours binding in 3-day period

**Splits** (temporal ordering preserved):
```
Train:      2017-2023  (default, adjust in config)
Validation: 2024 H1
Test:       2024 H2
```

## 🔧 Configuration Options

Edit `src/utils/config.py` or pass custom Config:

```python
from src.utils.config import Config

config = Config()

# Data paths
config.data.density_base_path = "/path/to/density"

# Shadow price settings
config.data.binding_threshold = 0.5  # $/MW threshold for "binding"
config.data.shadow_price_aggregation = "mean"  # or 'max', 'median', 'p95'

# Date ranges (adjust for your data)
config.data.train_start_date = "2020-01-01"
config.data.train_end_date = "2023-12-31"
config.data.val_start_date = "2024-01-01"
config.data.val_end_date = "2024-06-30"
config.data.test_start_date = "2024-07-01"
config.data.test_end_date = "2024-12-31"

# Features
config.features.use_temporal_features = True
config.features.use_cyclical_encoding = True
```

## ⚠️ Critical: Class Imbalance!

**Your data will have severe class imbalance:**

```
Binding events:     ~5-15% of data  ← minority class
Non-binding events: ~85-95% of data ← majority class
```

**When training models, you MUST:**

1. ✅ Use **class weights** in classification:
   ```python
   from lightgbm import LGBMClassifier

   clf = LGBMClassifier(
       class_weight='balanced',  # ← Critical!
       n_estimators=500,
       max_depth=10
   )
   ```

2. ✅ Evaluate with **F1 score, PR-AUC** (NOT accuracy!):
   ```python
   from sklearn.metrics import f1_score, average_precision_score

   f1 = f1_score(y_true, y_pred)  # Target: > 0.75
   pr_auc = average_precision_score(y_true, y_pred_proba)  # Target: > 0.75
   ```

3. ✅ Use **two-stage approach**:
   - Stage 1: Classify binding (with class weights)
   - Stage 2: Predict magnitude (only on binding events)

4. ❌ **Don't use accuracy** as primary metric (90% accuracy by always predicting 0!)

## 📝 Example Usage After Building Data

```python
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import f1_score, classification_report

# Load training data
train = pd.read_parquet('results/data/processed/train.parquet')
val = pd.read_parquet('results/data/processed/val.parquet')

# Get feature columns (exclude targets and metadata)
from src.data.dataset_builder import DatasetBuilder
builder = DatasetBuilder()
feature_cols = builder.get_feature_columns(train)

# Prepare data
X_train = train[feature_cols]
y_train_binding = train['target_binding']
y_train_shadow = train['target_shadow_price']

X_val = val[feature_cols]
y_val_binding = val['target_binding']

# Stage 1: Classify binding events
clf = LGBMClassifier(
    class_weight='balanced',  # ← Handles imbalance!
    n_estimators=500,
    max_depth=10,
    learning_rate=0.05
)
clf.fit(X_train, y_train_binding)

y_pred = clf.predict(X_val)
print(f"F1 Score: {f1_score(y_val_binding, y_pred):.3f}")
print("\nClassification Report:")
print(classification_report(y_val_binding, y_pred))

# Stage 2: Predict shadow price magnitude (on binding events)
binding_mask_train = y_train_binding == 1
X_train_binding = X_train[binding_mask_train]
y_train_magnitude = y_train_shadow[binding_mask_train]

reg = LGBMRegressor(n_estimators=500, max_depth=8)
reg.fit(X_train_binding, y_train_magnitude)

# Combined prediction
y_pred_proba = clf.predict_proba(X_val)[:, 1]
predicted_binding = y_pred_proba > 0.5
y_pred_shadow = np.zeros(len(X_val))
if predicted_binding.sum() > 0:
    y_pred_shadow[predicted_binding] = reg.predict(X_val[predicted_binding])

print(f"\nOverall MAE: {np.abs(val['target_shadow_price'] - y_pred_shadow).mean():.2f} $/MW")
```

## 🐛 Troubleshooting

### Error: "Base path does not exist"
```bash
# Check that path exists
ls /opt/temp/tmp/pw_data/spice6/prod_f0p_model_miso/density/

# If path is different, update config:
config.data.density_base_path = "/your/actual/path"
```

### Error: "No score files found"
```bash
# Check date range has data
ls /opt/temp/tmp/pw_data/spice6/prod_f0p_model_miso/density/auction_month=2024-01/

# Adjust date ranges in config to match available data
```

### Error: Shadow price function fails
```python
# Test your get_da_shadow function first
shadow_df = get_da_shadow('2024-01-01', '2024-01-31', 'constraint')
print(shadow_df.head())
print(shadow_df.columns)
# Expected columns: timestamp, constraint_id, shadow_price
```

### Low binding rate (<1%)
```python
# Check if you're filtering correctly
shadow_df = get_da_shadow('2024-01-01', '2024-01-31', 'constraint')
binding_rate = (shadow_df['shadow_price'] > 0.5).mean()
print(f"Binding rate: {binding_rate:.2%}")

# If too low, check:
# 1. Correct constraint class_type
# 2. Shadow price units ($/MW not $/MWh)
# 3. Binding threshold (try 0.1 or 0.01)
```

## 📚 Documentation

- **Code documentation**: `src/README.md`
- **Implementation plan**: `document/miso_shadow_price_implementation_plan.md`
- **Research methodology**: `document/shadow_price_prediction_research_plan.md`
- **Getting started**: `document/GETTING_STARTED.md`

## 🎯 Next Steps

1. **Build your dataset**:
   ```bash
   python examples/build_dataset_example.py
   ```

2. **Inspect the data**:
   ```python
   import pandas as pd
   train = pd.read_parquet('results/data/processed/train.parquet')
   print(train.info())
   print(train['target_binding'].value_counts())
   ```

3. **Train two-stage model**:
   - Follow example in `document/miso_shadow_price_implementation_plan.md` Section 2.4
   - Or use notebook in `notebooks/` (create one based on example above)

4. **Evaluate properly**:
   - F1 score > 0.75 for binding classification
   - PR-AUC > 0.75
   - MAE < 8 $/MW on binding events
   - Hit rate > 70% for trading simulation

Good luck! 🚀
