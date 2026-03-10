# Pipeline Optimization: Group by Auction Month

## Current Issue

The pipeline currently processes each `(auction_month, market_month)` pair independently, which means:

**Problem**: For the same auction month with multiple market months, models are trained multiple times unnecessarily.

### Example of Wasteful Computation:

```python
test_periods = [
    (Jul-2025, Sep-2025),  # q2 month 1
    (Jul-2025, Oct-2025),  # q2 month 2
    (Jul-2025, Nov-2025),  # q2 month 3
]
```

**Current behavior**:
1. Train models for Jul-2025 → Predict Sep-2025
2. **Train models for Jul-2025 again** → Predict Oct-2025 ❌
3. **Train models for Jul-2025 again** → Predict Nov-2025 ❌

**Result**: 3x training time for the same models!

## Proposed Solution

### Option A: Group Test Periods by Auction Month (Recommended)

Modify the pipeline to:
1. Group test periods by auction month
2. Train models once per auction month
3. Predict for all market months associated with that auction month

```python
# Group test periods
from collections import defaultdict

periods_by_auction = defaultdict(list)
for auction_month, market_month in test_periods:
    periods_by_auction[auction_month].append(market_month)

# Process each auction month
for auction_month, market_months in periods_by_auction.items():
    # Train once
    train_data = load_training_data(auction_month)
    models = train_models(train_data)

    # Predict for all market months
    for market_month in market_months:
        test_data = load_test_data(auction_month, market_month)
        predictions = models.predict(test_data)
        save_results(predictions, auction_month, market_month)
```

**Benefits**:
- ✅ Trains each model set only once
- ✅ Significant speedup for quarterly periods (3x faster)
- ✅ Lower memory usage (fewer model copies)
- ✅ Consistent models across market months

### Option B: Cache Models by Auction Month

Add a caching layer:
```python
model_cache = {}

for auction_month, market_month in test_periods:
    if auction_month not in model_cache:
        # Train and cache
        train_data = load_training_data(auction_month)
        model_cache[auction_month] = train_models(train_data)

    # Use cached models
    models = model_cache[auction_month]
    test_data = load_test_data(auction_month, market_month)
    predictions = models.predict(test_data)
```

**Benefits**:
- ✅ Minimal code changes
- ✅ Automatic caching
- ⚠️ Requires more memory (all models in cache)

## Impact Analysis

### For Quarterly Periods (q2, q3, q4)
Each quarterly period has **3 market months**:
- **Current**: 3 training runs per quarter
- **Optimized**: 1 training run per quarter
- **Speedup**: **3x faster**

### For Monthly Periods (f0, f1, f2, f3)
Each period has **1 market month**:
- **Current**: 1 training run
- **Optimized**: 1 training run
- **Speedup**: No change

### Overall Speedup
If you're testing a mix of periods:
```python
test_periods = [
    (Jul, Jul),    # f0 - 1 training
    (Aug, Sep),    # f1 - 1 training
    (Sep, Nov),    # f2 - 1 training
    (Jul, Sep),    # q2 - 3 trainings currently
    (Jul, Oct),    # q2 - (same as above)
    (Jul, Nov),    # q2 - (same as above)
]
```

- **Current**: 6 training runs
- **Optimized**: 4 training runs
- **Speedup**: 33% faster

## Implementation Recommendation

**Use Option A (Group by Auction Month)** because:
1. More efficient memory usage
2. Clearer code structure
3. Better parallelization opportunities (can parallelize by auction month)
4. Easier to debug (one training log per auction month)

## Code Changes Needed

### 1. Modify `pipeline.py` - `run()` method

```python
def run(self, test_periods, ...):
    # Group test periods by auction month
    from collections import defaultdict
    periods_by_auction = defaultdict(list)

    for auction_month, market_month in test_periods:
        periods_by_auction[auction_month].append(market_month)

    # Process each auction month
    all_results = []
    for auction_month, market_months in periods_by_auction.items():
        # Train once per auction month
        result = self._process_auction_month(
            auction_month,
            market_months,
            verbose,
            use_parallel
        )
        all_results.append(result)

    # Combine results
    return combine_results(all_results)
```

### 2. Create new method `_process_auction_month()`

```python
def _process_auction_month(
    self,
    auction_month,
    market_months,
    verbose,
    use_parallel
):
    # Load training data once
    train_data = self.data_loader.load_training_data(...)

    # Train models once
    models = ShadowPriceModels(self.config)
    models.train_classifiers(train_data, ...)
    models.train_regressors(train_data, ...)

    # Predict for all market months
    results = []
    for market_month in market_months:
        test_data = self.data_loader.load_test_data_for_period(
            auction_month, market_month
        )
        pred = models.predict(test_data)
        results.append(pred)

    return results
```

## Migration Path

1. **Phase 1**: Implement grouping logic (backward compatible)
2. **Phase 2**: Test with quarterly periods
3. **Phase 3**: Enable by default
4. **Phase 4**: Remove old per-period training code

## Verification

After implementation, verify:
- Training logs show 1 training per unique auction month
- Results are identical to before (same predictions)
- Speedup is achieved for quarterly periods
