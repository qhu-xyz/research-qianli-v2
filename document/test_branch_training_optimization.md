# Training Optimization: Test-Branch-Only Training

## Status: ✅ Already Implemented

The codebase **already implements** the optimization to only train branch-specific models for branches that appear in the test dataset.

## Implementation Details

### 1. Pipeline Extracts Test Branches
**File**: `src/shadow_price_prediction/pipeline.py`, lines 138-141

```python
# Extract unique branches from test data
test_branches = set(test_data["branch_name"].unique())

if verbose:
    print(f"  Found {len(test_branches)} unique branches in test set.")
```

### 2. Test Branches Passed to Training Methods
**File**: `src/shadow_price_prediction/pipeline.py`, line 148

```python
# Train classifiers with test_branches filter
period_models.train_classifiers(train_data, test_branches, verbose)

# Train regressors with test_branches filter
period_models.train_regressors(train_data, test_branches, verbose)
```

### 3. Classifier Training Filters by Test Branches
**File**: `src/shadow_price_prediction/models.py`, lines 410-413

```python
for branch_name, branch_data in train_data_by_branch.items():
    # Only train models for branches in test set
    if branch_name not in test_branches:
        skipped_count += 1
        continue
```

### 4. Regressor Training Filters by Test Branches
**File**: `src/shadow_price_prediction/models.py`, lines 533-536

```python
for branch_name, branch_data in train_data_by_branch.items():
    if branch_name not in test_branches:
        skipped_reg_count += 1
        continue
```

## Benefits

1. **Reduced Training Time**: Only trains models for branches that will actually be used
2. **Reduced Memory Usage**: Fewer models stored in memory
3. **Faster Prediction**: Smaller model dictionaries to search through
4. **No Wasted Computation**: Every trained model is guaranteed to be used

## Example Output

When training, you'll see messages like:

```
[Training Classification Models + Threshold Optimization]
--------------------------------------------------------------------------------
Training default fallback classifier on all training data...
  ✓ Default ensemble trained (2 models: XGBClassifier, LogisticRegression)
    Total samples: 150,000

Training branch-specific classifiers + optimizing thresholds...
  Total branches to train: 5,000

✓ Classification Training Complete
  Models trained: 1,200
  Optimal thresholds computed: 1,200
  Branches skipped: 3,800  ← These are branches NOT in test set
  Default optimal threshold: 0.450
```

The "Branches skipped" count includes:
- Branches not in test set (filtered out)
- Branches with insufficient samples (< `min_samples_for_branch_model`)

## Verification

To verify this is working correctly, you can check the verbose output during training:

1. **Before Training**: Note the number of unique branches in test set
2. **After Training**: Check "Models trained" count
3. **Skipped Count**: Should include all branches not in test set

The number of models trained should be ≤ number of test branches (some test branches might not have enough training samples).

## No Changes Needed

The optimization you requested is **already fully implemented** and working correctly. No modifications are necessary.
