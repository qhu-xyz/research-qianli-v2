# Dynamic Threshold Optimization Plan

## Executive Summary

Replace fixed 0.5 classification threshold with **optimized thresholds** that maximize F1-score on training data. This will improve binding detection accuracy by accounting for severe class imbalance (97% non-binding constraints).

**Expected Impact**: 5-15% improvement in F1-score, better balance between precision and recall.

---

## 1. Problem Statement

### Current Approach
- **Default threshold**: 0.5 (from `clf.predict()`)
- **Issue**: Suboptimal for imbalanced datasets
  - 97% of constraints are non-binding
  - Fixed 0.5 threshold may miss binding events (low recall)
  - OR produce too many false alarms (low precision)

### Proposed Solution
- **Find optimal threshold** that maximizes F1-score on training data
- **Store threshold** with each model
- **Apply threshold** consistently during prediction

---

## 2. Optimization Strategy

### Option A: Global Threshold (Simpler)
- **One threshold** for all branches
- Find optimal threshold on **all training data** using default classifier
- **Pros**: Simple, fast, consistent
- **Cons**: Ignores branch-specific characteristics

### Option B: Branch-Specific Thresholds (Recommended)
- **One threshold per branch**
- Find optimal threshold for each branch using its training data
- **Pros**: Accounts for branch-specific binding patterns
- **Cons**: More complex, requires sufficient binding samples per branch

### Recommendation: **Hybrid Approach**
- Use **branch-specific thresholds** where training data is sufficient (≥20 samples with mixed classes)
- Fall back to **global threshold** for branches with insufficient data

---

## 3. Methodology

### Step 1: Threshold Search on Training Data

For each branch (or globally):

```python
# After training classifier
y_proba_train = clf.predict_proba(X_train)[:, 1]
y_true_train = y_train_binary

# Search thresholds from 0.01 to 0.99
thresholds = np.linspace(0.01, 0.99, 99)
f1_scores = []

for threshold in thresholds:
    y_pred = (y_proba_train >= threshold).astype(int)
    f1 = f1_score(y_true_train, y_pred, zero_division=0)
    f1_scores.append(f1)

# Find threshold that maximizes F1
optimal_threshold = thresholds[np.argmax(f1_scores)]
max_f1 = max(f1_scores)
```

### Step 2: Store Optimal Thresholds

```python
# Store with each model
optimal_thresholds = {}  # {branch_name: threshold}

# For branch-specific
optimal_thresholds[branch_name] = optimal_threshold

# For default/global
optimal_threshold_default = 0.5  # or optimized global threshold
```

### Step 3: Apply During Prediction

```python
# Get probabilities
y_proba = clf.predict_proba(X_test)[:, 1]

# Apply optimized threshold
threshold = optimal_thresholds.get(branch_name, optimal_threshold_default)
y_pred = (y_proba >= threshold).astype(int)
```

---

## 4. Implementation Details

### 4.1 Data Structures

```python
# Global storage
optimal_thresholds = {}  # {branch_name: float}
optimal_threshold_default = 0.5  # fallback

# Per-branch storage (alternative)
class BranchModel:
    def __init__(self):
        self.classifier = None
        self.regressor = None
        self.optimal_threshold = 0.5
        self.threshold_f1_score = 0.0
```

### 4.2 Training Phase Modifications

**Location**: STEP 6 - Train Branch-Specific Classifiers

```python
# After: clf_branch.fit(X_branch, y_branch_binary)

# Add threshold optimization
y_proba_train = clf_branch.predict_proba(X_branch)[:, 1]
optimal_threshold, max_f1 = find_optimal_threshold(
    y_true=y_branch_binary,
    y_proba=y_proba_train,
    metric='f1'
)

# Store threshold
optimal_thresholds[branch_name] = optimal_threshold

# Track statistics
threshold_stats[branch_name] = {
    'optimal_threshold': optimal_threshold,
    'train_f1_at_optimal': max_f1,
    'train_f1_at_0.5': f1_score(y_branch_binary, (y_proba_train >= 0.5).astype(int))
}
```

### 4.3 Prediction Phase Modifications

**Location**: STEP 8 - Make Predictions

```python
# Current:
y_pred_binary_branch = clf_to_use.predict(X_branch)

# Modified:
y_pred_proba_branch = clf_to_use.predict_proba(X_branch)[:, 1]

# Get optimal threshold for this branch
threshold = optimal_thresholds.get(branch_name, optimal_threshold_default)

# Apply threshold
y_pred_binary_branch = (y_pred_proba_branch >= threshold).astype(int)
```

---

## 5. Validation Strategy

### 5.1 Training Set Analysis

For each branch, report:
- Optimal threshold value
- F1-score at optimal threshold (training data)
- F1-score at 0.5 threshold (training data)
- Improvement: `(F1_optimal - F1_0.5) / F1_0.5 * 100`

### 5.2 Test Set Evaluation

Compare performance with/without optimization:

| Metric | Fixed Threshold (0.5) | Optimized Threshold | Improvement |
|--------|----------------------|---------------------|-------------|
| F1-Score | ? | ? | ? |
| Precision | ? | ? | ? |
| Recall | ? | ? | ? |
| Accuracy | ? | ? | ? |

### 5.3 Threshold Distribution Analysis

Analyze optimal threshold distribution:
- Mean, median, std of optimal thresholds across branches
- Histogram of optimal thresholds
- Correlation between threshold and binding rate

---

## 6. Edge Cases & Fallback Logic

### Case 1: Insufficient Training Data
**Condition**: Branch has <20 training samples
**Action**: Use global default threshold (0.5 or optimized global)

### Case 2: All Same Class in Training
**Condition**: All binding OR all non-binding in training
**Action**: Use global default threshold

### Case 3: F1-Score is 0 for All Thresholds
**Condition**: Perfect class separation OR no overlap
**Action**: Use threshold that minimizes classification error

### Case 4: Multiple Thresholds Yield Same Max F1
**Condition**: Flat F1 curve
**Action**: Choose median threshold among tied values

---

## 7. Alternative Optimization Metrics

While F1-score is recommended, the framework should support:

### Option 1: F1-Score (Default)
- **Use Case**: Balanced precision and recall
- **Formula**: `2 * (precision * recall) / (precision + recall)`

### Option 2: F-Beta Score
- **Use Case**: Custom precision/recall weight
- **Formula**: `(1 + β²) * (precision * recall) / (β² * precision + recall)`
- **Example**: β=2 emphasizes recall (catch more binding events)

### Option 3: Youden's J Statistic
- **Use Case**: Maximize true positive rate while minimizing false positive rate
- **Formula**: `sensitivity + specificity - 1`

### Option 4: Cost-Based Threshold
- **Use Case**: Minimize economic cost
- **Formula**: Minimize `(FN_cost * FN + FP_cost * FP)`
- **Example**: Missing binding event costs more than false alarm

---

## 8. Implementation Phases

### Phase 1: Prototype (Global Threshold)
1. Implement `find_optimal_threshold()` function
2. Find optimal threshold on all training data (using default classifier)
3. Apply to all branches during prediction
4. Evaluate improvement vs. fixed 0.5 threshold

**Estimated Time**: 30 minutes
**Risk**: Low

### Phase 2: Branch-Specific Thresholds
1. Modify STEP 6 to optimize threshold per branch
2. Store thresholds in dictionary: `{branch_name: threshold}`
3. Modify STEP 8 to lookup and apply branch-specific thresholds
4. Add fallback logic for insufficient data

**Estimated Time**: 1 hour
**Risk**: Medium (need robust fallback logic)

### Phase 3: Visualization & Analysis
1. Create threshold distribution histogram
2. Show F1 improvement by branch
3. Analyze threshold vs. binding rate correlation
4. Add diagnostic cell showing top/bottom performers

**Estimated Time**: 30 minutes
**Risk**: Low

---

## 9. Code Modifications

### New Functions Needed

```python
def find_optimal_threshold(y_true, y_proba, metric='f1', thresholds=None):
    """
    Find optimal classification threshold.

    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_proba : array-like
        Predicted probabilities for positive class
    metric : str
        Optimization metric ('f1', 'f_beta', 'youden', 'cost')
    thresholds : array-like, optional
        Threshold values to search (default: np.linspace(0.01, 0.99, 99))

    Returns:
    --------
    optimal_threshold : float
        Threshold that maximizes the metric
    max_metric_value : float
        Maximum metric value achieved
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    metric_values = []
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)

        if metric == 'f1':
            value = f1_score(y_true, y_pred, zero_division=0)
        elif metric == 'youden':
            # Sensitivity + Specificity - 1
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            value = sensitivity + specificity - 1
        # ... other metrics

        metric_values.append(value)

    optimal_idx = np.argmax(metric_values)
    optimal_threshold = thresholds[optimal_idx]
    max_metric_value = metric_values[optimal_idx]

    return optimal_threshold, max_metric_value
```

### Cells to Modify

1. **STEP 6** (Train Classifiers): Add threshold optimization after training
2. **STEP 7** (Train Regressors): No changes needed
3. **STEP 8** (Predictions): Replace `predict()` with `predict_proba()` + threshold
4. **New Cell** (after STEP 7): Diagnostic cell analyzing threshold distribution

---

## 10. Expected Results

### Baseline (Fixed 0.5 Threshold)
- F1-Score: ~0.15-0.30 (typical for imbalanced data)
- Precision: ~0.20-0.40
- Recall: ~0.10-0.25

### After Optimization (Predicted)
- F1-Score: **+5-15% improvement**
- Precision: May increase or decrease (trade-off with recall)
- Recall: May increase or decrease (trade-off with precision)
- **Overall**: Better balance for imbalanced data

### Key Insights Expected
- Most optimal thresholds will be **< 0.5** (due to class imbalance)
- Typical range: **0.1 - 0.4** for imbalanced datasets
- Branches with higher binding rates → higher optimal thresholds
- Branches with lower binding rates → lower optimal thresholds

---

## 11. Risks & Mitigation

### Risk 1: Overfitting to Training Data
**Description**: Optimal threshold on training may not generalize
**Mitigation**:
- Use cross-validation for threshold optimization
- Monitor test set performance
- Consider using validation set if available

### Risk 2: Insufficient Data for Optimization
**Description**: Branches with <20 samples can't be optimized reliably
**Mitigation**:
- Use global fallback threshold
- Require minimum samples for optimization
- Report which branches use fallback

### Risk 3: Instability Across Runs
**Description**: Optimal threshold may vary with different data splits
**Mitigation**:
- Use stable optimization (average over multiple folds)
- Set minimum improvement threshold (e.g., >2% F1 improvement)
- Default to 0.5 if optimization doesn't help

---

## 12. Success Criteria

### Minimum Success
- ✅ F1-score improves by ≥2% on aggregated test data
- ✅ No degradation in MAE/RMSE for shadow price prediction
- ✅ Code runs without errors

### Target Success
- ✅ F1-score improves by ≥5% on aggregated test data
- ✅ Precision AND recall both improve (or one improves significantly)
- ✅ Branch-specific thresholds provide additional gains over global

### Stretch Success
- ✅ F1-score improves by ≥10%
- ✅ Actionable insights about threshold distribution
- ✅ Clear correlation between binding rate and optimal threshold

---

## 13. Rollback Plan

If optimization doesn't improve results:

1. **Keep code structure** but set all thresholds to 0.5
2. **Add flag**: `USE_OPTIMIZED_THRESHOLDS = False`
3. **Document findings**: Why optimization didn't help
4. **Alternative approaches**: Try different metrics or cost-based optimization

---

## 14. Next Steps

### Immediate Actions
1. ✅ Review and approve this plan
2. ✅ Choose approach: Global, Branch-Specific, or Hybrid
3. ✅ Decide on optimization metric: F1 (recommended) or alternative

### Implementation Sequence
1. **Step 1**: Create `find_optimal_threshold()` function
2. **Step 2**: Test on small subset (1-2 branches)
3. **Step 3**: Implement full pipeline (STEP 6 + STEP 8)
4. **Step 4**: Add diagnostics and visualization
5. **Step 5**: Run full test and compare results
6. **Step 6**: Document findings and finalize

---

## 15. Questions for Decision

Before proceeding with implementation, please confirm:

1. **Approach**: Branch-specific, Global, or Hybrid?
   - **Recommendation**: Hybrid (branch-specific with global fallback)

2. **Metric**: F1-score or alternative?
   - **Recommendation**: F1-score (balanced precision/recall)

3. **Minimum samples**: How many samples required for branch-specific optimization?
   - **Recommendation**: 20 samples minimum

4. **Fallback threshold**: Use 0.5 or optimize global threshold?
   - **Recommendation**: Optimize global threshold as fallback

5. **Implementation phase**: Start with global prototype or go directly to branch-specific?
   - **Recommendation**: Start with Phase 1 (global) to validate approach, then Phase 2

---

## Appendix A: Mathematical Background

### F1-Score Optimization

Given:
- True positives: TP(t) = count where y_true=1 AND y_pred(t)=1
- False positives: FP(t) = count where y_true=0 AND y_pred(t)=1
- False negatives: FN(t) = count where y_true=1 AND y_pred(t)=0

Where y_pred(t) = 1 if y_proba ≥ t, else 0

Then:
```
Precision(t) = TP(t) / (TP(t) + FP(t))
Recall(t) = TP(t) / (TP(t) + FN(t))
F1(t) = 2 * Precision(t) * Recall(t) / (Precision(t) + Recall(t))
```

Optimal threshold:
```
t* = argmax_{t ∈ [0,1]} F1(t)
```

### Why F1-Score for Imbalanced Data?

- **Accuracy** misleading when 97% are non-binding (predicting all 0 → 97% accuracy!)
- **Precision** alone: May classify very conservatively → high precision, low recall
- **Recall** alone: May classify too aggressively → high recall, low precision
- **F1-Score**: Harmonic mean balances both → optimal for imbalanced classes

---

## Appendix B: Example Output Format

```
[THRESHOLD OPTIMIZATION RESULTS]

Global Default Threshold:
  Optimized from all training data: 0.23
  F1-score at 0.5: 0.185
  F1-score at 0.23: 0.267
  Improvement: +44.3%

Branch-Specific Thresholds:
  Total branches: 12,453
  Branches with optimized thresholds: 3,127 (25.1%)
  Branches using default threshold: 9,326 (74.9%)

Threshold Distribution:
  Mean: 0.28
  Median: 0.25
  Std: 0.12
  Min: 0.08
  Max: 0.73

Top 5 Branches by F1 Improvement:
  1. ANTA_ANITA16_1 1: 0.15 → 0.42 (+180%, threshold=0.18)
  2. GRAY XFMR_2: 0.22 → 0.51 (+132%, threshold=0.21)
  ...

Test Set Performance Comparison:
                    Fixed (0.5)  Optimized  Improvement
  F1-Score          0.189        0.245      +29.6%
  Precision         0.312        0.289      -7.4%
  Recall            0.138        0.215      +55.8%
  Accuracy          0.971        0.968      -0.3%
```

---

**Document Version**: 1.0
**Date**: 2025-11-16
**Status**: Ready for Review
**Next Action**: Await approval to proceed with implementation
