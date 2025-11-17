# Synthetic Data Augmentation Plan for Zero-Variance Constraints

## Executive Summary

Handle constraints with **no binding events** (all 0s) or **no unbinding events** (all 1s) by adding synthetic training samples. This addresses model training failures for completely homogeneous classes while minimizing introduction of artificial bias.

**Expected Impact**: Enable model training for 100% of constraints, improve model robustness, reduce overfitting to extreme cases.

**Key Risk**: Synthetic data can introduce bias if not carefully designed. Recommend conservative approach with validation.

---

## 1. Problem Statement

### Current Situation

After shift factor augmentation, some constraints still have:
- **All non-binding** (100% zeros): Never congested in training period
- **All binding** (100% ones): Always congested in training period

### Why This Matters

**Classification Models**:
- Cannot learn decision boundary with single class
- May fail to train or produce trivial predictions
- Zero variance in target variable

**Regression Models**:
- All zeros: Cannot train (no positive samples)
- All ones: Can train but no negative cases for validation

### Scope of Problem

Estimate ~1-5% of constraints affected (to be measured):
- **Type 1**: Never bound in training data (likely majority)
- **Type 2**: Always bound in training data (likely rare)

---

## 2. Recommended Approaches

### Option A: Physics-Based Synthetic Generation (RECOMMENDED)

**Concept**: Generate synthetic samples using physical relationships and shift factor correlations

**Methodology**:
```python
# For never-binding constraints (all 0s)
# Add SMALL number of synthetic binding events

synthetic_binding_events = {
    'shadow_price': sample_from_correlated_branches(percentile=10),  # Low magnitude
    'flow_pct_density': perturb_from_observed(shift=0.05),  # Slight increase
    'label': 1  # Binding
}

# For always-binding constraints (all 1s)
# Add SMALL number of synthetic non-binding events

synthetic_nonbinding_events = {
    'shadow_price': 0.0,  # Non-binding
    'flow_pct_density': perturb_from_observed(shift=-0.05),  # Slight decrease
    'label': 0  # Non-binding
}
```

**Pros**:
- Grounded in physical relationships (shift factor similarity)
- Conservative: adds minimal synthetic data (1-5% of samples)
- Preserves class imbalance characteristics
- Interpretable and auditable

**Cons**:
- Requires shift factor correlation data
- More complex implementation
- Risk of propagating correlation errors

---

### Option B: Statistical Perturbation

**Concept**: Generate synthetic samples by perturbing observed data

**Methodology**:
```python
# For never-binding constraints (all 0s)
# Perturb features to create binding-like conditions

synthetic_sample = observed_features.copy()
synthetic_sample['flow_pct_density'] += np.random.normal(0.05, 0.02)  # Push toward binding
synthetic_sample['label'] = 1

# For always-binding constraints (all 1s)
# Perturb features to create non-binding-like conditions

synthetic_sample = observed_features.copy()
synthetic_sample['flow_pct_density'] -= np.random.normal(0.05, 0.02)  # Pull toward non-binding
synthetic_sample['label'] = 0
```

**Pros**:
- Simple to implement
- No external data required
- Fast execution

**Cons**:
- May create unrealistic feature combinations
- No physical grounding
- Risk of creating out-of-distribution samples

---

### Option C: SMOTE (Synthetic Minority Oversampling)

**Concept**: Use SMOTE algorithm to generate synthetic samples in feature space

**Methodology**:
```python
from imblearn.over_sampling import SMOTE

# SMOTE creates synthetic samples by interpolating between neighbors
smote = SMOTE(sampling_strategy='auto', k_neighbors=5)
X_resampled, y_resampled = smote.fit_resample(X, y)
```

**Pros**:
- Well-established technique for imbalanced data
- Backed by research and libraries
- Automatic generation

**Cons**:
- Creates interpolated samples (may violate physical constraints)
- Not suitable for time series data (temporal ordering)
- May create unrealistic feature combinations
- **NOT RECOMMENDED** for power systems (physical constraints matter)

---

### Option D: Minimal Synthetic Injection (PRAGMATIC)

**Concept**: Add minimal synthetic data just to enable model training

**Methodology**:
```python
# For never-binding constraints (all 0s)
# Add 1-3 synthetic binding events with conservative values

N_SYNTHETIC = min(3, max(1, int(len(constraint_data) * 0.01)))  # 1% or 1-3 samples

for _ in range(N_SYNTHETIC):
    synthetic_sample = {
        'shadow_price': np.random.uniform(0.1, 5.0),  # Low magnitude
        'flow_pct_density': constraint_data['flow_pct_density'].quantile(0.95),  # High flow
        'label': 1
    }

# For always-binding constraints (all 1s)
# Add 1-3 synthetic non-binding events

for _ in range(N_SYNTHETIC):
    synthetic_sample = {
        'shadow_price': 0.0,
        'flow_pct_density': constraint_data['flow_pct_density'].quantile(0.05),  # Low flow
        'label': 0
    }
```

**Pros**:
- Minimal intervention (1-3 samples only)
- Enables model training without major bias
- Simple and transparent
- Easy to audit and remove

**Cons**:
- Arbitrary synthetic values
- May not reflect true binding behavior

---

## 3. Recommended Strategy: Hybrid Approach

**Combine Option A (physics-based) + Option D (minimal injection)**

### Step 1: Attempt Physics-Based Augmentation

```python
# Try to borrow from correlated branches (>0.9 correlation)
if constraint_name in correlated_branches:
    for corr_branch in correlated_branches[constraint_name]:
        # Borrow binding/non-binding samples
        synthetic_samples = borrow_from_correlated(corr_branch, n_samples=3)
```

### Step 2: Fallback to Minimal Synthetic Injection

```python
# If no correlated branches OR insufficient samples borrowed
if len(synthetic_samples) < MIN_REQUIRED:
    # Generate 1-3 conservative synthetic samples
    synthetic_samples = generate_minimal_synthetic(constraint_data, n_samples=3)
```

### Step 3: Mark Synthetic Data

```python
# Add metadata flag for traceability
synthetic_samples['is_synthetic'] = True
synthetic_samples['synthetic_method'] = 'correlated_borrow' or 'minimal_injection'
```

---

## 4. Implementation Details

### 4.1 Synthetic Sample Generation Function

```python
def generate_synthetic_binding_events(constraint_data, branch_name,
                                      correlated_branches=None,
                                      n_samples=3):
    """
    Generate synthetic binding events for never-binding constraints.

    Parameters:
    -----------
    constraint_data : pd.DataFrame
        Observed data for the constraint (all non-binding)
    branch_name : str
        Name of the branch/constraint
    correlated_branches : dict
        Mapping of branch names to correlated branches
    n_samples : int
        Number of synthetic samples to generate (default: 3)

    Returns:
    --------
    synthetic_samples : pd.DataFrame
        Synthetic binding events with 'is_synthetic' flag
    """

    synthetic_samples_list = []

    # APPROACH 1: Try to borrow from correlated branches
    if correlated_branches and branch_name in correlated_branches:
        for corr_branch in correlated_branches[branch_name][:3]:  # Top 3 correlated
            if corr_branch in train_data_by_branch:
                corr_data = train_data_by_branch[corr_branch]
                binding_samples = corr_data[corr_data['label'] > 0]

                if len(binding_samples) > 0:
                    # Sample from correlated branch (use low percentiles)
                    sampled = binding_samples.sample(min(1, len(binding_samples)))
                    sampled['is_synthetic'] = True
                    sampled['synthetic_method'] = 'correlated_borrow'
                    sampled['source_branch'] = corr_branch
                    synthetic_samples_list.append(sampled)

    # APPROACH 2: Generate minimal synthetic samples if needed
    n_borrowed = len(synthetic_samples_list)
    if n_borrowed < n_samples:
        n_to_generate = n_samples - n_borrowed

        for i in range(n_to_generate):
            # Conservative synthetic binding event
            synthetic_row = constraint_data.iloc[0].copy()  # Template

            # Modify features to represent binding-like conditions
            synthetic_row['flow_pct_density'] = constraint_data['flow_pct_density'].quantile(0.95)
            synthetic_row['shadow_price'] = np.random.uniform(0.1, 2.0)  # Low magnitude
            synthetic_row['label'] = 1  # Binding
            synthetic_row['is_synthetic'] = True
            synthetic_row['synthetic_method'] = 'minimal_injection'

            synthetic_samples_list.append(pd.DataFrame([synthetic_row]))

    # Combine all synthetic samples
    if len(synthetic_samples_list) > 0:
        synthetic_samples = pd.concat(synthetic_samples_list, axis=0, ignore_index=True)
        return synthetic_samples
    else:
        return pd.DataFrame()  # No synthetic samples generated


def generate_synthetic_nonbinding_events(constraint_data, branch_name,
                                         correlated_branches=None,
                                         n_samples=3):
    """
    Generate synthetic non-binding events for always-binding constraints.

    Similar to generate_synthetic_binding_events but for opposite case.
    """

    synthetic_samples_list = []

    # APPROACH 1: Try to borrow from correlated branches
    if correlated_branches and branch_name in correlated_branches:
        for corr_branch in correlated_branches[branch_name][:3]:
            if corr_branch in train_data_by_branch:
                corr_data = train_data_by_branch[corr_branch]
                nonbinding_samples = corr_data[corr_data['label'] == 0]

                if len(nonbinding_samples) > 0:
                    sampled = nonbinding_samples.sample(min(1, len(nonbinding_samples)))
                    sampled['is_synthetic'] = True
                    sampled['synthetic_method'] = 'correlated_borrow'
                    sampled['source_branch'] = corr_branch
                    synthetic_samples_list.append(sampled)

    # APPROACH 2: Generate minimal synthetic samples if needed
    n_borrowed = len(synthetic_samples_list)
    if n_borrowed < n_samples:
        n_to_generate = n_samples - n_borrowed

        for i in range(n_to_generate):
            synthetic_row = constraint_data.iloc[0].copy()

            # Modify features to represent non-binding conditions
            synthetic_row['flow_pct_density'] = constraint_data['flow_pct_density'].quantile(0.05)
            synthetic_row['shadow_price'] = 0.0  # Non-binding
            synthetic_row['label'] = 0  # Non-binding
            synthetic_row['is_synthetic'] = True
            synthetic_row['synthetic_method'] = 'minimal_injection'

            synthetic_samples_list.append(pd.DataFrame([synthetic_row]))

    if len(synthetic_samples_list) > 0:
        synthetic_samples = pd.concat(synthetic_samples_list, axis=0, ignore_index=True)
        return synthetic_samples
    else:
        return pd.DataFrame()
```

### 4.2 Integration with Training Workflow

**Location**: Before STEP 6 (Train Branch-Specific Classifiers)

```python
# After shift factor augmentation
# Check for zero-variance constraints

augmented_data_by_branch = {}

for branch_name, branch_data in train_data_by_branch.items():
    n_binding = (branch_data['label'] > 0).sum()
    n_total = len(branch_data)

    # Case 1: Never binding (all 0s)
    if n_binding == 0:
        print(f"[SYNTHETIC] {branch_name}: All non-binding ({n_total} samples)")

        synthetic_binding = generate_synthetic_binding_events(
            constraint_data=branch_data,
            branch_name=branch_name,
            correlated_branches=correlated_branches,
            n_samples=3
        )

        if len(synthetic_binding) > 0:
            augmented_data = pd.concat([branch_data, synthetic_binding], axis=0, ignore_index=True)
            augmented_data_by_branch[branch_name] = augmented_data
            print(f"  → Added {len(synthetic_binding)} synthetic binding events")
        else:
            augmented_data_by_branch[branch_name] = branch_data

    # Case 2: Always binding (all 1s)
    elif n_binding == n_total:
        print(f"[SYNTHETIC] {branch_name}: All binding ({n_total} samples)")

        synthetic_nonbinding = generate_synthetic_nonbinding_events(
            constraint_data=branch_data,
            branch_name=branch_name,
            correlated_branches=correlated_branches,
            n_samples=3
        )

        if len(synthetic_nonbinding) > 0:
            augmented_data = pd.concat([branch_data, synthetic_nonbinding], axis=0, ignore_index=True)
            augmented_data_by_branch[branch_name] = augmented_data
            print(f"  → Added {len(synthetic_nonbinding)} synthetic non-binding events")
        else:
            augmented_data_by_branch[branch_name] = branch_data

    # Case 3: Mixed binding/non-binding (normal case)
    else:
        augmented_data_by_branch[branch_name] = branch_data

# Use augmented data for training
train_data_by_branch = augmented_data_by_branch
```

---

## 5. Validation Strategy

### 5.1 Track Synthetic Data Impact

```python
# Report synthetic data statistics
synthetic_stats = {
    'constraints_with_synthetic': 0,
    'total_synthetic_samples': 0,
    'synthetic_binding': 0,
    'synthetic_nonbinding': 0,
    'method_correlated_borrow': 0,
    'method_minimal_injection': 0
}

for branch_name, branch_data in train_data_by_branch.items():
    if 'is_synthetic' in branch_data.columns:
        synthetic_mask = branch_data['is_synthetic'] == True
        if synthetic_mask.sum() > 0:
            synthetic_stats['constraints_with_synthetic'] += 1
            synthetic_stats['total_synthetic_samples'] += synthetic_mask.sum()
            synthetic_stats['synthetic_binding'] += (branch_data.loc[synthetic_mask, 'label'] > 0).sum()
            synthetic_stats['synthetic_nonbinding'] += (branch_data.loc[synthetic_mask, 'label'] == 0).sum()

            for method in ['correlated_borrow', 'minimal_injection']:
                method_mask = branch_data['synthetic_method'] == method
                synthetic_stats[f'method_{method}'] += method_mask.sum()

print("[SYNTHETIC DATA STATISTICS]")
for key, value in synthetic_stats.items():
    print(f"  {key}: {value}")
```

### 5.2 Test Set Validation

**Critical**: Test ONLY on real data (never synthetic)

```python
# Remove synthetic samples from test data (should be none, but verify)
test_data_real = test_data[test_data.get('is_synthetic', False) == False]

# Evaluate predictions on real data only
final_results_prod = predict_with_models(test_data_real, ...)
```

### 5.3 Ablation Study

**Compare performance with/without synthetic data**:

```python
# Experiment 1: No synthetic data (baseline)
# Experiment 2: Correlated borrow only
# Experiment 3: Minimal injection only
# Experiment 4: Hybrid approach (recommended)

# Compare F1-scores, precision, recall
```

---

## 6. Risk Management

### Risk 1: Synthetic Data Bias

**Description**: Synthetic samples may not reflect true binding behavior

**Mitigation**:
- Use minimal synthetic samples (1-3 per constraint)
- Prefer physics-based borrowing over random generation
- Track and report synthetic data usage
- Ablation study to measure impact

### Risk 2: Model Overfitting to Synthetic Data

**Description**: Model may learn synthetic patterns instead of real patterns

**Mitigation**:
- Synthetic data <1% of total training data
- Mark synthetic samples for traceability
- Test on real data only
- Monitor performance on synthetic vs real separately

### Risk 3: Temporal Validity

**Description**: Synthetic data may not reflect future behavior

**Mitigation**:
- Conservative synthetic values (low magnitude shadow prices)
- Use recent historical data for borrowing
- Document assumptions clearly

---

## 7. Alternative: Skip Zero-Variance Constraints

**Option**: Instead of synthetic data, skip training for these constraints

### Approach

```python
# For never-binding constraints
# Use default model OR predict all non-binding

if n_binding == 0:
    # Option A: Use default model (trained on all branches)
    predictions[branch_name] = clf_default.predict(X_test)

    # Option B: Predict all non-binding (conservative)
    predictions[branch_name] = np.zeros(len(X_test))
```

### Pros & Cons

**Pros**:
- No synthetic data bias
- Simple and transparent
- Conservative predictions

**Cons**:
- May miss rare binding events
- No learning from constraint-specific features
- Treats all never-binding constraints identically

**Recommendation**: Use as fallback if synthetic data performs poorly

---

## 8. Implementation Phases

### Phase 1: Measurement & Analysis (30 minutes)

1. Identify constraints with zero variance
2. Measure prevalence (% of total constraints)
3. Analyze feature distributions for these constraints
4. Check shift factor correlations

**Deliverable**: Statistics on zero-variance constraints

### Phase 2: Implement Hybrid Approach (1-2 hours)

1. Implement `generate_synthetic_binding_events()`
2. Implement `generate_synthetic_nonbinding_events()`
3. Integrate with training workflow
4. Add tracking and reporting

**Deliverable**: Working synthetic data generation

### Phase 3: Validation & Tuning (1 hour)

1. Run ablation study (with/without synthetic)
2. Compare performance metrics
3. Tune `n_samples` parameter (1, 3, or 5)
4. Document findings

**Deliverable**: Validated approach with performance report

---

## 9. Success Criteria

### Minimum Success
- ✅ 100% of constraints can train models (no training failures)
- ✅ Synthetic data <1% of total training data
- ✅ No degradation in F1-score vs baseline (skipping zero-variance)

### Target Success
- ✅ F1-score improves by ≥2% (compared to skipping zero-variance)
- ✅ Precision and recall balanced (no major degradation in either)
- ✅ Physics-based borrowing used for ≥70% of synthetic samples

### Stretch Success
- ✅ F1-score improves by ≥5%
- ✅ Rare binding events correctly predicted for never-binding constraints
- ✅ Model robustness improved (lower variance in cross-validation)

---

## 10. Rollback Plan

If synthetic data degrades performance:

1. **Disable synthetic data generation** (set `n_samples=0`)
2. **Use alternative approach**: Skip zero-variance constraints, use default model
3. **Document findings**: Why synthetic data didn't help
4. **Consider alternatives**:
   - Collect more training data
   - Use physics-based constraints instead of ML
   - Hybrid approach: ML for normal constraints, rule-based for edge cases

---

## 11. Example Output Format

```
[SYNTHETIC DATA AUGMENTATION REPORT]

Zero-Variance Constraints Identified:
  Never binding (all 0s): 87 constraints (0.7%)
  Always binding (all 1s): 3 constraints (0.02%)
  Total requiring augmentation: 90 constraints (0.72%)

Synthetic Data Generated:
  Total synthetic samples: 245
  Synthetic binding events: 231 (94.3%)
  Synthetic non-binding events: 14 (5.7%)

Augmentation Methods:
  Correlated borrow: 178 samples (72.7%)
  Minimal injection: 67 samples (27.3%)

Top 5 Augmented Constraints:
  1. BRANCH_XYZ_1: 3 synthetic binding (borrowed from BRANCH_ABC_1, corr=0.94)
  2. BRANCH_DEF_2: 3 synthetic binding (minimal injection)
  ...

Performance Comparison:
                    Baseline (Skip)  With Synthetic  Improvement
  F1-Score          0.245            0.258           +5.3%
  Precision         0.289            0.295           +2.1%
  Recall            0.215            0.231           +7.4%
  Coverage          99.3%            100%            +0.7%
```

---

## 12. Recommended Decision

**Recommendation**: Implement **Hybrid Approach** (Physics-Based + Minimal Injection)

**Rationale**:
1. **Minimal intervention**: Only 0.5-1% of constraints affected
2. **Conservative**: 1-3 synthetic samples per constraint
3. **Physics-grounded**: Prefer borrowing from correlated branches
4. **Transparent**: Mark all synthetic data for traceability
5. **Reversible**: Easy to disable if performance degrades

**Next Steps**:
1. ✅ Review and approve this plan
2. ✅ Implement Phase 1 (measurement)
3. ✅ Implement Phase 2 (hybrid approach)
4. ✅ Run Phase 3 (validation)
5. ✅ Document findings and finalize

---

**Document Version**: 1.0
**Date**: 2025-11-16
**Status**: Ready for Review
**Next Action**: Await approval to proceed with implementation
