# Phase 7: Learned Weights — Linear/Logistic Models with Incremental Features

**Date**: 2026-03-15
**Status**: Draft (rev 2 — review fixes)
**Depends on**: M1-M2 (class-specific model table + v0c baseline)

---

## 1. Motivation

v0c uses hand-tuned weights (0.40/0.30/0.30) on 3 features. These weights were
chosen by intuition, not optimized. A simple learned model (logistic/linear regression)
could find better weights and incorporate more features — these are 3-12 parameter
models with low overfitting risk, though not zero (3 holdout groups, feature
collinearity between shadow_price_da and da_rank_value).

The question: **does learning the weights improve over hand-tuning, and do additional
features help when weights are learned?**

## 2. Feature Normalization Contract

**All features are min-max normalized within each (PY, aq) group before training.**
This matches v0c's per-group normalization in `scoring.py`. Without this, a pooled
model would learn scale artifacts across groups rather than meaningful weights.

```python
# Per group: normalize each feature to [0, 1]
for feat in feature_cols:
    mn, mx = group_df[feat].min(), group_df[feat].max()
    if mx > mn:
        group_df[feat] = (group_df[feat] - mn) / (mx - mn)
    else:
        group_df[feat] = 0.5
```

Training pools normalized rows from all training PY groups. Evaluation uses
per-group normalization on each eval group independently.

## 3. Feature Ladder

Add features incrementally. Each step adds one feature group to the previous step.
All features listed are available in the current Phase 6 model table unless noted.

### Step 1: v0c features, learned weights (3 features)

Same normalized inputs as v0c, but weights learned instead of 0.40/0.30/0.30.

| Feature | Source | In model table? |
|---|---|---|
| `da_rank_value` | History (DA) | Yes (class-specific) |
| `rt_max` = `max(bin_80..110_cid_max)` | Density (SPICE) | Derived from existing columns |
| `bf` (`bf_12` / `bfo_12`) | History (binding freq) | Yes (class-specific) |

### Step 2: + shadow_price_da (4 features)

Raw historical DA congestion value (not just the rank). Gives the model access to
magnitude, not just ordering. Note: collinear with da_rank_value — ridge handles this.

| Added | In model table? |
|---|---|
| `shadow_price_da` | Yes (class-specific) |

### Step 3: + density bins (7 features)

Additional density bin features already in the model table.

| Added | In model table? |
|---|---|
| `bin_60_cid_max` | Yes |
| `bin_70_cid_max` | Yes |
| `bin_120_cid_max` | Yes |

Note: `ori_mean` and `mix_mean` are NOT in the current Phase 6 model table
(they exist in V6.1 signal but not in our density pipeline output). Excluded
from the ladder to avoid adding a feature-engineering step.

### Step 4: + structural features (9 features)

| Added | In model table? |
|---|---|
| `count_active_cids` | Yes |
| `limit_mean` | Yes |

### Step 5: + cross-class features (11 features)

The other class's BF and shadow_price_da.

| Added | In model table? |
|---|---|
| `cross_class_bf` | Yes |
| `shadow_price_da` (other class) | Needs: join other class's spda. See §7. |

### Step 6: + counter-flow density (13 features)

| Added | In model table? |
|---|---|
| `bin_-50_cid_max` | Yes |
| `bin_-100_cid_max` | Yes |

## 4. Models per Step

For each feature step, train and evaluate:

| Model | Target | Method |
|---|---|---|
| **Logistic** | `SP > 0` (binary) | `LogisticRegression(C=1.0, class_weight="balanced")` |
| **Ridge** | `log1p(SP)` (continuous) | `Ridge(alpha=1.0)` |
| **v0c formula** | N/A (hand-tuned) | Only at Step 1 for comparison |

Both are simple, few-parameter models. Logistic has 1 weight per feature + intercept.
Ridge same. Overfitting risk is low but not zero — 3 holdout groups and collinear
features (shadow_price_da ↔ da_rank_value) mean we should not over-interpret small
holdout improvements.

## 5. Evaluation

Same framework as M2:
- K=150/200/300/400
- Metrics: VC, Recall, NB12_SP, Dg20k_Recall, Dg40k_Recall
- Dev (12 groups) + holdout (3 groups)
- Per class_type (onpeak, offpeak)
- Expanding window (train on prior PYs, eval on target PY)
- **Paired scorecard** per class_type (same as Phase 6 design at
  `2026-03-15-class-specific-pipeline-design.md`)

## 6. Success Criteria

Uses the **same paired scorecard** as Phase 6 M2. A Phase 7 model is promoted if
on holdout:

1. **Paired score > v0c paired score** for the same class_type
2. **VC@K ≥ v0c VC@K - 0.02** at each K in the pair (max 2% regression)
3. **Dg20k_Recall@K ≥ v0c Dg20k_Recall@K - 0.05** at each K (max 5% regression)

If no model meets all 3, v0c remains the champion and Phase 7 is a negative result
(confirming the formula weights are already near-optimal for these features).

## 7. Implementation Notes

### Step 5 cross-class shadow_price_da

Step 5 adds the other class's `shadow_price_da` as a feature. This requires building
both class tables and cross-joining the shadow_price_da column:

```python
mt_on = build_class_model_table(py, aq, "onpeak")
mt_off = build_class_model_table(py, aq, "offpeak")
# For onpeak model: add offpeak shadow_price_da
mt_on = mt_on.join(
    mt_off.select(["branch_name", pl.col("shadow_price_da").alias("cross_shadow_price_da")]),
    on="branch_name", how="left"
)
```

### Script

`scripts/phase6/run_learned_weights.py`

### Output

`registry/{onpeak,offpeak}/phase7/results.json`
