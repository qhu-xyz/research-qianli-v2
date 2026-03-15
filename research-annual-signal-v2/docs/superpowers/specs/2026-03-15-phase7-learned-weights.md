# Phase 7: Learned Weights — Linear/Logistic Models with Incremental Features

**Date**: 2026-03-15
**Status**: Draft
**Depends on**: M1-M2 (class-specific model table + v0c baseline)

---

## 1. Motivation

v0c uses hand-tuned weights (0.40/0.30/0.30) on 3 features. These weights were
chosen by intuition, not optimized. A simple learned model (logistic/linear regression)
could find better weights and incorporate more features without overfitting risk —
these are 3-10 parameter models, not 200-tree LightGBM.

The question: **does learning the weights improve over hand-tuning, and do additional
features help when weights are learned?**

## 2. Feature Ladder

Add features incrementally. Each step adds one feature group to the previous step.

### Step 1: v0c features, learned weights (3 features)

Same inputs as v0c, but weights learned by logistic regression instead of 0.40/0.30/0.30.

| Feature | Source | Class-specific? |
|---|---|---|
| `da_rank_value` | History (DA) | Yes |
| `rt_max` | Density (SPICE) | No |
| `bf` (`bf_12` / `bfo_12`) | History (binding freq) | Yes |

### Step 2: + shadow_price_da (4 features)

Raw historical DA congestion value (not just the rank). Gives the model access to
magnitude, not just ordering.

| Added | Source |
|---|---|
| `shadow_price_da` | History (DA), class-specific |

### Step 3: + density scores (6 features)

Forward-looking density features from SPICE simulation.

| Added | Source |
|---|---|
| `ori_mean` | SPICE density (class-agnostic) |
| `count_active_cids` | SPICE density (structural) |

### Step 4: + cross-class BF (7 features)

The other class's binding frequency — does cross-class history help when weights
are learned?

| Added | Source |
|---|---|
| `cross_class_bf` | History (other class BF) |

### Step 5: + more density bins (10 features)

Additional density bin features that the NB model found useful.

| Added | Source |
|---|---|
| `bin_60_cid_max` | SPICE density |
| `bin_70_cid_max` | SPICE density |
| `bin_120_cid_max` | SPICE density |

### Step 6: + limits (12 features)

Constraint MW limits — structural features.

| Added | Source |
|---|---|
| `limit_min` | SPICE density |
| `limit_mean` | SPICE density |

## 3. Models per Step

For each feature step, train and evaluate:

| Model | Target | Method |
|---|---|---|
| **Logistic** | `SP > 0` (binary) | `LogisticRegression(C=1.0, class_weight="balanced")` |
| **Ridge** | `log1p(SP)` (continuous) | `Ridge(alpha=1.0)` |
| **v0c formula** | N/A (hand-tuned) | Only at Step 1 for comparison |

Both are simple, few-parameter models. Logistic has 1 weight per feature + intercept.
Ridge same. No overfitting risk with 3-12 features on 800+ branches.

**NOT tested**: LightGBM, random forest, neural nets. Those were already tested in
Phase 4-5 and showed overfitting on the dormant subpopulation. The point of Phase 7
is simple learned models, not complex ones.

## 4. Evaluation

Same as M2:
- K=150/200/300/400
- Metrics: VC, Recall, NB12_SP, Dg20k_Recall, Dg40k_Recall
- Dev (12 groups) + holdout (3 groups)
- Per class_type (onpeak, offpeak)
- Expanding window (train on prior PYs, eval on target PY)

**Key comparison**: does any learned model consistently beat v0c across ALL K levels?
Not just on one K or one metric — across the board.

## 5. Success Criteria

A Phase 7 model is promoted if on holdout:

1. **VC@K ≥ v0c VC@K - 0.01** at every K level (no regression)
2. **At least one metric strictly improves** (NB12_SP, DangR, or Recall)
3. **Consistent across both class types** (wins onpeak AND offpeak)

If no model meets all 3, v0c remains the champion and Phase 7 is a negative result
(confirming the formula weights are already near-optimal).

## 6. Implementation

### Script

`scripts/phase6/run_learned_weights.py` — one script, runs all steps × models × classes.

### Output

`registry/{onpeak,offpeak}/m2_phase7/results.json` — per-step, per-model metrics.

### Runtime

~2 minutes per class (logistic + ridge are instant; data loading is the bottleneck).
Total: ~5 minutes for both classes × dev + holdout.
