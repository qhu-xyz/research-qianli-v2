# Evaluation Metrics: Classification & Regression

> **Last updated**: 2026-02-20
> **Applies to**: Two-stage shadow price prediction pipeline

---

## 1. What Is a "Constraint" and What Is a "Match"?

### Identity hierarchy

```
constraint_id          = "monitorLine : contingency"     (MISO's raw identifier)
                          e.g., "AEC.JOPPA3451UNIT3-TOP : AEC.COFFN1-MARKE1"

branch_name            = spice_map[constraint_id]         (mapped from constraint_info)
                          e.g., "AEC.JOPPA3451UNIT3-TOP"
                          (one branch_name can have multiple constraint_ids)

flow_direction         = {0, 1}                           (direction of power flow)

Unique constraint key  = (constraint_id, flow_direction)  — per outage date
Unique branch key      = (branch_name, flow_direction)    — aggregated across outage dates
```

### How matching works in this pipeline

The pipeline operates at the **constraint_id level per outage date**. Each row in the results DataFrame is one `(constraint_id, outage_date)` pair. The model predicts:

1. **Will this constraint bind?** — binary {0, 1}
2. **If yes, what shadow price?** — continuous $/MWh

A **match** (true positive) is:
- The model predicts binding (`predicted_binding = 1`)
- The actual outcome is binding (`actual_binding = 1`, i.e., `label > 0`)
- Both are for the **same `constraint_id`** on the **same `outage_date`**

This is a **strict match** — the `constraint_id` (which is `monitorLine:contingency`) must be identical, not just the `branch_name`. However, for aggregated metrics (monthly level, constraint-level ValCap), we group by `(branch_name, flow_direction)` and use:
- `actual_shadow_price`: SUM across outage dates
- `binding_probability`: MAX across outage dates
- `actual_binding`: MAX (1 if bound on any outage date)

### Why strict matching

Each `constraint_id` represents a specific `monitorLine:contingency` pair. The same `branch_name` (monitor line) can appear with different contingencies. A model that gets the branch right but the contingency wrong would produce wrong shadow prices, since different contingencies bind at different severity levels. Hence, we match at the `constraint_id` level.

---

## 2. Stage 1 — Classification Metrics

### What we're measuring

The classifier answers: "For each `(constraint_id, outage_date)`, will this constraint bind?"

| Metric | Formula | What it tells you |
|--------|---------|-------------------|
| **AUC-ROC** | Area under ROC curve | Threshold-free discrimination: can the model separate binders from non-binders? |
| **AUC-PR** | Area under Precision-Recall curve | Better than AUC-ROC for imbalanced data (~7% binding rate) |
| **Brier Score** | Mean squared error of probabilities | How well-calibrated are the predicted probabilities? |
| **Precision** | TP / (TP + FP) | Of constraints predicted binding, what fraction actually binds? |
| **Recall** | TP / (TP + FN) | Of constraints that actually bind, what fraction did we predict? |
| **F1** | 2 * P * R / (P + R) | Harmonic mean of precision and recall |
| **F-beta** | (1 + beta^2) * P * R / (beta^2 * P + R) | Weighted F-score. beta=0.5 favors precision, beta=2.0 favors recall |

### Precision vs Recall tradeoff in this context

- **High precision** means: when we say "this will bind," we're usually right. But we might miss many actual binders. Good for: avoiding false alarms that waste trading capital.
- **High recall** means: we catch most actual binders. But we might also flag many non-binders. Good for: not missing profitable FTR trading opportunities.

**Legacy baseline** used F0.5 (precision-weighted) → high precision (~0.33) but low recall (~0.27). The proposed approach uses F2.0 (recall-weighted) to catch more binders and rely on the regression stage to rank them.

### Can we define precision/recall for the regression task?

Not directly in the standard sense, because regression predicts a continuous value. However, we can define derived binary metrics:

1. **Shadow price > X threshold**: Convert regression to binary ("did we predict shadow price > $500?") and compute precision/recall. But this conflates the classifier's decision with the regressor's accuracy.

2. **Top-K precision**: "Of the top K constraints ranked by predicted shadow price, how many actually bind?" This is what **Precision@K** captures.

3. **Value Capture@K**: "Of the top K constraints ranked by predicted probability, what fraction of total dollar value did we capture?" This is the most useful regression-adjacent metric because it combines the classifier's ranking with actual dollar outcomes.

In practice, the regression task is evaluated via Spearman correlation, MAE, and RMSE on true positives — not precision/recall.

---

## 3. Stage 2 — Regression Metrics (True Positives Only)

### Why "true positives only"?

The regressor only runs on constraints predicted as binding (Stage 1 output). Evaluating regression on ALL samples would conflate classification errors with regression errors. For true-positive constraints, we know:
- The model correctly identified them as binding
- We can meaningfully compare predicted vs actual shadow prices

| Metric | Formula | What it tells you |
|--------|---------|-------------------|
| **Spearman (TP)** | Rank correlation of predicted vs actual shadow prices | Are high-value constraints ranked above low-value ones? (Most important for FTR trading) |
| **MAE (TP)** | Mean absolute dollar error | Average absolute error in predicted shadow price |
| **RMSE (TP)** | Root mean squared dollar error | Average error (penalizes large misses more) |
| **Bias (TP)** | Mean(predicted) - Mean(actual) | Systematic over/under-prediction? |
| **Residual Std** | Std(actual - predicted) | Error dispersion |
| **Residual Skew** | Skewness(actual - predicted) | Are errors asymmetric? |

### Why Spearman is the key regression metric

For FTR trading, we don't need to predict the exact dollar shadow price. We need to **rank** constraints: which ones have the highest expected congestion value. Spearman correlation measures rank agreement between predicted and actual values, regardless of scale.

A model with Spearman = 0.45 and MAE = $2,000 can be more valuable than one with Spearman = 0.30 and MAE = $1,000, because the first model puts the right constraints at the top.

---

## 4. Combined Pipeline Metrics (All Samples)

These evaluate the end-to-end system including both classification and regression.

| Metric | Formula | What it tells you |
|--------|---------|-------------------|
| **RMSE (all)** | RMSE of predicted vs actual shadow price, all constraints | Combined error including zero-valued non-binders |
| **MAE (all)** | MAE of predicted vs actual, all constraints | Combined error |

For non-binding constraints (`actual_shadow_price = 0`):
- If classifier correctly says "not binding" → predicted = 0, error = 0
- If classifier incorrectly says "binding" → predicted > 0, contributes to RMSE

This metric captures false positive cost: predicting shadow price where there is none.

---

## 5. Ranking Metrics (Portfolio-Level)

These are the most relevant metrics for the FTR trading use case.

### Per-outage level

For a single `outage_date`, rank all ~5,400 constraints by predicted probability:

| Metric | Formula | What it tells you |
|--------|---------|-------------------|
| **Precision@K** | (# true binders in top K) / K | What fraction of our top picks actually bind? |
| **Recall@K** | (# true binders in top K) / (total true binders) | What fraction of all binders did we capture in top K? |
| **Lift@K** | Precision@K / base_binding_rate | How much better than random selection? |
| **ValCap@K** | Sum(actual_SP, top K) / Sum(actual_SP, all) | **The money metric**: what % of total $ value did we capture? |
| **NDCG** | Normalized discounted cumulative gain | Overall ranking quality using actual SP as graded relevance |

### Per-constraint level (aggregated across outage dates)

Group by `(branch_name, flow_direction)` across all outage dates in the test period. For each constraint:
- `actual_sp_sum` = total actual shadow price across outage dates
- `binding_prob_max` = max predicted probability across outage dates

Then rank constraints by `binding_prob_max` and compute ValCap@K at the constraint level.

This is closer to the actual signal output: the trading signal ranks ~1,300 constraints into 5 tiers, and the portfolio is constructed from the top tiers.

---

## 6. Concrete Examples

### Example 1: Classification

Test data for auction_month=2020-07, onpeak, f0:
- 142,660 total `(constraint_id, outage_date)` pairs
- 8,924 actually bind (6.3% binding rate)

The legacy model:
- Predicts 7,529 as binding
- 2,562 are correct (true positives)
- 4,967 are wrong (false positives)
- 6,362 actual binders were missed (false negatives)

```
Precision = 2,562 / 7,529 = 0.340   (34% of predictions correct)
Recall    = 2,562 / 8,924 = 0.287   (29% of actual binders caught)
```

### Example 2: Regression (on the 2,562 true positives)

For the correctly-identified binding constraints:
- Mean actual shadow price: $1,307/MWh
- Mean predicted shadow price: $701/MWh
- Bias: -$607 (systematic under-prediction)
- Spearman: 0.205 (weak but positive ranking)

### Example 3: Value Capture (constraint-level)

Rank all ~5,400 constraints by predicted probability, pick the top 1,000:
- These top 1,000 contain 84% of total actual shadow-price dollars
- This means: even with mediocre recall, the model correctly identifies the highest-value constraints

### Example 4: What "match" means at different levels

```
Scenario: Branch "AEC.JOPPA3451UNIT3-TOP" has 3 constraint_ids:
  - "AEC.JOPPA3451UNIT3-TOP : AEC.COFFN1-MARKE1"   (contingency A)
  - "AEC.JOPPA3451UNIT3-TOP : AEC.NEWTN1-CENTR1"   (contingency B)
  - "AEC.JOPPA3451UNIT3-TOP : AEC.MERID1-SOUTH1"   (contingency C)

On outage_date 2020-07-15:
  - Contingency A actually binds with SP = $2,000
  - Contingency B actually binds with SP = $500
  - Contingency C does not bind (SP = $0)

Model predictions:
  - Contingency A: P(bind) = 0.85, predicted SP = $1,200  → TP
  - Contingency B: P(bind) = 0.30, predicted SP = $0      → FN (missed)
  - Contingency C: P(bind) = 0.60, predicted SP = $800    → FP (false alarm)

Per-constraint metrics for this branch on this date:
  Precision: 1/2 = 0.50
  Recall:    1/2 = 0.50

At the branch-aggregated level (monthly):
  actual_sp_sum   = $2,000 + $500 + $0 = $2,500
  pred_sp_sum     = $1,200 + $0 + $800 = $2,000
  binding_prob_max = 0.85
  actual_binding  = 1 (at least one contingency binds)
```

---

## 7. Metric Hierarchy

```
           Most important for trading
                    │
    ┌───────────────┼───────────────┐
    │               │               │
 ValCap@K     Spearman(TP)     Recall
    │               │               │
    │               │               │
 ─ captures     ─ ranks           ─ catches
   dollar         constraints       binders
   value          correctly

           Supporting metrics
                    │
    ┌───────┬───────┼───────┬──────────┐
    │       │       │       │          │
  AUC-ROC  NDCG  Precision  RMSE   Brier
    │       │       │       │          │
    │       │       │       │          │
 ─ discrim ─ full ─ false ─ dollar  ─ calibr.
   ability   rank   alarm    error    quality
```

The 5 hard gates map to this hierarchy:
- **S1-AUC** → discrimination ability
- **S1-REC** → binder coverage
- **S2-SPR** → constraint ranking quality
- **C-VC@1000** → dollar value capture
- **C-RMSE** → dollar error magnitude
