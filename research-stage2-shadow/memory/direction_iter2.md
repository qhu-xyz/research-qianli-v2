# Direction — Iteration 2 (feat-eng-3-20260304-121042)

## Champion: v0011 (34 features, all non-zero)

Mean EV-VC@100=0.0801, EV-VC@500=0.2270, EV-NDCG=0.7499, Spearman=0.3925

## Batch Constraint Update

**Batch constraint relaxed for iter 2.** HP changes are now allowed alongside the 34-feature set. Feature set is frozen at 34 — no feature additions or removals this iteration.

**Rationale**: The feature cleanup from iter 1 is complete. HPs (n_estimators=400, lr=0.05, colsample_bytree=0.8, mcw=25) were never re-tuned after the feature set changed from 29→34 effective features. HP tuning is the highest-value next experiment.

---

## Priority: Recover EV-VC@500 Breadth

v0011's EV-VC@500 degraded -2.5% vs v0009 (0.2270 vs 0.2329). This is the **binding gate constraint**:
- L1 margin: only +4.2%
- L2: at exact limit (1 tail failure, max is 1)
- L3: margin only +0.0023

Any further EV-VC@500 degradation will block promotion. Iter 2 must target EV-VC@500 recovery while preserving EV-VC@100 gains.

---

## Hypothesis A (Primary): More Ensemble Capacity

**What**: Increase `n_estimators` from 400 to 600, decrease `learning_rate` from 0.05 to 0.035. All other HPs unchanged.

**Why**: More trees + slower learning gives the model more capacity for nuanced discrimination in the 100-500 tier. With 34 clean features, the current 400 trees may underfit at the mid-tier — constraints ranked 100-500 need finer distinctions that additional trees can capture. The lower learning rate ensures each tree contributes less, reducing overfitting risk from the additional capacity.

**Prior evidence**: Learning #4 (lr=0.03/n_est=700 WORSE than baseline) was tested with L2=5 — the L2 compression interfered. With L2=1.0 (current), the more-trees approach has NOT been tested. This is a clean experiment.

**Hypothesis A overrides**:
```json
{"regressor": {"n_estimators": 600, "learning_rate": 0.035}}
```

---

## Hypothesis B (Alternative): More Capacity + Lower colsample

**What**: Same as Hypothesis A (n_estimators=600, lr=0.035), PLUS decrease `colsample_bytree` from 0.8 to 0.7.

**Why**: With 34 features, colsample=0.8 means 27 features per tree. Reducing to 0.7 (24 features/tree) forces each tree to use different feature subsets, increasing diversity in the ensemble. This may help the 100-500 tier by preventing over-reliance on the top features that dominate the top-100 ranking. Combined with more trees, the ensemble gets both more capacity AND more diversity.

**Risk**: subsample=0.6 was tested and caused signal starvation (Learning #5). colsample=0.7 is a milder reduction and acts on feature dimension, not row dimension, so the risk profile is different.

**Hypothesis B overrides**:
```json
{"regressor": {"n_estimators": 600, "learning_rate": 0.035, "colsample_bytree": 0.7}}
```

---

## Screen Months

- **Weak month: 2021-11** — EV-VC@500 degraded from 0.1790 (v0009) to 0.1401 (v0011), a -21.7% drop. Worst Spearman month (0.2616). Tests whether HP changes recover mid-tier value capture on the weakest breadth month. Not screened in prior iterations of this batch.
- **Strong month: 2020-09** — EV-VC@500 improved from 0.2809 (v0009) to 0.2990 (v0011), a +6.4% gain. Strong EV-VC@100 (0.0900). Tests that we don't regress on a month that already improved. Not screened in prior iterations of this batch.

**Rationale**: Both months are fresh (not used in prior screens: 2022-06, 2022-12, 2022-09, 2021-09 already used). The weak month (2021-11) was the second-worst EV-VC@500 degradation month and has the worst Spearman, making it highly diagnostic. The strong month (2020-09) improved on both EV-VC@100 and EV-VC@500, serving as a regression sentinel.

---

## Winner Criteria

Pick the hypothesis with **higher mean EV-VC@500 across the 2 screen months** (since EV-VC@500 recovery is the priority), with these tiebreakers/vetos:

1. If both are within ±5% of each other on EV-VC@500, prefer the one with higher EV-VC@100.
2. **Veto**: If a hypothesis drops Spearman > 0.02 on either screen month vs champion (v0011), disqualify it.
3. **Veto**: If a hypothesis drops EV-VC@100 > 10% on either screen month vs champion (v0011), disqualify it (protect the +5.2% gain).
4. If both pass or both fail veto checks, use EV-VC@500 as the primary selector.

---

## Code Changes for Winner

### If Hypothesis A wins:

**File: `ml/config.py`**
- In `RegressorConfig` (the dataclass or wherever defaults are defined):
  - Change `n_estimators` from `400` to `600`
  - Change `learning_rate` from `0.05` to `0.035`

**Verification**: After code change, run `python -c "from ml.config import RegressorConfig; c = RegressorConfig(); print(c.n_estimators, c.learning_rate)"` and confirm output is `600 0.035`.

### If Hypothesis B wins:

All changes from Hypothesis A, PLUS:

**File: `ml/config.py`**
- Change `colsample_bytree` from `0.8` to `0.7`

**Verification**: After code change, run `python -c "from ml.config import RegressorConfig; c = RegressorConfig(); print(c.n_estimators, c.learning_rate, c.colsample_bytree)"` and confirm output is `600 0.035 0.7`.

### Tests
- No feature count assertions affected (feature set unchanged at 34)
- Run `pytest ml/tests/` after code change to confirm all tests pass

---

## Expected Impact

| Metric | Hyp A (more trees) | Hyp B (more trees + lower colsample) |
|--------|-------------------|--------------------------------------|
| EV-VC@100 | Neutral to +1% | -1% to +1% (less top-feature concentration) |
| EV-VC@500 | +1-3% (more mid-tier capacity) | +2-4% (more diversity + capacity) |
| EV-NDCG | +0-1% | +0-1% |
| Spearman | Neutral | Neutral to +0.5% |
| C-RMSE | -1-2% (better fit) | -1-3% (more robust) |

Conservative estimates. The main uncertainty is whether 400→600 trees provides meaningful additional capacity or just adds compute time.

---

## Risk Assessment

1. **Low risk (Hyp A)**: More trees + lower LR is a standard ensemble scaling technique. Worst case is neutral performance with longer training time.
2. **Medium risk (Hyp B)**: colsample=0.7 reduces feature diversity per tree. If the top features dominate because they carry most signal, reducing their per-tree representation could hurt. But with 34 clean features, 24/tree should still cover the key signals.
3. **Spearman safety**: With 5.1% margin to floor and only HP changes (no feature changes), Spearman risk is low. The veto check at screen stage provides early warning.
4. **EV-VC@100 protection**: The veto check (>10% degradation) ensures we don't give back the +5.2% gain from iter 1.

---

## Current Regressor Config (v0011 baseline)
```json
{
  "n_estimators": 400,
  "max_depth": 5,
  "learning_rate": 0.05,
  "subsample": 0.8,
  "colsample_bytree": 0.8,
  "reg_alpha": 1.0,
  "reg_lambda": 1.0,
  "min_child_weight": 25,
  "unified_regressor": false,
  "value_weighted": false
}
```

Only the overrides specified above should change. All other parameters remain at v0011 values.
