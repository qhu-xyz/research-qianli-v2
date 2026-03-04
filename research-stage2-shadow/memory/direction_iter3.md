# Direction — Iteration 3 (feat-eng-3-20260304-121042)

## Champion: v0012 (34 features, n_estimators=600, lr=0.03)

Mean EV-VC@100=0.0758, EV-VC@500=0.2348, EV-NDCG=0.7518, Spearman=0.3940

## Batch Constraint

Feature set frozen at 34. HP changes allowed. This is the **final iteration** of the batch.

## Problem Statement

v0012 recovered EV-VC@500 breadth (+3.5%) but traded EV-VC@100 precision (-5.3%). Over the two iterations, EV-VC@100 has gone: v0009=0.0762 → v0011=0.0801 (+5.2%) → v0012=0.0758 (-5.3%). The net movement is slightly negative vs v0009.

EV-VC@100 margin to floor is +14.2% — still comfortable, but the business objective prioritizes top-100 capital allocation. Recovering EV-VC@100 without surrendering the EV-VC@500 tail safety gains is the goal.

**Constraint**: n_estimators=600 and learning_rate=0.03 are settled (D11). Feature set is frozen at 34. Only other regressor HPs may change.

---

## Hypothesis A (Primary): Reduce min_child_weight (25→15)

**What**: Lower the minimum samples per leaf to allow more granular predictions at the top of the ranking.

**Hypothesis A overrides**:
```json
{"regressor": {"min_child_weight": 15}}
```

**Rationale**: mcw=25 was established as beneficial (learning #1) for the 29-effective-feature regime (v0007 era). With the current 34-feature, 600-tree, lr=0.03 config, the ensemble averaging is much stronger. The large ensemble (600 trees) provides natural regularization against leaf overfitting, making the conservative mcw=25 potentially over-regularized for top-100 discrimination.

At mcw=25 with ~10K training samples per month, leaf nodes average ~400 samples (assuming depth-5 trees with ~25 leaves). At mcw=15, leaves can split further, creating more precise value estimates for the rarest high-value constraints — exactly the ones that matter for EV-VC@100.

**Prior evidence**: mcw=25 was tested against mcw=10 (v0 default) with 400 trees at lr=0.05. With 600 trees at lr=0.03, the ensemble is 50% larger and the learning rate 40% lower, providing more overfitting protection. mcw=15 is a moderate step back (not all the way to 10).

**Risk**: If mcw=15 causes overfitting on training data, EV-VC@500 and Spearman could degrade. The 600-tree ensemble should mitigate this, but the screen will catch it.

---

## Hypothesis B (Alternative): Enable value_weighted training

**What**: Weight training samples by shadow price magnitude so high-dollar constraints dominate the loss function.

**Hypothesis B overrides**:
```json
{"regressor": {"value_weighted": true}}
```

**Rationale**: The standard loss function (MSE on all binding rows) treats a $1 constraint the same as a $50 constraint. Value weighting emphasizes high-dollar rows in gradient computation, directly targeting the constraints that drive EV-VC@100 (top-100 by expected value). This should improve prediction accuracy for high-value constraints without changing ensemble structure.

**Prior evidence**: value_weighted has never been tested. It's a clean, orthogonal axis to all prior HP experiments.

**Risk**: Could reduce prediction accuracy for mid-tier constraints (lower-dollar but still binding), potentially degrading EV-VC@500. The screen will catch this. Also could increase sensitivity to outlier training rows (very high shadow prices distorting gradient).

---

## Screen Months

- **Weak month: 2022-06** — Weakest month across all metrics: EV-VC@100=0.0166, EV-VC@500=0.0676, Spearman=0.2742. Tests whether HP changes help in structurally difficult regimes. Fresh as a screen month for HP experiments (not used in iter 2 screening).

- **Strong month: 2020-11** — Strong EV-VC@100 (0.1466), strong Spearman (0.5281). Fresh screen month (never used in this batch). Tests that HP changes don't regress on well-performing months.

**Note**: Avoiding 2022-09 and 2022-12 which were used in iter 2 screening. Using fresh months for independent signal.

---

## Winner Criteria

Pick the hypothesis with **higher mean EV-VC@100 across the 2 screen months** (EV-VC@100 recovery is the primary objective):

1. **Primary**: Higher mean EV-VC@100 wins.
2. **Veto on EV-VC@500**: If a hypothesis drops EV-VC@500 > 10% on either screen month vs champion, disqualify it (protecting iter 2 gains). Tighter veto than iter 2 because EV-VC@500 tail safety is now hard-won.
3. **Veto on Spearman**: If a hypothesis drops Spearman > 0.02 on either screen month vs champion, disqualify it.
4. **Tiebreaker**: If both are within ±3% on mean EV-VC@100, prefer the one with higher mean EV-VC@500 (protecting breadth).
5. **If both vetoed**: Run the champion config unchanged (no code changes), and document as a no-op iteration.

---

## Code Changes for Winner

### If Hypothesis A wins (min_child_weight=15):

**File: `ml/config.py`** — in `RegressorConfig` dataclass:
- Change `min_child_weight: int = 25` → `min_child_weight: int = 15`

**Verification**: `python -c "from ml.config import RegressorConfig; c = RegressorConfig(); print(c.min_child_weight)"` → expect `15`

### If Hypothesis B wins (value_weighted=True):

**File: `ml/config.py`** — in `RegressorConfig` dataclass:
- Change `value_weighted: bool = False` → `value_weighted: bool = True`

**Verification**: `python -c "from ml.config import RegressorConfig; c = RegressorConfig(); print(c.value_weighted)"` → expect `True`

### Tests
- No feature count assertions affected (34 features unchanged)
- Run `pytest ml/tests/` after code change to confirm tests pass

---

## Expected Impact

| Metric | Hyp A (mcw=15) | Hyp B (value_weighted=True) |
|--------|----------------|----------------------------|
| EV-VC@100 | +2-5% (sharper leaves for high-value) | +1-3% (loss emphasizes high-$) |
| EV-VC@500 | -1% to neutral (slight overfitting risk) | -2-3% (mid-tier de-emphasized) |
| EV-NDCG | +0-1% | Neutral |
| Spearman | Neutral to -0.5% | -0.5-1% (rank distortion from weight skew) |
| C-RMSE | -1-2% (more precise predictions) | +1-3% (biased toward high-$ accuracy) |
| Training time | Negligible | Negligible |

---

## Risk Assessment

1. **Moderate risk (Hyp A)**: mcw reduction increases overfitting potential. Mitigated by 600-tree ensemble + lr=0.03. If mcw=15 is too aggressive, result will show on screen months. No catastrophic risk.
2. **Moderate risk (Hyp B)**: Value weighting is completely untested. Could help or hurt — truly uncertain. Shadow price distributions are heavy-tailed (a few $100+ constraints among many $1-10 constraints), so value weighting could over-fit to outliers. Screen will reveal.
3. **Gate safety**: EV-VC@100 has +14.2% margin — even a neutral result is acceptable. EV-VC@500 has +7.8% margin — can absorb ~2% degradation without hitting floor. Spearman +5.5% margin.
4. **No feature risk**: Feature set unchanged at 34. Clean HP-only experiment.
5. **Final iteration**: This is the last chance for the batch. If both hypotheses are vetoed, accept v0012 as the batch champion and close cleanly.

---

## Current Regressor Config (v0012 baseline)
```json
{
  "n_estimators": 600,
  "max_depth": 5,
  "learning_rate": 0.03,
  "subsample": 0.8,
  "colsample_bytree": 0.8,
  "reg_alpha": 1.0,
  "reg_lambda": 1.0,
  "min_child_weight": 25,
  "unified_regressor": false,
  "value_weighted": false
}
```

Only the overrides specified above should change. All other parameters remain at v0012 values.
