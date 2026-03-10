# Direction — Iteration 3 (feat-eng-3-20260304-121042)

## Champion: v0012 (34 features, n_estimators=600, lr=0.03, mcw=25)

Mean EV-VC@100=0.0758, EV-VC@500=0.2348, EV-NDCG=0.7518, Spearman=0.3940

## Batch Constraint

Feature set frozen at 34. HP changes allowed (relaxed since D6/iter 2). This is the **final iteration** of the batch.

## Problem Statement

v0012 recovered EV-VC@500 breadth (+3.5%) but traded EV-VC@100 precision (-5.3%). Over the two iterations, EV-VC@100 has gone: v0009=0.0762 → v0011=0.0801 (+5.2%) → v0012=0.0758 (-5.3%). The net movement is slightly negative vs v0009.

EV-VC@100 margin to floor is +14.2% — still comfortable, but the business objective prioritizes top-100 capital allocation. Recovering EV-VC@100 without surrendering the EV-VC@500 tail safety gains is the goal.

**Settled parameters (D11)**: n_estimators=600, learning_rate=0.03. Feature set frozen at 34. Only other regressor HPs may change.

**Remaining unexplored HP axes**: min_child_weight (currently 25), value_weighted (currently false). Both recommended by both reviewers for iter 3.

---

## Hypothesis A (Primary): Reduce min_child_weight (25→15)

**What**: Lower the minimum samples per leaf to allow more granular predictions at the top of the ranking.

**Hypothesis A overrides**:
```json
{"regressor": {"min_child_weight": 15}}
```

**Rationale**: mcw=25 was established as beneficial (learning #1) for the 29-effective-feature / 400-tree / lr=0.05 regime (v0007 era). With the current 34-feature, 600-tree, lr=0.03 config, the ensemble averaging is much stronger. The large ensemble (600 trees) provides natural regularization against leaf overfitting, making the conservative mcw=25 potentially over-regularized for top-100 discrimination.

At mcw=25 with ~10K binding training samples per month, leaf nodes contain at minimum 25 samples. At mcw=15, leaves can split further, creating more precise value estimates for rare high-value constraints — exactly the ones that matter for EV-VC@100.

**Prior evidence**: mcw was tested at 10 (v0 default) vs 25. Going from 10→25 helped EV-VC@500 +6.2% with a 400-tree ensemble. With 600 trees providing 50% more averaging, mcw=15 is a moderate step back toward sharper predictions while retaining most of the mcw=25 conservative benefit.

**Risk**: If mcw=15 causes overfitting in low-signal months, EV-VC@500 and Spearman could degrade on those months. The 600-tree ensemble should mitigate this, and the screen will catch it.

---

## Hypothesis B (Alternative): Enable value_weighted training

**What**: Weight training samples by shadow price magnitude so high-dollar constraints dominate the loss function.

**Hypothesis B overrides**:
```json
{"regressor": {"value_weighted": true}}
```

**Rationale**: The standard loss function (MSE on all binding rows) treats a $1 constraint the same as a $50 constraint. Value weighting emphasizes high-dollar rows in gradient computation, directly targeting the constraints that drive EV-VC@100 (top-100 by expected value). This should improve prediction accuracy for high-value constraints without changing ensemble structure.

**Prior evidence**: value_weighted has never been tested. It's a clean, orthogonal axis to all prior HP experiments. This is the more exploratory hypothesis.

**Risk**: Could reduce prediction accuracy for mid-tier constraints (lower-dollar but still binding), potentially degrading EV-VC@500. Shadow price distributions are heavy-tailed (a few $100+ constraints among many $1-10 constraints), so value weighting could over-fit to outlier training rows. The screen will catch this.

---

## Screen Months

- **Weak month: 2021-03** — EV-VC@100=0.0165 (bottom quartile), EV-VC@500=0.2415 (above mean), Spearman=0.4224 (healthy). This month has poor top-100 discrimination but adequate overall signal — diagnostic for whether the HP change specifically helps top-tier separation. Not anomalous like 2021-05 (EV-VC@100=0.0006).

- **Strong month: 2022-12** — EV-VC@100=0.1236 (top quartile), EV-VC@500=0.3444 (strong), Spearman=0.3866. This month saw the largest EV-VC@100 regression from v0011→v0012 (-21.8%), making it the best single indicator of whether the iter 2 dilution can be reversed. Also a strong EV-VC@500 month, so we can confirm non-regression on breadth.

**Rationale**: This pair directly targets the iter 3 objective. If an HP change recovers EV-VC@100 on 2022-12 (where v0012 lost -21.8%) without regressing on 2021-03's EV-VC@500 (0.2415), the hypothesis is strongly supported for full benchmark.

---

## Winner Criteria

Pick the hypothesis with **higher mean EV-VC@100 across the 2 screen months** (EV-VC@100 recovery is the primary objective):

1. **Primary**: Higher mean EV-VC@100 wins.
2. **Veto on EV-VC@500**: If a hypothesis drops EV-VC@500 > 10% on either screen month vs champion, disqualify it (protecting iter 2 gains).
3. **Veto on Spearman**: If a hypothesis drops Spearman > 0.03 on either screen month vs champion, disqualify it.
4. **Tiebreaker**: If both are within ±3% on mean EV-VC@100, prefer the one with higher mean EV-VC@500 (protecting breadth).
5. **If both vetoed**: Run the champion config unchanged (no code changes), and document as a no-op iteration. Accept v0012 as batch champion.

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

**Additional check for Hyp B**: Before running full benchmark, verify that `ml/pipeline.py` actually uses `value_weighted` during training. Search for `value_weighted` or `sample_weight` in pipeline.py. If the implementation is missing/stubbed, this hypothesis cannot be tested via overrides alone and would need code changes to pipeline.py.

### Tests
- No feature count assertions affected (34 features unchanged)
- Run `pytest ml/tests/` after code change to confirm tests pass

---

## Expected Impact

| Metric | Hyp A (mcw=15) | Hyp B (value_weighted=True) |
|--------|----------------|----------------------------|
| EV-VC@100 | +2-5% (sharper leaves for high-value) | +3-8% (loss emphasizes high-$) |
| EV-VC@500 | -1% to neutral (slight overfitting risk) | -2-3% (mid-tier de-emphasized) |
| EV-NDCG | +0-1% | Neutral to +0.5% |
| Spearman | Neutral to -0.5% | -0.5-1% (rank distortion from weight skew) |
| C-RMSE | -1-2% (more precise predictions) | +1-3% (biased toward high-$ accuracy) |
| Training time | Negligible | Negligible |

---

## Risk Assessment

1. **Moderate risk (Hyp A)**: mcw reduction increases overfitting potential. Mitigated by 600-tree ensemble + lr=0.03. If mcw=15 is too aggressive, result will show on screen months. No catastrophic risk.
2. **Higher uncertainty (Hyp B)**: Value weighting is completely untested. Could help or hurt — truly uncertain. Shadow price distributions are heavy-tailed, so value weighting could over-fit to outliers. Additionally, the implementation may be incomplete (needs pipeline.py verification).
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
