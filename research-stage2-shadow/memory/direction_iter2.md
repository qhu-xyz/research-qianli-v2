# Direction — Iteration 2 (feat-eng-3-20260304-121042)

## Champion: v0011 (34 features, pruned from 39)

Mean EV-VC@100=0.0801, EV-VC@500=0.2270, EV-NDCG=0.7499, Spearman=0.3925

## Batch Constraint Update

**HP changes now allowed** (per D6). Feature set is frozen at 34 — no feature additions or removals this iteration. Only regressor hyperparameters may change.

## Problem Statement

v0011 traded breadth for precision: EV-VC@100 +5.2% but EV-VC@500 -2.5%. EV-VC@500 is now the **binding gate constraint**:
- **L1**: margin +4.2% (one bad iteration could fail)
- **L2**: at exact limit (1 tail failure: 2022-09 at 0.0527 < tail_floor 0.0536)
- **L3**: margin only +0.0023 above threshold

Goal: recover EV-VC@500 breadth while preserving EV-VC@100 gains.

---

## Hypothesis A (Primary): More Trees + Lower Learning Rate

**What**: Increase ensemble size and reduce step size for finer-grained mid-tier value discrimination.

**Hypothesis A overrides**:
```json
{"regressor": {"n_estimators": 600, "learning_rate": 0.03}}
```

**Rationale**: With 400 trees at lr=0.05 (total boosting budget=20), the model converges quickly to strong top-100 discrimination but may not develop enough granularity for the 100-500 ranking tier. With 600 trees at lr=0.03 (budget=18), the model takes smaller steps across more rounds, capturing subtler value distinctions among mid-ranked constraints.

**Prior evidence**: Learning #4 (lr=0.03/n_est=700 + L2=5 was WORSE) does NOT apply here — that test compounded L2=5 compression with slow learning. With L2=1.0 (current), more-trees-lower-LR is a clean, untested direction.

---

## Hypothesis B (Alternative): Higher Column Sampling + Modest Tree Increase

**What**: Increase colsample_bytree from 0.8 to 0.9 to restore per-tree feature breadth, with modest tree/LR adjustment.

**Hypothesis B overrides**:
```json
{"regressor": {"colsample_bytree": 0.9, "n_estimators": 500, "learning_rate": 0.04}}
```

**Rationale**: The v0011 EV-VC@500 degradation was systematic (7/12 months). The pruning from 39→34 features improved sampling efficiency for top-100 but reduced per-tree combinatorial diversity. At colsample=0.8 with 34 features, each tree sees 27 features. At colsample=0.9, each tree sees ~31/34 features — ensuring nearly all signal dimensions are represented in every tree. This more closely matches v0009's effective regime (where 31/39 features were sampled, ~26 real), but with ALL features carrying real signal. The 500 trees at lr=0.04 (budget=20, same as current) provide stability.

**Why not colsample=0.7?** Lower colsample risks the same "starvation" direction as subsample=0.6 (learning #5). For breadth recovery, we need MORE signal coverage per tree, not less.

---

## Screen Months

- **Weak month: 2022-09** — The sole EV-VC@500 tail failure (0.0527 < tail_floor 0.0536). Also weak on EV-VC@100 (0.028), EV-NDCG (0.674), Spearman (0.328). If HP changes lift EV-VC@500 above 0.0536 here, the L2 at-limit problem is eliminated. There is no substitute for testing on the actual failure month.

- **Strong month: 2022-12** — Strong breadth metrics (EV-VC@100=0.158, EV-VC@500=0.346, Spearman=0.386). Fresh screen month (not used in iter 1's 2022-09/2021-09 or prior batch's 2022-06/2022-12). Tests that HP changes don't regress on well-performing months.

**Note**: 2022-09 was used in iter 1 screening (for feature pruning). Re-using it is justified because it's the only EV-VC@500 tail failure and the iter 2 hypotheses (HP changes) are completely different from iter 1 (feature changes).

---

## Winner Criteria

Pick the hypothesis with **higher mean EV-VC@500 across the 2 screen months** (EV-VC@500 recovery is the primary objective):

1. **Primary**: Higher mean EV-VC@500 wins.
2. **Veto on EV-VC@100**: If a hypothesis drops EV-VC@100 > 15% on either screen month vs champion, disqualify it (protecting iter 1 gains).
3. **Veto on Spearman**: If a hypothesis drops Spearman > 0.02 on either screen month vs champion, disqualify it.
4. **Tiebreaker**: If both are within ±3% on mean EV-VC@500, prefer the one with higher mean EV-VC@100.

---

## Code Changes for Winner

### If Hypothesis A wins (n_estimators=600, lr=0.03):

**File: `ml/config.py`** — in `RegressorConfig` dataclass:
- Change `n_estimators: int = 400` → `n_estimators: int = 600`
- Change `learning_rate: float = 0.05` → `learning_rate: float = 0.03`

**Verification**: `python -c "from ml.config import RegressorConfig; c = RegressorConfig(); print(c.n_estimators, c.learning_rate)"` → expect `600 0.03`

### If Hypothesis B wins (colsample=0.9, n_estimators=500, lr=0.04):

**File: `ml/config.py`** — in `RegressorConfig` dataclass:
- Change `colsample_bytree: float = 0.8` → `colsample_bytree: float = 0.9`
- Change `n_estimators: int = 400` → `n_estimators: int = 500`
- Change `learning_rate: float = 0.05` → `learning_rate: float = 0.04`

**Verification**: `python -c "from ml.config import RegressorConfig; c = RegressorConfig(); print(c.n_estimators, c.learning_rate, c.colsample_bytree)"` → expect `500 0.04 0.9`

### Tests
- No feature count assertions affected (34 features unchanged)
- Run `pytest ml/tests/` after code change to confirm tests pass

---

## Expected Impact

| Metric | Hyp A (600 trees, lr=0.03) | Hyp B (colsample=0.9, 500 trees, lr=0.04) |
|--------|---------------------------|-------------------------------------------|
| EV-VC@100 | Neutral to +1% | Neutral (possibly slight loss from less tree diversity) |
| EV-VC@500 | +1-3% (more mid-tier capacity) | +1-3% (restored per-tree signal coverage) |
| EV-NDCG | +0-1% | Neutral to +0.5% |
| Spearman | Neutral | Neutral |
| C-RMSE | -1-2% (better fit) | -0.5-1% (more robust) |
| Training time | ~50% longer | ~25% longer |

---

## Risk Assessment

1. **Low risk (Hyp A)**: More trees + lower LR is the most conservative HP change. Strictly more ensemble averaging. Worst case: no improvement, longer training.
2. **Low risk (Hyp B)**: colsample 0.8→0.9 is a small increase. Trees see 31/34 features instead of 27/34. Cannot cause starvation. Slight overfitting risk offset by more trees + lower LR.
3. **Gate safety**: EV-VC@100 has +20.6% margin to floor — enormous headroom. Spearman has +5.1% margin. Only EV-VC@500 is tight, and that's what we're targeting.
4. **No feature risk**: Feature set is unchanged at 34. Clean HP-only experiment.

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
