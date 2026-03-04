# Direction — Iteration 1 (ralph-v1-20260304-003317)

## Goal
Establish whether basic hyperparameter tuning of the regressor can improve over v0 baseline. Both hypotheses are pure `--overrides` changes — no code modifications needed for screening or full benchmark.

## v0 Baseline Summary (champion)
- Regressor: XGBoost 400 trees, max_depth=5, lr=0.05, reg_lambda=1.0, min_child_weight=10
- Mode: gated (binding-only), no value weighting
- 34 features, full monotone constraints
- Means: EV-VC@100=0.069, EV-VC@500=0.216, EV-NDCG=0.747, Spearman=0.393
- Catastrophic months: 2022-06 (universal worst), 2021-05 (near-zero EV-VC@100)

---

## Hypothesis A (Primary): Slower learning rate + more trees

**Rationale**: v0 uses lr=0.05 with 400 trees. Reducing lr to 0.03 with 700 trees creates a smoother, higher-capacity ensemble that should reduce overfitting on training windows. The lr×n_estimators product increases from 20 to 21, maintaining similar total learning while distributing it across more, smaller steps. This is the safest variance-reduction move — standard ML practice for improving generalization on tail months.

**Expected effect**: Modest improvement on tail months (2022-06, 2021-05, 2022-09) through better generalization. Should not regress on strong months.

**Overrides**:
```json
{"regressor": {"learning_rate": 0.03, "n_estimators": 700}}
```

---

## Hypothesis B (Alternative): Heavier L2 regularization + larger leaves

**Rationale**: v0 uses reg_lambda=1.0 and min_child_weight=10. Increasing reg_lambda to 5.0 and min_child_weight to 25 constrains the model more aggressively — preventing the regressor from fitting noisy leaf-level patterns in small binding subsets. The catastrophic months (2022-06 with C-RMSE=5918) suggest the model overfits on some training windows. Heavier regularization is a different lever than ensemble smoothing (Hypothesis A) and targets the same root cause (overfitting) through a different mechanism (penalty-based vs averaging-based).

**Expected effect**: Reduced variance on tail months at possible cost of slightly lower peak performance. Should help C-RMSE/C-MAE (monitored) and may improve EV metrics on bad months.

**Overrides**:
```json
{"regressor": {"reg_lambda": 5.0, "min_child_weight": 25}}
```

---

## Screen Months

| Role | Month | Rationale |
|------|-------|-----------|
| **Weak** | 2022-06 | Universal worst: EV-NDCG=0.604, EV-VC@100=0.014, Spearman=0.273, C-RMSE=5918. Any improvement should show here first. |
| **Strong** | 2022-12 | Best EV-VC@100=0.194, strong EV-NDCG=0.815, Spearman=0.385. Changes must NOT regress here. |

---

## Winner Criteria

1. **Primary**: Higher mean EV-VC@100 across the 2 screen months
2. **Safety**: Spearman must not drop >0.05 on either screen month vs v0
3. **Tiebreak**: If EV-VC@100 is within 0.005, prefer higher mean EV-NDCG
4. **Both fail**: If both hypotheses regress EV-VC@100 mean below v0 on screen months, pick the one that regresses least and proceed to full benchmark anyway (we need data)

---

## Code Changes for Winner

**Neither hypothesis requires code changes.** Both are pure hyperparameter overrides.

For the winning config, the worker should:
1. Update `ml/config.py` `RegressorConfig` defaults to match the winning hyperparameters
2. That's it — no other file changes needed

Specifically:
- **If A wins**: In `ml/config.py`, change `RegressorConfig` fields:
  - `n_estimators: int = 400` → `n_estimators: int = 700`
  - `learning_rate: float = 0.05` → `learning_rate: float = 0.03`
- **If B wins**: In `ml/config.py`, change `RegressorConfig` fields:
  - `reg_lambda: float = 1.0` → `reg_lambda: float = 5.0`
  - `min_child_weight: int = 10` → `min_child_weight: int = 25`

---

## Expected Gate Impact

| Gate | v0 Mean | Direction | Expected Δ |
|------|---------|-----------|------------|
| EV-VC@100 | 0.069 | ↗ | +0.005 to +0.015 (better top-ranking from smoother predictions) |
| EV-VC@500 | 0.216 | ↗ | +0.005 to +0.010 |
| EV-NDCG | 0.747 | ↗ | +0.005 to +0.015 |
| Spearman | 0.393 | → | ±0.01 (rank correlation is robust to smoothing) |
| C-RMSE | 3133 | ↘ | -50 to -200 (reduced overfitting → lower error) |

Layer 1 (mean quality) is tight — floor equals v0 mean. Even a small improvement passes. Layer 2 (tail safety) is permissive on EV-VC@100 (tail_floor=0.0001). Layer 3 (tail non-regression) needs bottom-2 months to stay within 0.02 of v0's bottom-2.

---

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Both hypotheses produce near-identical results to v0 | Medium | We still get data to calibrate next iteration; proceed to full benchmark with the marginal winner |
| Slower lr + more trees increases training time significantly | Low | 700 trees at lr=0.03 adds ~75% more training time; still well within 35-min budget for 12 months |
| Heavier regularization underfits, hurting strong months | Low-Medium | Screen month 2022-12 (strong) catches this; safety criterion protects against it |
| Neither hypothesis helps the catastrophic months (structural issue, not overfitting) | Medium | If screen results on 2022-06 show no improvement, iter 2 should try value-weighted training (requires pipeline.py code change) or feature selection |
