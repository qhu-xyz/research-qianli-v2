# Direction — Iteration 2 (ralph-v1-20260304-003317)

## ⚠️ CRITICAL RULES — READ BEFORE ANYTHING ELSE

### DO NOT MODIFY these files (frozen / human-owned):
- `ml/evaluate.py` — HUMAN-WRITE-ONLY
- `ml/data_loader.py` — no changes needed
- `ml/features.py` — no changes needed
- `ml/pipeline.py` — no changes needed
- `ml/train.py` — no changes needed
- `ml/benchmark.py` — no changes needed
- `registry/gates.json` — HUMAN-WRITE-ONLY
- `registry/v0/` — baseline is immutable

### The classifier is FROZEN:
- **NEVER** modify `ClassifierConfig` in `ml/config.py`
- **NEVER** add classifier features, change classifier hyperparams, or touch `threshold_beta`
- The ONLY class you may edit is `RegressorConfig` — and ONLY its default field values

### YOU MUST:
1. Revert ALL uncommitted changes before starting: `git checkout -- ml/`
2. Verify clean state: `git diff --name-only` should show NO ml/ files
3. Run screens with `--overrides` (ZERO code changes for screening)
4. Only edit `ml/config.py` after picking the winner
5. Commit before writing handoff
6. Verify `registry/{VERSION_ID}/metrics.json` exists before writing handoff

---

## Goal

Test two regressor hyperparameter changes via 2-month screening, pick the winner, run full 12-month benchmark. Both hypotheses use `--overrides` — NO code changes needed for screening.

## v0 Baseline (champion)

Regressor: XGBoost 400 trees, max_depth=5, lr=0.05, reg_lambda=1.0, min_child_weight=10. Gated mode, 34 features.

Key metrics (12-month means): EV-VC@100=0.069, EV-VC@500=0.216, EV-NDCG=0.747, Spearman=0.393.

v0 on screen months:
- **2022-06** (weak): EV-VC@100=0.014, EV-VC@500=0.076, EV-NDCG=0.604, Spearman=0.273
- **2022-12** (strong): EV-VC@100=0.194, EV-VC@500=0.345, EV-NDCG=0.815, Spearman=0.385

---

## Hypothesis A (Primary): Slower learning rate + more trees

**Rationale**: lr 0.05→0.03 with n_estimators 400→700 creates a smoother ensemble. The lr×trees product increases from 20 to 21, maintaining total learning capacity while distributing it over more, smaller steps. Standard variance-reduction technique. Should help tail months (2022-06, 2021-05) through better generalization without hurting strong months.

**Overrides**:
```json
{"regressor": {"learning_rate": 0.03, "n_estimators": 700}}
```

---

## Hypothesis B (Alternative): Heavier L2 regularization + larger leaves

**Rationale**: reg_lambda 1→5 and min_child_weight 10→25 constrain the model from fitting noise in small binding subsets (~10k samples/month). The catastrophic C-RMSE on 2022-06 (5918) suggests overfitting on some training windows. This attacks the same problem as Hypothesis A but through penalty-based regularization rather than ensemble smoothing.

**Overrides**:
```json
{"regressor": {"reg_lambda": 5.0, "min_child_weight": 25}}
```

---

## Screen Months

| Role | Month | v0 EV-VC@100 | v0 EV-NDCG | v0 Spearman | Rationale |
|------|-------|-------------|-----------|------------|-----------|
| **Weak** | 2022-06 | 0.014 | 0.604 | 0.273 | Universal worst across all Group A gates. Improvements should show here. |
| **Strong** | 2022-12 | 0.194 | 0.815 | 0.385 | Highest EV-VC@100. Changes must NOT regress here. |

---

## Winner Criteria

1. **Primary**: Higher mean EV-VC@100 across 2022-06 and 2022-12
2. **Safety**: Spearman must not drop >0.05 on either screen month vs v0 (i.e., Spearman ≥ 0.223 on 2022-06, ≥ 0.335 on 2022-12)
3. **Tiebreak**: If EV-VC@100 within 0.005, prefer higher mean EV-NDCG
4. **Both regress**: If both drop EV-VC@100 mean below v0 on screen months, pick the one that regresses least — we need data

---

## Code Changes for Winner

**Screening requires ZERO code changes.** Use `--overrides` flag only.

After picking the winner, update `ml/config.py` RegressorConfig defaults:

- **If A wins**: change `n_estimators: int = 400` → `700` and `learning_rate: float = 0.05` → `0.03`
- **If B wins**: change `reg_lambda: float = 1.0` → `5.0` and `min_child_weight: int = 10` → `25`

No other files should be changed. No new features, no new functions, no pipeline restructuring.

---

## Expected Gate Impact

| Gate | v0 Mean | Expected Δ | Reasoning |
|------|---------|------------|-----------|
| EV-VC@100 | 0.069 | +0.005 to +0.015 | Smoother/regularized predictions → better top-ranking |
| EV-VC@500 | 0.216 | +0.005 to +0.010 | Same mechanism, broader cutoff |
| EV-NDCG | 0.747 | +0.005 to +0.015 | Ranking quality benefits from less noise |
| Spearman | 0.393 | ±0.01 | Rank correlation is robust to smoothing/regularization |
| C-RMSE | 3133 | -50 to -200 | Reduced overfitting → lower error |

Gate floors = v0 means, so even small improvement passes Layer 1. Layer 2 (tail safety) is very permissive on EV-VC@100 (tail_floor=0.0001). Layer 3 (tail non-regression) needs bottom-2 within 0.02 of v0.

---

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Both hypotheses ~identical to v0 | Medium | We still get data for calibration; pick marginal winner |
| 700 trees increases training time | Low | ~75% more trees; well within 35-min budget |
| Heavier regularization underfits strong months | Low-Medium | Screen month 2022-12 catches this |
| Structural issue on 2022-06 (not fixable by hyperparams) | Medium | If no screen improvement, iter 3 tries feature selection or depth reduction |
