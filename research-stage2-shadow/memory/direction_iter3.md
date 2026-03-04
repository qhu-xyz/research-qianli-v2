# Direction — Iteration 3 (ralph-v1-20260304-003317)

## CRITICAL RULES — READ BEFORE ANYTHING ELSE

### Step 0: Revert dirty state (MANDATORY, DO THIS FIRST)

The main working tree has extensive UNCOMMITTED changes from a prior failed worker. You MUST clean before doing ANYTHING else.

```bash
cd /home/xyz/workspace/research-qianli-v2/research-stage2-shadow

# Revert ALL uncommitted changes to ml/ and registry/
git checkout -- ml/ registry/

# Verify clean state — should show NO ml/ or registry/ files
git diff --name-only | grep -E '^(ml/|registry/)' && echo "DIRTY — STOP AND FIX" || echo "CLEAN — proceed"
```

**DO NOT proceed past Step 0 if the verification shows DIRTY.**

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

---

## Goal

Test two regressor hyperparameter configurations via 2-month screening, pick the winner, run full 12-month benchmark. Both hypotheses build on v0003's proven L2 regularization (reg_lambda=5.0, min_child_weight=25) and add a new regularization dimension. Screening uses `--overrides` — NO code changes until after picking the winner.

---

## v0 Baseline (COMMITTED, correct values — 10/2 train/val, 24 features)

Regressor: XGBoost 400 trees, max_depth=5, lr=0.05, reg_lambda=1.0, min_child_weight=10. Gated mode, 24 features.

| Metric | v0 Mean | v0 Bottom-2 | Gate Floor |
|--------|---------|-------------|------------|
| EV-VC@100 | 0.0303 | 0.0035 | 0.0223 |
| EV-VC@500 | 0.1180 | 0.0488 | 0.0880 |
| EV-NDCG | 0.7400 | 0.6735 | 0.6900 |
| Spearman | 0.3921 | 0.3296 | 0.3421 |
| C-RMSE | 3400.4 | 5967.6 | 3900.4 (ceiling) |

v0 on screen months:
- **2022-09** (weak): EV-VC@100=0.002, EV-VC@500=0.033, EV-NDCG=0.692, Spearman=0.382, C-RMSE=2696
- **2021-01** (strong): EV-VC@100=0.066, EV-VC@500=0.194, EV-NDCG=0.794, Spearman=0.458, C-RMSE=2516

### v0003 reference (iter 2 worktree, not on main — for comparison only)
v0003 used reg_lambda=5.0, min_child_weight=25 (all other params = v0 defaults).
Mean: EV-VC@100=0.0337 (+0.0034), EV-NDCG=0.7435 (+0.0035), Spearman=0.3921 (unchanged), C-RMSE=3377.7 (-23).

---

## Hypothesis A (Primary): Combined ensemble smoothing + L2 regularization

**What**: Stack slower learning rate + more trees (ensemble smoothing) ON TOP OF heavier L2 regularization (v0003's proven config).

**Rationale**: In the iter 2 screen, lr=0.03/n_est=700 and reg_lambda=5/mcw=25 performed nearly identically (EV-VC@100: 0.0242 vs 0.0244, EV-NDCG: 0.7284 vs 0.7303). They attack overfitting through complementary mechanisms:
- L2 penalty constrains individual tree weights
- Slower lr with more trees smooths the ensemble

Combining them may stack benefits. The lr x n_estimators product increases from 20 to 21 (similar total capacity but spread over smaller, more regularized steps).

**Overrides**:
```json
{"regressor": {"learning_rate": 0.03, "n_estimators": 700, "reg_lambda": 5.0, "min_child_weight": 25}}
```

---

## Hypothesis B (Alternative): Shallower trees + L2 regularization

**What**: Reduce tree depth from 5 to 3 while keeping v0003's L2 regularization.

**Rationale**: max_depth is a qualitatively different regularization axis than L2 penalty or ensemble size. At depth=3, each tree has at most 8 leaf nodes (vs 32 at depth=5), limiting interaction complexity. This forces the model to rely on simpler, more robust feature relationships. For a regression problem with ~10k binding samples/month, depth=3 may be sufficient to capture the dominant price signals (hist_da, expected_overload, prob_exceed_*) while avoiding spurious higher-order interactions that overfit small training sets.

**Overrides**:
```json
{"regressor": {"max_depth": 3, "reg_lambda": 5.0, "min_child_weight": 25}}
```

---

## Screen Months

| Role | Month | v0 EV-VC@100 | v0 EV-NDCG | v0 Spearman | Rationale |
|------|-------|-------------|-----------|------------|-----------|
| **Weak** | 2022-09 | 0.002 | 0.692 | 0.382 | Universally worst EV-VC@100 and 2nd-worst EV-NDCG. Improvements should show here first. |
| **Strong** | 2021-01 | 0.066 | 0.794 | 0.458 | Strong across all Group A gates. Changes must NOT regress here. |

Different from iter 2 screen months (2022-06, 2022-12) for broader diagnostic coverage.

---

## Winner Criteria

1. **Primary**: Higher mean EV-VC@100 across 2022-09 and 2021-01
2. **Safety**: Spearman must not drop > 0.05 on either screen month vs v0 (i.e., Spearman >= 0.332 on 2022-09, >= 0.408 on 2021-01)
3. **Tiebreak**: If EV-VC@100 within 0.005, prefer higher mean EV-NDCG
4. **Both regress**: If both drop mean EV-VC@100 below v0 on screen months, pick the one that regresses least (we still get data)

---

## Code Changes for Winner

**Screening requires ZERO code changes.** Use `--overrides` flag only.

After picking the winner, update `ml/config.py` RegressorConfig defaults:

- **If A wins**: change `learning_rate: float = 0.05` to `0.03`, `n_estimators: int = 400` to `700`, `reg_lambda: float = 1.0` to `5.0`, `min_child_weight: int = 10` to `25`
- **If B wins**: change `max_depth: int = 5` to `3`, `reg_lambda: float = 1.0` to `5.0`, `min_child_weight: int = 10` to `25`

Also update `ml/tests/test_config.py` if it has assertions on these default values.

No other files should be changed. No new features, no new functions, no pipeline restructuring.

---

## Expected Gate Impact

Both hypotheses include v0003's proven reg_lambda=5/mcw=25 base, which alone gave +0.0034 EV-VC@100 and +0.0035 EV-NDCG.

| Gate | v0 Mean | Expected Δ (A) | Expected Δ (B) | Reasoning |
|------|---------|----------------|----------------|-----------|
| EV-VC@100 | 0.0303 | +0.003 to +0.006 | +0.002 to +0.005 | A stacks two regularizations; B limits interactions |
| EV-VC@500 | 0.1180 | +0.000 to +0.005 | -0.005 to +0.005 | Broader ranking less sensitive; B might lose some interaction effects |
| EV-NDCG | 0.7400 | +0.003 to +0.008 | +0.002 to +0.006 | Both reduce overfitting → cleaner rankings |
| Spearman | 0.3921 | -0.005 to +0.005 | -0.01 to +0.005 | Rank correlation robust; B's depth cut might slightly reduce expressiveness |
| C-RMSE | 3400.4 | -20 to -100 | -50 to -200 | Both reduce variance; B more aggressive |

Both hypotheses should pass all Group A gates (v0003 base alone passes). The question is which pushes EV-VC@100 further.

---

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Combining lr+trees with L2 over-regularizes (A) | Low-Medium | Screen catches this; v0003 base is conservative |
| max_depth=3 under-fits, losing Spearman (B) | Medium | Safety check requires Spearman within 0.05 of v0 |
| Both hypotheses ~identical (stacked regularization saturates) | Medium | Even if no additional gain, winner still has v0003's proven +0.0034 |
| 700 trees increases training time (A) | Low | ~75% more trees, well within 35-min budget |
| Dirty state not fully reverted | Low | Step 0 has explicit verification gate |
| Worker modifies frozen files | Low | DO NOT MODIFY list is explicit; worktree isolation helps |
