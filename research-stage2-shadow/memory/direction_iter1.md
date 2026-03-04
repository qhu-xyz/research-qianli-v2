# Direction — Iteration 1 (batch ralph-v2-20260304-031811)

## Context

Champion is **v0** (baseline, 6/2 train/val, 34 features, default regressor hyperparams).

Previous batch (ralph-v1) found L2 regularization (reg_lambda=5, mcw=25) to be the single best lever (+11% EV-VC@100 mean in 10/2/24feat config). That finding was never promoted to main due to infrastructure issues. This iteration re-validates L2 in the current 6/2/34feat pipeline config and tests whether row/column subsampling provides additional benefit.

### v0 Baseline (current champion)

| Metric | Mean | Std | Min | Bottom-2 |
|--------|------|-----|-----|----------|
| EV-VC@100 | 0.0690 | 0.056 | 0.00011 | 0.0068 |
| EV-VC@500 | 0.2160 | 0.111 | 0.0407 | 0.0558 |
| EV-NDCG | 0.7472 | 0.060 | 0.6045 | 0.6476 |
| Spearman | 0.3928 | 0.071 | 0.2649 | 0.2689 |

---

## Hypothesis A (primary) — L2 Regularization

**What**: Increase L2 penalty and minimum leaf weight to constrain overfitting on small binding subsets.

**Rationale**: L2 regularization was the strongest lever found in the previous batch (reg_lambda 1.0→5.0, mcw 10→25). With 34 features and only 6 months of training data (~10k binding samples/month), overfitting is the primary quality limiter. L2 shrinkage constrains extreme leaf weights that cause mis-ranking on out-of-sample months.

**Overrides**:
```json
{"regressor": {"reg_lambda": 5.0, "min_child_weight": 25}}
```

**Expected impact**: +5-15% EV-VC@100 mean, improved tail safety (bottom-2). Previous batch saw +11% mean and +37% bottom-2 in a 10/2/24feat config; effect should be similar or stronger with 34 features (more overfitting surface for L2 to control).

**Risk**: Low — confirmed across 12 months in previous batch (different config but same mechanism).

---

## Hypothesis B (alternative) — L2 + Subsampling Reduction

**What**: Combine L2 regularization with reduced row and column sampling to force tree diversity.

**Rationale**: With 34 features, colsample_bytree=0.6 means each tree sees ~20 features — reducing co-adapted feature splits. With 6 months of training data, subsample=0.6 forces each tree to learn from a different 60% of the binding sample, improving generalization. This is a DIFFERENT regularization mechanism from L2 (diversity via randomization vs shrinkage via penalty). Unlike the ensemble smoothing approach (lr/trees) tested in the previous batch — which COMPETED with L2 — subsampling should COMPLEMENT L2 because they operate on different axes.

**Overrides**:
```json
{"regressor": {"reg_lambda": 5.0, "min_child_weight": 25, "subsample": 0.6, "colsample_bytree": 0.6}}
```

**Expected impact**: Could add 3-8% on top of L2 alone, especially on weak months where the model overfits to idiosyncratic training patterns. If it works, this is strictly better than A.

**Risk**: Medium — too aggressive subsampling (0.6 is 25% less data per tree) could starve trees of signal on already-small binding subsets. Could hurt strong months if the model needs full feature/sample access to capture the signal.

---

## Screen Months

**Weak month: 2022-06**
- Consistently worst across ALL Group A gates: EV-NDCG=0.6045 (lowest), Spearman=0.2728 (2nd lowest), EV-VC@100=0.0136 (2nd lowest), EV-VC@500=0.0756 (2nd lowest)
- Systemic weakness (not just one metric), making it ideal for detecting overall improvements
- Also used as diagnostic month in previous batch — enables comparison

**Strong month: 2022-12**
- Strongest EV-VC@100 (0.1942), strong EV-NDCG (0.8153)
- Regression canary: if a config hurts this month, the regularization is too aggressive
- Also used in previous batch — enables comparison

---

## Winner Criteria

1. **Primary**: Higher mean EV-VC@100 across the 2 screen months (2022-06 and 2022-12)
2. **Tiebreak** (if EV-VC@100 difference < 0.002): Higher mean EV-NDCG across screen months
3. **Override**: If the winner has Spearman drop > 0.05 below v0 on either screen month, pick the other hypothesis instead (rank correlation collapse is unacceptable)
4. **Default**: If both are within noise of each other, prefer A (simpler config, lower risk)

---

## Code Changes for Winner

### If Hypothesis A wins

Update `ml/config.py` — `RegressorConfig` class defaults:
- `reg_lambda: float = 1.0` → `reg_lambda: float = 5.0`
- `min_child_weight: int = 10` → `min_child_weight: int = 25`

Update `tests/test_config.py` — adjust any assertions on default values.

### If Hypothesis B wins

Update `ml/config.py` — `RegressorConfig` class defaults:
- `reg_lambda: float = 1.0` → `reg_lambda: float = 5.0`
- `min_child_weight: int = 10` → `min_child_weight: int = 25`
- `subsample: float = 0.8` → `subsample: float = 0.6`
- `colsample_bytree: float = 0.8` → `colsample_bytree: float = 0.6`

Update `tests/test_config.py` — adjust any assertions on default values.

---

## DO NOT MODIFY

- Any file in `ml/` other than `config.py` (and `tests/test_config.py` for test assertions)
- `registry/gates.json`
- `ml/evaluate.py`
- Any classifier configuration (ClassifierConfig is FROZEN)
- `state.json` (orchestrator manages this)
