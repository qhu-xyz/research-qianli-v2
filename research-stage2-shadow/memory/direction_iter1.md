# Direction — Iteration 1

**Batch**: screen-test-20260304-000123
**Champion**: v0 (baseline)
**Objective**: Improve regressor quality to boost EV ranking metrics while maintaining tail safety

## Context

This is the first iteration of a new batch. The previous batch's iter 1 failed (phantom worker completion). No successful regressor experiments exist yet.

**v0 diagnosis**: The baseline shows extreme per-month variance in EV-VC@100 (std=0.056 on mean=0.069), suggesting the regressor overfits to training-window patterns that don't generalize. The universal worst month is 2022-06 (EV-NDCG=0.604, Spearman=0.273, C-RMSE=5918). Gate floors equal v0 means — any regression fails promotion.

**Strategy**: Since this is the first iteration with no prior signal, we test two orthogonal regularization strategies — structural regularization (fewer parameters per tree) vs penalty-based regularization (stronger L2/leaf constraints). Both aim to reduce variance and improve weak-month performance.

## Screen Months

- **Weak month: 2022-06** — Universal worst: EV-NDCG=0.604, Spearman=0.273, C-RMSE=5918, C-MAE=2283. Regression quality collapses here. Any improvement in regularization should help this month most.
- **Strong month: 2022-12** — Best EV-VC@100=0.194, strong EV-NDCG=0.815, strong EV-VC@500=0.344. Must NOT regress here. Good sanity check that regularization doesn't kill strong-month performance.

**Rationale**: 2022-06 is the clear stress test — it drives the bottom-2 mean for EV-NDCG and Spearman. If regularization helps here, it directly improves Layer 3 gate checks. 2022-12 is the best-performing month; if regularization kills performance here, the hypothesis is a net negative.

## Hypothesis A (Primary): Slower learning with more trees

**Idea**: Reduce learning rate and compensate with more trees. Lower lr means each tree makes smaller corrections → smoother ensemble → better ranking stability. The product `lr × n_estimators` controls total learning capacity; keeping it roughly constant (0.05×400=20 → 0.03×700=21) preserves capacity while smoothing.

**Override JSON**:
```json
{"regressor": {"learning_rate": 0.03, "n_estimators": 700}}
```

**Screen command**:
```bash
cd /home/xyz/workspace/pmodel && source .venv/bin/activate
cd /home/xyz/workspace/research-qianli-v2/research-stage2-shadow
python -m ml.benchmark --version-id _screenA --ptype f0 --class-type onpeak \
  --eval-months 2022-06 2022-12 \
  --overrides '{"regressor": {"learning_rate": 0.03, "n_estimators": 700}}'
```

**Why this might work**:
- XGBoost with lower lr + more trees is a well-known recipe for better generalization
- Smoother model → fewer extreme predictions → better ranking in volatile months
- Conservative change: doesn't touch feature set, regularization penalties, or tree depth

**Risk**: ~75% longer training time per month (700 vs 400 trees). Acceptable for screening (2 months). Manageable for full 12-month run.

## Hypothesis B (Alternative): Stronger L2 + larger leaf size

**Idea**: Increase L2 penalty (reg_lambda 1→5) and minimum leaf weight (min_child_weight 10→20). This prevents the model from fitting small groups of training samples with extreme predictions — exactly the failure mode that causes volatile month-to-month performance.

**Override JSON**:
```json
{"regressor": {"reg_lambda": 5.0, "min_child_weight": 20}}
```

**Screen command**:
```bash
cd /home/xyz/workspace/pmodel && source .venv/bin/activate
cd /home/xyz/workspace/research-qianli-v2/research-stage2-shadow
python -m ml.benchmark --version-id _screenB --ptype f0 --class-type onpeak \
  --eval-months 2022-06 2022-12 \
  --overrides '{"regressor": {"reg_lambda": 5.0, "min_child_weight": 20}}'
```

**Why this might work**:
- Higher reg_lambda penalizes large leaf weights → shrinks extreme predictions toward zero
- Higher min_child_weight requires more samples per leaf → prevents overfitting to rare patterns
- Together, they make the model more conservative on uncertain predictions, which should help ranking

**Risk**: Over-regularization could flatten predictions, hurting Spearman (rank correlation needs spread in predictions). If both screen months show Spearman drop > 0.05, this hypothesis is too aggressive.

## Winner Criteria

1. **Primary**: Higher mean EV-VC@100 across the 2 screen months (2022-06 + 2022-12)
2. **Safety gate**: Spearman must not drop > 0.05 vs v0 on EITHER screen month. If both hypotheses fail this, pick the one with smaller Spearman drop.
3. **Tiebreaker**: Higher mean EV-NDCG across the 2 screen months

**v0 baselines for screen months** (for comparison):
| Month | EV-VC@100 | EV-VC@500 | EV-NDCG | Spearman |
|-------|-----------|-----------|---------|----------|
| 2022-06 | 0.0136 | 0.0756 | 0.6045 | 0.2728 |
| 2022-12 | 0.1942 | 0.3445 | 0.8153 | 0.3847 |

## Code Changes for Winner

Both hypotheses are pure hyperparameter changes. For the winning config:

1. **Update `ml/config.py`** — Modify `RegressorConfig.__init__` defaults to match the winning override values.
   - For Hypothesis A winner: change `learning_rate=0.05` → `0.03`, `n_estimators=400` → `700`
   - For Hypothesis B winner: change `reg_lambda=1.0` → `5.0`, `min_child_weight=10` → `20`

2. **No other code changes needed** — these are pure config defaults, no pipeline wiring required.

3. **Full benchmark command** (after code change):
```bash
cd /home/xyz/workspace/pmodel && source .venv/bin/activate
cd /home/xyz/workspace/research-qianli-v2/research-stage2-shadow
python -m ml.benchmark --version-id v0001 --ptype f0 --class-type onpeak
```

## Expected Impact

| Gate | v0 Mean | Expected Direction | Reasoning |
|------|---------|-------------------|-----------|
| EV-VC@100 | 0.069 | +0.005 to +0.015 | Better ranking from smoother predictions |
| EV-VC@500 | 0.216 | +0.005 to +0.010 | Same mechanism, broader scope |
| EV-NDCG | 0.747 | +0.005 to +0.015 | Ranking quality improves with regularization |
| Spearman | 0.393 | ±0.010 | May slightly improve or be neutral |
| C-RMSE | 3133 | -100 to +100 | Regularization may slightly increase error on train but reduce on test |

**Bottom-2 mean (tail safety)**: Main expected benefit — regularization should improve 2022-06 performance, directly lifting the bottom-2 mean for all Group A metrics.

## Risk Assessment

1. **Over-regularization**: Too much smoothing could make all predictions similar, destroying ranking ability. Screen month check catches this — watch Spearman.
2. **Training time**: Hypothesis A (700 trees) takes ~75% longer. Full 12-month run: ~60 min instead of ~35 min. Acceptable.
3. **Marginal improvement**: Both hypotheses are conservative. If neither moves the needle meaningfully on screen months, future iterations should try bolder changes (feature selection, value-weighted training with pipeline.py wiring, unified regressor mode).
4. **2022-06 may be classifier-limited**: The frozen classifier has its worst S1-NDCG (0.606) in this month. Even perfect regression won't fully fix EV metrics if the classifier's ranking is poor. But improving regression quality in this month still helps.

## Notes for Worker

- **Both hypotheses use `--overrides` only** — no code changes needed for screening
- **Save screen results** to compare: note the per-month EV-VC@100, EV-NDCG, and Spearman for both hypotheses
- **If BOTH hypotheses lose to v0** on screen months: still pick the one that's closest to v0, implement it, and run full benchmark. The 2-month screen is noisy — full 12-month may show different results.
- **value_weighted is intentionally deferred** to a future iteration — it requires pipeline.py code changes that can't be screened via --overrides
