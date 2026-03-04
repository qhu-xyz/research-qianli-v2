# Hypothesis Log

## H1: Value-weighted regressor training improves EV ranking quality
- **Iteration**: 1 (smoke-test-20260303-223300)
- **Status**: UNTESTED (worker failed)
- **Rationale**: Upweight high-shadow-price training samples so model prioritizes accuracy on high-value constraints, improving EV-VC@100 and EV-VC@500
- **Planned changes**: value_weighted=True, n_estimators 400→600, learning_rate 0.05→0.03, reg_lambda 1.0→3.0, min_child_weight 10→15
- **Result**: Worker phantom-completed without producing artifacts. No data to confirm or reject.
- **Next action**: Deprioritized — requires pipeline.py code edit, which is higher-risk for worker failure. Try pure-override hypotheses first.

## H2: Slower learning rate + more trees improves tail months
- **Iteration**: 1 (ralph-v1-20260304-003317)
- **Status**: UNTESTED (worker failed — ignored direction entirely)
- **Rationale**: lr 0.05→0.03, n_estimators 400→700 creates smoother ensemble, reduces overfitting on tail months (2022-06, 2021-05)
- **Planned changes**: Pure hyperparameter override, no code edits
- **Result**: Worker ignored direction, attempted unauthorized classifier changes instead. No data produced.
- **Next action**: Retry in iter 2 with maximum guardrails — exact commands, DO NOT MODIFY file list

## H2+H3 Screen: lr+trees vs L2+leaves (iter 2)
- **Iteration**: 2 (ralph-v1-20260304-003317)
- **Status**: CONFIRMED (both tested, B=L2+leaves won tiebreak, full benchmark completed)
- **Rationale**: Screen two variance-reduction approaches — ensemble smoothing (H2) vs penalty regularization (H3)
- **Screen results** (2022-06 + 2022-12):
  - H2 (lr=0.03, n_est=700): mean EV-VC@100=0.0242, mean EV-NDCG=0.7284
  - H3 (lambda=5, mcw=25): mean EV-VC@100=0.0244, mean EV-NDCG=0.7303
  - H3 won on tiebreak (EV-VC@100 within 0.005, EV-NDCG +0.0019)
- **Full 12-month results (v0003, H3=winner, vs committed v0)**:
  - EV-VC@100: 0.0337 vs 0.0303 = **+0.0034 (+11%)** ← CONFIRMED improvement
  - EV-NDCG: 0.7435 vs 0.7400 = **+0.0035** ← CONFIRMED improvement
  - Spearman: 0.3921 vs 0.3921 = unchanged
  - EV-VC@500: 0.1174 vs 0.1180 = -0.0006 (flat)
  - C-RMSE: 3377.7 vs 3400.4 = -22.7 (improved)
  - Bottom-2 EV-VC@100: 0.0048 vs 0.0035 = **+0.0013 (+37%)**
- **Conclusion**: Heavier L2 regularization provides small but consistent improvement on EV ranking quality and tail safety. The mechanism (constraining overfitting on small binding subsets) is validated.
- **Infrastructure issue**: Results exist on worktree branch only; WORKER_FAILED=1 due to merge failure, not model quality.
- **Next action**: Cherry-pick v0003 commit to main and promote. If that fails, retry from clean state in iter 3.
