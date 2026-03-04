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

## H3: Heavier L2 + larger leaves reduces tail variance
- **Iteration**: 1 (ralph-v1-20260304-003317) — planned as alternative to H2
- **Status**: UNTESTED (worker failed — never reached screening)
- **Rationale**: reg_lambda 1→5, min_child_weight 10→25 constrains model from fitting noise in small binding subsets
- **Planned changes**: Pure hyperparameter override, no code edits
- **Result**: Never tested — worker failed before screening phase
- **Next action**: If H2 wins screening in iter 2, H3 becomes iter 3 hypothesis
