# Hypothesis Log

## H1: Value-weighted regressor training improves EV ranking quality
- **Iteration**: 1 (smoke-test-20260303-223300)
- **Status**: UNTESTED (worker failed)
- **Rationale**: Upweight high-shadow-price training samples so model prioritizes accuracy on high-value constraints, improving EV-VC@100 and EV-VC@500
- **Planned changes**: value_weighted=True, n_estimators 400→600, learning_rate 0.05→0.03, reg_lambda 1.0→3.0, min_child_weight 10→15
- **Result**: Worker phantom-completed without producing artifacts. No data to confirm or reject.
- **Next action**: Retry in iteration 2 with simplified instructions (fewer simultaneous hyperparameter changes, more explicit pipeline.py edit)
