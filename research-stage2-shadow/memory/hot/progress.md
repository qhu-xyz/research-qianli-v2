## Status: ORCHESTRATOR_PLANNING
**Batch**: smoke-test-20260303-223300
**Iteration**: 1
**State**: Direction written, awaiting worker execution

### Iteration 1 Plan
- **Hypothesis**: Value-weighted regressor training improves EV ranking quality
- **Key changes**: value_weighted=True, n_estimators 400→600, learning_rate 0.05→0.03, reg_lambda 1.0→3.0, min_child_weight 10→15
- **Pipeline code change needed**: Wire up sample_weight computation in pipeline.py Phase 4
- **Direction file**: `memory/direction_iter1.md`
- **Primary target**: EV-VC@100, EV-VC@500 improvement
- **Key risk**: Spearman regression from over-weighting the tail
