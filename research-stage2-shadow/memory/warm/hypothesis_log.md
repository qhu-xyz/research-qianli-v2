# Hypothesis Log

(Reset after batch feat-eng-3-20260304-121042 closure. See memory/archive/ for history.)

## Untested Hypotheses (carry forward from prior batch)

### H7: min_child_weight 25→15
- **Priority**: HIGH (next-batch primary)
- **Rationale**: Sharper leaves for top-100 discrimination. 600-tree ensemble provides natural regularization.
- **Risk**: Moderate — overfitting in low-signal months possible but mitigated by ensemble size
- **Override**: `{"regressor": {"min_child_weight": 15}}`

### H8: value_weighted=True
- **Priority**: MEDIUM (next-batch alternative)
- **Rationale**: Weight training loss by shadow price magnitude to emphasize high-$ constraints
- **Risk**: Higher uncertainty — heavy-tailed distribution could cause overfitting to outliers
- **Prerequisite**: Verify pipeline.py actually uses value_weighted during training
- **Override**: `{"regressor": {"value_weighted": true}}`
