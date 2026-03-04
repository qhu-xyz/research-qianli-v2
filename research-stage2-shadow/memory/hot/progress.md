## Status: ORCHESTRATOR_PLANNING_DONE (iter 1)
**Batch**: ralph-v1-20260304-003317
**Iteration**: 1
**Champion**: v0 (unchanged)

### Iteration History
| Iter | Version | Hypothesis | Result |
|------|---------|-----------|--------|
| 1 | (pending) | A: slower lr + more trees; B: stronger L2 + larger leaf | Screening → awaiting worker |

### Current Iteration Plan
- **Hypothesis A**: learning_rate 0.05→0.03, n_estimators 400→700 (smoother ensemble)
- **Hypothesis B**: reg_lambda 1→5, min_child_weight 10→25 (penalty-based regularization)
- **Screen months**: 2022-06 (weak) + 2022-12 (strong)
- **Winner criteria**: Higher mean EV-VC@100 across screen months; Spearman safety check (no drop > 0.05)
- **Direction file**: memory/direction_iter1.md
