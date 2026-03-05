# Progress

## Current State
- **Champion**: v0 (baseline, unchanged)
- **Current batch**: tier-fe-2-20260304-225923 (FE only)
- **Current iteration**: 2 (planning complete, worker next)
- **Version counter**: next_id=4 (leaked from 3 consecutive worker failures)
- **Consecutive worker failures**: 3 (2 from tier-fe-1, 1 from tier-fe-2)

## Batch History

### Previous batch: tier-fe-1 (3 iterations, ALL FAILED)
- Worker execution failures -- no artifacts produced in any iteration
- Hypotheses (interaction features + pruning) remain untested
- Version counter leaked to next_id=3

### Current batch: tier-fe-2

**Iter 1 -- WORKER FAILED**
- Direction: Add 3 interaction features (overload_x_hist, prob110_x_recent_hist, tail_x_hist), screen 2 months, full benchmark
- Worker wrote handoff claiming done but made zero code changes and produced zero artifacts
- Version counter leaked: 3->4

**Iter 2 -- PLANNED (orchestrator plan complete)**
- Hypothesis A: Add 3 interaction features (34->37)
- Hypothesis B: Add 3 interactions + prune 4 low-importance (34->33)
- Screen months: 2022-06 (weak) + 2021-09 (strong)
- Full 2-hypothesis screening protocol restored
- Direction written to memory/direction_iter2.md

## Priority Improvement Areas
1. Tier-VC@100 below floor (0.071 vs 0.075) -- only Group A gate failing Layer 1
2. Tier0-AP mean 0.306 -- high variance (0.114 to 0.594), worst months late 2022
3. Tier01-AP mean 0.311 -- barely passing, worst months 2022-06 (0.195), 2022-12 (0.194)
4. Tier-Recall@1 catastrophically low (0.047) -- missing most strongly binding constraints
5. High variance across months -- 2022-06 is the worst month across most metrics
