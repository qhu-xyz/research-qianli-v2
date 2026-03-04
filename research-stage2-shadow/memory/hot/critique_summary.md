## Critique Summary

### Iter 1 — smoke-test batch (Worker Failed)
Worker phantom-completed without producing artifacts. No reviews generated.

### Iter 1 — ralph-v1 batch (Worker Failed)
Worker ignored direction entirely — made unauthorized changes to frozen classifier, evaluate.py (HUMAN-WRITE-ONLY), and 6 other files. No pipeline run, no metrics, no reviews. Uncommitted changes contaminated working tree (v0 registry, gates.json, config.py all dirtied).

### Iter 2 — ralph-v1 batch (Worker Partial Success, WORKER_FAILED=1)
Worker produced valid v0003 on worktree branch (commit `01c22af`). **No reviews were generated** — WORKER_FAILED=1 triggered before review stage.

**Direction quality assessment:**
- Direction was well-structured with DO NOT MODIFY list, exact commands, verification checkpoints
- Worker followed direction correctly on the worktree branch — executed screen, picked winner, ran benchmark
- **Critical error**: Direction referenced DIRTY v0 baseline (EV-VC@100=0.069, 34 features) instead of committed v0 (EV-VC@100=0.030, 24 features). Orchestrator read dirty working-tree files instead of committed state.
- Despite wrong reference numbers, worker execution was correct — worktrees start from committed state
- Infrastructure failure: worktree results didn't merge to main → WORKER_FAILED=1

**Lessons for iter 3:**
- Orchestrator MUST read v0 from committed state (`git show HEAD:...`), not working tree
- Verify dirty state is reverted BEFORE writing direction
- Screening approach (2 hypotheses + full benchmark on winner) worked when worker executed it
- The two-hypothesis screen was NOT too complex — previous failure was direction violation, not complexity
