## Status: BATCH COMPLETE (ralph-v1-20260304-003317, 3 iterations)
**Champion**: v0 (unchanged — v0003 and v0004 both passed gates but neither landed on main)
**Recommended action**: Cherry-pick v0003 (commit `01c22af` on branch `worker-iter2-ralph-v1-20260304-003317`) to main and promote. v0004 (commit `fca00aa`) is alternative if tail safety prioritized.

### Iteration History
| Iter | Version | Hypothesis | Result | Metrics on worktree |
|------|---------|-----------|--------|---------------------|
| 1 | v0002 | lr+trees OR L2+leaves screen | FAILED: direction violation, dirty state left | None |
| 2 | v0003 | Screen: A=lr+trees, B=L2+leaves → B wins | PARTIAL: valid on worktree, infra fail | EV-VC@100 mean=0.0337 (+11%) |
| 3 | v0004 | Screen: A=lr+trees+L2, B=depth=3+L2 → B wins | PARTIAL: valid on worktree, infra fail | EV-VC@100 mean=0.0306 (+1%), bottom_2=0.0070 (+100%) |

### Key Findings
1. L2 regularization (reg_lambda=5, mcw=25) confirmed as beneficial (+11% EV-VC@100 mean)
2. Depth=3 trades mean quality for tail safety (bottom_2 doubled, mean flat)
3. Stacking lr/trees smoothing on L2 provides zero benefit (mechanisms compete)
4. Worker worktree isolation works for execution but integration to main is broken
5. 2 of 3 iterations had correct worker execution; 1 had complete direction violation

### Infrastructure Issue
All 3 iterations ended with WORKER_FAILED=1. Iters 2-3 produced valid results on worktree branches that never merged to main. Root cause: worktree→main integration gap in the pipeline automation.
