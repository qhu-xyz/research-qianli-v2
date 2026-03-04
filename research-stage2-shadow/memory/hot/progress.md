## Status: ORCHESTRATOR_PLAN_DONE (iter 3 planned)
**Batch**: ralph-v1-20260304-003317
**Iteration**: 3 (plan written, awaiting worker)
**Champion**: v0 (unchanged — v0003 passed gates but never landed on main)

### Iteration History
| Iter | Batch | Version | Hypothesis | Result |
|------|-------|---------|-----------|--------|
| 1 (smoke-test) | smoke-test-20260303-223300 | v0001 | Value-weighted training | FAILED: phantom completion |
| 1 (ralph-v1) | ralph-v1-20260304-003317 | v0002 | lr+trees OR L2+leaves screen | FAILED: direction violation, dirty state left |
| 2 (ralph-v1) | ralph-v1-20260304-003317 | v0003 | Screen: A=lr+trees, B=L2+leaves → B wins | PARTIAL SUCCESS: valid results on worktree branch, infra failure |
| 3 (ralph-v1) | ralph-v1-20260304-003317 | v0003 | Screen: A=combined lr+trees+L2, B=depth=3+L2 | PLANNED |

### Iter 3 Plan
- **Hypothesis A**: Combined ensemble smoothing + L2 (lr=0.03, n_est=700, reg_lambda=5.0, mcw=25)
- **Hypothesis B**: Shallower trees + L2 (max_depth=3, reg_lambda=5.0, mcw=25)
- **Screen months**: 2022-09 (weak) + 2021-01 (strong)
- **Key insight**: Both build on v0003's proven reg_lambda=5/mcw=25 base, ensuring at minimum v0003-level results
- **Worker must**: Revert dirty state first (`git checkout -- ml/ registry/`), then screen with overrides

### Critical Issue: Dirty Codebase (STILL PRESENT)
Main working tree has uncommitted changes from iter-1 worker:
- `registry/v0/`, `registry/gates.json`: wrong v0 baseline (6/2/34feat instead of 10/2/24feat)
- `ml/config.py` + 7 other ml/ files: unauthorized modifications
- **Fix**: `git checkout -- ml/ registry/` (mandatory Step 0 in direction)
