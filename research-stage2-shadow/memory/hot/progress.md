## Status: ORCHESTRATOR_SYNTHESIZING (iter 2 complete)
**Batch**: ralph-v1-20260304-003317
**Iteration**: 2 (synthesis complete)
**Champion**: v0 (unchanged — v0003 passes gates but artifacts not on main)

### Iteration History
| Iter | Batch | Version | Hypothesis | Result |
|------|-------|---------|-----------|--------|
| 1 (smoke-test) | smoke-test-20260303-223300 | v0001 | Value-weighted training | FAILED: phantom completion |
| 1 (ralph-v1) | ralph-v1-20260304-003317 | v0002 | lr+trees OR L2+leaves screen | FAILED: direction violation, dirty state left |
| 2 (ralph-v1) | ralph-v1-20260304-003317 | v0003 | Screen: A=lr+trees, B=L2+leaves → B wins | PARTIAL SUCCESS: valid results on worktree branch, infra failure |

### Iter 2 Results (v0003, on worktree branch `worker-iter2-ralph-v1-20260304-003317`)
- **Winner**: Hypothesis B (reg_lambda=5.0, min_child_weight=25)
- **vs committed v0**: EV-VC@100 +0.0034, EV-NDCG +0.0035, Spearman ±0, C-RMSE -23
- **Gate status**: Passes ALL 3 layers on ALL 4 Group A gates (committed baseline)
- **Blocked by**: Artifacts on worktree branch only, dirty main working tree

### Critical Issue: Dirty Codebase
Main working tree has uncommitted changes from iter-1 worker:
- `registry/v0/`, `registry/gates.json`: wrong v0 baseline (6/2/34feat instead of 10/2/24feat)
- `ml/config.py` + 7 other ml/ files: unauthorized modifications
- **Fix**: `git checkout -- ml/ registry/` + cherry-pick `01c22af`

### Next Steps (iter 3)
1. Revert dirty state on main
2. Cherry-pick v0003 commit `01c22af` from worktree branch
3. If successful → promote v0003
4. If cherry-pick fails → retry L2 regularization from clean state
