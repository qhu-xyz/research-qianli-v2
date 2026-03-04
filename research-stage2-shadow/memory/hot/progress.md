## Status: ORCHESTRATOR_PLANNING → WORKER (iter 2)
**Batch**: ralph-v1-20260304-003317
**Iteration**: 2 (planning complete, handing off to worker)
**Champion**: v0 (unchanged — no successful experiments yet)

### Iteration History
| Iter | Batch | Version | Hypothesis | Result |
|------|-------|---------|-----------|--------|
| 1 (smoke-test) | smoke-test-20260303-223300 | v0001 | Value-weighted training | FAILED: phantom completion |
| 1 (ralph-v1) | ralph-v1-20260304-003317 | v0002 | lr+trees OR L2+leaves screen | FAILED: direction violation, unauthorized changes |
| 2 (ralph-v1) | ralph-v1-20260304-003317 | v0002 | Screen: A=lr+trees, B=L2+leaves | IN PROGRESS |

### Iter 2 Plan
- **Hypothesis A**: lr=0.03, n_estimators=700 (smoother ensemble)
- **Hypothesis B**: reg_lambda=5.0, min_child_weight=25 (heavier regularization)
- **Screen months**: 2022-06 (weak) + 2022-12 (strong)
- **Winner criteria**: Higher mean EV-VC@100 across screen months; Spearman safety check
- Direction file: memory/direction_iter2.md

### Key Context
- Two consecutive worker failures — zero metrics produced across all iterations
- Both hypotheses are pure `--overrides` — no code changes needed for screening
- Codebase may be dirty from failed iter 1 worker — direction instructs revert first
