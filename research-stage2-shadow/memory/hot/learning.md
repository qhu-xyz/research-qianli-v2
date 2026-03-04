## Accumulated Learning

### Worker Reliability (3 iterations: 2 complete failures, 1 partial success)
1. **Phantom completion is the dominant failure mode**: Iters 1+1b had workers write `worker_done.json` claiming "done" while producing zero artifacts.
2. **Workers can make unauthorized changes that persist**: Iter 1b worker modified 8+ files including v0 registry and gates.json. These UNCOMMITTED changes contaminated the working tree through iters 2 and 3, causing the orchestrator to read wrong baseline values.
3. **Worktree isolation works but has integration gap**: Iter 2 worker ran correctly in an isolated worktree (starting from clean committed state), but results on the worktree branch weren't merged back to main → WORKER_FAILED=1.
4. **Direction quality matters but isn't the only factor**: Iter 2's well-structured direction (DO NOT MODIFY list, exact commands, verification gates) resulted in correct worker execution. The iter 1b failure was direction violation, not direction complexity.
5. **Revert dirty state is MANDATORY before each iteration**: Dirty working tree files caused the orchestrator to write direction with wrong v0 reference numbers. Must use `git checkout -- ml/ registry/` or read from committed state.

### Dirty State Cascade (CRITICAL)
The iter-1b worker's uncommitted changes created a cascade of errors:
- `registry/v0/config.json`: train_months 10→6, features 24→34
- `registry/v0/metrics.json`: recalculated with wrong config (EV-VC@100 0.030→0.069)
- `registry/gates.json`: recalibrated from wrong v0 (floors ~2-3x too high)
- `ml/config.py` + 7 other ml/ files: extensive unauthorized modifications
- Iter 2 direction referenced dirty v0 (EV-VC@100=0.069 instead of 0.030)
- Any gate evaluation against dirty gates.json would fail ALL Group A gates

**Fix**: `git checkout -- ml/ registry/` restores all files to committed state.

### Pipeline Architecture
1. **Pipeline.py requires code modification for value_weighted**: `train.py` accepts `sample_weight`, `config.py` has the flag, but `pipeline.py` Phase 4 does NOT wire it. Defer this hypothesis until override-only hypotheses are exhausted.

### v0 Baseline Characteristics (COMMITTED, CORRECT values)
- EV-VC@100 mean=0.030, high variance (std=0.022), worst months: 2022-09 (0.002), 2021-05 (0.007)
- EV-VC@500 mean=0.118, worst months: 2022-09 (0.032), 2022-12 (0.054)
- EV-NDCG mean=0.740, worst months: 2021-03 (0.656), 2022-09 (0.691)
- Spearman mean=0.392, worst months: 2021-03 (0.320), 2022-12 (0.340)
- 2022-09 is the weakest month for EV-VC metrics; 2021-03 for EV-NDCG/Spearman
- Gate floors set to ~0.7x v0 mean, tail floors match worst single months

### Experimental Results
1. **Heavier L2 regularization works (v0003)**: reg_lambda 1→5, min_child_weight 10→25
   - EV-VC@100: +0.0034 mean (+11%), +0.0013 bottom-2 (+37%)
   - EV-NDCG: +0.0035 mean
   - Spearman: unchanged
   - C-RMSE: -22.7 (improved)
   - EV-VC@500: -0.0006 mean (flat), -0.0059 bottom-2 (within noise)
   - Mechanism: reduces overfitting on small binding subsets → better generalization
   - Passes all 3 layers on all 4 Group A gates (committed baseline)
2. **Screening on 2 months is effective**: 2022-06 + 2022-12 screen correctly identified winner (H3/B) that held up over full 12-month benchmark
