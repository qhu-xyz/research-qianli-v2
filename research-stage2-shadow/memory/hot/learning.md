## Accumulated Learnings

### Regressor Hyperparameter Findings (from batch ralph-v1-20260304-003317)

1. **L2 regularization (reg_lambda=5, mcw=25) is the single best lever found so far**
   - v0003: EV-VC@100 mean +0.0034 (+11%), EV-NDCG +0.0035, Spearman unchanged
   - Bottom_2 EV-VC@100 +0.0013 (+37%) — tail safety also improves
   - Mechanism: constrains overfitting on small binding subsets (~10k samples/month)
   - No regressions on any Group A gate mean

2. **Reducing max_depth (5→3) trades mean quality for tail safety**
   - v0004: EV-VC@100 bottom_2 doubled (+100%), but mean EV-VC@100 only +1%, EV-VC@500 -6%
   - 8 leaf nodes limits spurious higher-order interactions; helps weak months
   - depth=3 + L2 (v0004) < depth=5 + L2 (v0003) for mean ranking quality
   - depth=4 untested — possible sweet spot between mean and tail

3. **Stacking ensemble smoothing (lr+trees) on L2 provides zero benefit**
   - lr=0.03/n_est=700 + reg_lambda=5/mcw=25 performed WORSE than either alone
   - Over-regularizes weak months (2022-09: 0.0016 vs v0's 0.002)
   - The two mechanisms compete rather than complement

4. **Ensemble smoothing alone (lr=0.03, n_est=700) ≈ L2 alone on screen months**
   - Iter 2 screen: lr+trees mean EV-VC@100=0.0242 vs L2 mean=0.0244
   - L2 won on tiebreak (EV-NDCG +0.0019) and was confirmed in full benchmark
   - Ensemble smoothing is a valid axis but L2 is slightly better

### Screening Methodology
- **Two-month screening works**: 2/2 screens correctly predicted full-benchmark winner
- **Screen month selection matters**: Use one weak + one strong month for diagnostic coverage
- **Weak months used**: 2022-06, 2022-09, 2022-12 (EV-VC weakness); 2021-03 (EV-NDCG/Spearman weakness)
- **Strong months used**: 2021-01, 2022-12 (strong across all gates)

### v0 Baseline Characteristics
- EV-VC@100: highest relative variance (std/mean ≈ 0.88), hardest to improve consistently
- Worst months: 2022-09 (EV-VC@100=0.002), 2021-03 (EV-NDCG=0.656, Spearman=0.320)
- Gate floors at ~0.7x v0 mean — well-calibrated with ~30% headroom

### Worker Reliability Patterns
1. **Phantom completion** is dominant failure mode — workers claim "done" without artifacts
2. **Direction violation** occurred once (iter 1): worker attempted unauthorized classifier changes
3. **Worktree isolation works for execution** but has integration gap to main
4. **Direction quality correlates with worker compliance**: explicit DO NOT MODIFY lists + exact commands → correct execution
5. **Dirty state from failed workers persists** and cascades — must revert before each iteration
6. **ALWAYS read baselines from committed state** (`git show HEAD:...`), never from working tree

### Infrastructure Issues to Fix
1. **Worktree→main integration**: Auto-merge/cherry-pick after worker success on worktree
2. **Dirty state prevention**: Mandatory `git checkout -- ml/ registry/` at batch start
3. **Worker_done.json validation**: Pre-flight check that claimed artifacts actually exist
4. **File permissions**: Read-only on HUMAN-WRITE-ONLY files in worktrees

### Unexplored Directions
1. **Value-weighted training**: Requires pipeline.py edit. High potential, higher risk.
2. **subsample/colsample tuning**: 0.8→0.6-0.7 — untouched regularization axis
3. **reg_alpha (L1)**: Currently 0.1 — could zero out noisy features
4. **Feature selection**: Drop low-importance features with reg_lambda=5 base
5. **depth=4 + L2**: Compromise between v0003 (depth=5) and v0004 (depth=3)
