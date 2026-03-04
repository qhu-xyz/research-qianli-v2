## Accumulated Learnings

### Regressor Hyperparameter Findings

1. **L2 regularization (reg_lambda=5, mcw=25) is the single best lever found so far**
   - v0003 (ralph-v1, 10/2/24feat): EV-VC@100 mean +0.0034 (+11%), bot-2 +37%
   - v0005 (ralph-v2, 6/2/34feat): EV-VC@100 mean +0.0045 (+6.5%), bot-2 +23%
   - **Confirmed across 2 pipeline configs** — robust finding
   - Mechanism: constrains overfitting on small binding subsets (~10k samples/month)
   - Tradeoff: Spearman drops ~0.0008 (noise-level) — L2 compresses predictions, slightly hurting rank correlation while improving top-K value capture. Both reviewers noted this pattern.

2. **Reducing max_depth (5→3) trades mean quality for tail safety**
   - v0004 (ralph-v1): EV-VC@100 bottom_2 doubled (+100%), but mean only +1%, EV-VC@500 -6%
   - depth=4 UNTESTED — possible sweet spot (16 leaves vs 32@depth5 or 8@depth3)

3. **Stacking ensemble smoothing (lr+trees) on L2 provides zero benefit**
   - lr=0.03/n_est=700 + reg_lambda=5/mcw=25 performed WORSE than either alone
   - The two mechanisms COMPETE rather than complement

4. **Aggressive subsampling (0.6) causes signal starvation with 34 features**
   - v0005 iter1 screen + full v1 benchmark confirmed: strong month degraded -24% EV-VC@100
   - Spearman collapsed -12% with 2 months below tail_floor
   - At subsample=0.6, each tree sees ~36k samples — too few for 34-feature regression
   - subsample=0.7 is UNTESTED — less aggressive, might complement L2 without starvation

5. **reg_alpha (L1) is UNTESTED**
   - Currently 0.1, could increase to 0.5-1.0 for implicit feature selection
   - Different mechanism from L2: sparsity (zeroing features) vs shrinkage (reducing weights)
   - Could improve Spearman by eliminating noisy features

### Pipeline Findings (from Codex code review)

6. **Train-inference mismatch in gated mode** (NOT YET FIXED)
   - Regressor trains on rows where `y_train_binary == 1` (true binding labels)
   - But at inference, regressor applies to rows where classifier predicts positive
   - This creates distribution shift between training and inference
   - Affects both v0 and v0005 equally — structural issue, not a regression
   - Fix requires modifying pipeline.py (out of scope for config-only changes)
   - Location: `ml/pipeline.py:195-209` (training) vs `ml/pipeline.py:249-253` (inference)

7. **Feature importance pipeline is wired but never populated**
   - benchmark.py expects `_feature_importance` in month metrics but nothing emits it
   - Cannot validate whether regressor-specific features contribute
   - Fix: add `model.get_score()` export to per-month metrics

### Screening Methodology
- **Two-month screening works**: 3/3 screens correctly predicted full-benchmark winner (iters 2-3 ralph-v1, iter 1 ralph-v2)
- **Screen month selection**: Use one weak + one strong month for diagnostic coverage
- **Weak months tested**: 2022-06, 2022-09, 2021-01
- **Strong months tested**: 2022-12, 2021-01

### Gate System Issues
- **Current gates are dysfunctional**: floors at v0 exact mean → v0 fails its own L1 gates
- **noise_tolerance=0.02 is not scale-aware**: meaningless for C-RMSE (~3000 scale), generous for EV-VC@100 (~0.07 scale)
- **Both reviewers independently flagged gate calibration** as the primary blocker
- Previous gate_calibration.md documented correct gates (floor ≈ 0.87x v0 mean) but they were never applied to gates.json

### Worker Reliability Patterns
1. **Phantom completion** is dominant failure mode in earlier batches
2. **Direction quality correlates with worker compliance**: explicit DO NOT MODIFY lists + exact commands → correct execution
3. **ralph-v2 iter 1**: First successful worker execution in this pipeline — full screen + benchmark + clean commit
4. **ALWAYS read baselines from committed state** (`git show HEAD:...`), never from working tree
