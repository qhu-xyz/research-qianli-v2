## Accumulated Learnings

### Regressor Hyperparameter Findings

1. **L2 regularization (reg_lambda=5, mcw=25) is the single best lever found so far**
   - v0003 (ralph-v1, 10/2/24feat): EV-VC@100 mean +0.0034 (+11%), bot-2 +37%
   - v0005 (ralph-v2, 6/2/34feat): EV-VC@100 mean +0.0045 (+6.5%), bot-2 +23%
   - **Confirmed across 2 pipeline configs** — robust finding
   - Mechanism: constrains overfitting on small binding subsets (~10k samples/month)
   - Tradeoff: Spearman drops ~0.0008 (noise-level) — L2 compresses predictions, slightly hurting rank correlation while improving top-K value capture. Both reviewers noted this pattern.

2. **Reducing max_depth trades mean quality for tail safety — and depth=4 is NOT a sweet spot**
   - v0004 (ralph-v1, depth=3): EV-VC@100 bottom_2 doubled (+100%), but mean only +1%, EV-VC@500 -6%
   - v0006 screen (depth=4): EV-VC@100 weak month -42%, Spearman negligible change
   - **Depth reduction is not a viable Spearman lever** — it just destroys value capture on weak months

3. **Stacking ensemble smoothing (lr+trees) on L2 provides zero benefit**
   - lr=0.03/n_est=700 + reg_lambda=5/mcw=25 performed WORSE than either alone
   - The two mechanisms COMPETE rather than complement

4. **Aggressive subsampling (0.6) causes signal starvation with 34 features**
   - v0005 iter1 screen + full v1 benchmark confirmed: strong month degraded -24% EV-VC@100
   - Spearman collapsed -12% with 2 months below tail_floor
   - At subsample=0.6, each tree sees ~36k samples — too few for 34-feature regression
   - subsample=0.7 is UNTESTED — less aggressive, might complement L2 without starvation

5. **L1 regularization (reg_alpha=1.0) has negligible effect**
   - Iter 2 screen: Spearman ≈ +0.001 on strong month, +0.0 on weak month; EV-VC@100 -23% weak, -2.8% strong
   - Full benchmark INCONCLUSIVE (config bug — ran with 0.1 instead of 1.0)
   - Screen suggests all 29 active features carry enough signal to survive L1=1.0 sparsity
   - **Feature set is NOT the Spearman problem** — all features contribute, none are pure noise

6. **REGULARIZATION AXIS IS EXHAUSTED FOR SPEARMAN RECOVERY**
   - Tested: L2 (λ=1→5), L1 (α=0.1→1.0), depth (3,4,5), subsampling (0.6)
   - None recovered the 0.0008 Spearman gap caused by L2=5/mcw=25
   - The Spearman compression is a structural property of L2 regularization compressing the prediction distribution
   - **Next iteration MUST explore non-regularization axes** (value-weighted, training mode, etc.)

### Pipeline Findings (from Code Reviews)

7. **Train-inference mismatch in gated mode** (NOT YET FIXED)
   - Regressor trains on rows where `y_train_binary == 1` (true binding labels)
   - But at inference, regressor applies to rows where classifier predicts positive
   - This creates distribution shift between training and inference
   - Affects both v0 and v0005 equally — structural issue, not a regression
   - Fix requires modifying pipeline.py (out of scope for config-only changes)
   - Location: `ml/pipeline.py:195-209` (training) vs `ml/pipeline.py:249-253` (inference)

8. **Feature importance pipeline is wired but never populated**
   - benchmark.py expects `_feature_importance` in month metrics but nothing emits it
   - Cannot validate whether regressor-specific features contribute
   - Fix: add `model.get_score()` export to per-month metrics

9. **Config provenance is not verified** (NEW from iter 2)
   - v0006 config bug: code change committed after benchmark, but benchmark used pre-change defaults
   - registry/v0006/config.json correctly records what was used (reg_alpha=0.1), contradicting changes_summary
   - **No automated check** that claimed overrides match saved config.json
   - Both reviewers independently flagged this — process improvement needed

10. **Test suite is misaligned with code** (NEW from iter 2)
    - Tests expect 13 classifier features / 24 regressor features
    - Code has 14 classifier features / 34 regressor features
    - Weak guardrails contributed to the config bug going undetected

### Screening Methodology
- **Two-month screening works**: 3/3 screens correctly predicted full-benchmark winner (iters 2-3 ralph-v1, iter 1 ralph-v2)
- **Screen month selection**: Use one weak + one strong month for diagnostic coverage
- **Weak months tested**: 2022-06, 2022-09, 2021-01, 2021-11
- **Strong months tested**: 2022-12
- **Screen can show effects the full benchmark doesn't** (iter 2: screen showed L1=1.0 difference, but full benchmark had config bug)

### Gate System Issues
- **Current gates are dysfunctional**: floors at v0 exact mean → v0 fails its own L1 gates
- **noise_tolerance=0.02 is not scale-aware**: meaningless for C-RMSE (~3000 scale), generous for EV-VC@100 (~0.07 scale)
- **Both reviewers independently flagged gate calibration** as the primary blocker in BOTH iterations
- Previous gate_calibration.md documented correct gates (floor ≈ 0.87x v0 mean) but they were never applied to gates.json
- **Spearman gate has now blocked 2 iterations** — EV-VC improvements of +6.5% / +5.9% remain stranded

### Worker Reliability Patterns
1. **Phantom completion** is dominant failure mode in earlier batches
2. **Direction quality correlates with worker compliance**: explicit DO NOT MODIFY lists + exact commands → correct execution
3. **ralph-v2 iter 1**: First successful worker execution in this pipeline — full screen + benchmark + clean commit
4. **ralph-v2 iter 2**: Worker correctly executed screening and winner selection, but CONFIG BUG in full benchmark (overrides not applied). New failure mode: benchmark infrastructure doesn't propagate config changes.
5. **ALWAYS read baselines from committed state** (`git show HEAD:...`), never from working tree
