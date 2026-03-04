## Accumulated Learnings

### Regressor Hyperparameter Findings

1. **mcw=25 (min_child_weight) is the best single lever found** (NEW — updated from batch ralph-v2)
   - v0007 (reg_lambda=1.0, mcw=25): EV-VC@500 +6.2%, EV-NDCG +0.5%, Spearman +0.1%
   - **Pure EV-VC lever**: mcw=25 improves value capture without affecting Spearman
   - Mechanism: larger leaves → more conservative predictions → better rank ordering at scale
   - The prior finding (L2+mcw together) was a compound effect; decomposition in iter 3 proved mcw was the beneficial component

2. **reg_lambda (L2) > 1.0 compresses Spearman** (REFINED — decomposed in batch ralph-v2)
   - v0005 (λ=5, mcw=25): EV-VC@100 +6.5%, but Spearman -0.0008 — L2 was the culprit
   - v0007 (λ=1, mcw=25): EV-VC@500 +6.2%, Spearman +0.0004 — L2 reversion fixed it
   - Screen Hyp A (λ=3, mcw=10): Even moderate L2 compresses predictions → hurts weak-month value capture
   - **L2 regularization compresses the prediction distribution**: helps top-K value capture by reducing noise, but hurts rank correlation by flattening the prediction surface
   - Optimal: reg_lambda=1.0 (v0 default). Do NOT increase L2 for regressor.

3. **Reducing max_depth trades mean quality for tail safety — depth=4 is NOT a sweet spot**
   - depth=3: EV-VC@100 bot-2 +100%, but mean only +1%, EV-VC@500 -6%
   - depth=4: EV-VC@100 weak month -42%, Spearman negligible
   - **Depth reduction is not a viable Spearman lever**

4. **Stacking ensemble smoothing (lr+trees) on L2 provides zero benefit**
   - lr=0.03/n_est=700 + reg_lambda=5/mcw=25 performed WORSE than either alone
   - The two mechanisms COMPETE rather than complement

5. **Aggressive subsampling (0.6) causes signal starvation with 34 features**
   - Spearman collapsed -12%, strong month degraded -24% EV-VC@100
   - subsample=0.7 is UNTESTED — possible future direction

6. **L1 regularization (reg_alpha=1.0) has negligible effect**
   - Screen suggests all features carry signal; feature set is NOT the Spearman problem
   - Full benchmark INCONCLUSIVE (config bug in iter 2)

7. **REGULARIZATION AXIS IS EXHAUSTED FOR SPEARMAN RECOVERY**
   - Tested: L2 (λ=1→5), L1 (α=0.1→1.0), depth (3,4,5), subsampling (0.6)
   - Spearman recovery came from REVERTING L2, not from any regularization increase
   - Future Spearman improvements must come from non-regularization axes

### Pipeline Findings (from Code Reviews)

8. **Train-inference mismatch in gated mode** (NOT YET FIXED)
   - Regressor trains on `y_train_binary == 1` but infers on classifier predictions
   - Structural issue affecting all versions equally
   - Location: `ml/pipeline.py:195-209` (training) vs `ml/pipeline.py:249-253` (inference)

9. **Feature importance pipeline wired but never populated**
   - benchmark.py expects `_feature_importance` but nothing emits it
   - Fix: add `model.get_score()` export to per-month metrics

10. **Config provenance vulnerability** (partially mitigated)
    - v0006 config bug: benchmark used pre-change defaults
    - v0007 verified correctly via config.json provenance check
    - Explicit direction file instructions prevented recurrence

11. **Test suite corrected in iter 3** (was misaligned: 13/24 → 14/34 features)
    - Frozen-dataclass test was removed — verify ClassifierConfig is still frozen

12. **Temporal leakage concern** (NEW from iter 3 Codex review)
    - data_loader.py train_end may be inclusive → potential leakage
    - Structural issue needing audit, not v0007-specific

### Screening Methodology
- **Two-month screening works**: 4/4 screens correctly predicted full-benchmark relative winner
- **Screen month selection**: one weak + one strong for diagnostic coverage
- **Weak months tested**: 2022-06, 2022-09, 2021-01, 2021-11
- **Strong months tested**: 2022-12

### Gate System Issues
- **Floors at v0 exact mean**: blocked 2 of 3 iterations despite clear EV-VC improvements
- **noise_tolerance=0.02 is not scale-aware**: meaningless for C-RMSE, generous for EV-VC@100
- **HUMAN_SYNC requested 3x** across this batch — still pending

### Worker Reliability Patterns
1. **Phantom completion** is dominant failure mode in earlier batches
2. **Direction quality correlates with worker compliance**: explicit constraints → correct execution
3. **Config bug prevention**: explicit verification instructions in direction file fixed the iter 2 bug
4. **ALWAYS read baselines from committed state** (`git show HEAD:...`), never from working tree
