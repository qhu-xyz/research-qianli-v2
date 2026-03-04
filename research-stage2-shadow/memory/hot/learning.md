## Accumulated Learnings

### Regressor Hyperparameter Findings

1. **mcw=25 (min_child_weight) is the best single lever found** (from batch ralph-v2)
   - v0007 (reg_lambda=1.0, mcw=25): EV-VC@500 +6.2%, EV-NDCG +0.5%, Spearman +0.1%
   - **Pure EV-VC lever**: mcw=25 improves value capture without affecting Spearman
   - Mechanism: larger leaves → more conservative predictions → better rank ordering at scale
   - The prior finding (L2+mcw together) was a compound effect; decomposition proved mcw was the beneficial component

2. **reg_lambda (L2) > 1.0 compresses Spearman** (REFINED)
   - v0005 (λ=5, mcw=25): EV-VC@100 +6.5%, but Spearman -0.0008 — L2 was the culprit
   - v0007 (λ=1, mcw=25): EV-VC@500 +6.2%, Spearman +0.0004 — L2 reversion fixed it
   - **L2 regularization compresses the prediction distribution**: helps top-K value capture but hurts rank correlation
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

### Feature Engineering Findings

8. **Distributional shape features (skewness, kurtosis, CV) carry real signal** (CONFIRMED)
    - Adding density_skewness, density_kurtosis, density_cv → EV-VC@100 +9.0%, C-RMSE -3.1%
    - Improvement is concentrated in ~3 months (2020-09, 2020-11, 2021-09 drive most of the +9%)

9. **season_hist_da_3 and prob_below_85 contributed alongside distributional features** (WEAK EVIDENCE)
    - 39 features beat 37 on screen (+32% EV-VC@100 mean on 2 months)
    - Not independently tested — could be noise or genuine additional signal

10. **Pruning 5 dead features improves top-100 precision but trades breadth** (NEW — confirmed in v0011)
    - Removing zero-filled features (hist_physical_interaction, overload_exceedance_product, band_severity, sf_exceed_interaction, hist_seasonal_band) from 39→34
    - EV-VC@100 +5.2% (concentrated in few months), EV-VC@500 -2.5% (systematic, 7/12 months)
    - Mechanism: colsample_bytree=0.8 now samples 27/34 useful features vs 27/34 useful + 4/34 wasted before
    - **This is a precision-vs-breadth tradeoff, not a free win**

11. **flow_direction is NOT useful for regression** (NEW — tested and rejected in v0011 screen)
    - Prune-only beat prune+flow_direction by +6.3% on screen mean EV-VC@100
    - Binding direction does not correlate with shadow price magnitude

12. **EV-VC@100 improvement is outlier-dependent across batches** (PATTERN — reinforced)
    - v0009: 3 months drove most of the +9% gain
    - v0011: 3 months drove most of the +5.2% gain (2021-05 +183%, 2021-11 +168%, 2021-09 +17.6%)
    - The model redistributes value capture rather than uniformly improving
    - Mean improvement is real but fragile — sensitive to individual month performance

### Pipeline Findings (from Code Reviews)

13. **Train-inference mismatch in gated mode** (NOT YET FIXED)
    - Regressor trains on `y_train_binary == 1` but infers on classifier predictions
    - Structural issue affecting all versions equally
    - Location: `ml/pipeline.py:195-209` (training) vs `ml/pipeline.py:249-253` (inference)

14. **Feature importance pipeline wired but never populated** (NOT YET FIXED)
    - benchmark.py expects `_feature_importance` but nothing emits it
    - Fix: add `model.get_score()` export to per-month metrics

15. **Temporal leakage concern** (NOT YET FIXED)
    - data_loader.py train_end may be inclusive → potential leakage
    - Structural issue needing audit, not version-specific

16. **R-REC@500 metric definition mismatch** (NOT YET FIXED)
    - Computed from ev_scores, not regressor-only ranking → doesn't match its label

17. **Pipeline docstring says classifier overrides supported but code ignores them** (NEW — LOW)
    - _apply_config_overrides only applies top-level and regressor keys
    - Correctness debt, not blocking

### Screening Methodology
- **Two-month screening works**: 6/6 screens correctly predicted full-benchmark relative winner
- **Screen month selection**: one weak + one strong for diagnostic coverage
- **Screen months used**: 2022-06, 2022-12, 2022-09, 2021-09 (latter two are new this batch)

### Gate System Issues
- **EV-VC@500 is now the binding gate constraint** (replacing Spearman): L1 margin +4.2%, L2 at exact limit (1 fail), L3 margin +0.0023
- **Gates v4 (calibrated to v0007)**: Do NOT recalibrate downward to v0011 — would loosen EV-VC@500, EV-NDCG, Spearman floors
- **noise_tolerance=0.02 is not scale-aware**: both reviewers flagged in both batches, still unfixed
- **EV-VC@100 tail_floor (0.000135) is non-protective**: effectively allows any value > 0

### Worker Reliability Patterns
1. **Phantom completion** is dominant failure mode in earlier batches
2. **Direction quality correlates with worker compliance**: explicit constraints → correct execution
3. **Config bug prevention**: explicit verification instructions in direction file fixed the iter 2 bug
4. **ALWAYS read baselines from committed state** (`git show HEAD:...`), never from working tree
