## Accumulated Learnings

### Regressor Hyperparameter Findings

1. **mcw=25 (min_child_weight) is the best single lever found** (from batch ralph-v2)
   - v0007 (reg_lambda=1.0, mcw=25): EV-VC@500 +6.2%, EV-NDCG +0.5%, Spearman +0.1%
   - **Pure EV-VC lever**: mcw=25 improves value capture without affecting Spearman
   - Mechanism: larger leaves → more conservative predictions → better rank ordering at scale
   - **UNTESTED**: mcw=15 with 600-tree ensemble — planned for iter 3, worker failed. Carry forward.

2. **reg_lambda (L2) > 1.0 compresses Spearman** (REFINED)
   - v0005 (λ=5, mcw=25): EV-VC@100 +6.5%, but Spearman -0.0008 — L2 was the culprit
   - v0007 (λ=1, mcw=25): EV-VC@500 +6.2%, Spearman +0.0004 — L2 reversion fixed it
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
   - Feature set carries signal; sparsity does not help

7. **REGULARIZATION AXIS IS EXHAUSTED FOR SPEARMAN RECOVERY**
   - Tested: L2 (λ=1→5), L1 (α=0.1→1.0), depth (3,4,5), subsampling (0.6)
   - Spearman recovery came from REVERTING L2, not from any regularization increase

8. **More trees + lower LR recovers breadth without harming Spearman** (confirmed in v0012)
   - n_estimators 400→600, lr 0.05→0.03 (budget 20→18)
   - EV-VC@500 +3.5%, critical tail failure eliminated (2022-09: 0.0527→0.0720)
   - EV-VC@100 -5.3% tradeoff (diluted top-100 discrimination)
   - **n_est=600, lr=0.03 is settled** — do not adjust further

9. **Higher colsample_bytree (0.8→0.9) does NOT help breadth**
   - Breadth recovery needs more ensemble rounds, not more features per tree

10. **value_weighted=True is UNTESTED** — carry forward. Verify pipeline.py implementation first.

### Feature Engineering Findings

11. **Distributional shape features (skewness, kurtosis, CV) carry real signal**
    - Adding to regressor set → EV-VC@100 +9.0%, C-RMSE -3.1%
    - Improvement concentrated in ~3 months — outlier-dependent

12. **Pruning dead features improves precision, trades breadth**
    - 39→34 (remove 5 zero-filled): EV-VC@100 +5.2%, EV-VC@500 -2.5%
    - Mechanism: colsample_bytree efficiency

13. **flow_direction has no regression signal** — tested and rejected

14. **EV-VC@100 improvement is outlier-dependent** (pattern across batches)
    - v0009: 3 months drove most of +9%; v0011: 3 months drove most of +5.2%
    - Mean improvement is real but fragile

### Pipeline Findings (Code Debt — Unfixed)

15. **HIGH**: Gated regressor train-inference mismatch (pipeline.py:195-208) — trains on true labels, infers on classifier predictions. Flagged 3+ times.
16. **LOW**: Feature importance pipeline wired but never populated
17. **LOW**: Temporal leakage concern in data_loader.py
18. **LOW**: R-REC@500 computed from ev_scores, not regressor-only ranking
19. **LOW**: Pipeline classifier override docstring mismatch

### Screening Methodology
- **Two-month screening works**: 6/6 screens correctly predicted full-benchmark relative winner
- **Screen month selection**: one weak + one strong for diagnostic coverage

### Gate System Issues
- **Gates v4 calibrated to v0007** — should recalibrate to v0012 before next batch
- **noise_tolerance=0.02 is not scale-aware**: flagged 3+ times by reviewers, still unfixed
- **EV-VC@100 tail_floor (0.000135) is non-protective**

### Worker Reliability
- **Phantom completion** is the dominant failure mode — worker reports "done" but produces no artifacts
- **Direction quality correlates with compliance**: explicit constraints → correct execution
- **Config bug prevention**: explicit verification instructions in direction file fixed prior bugs
- **ALWAYS read baselines from committed state** (`git show HEAD:...`), never working tree

### Current Config (v0012 champion)
```json
{
  "n_estimators": 600, "max_depth": 5, "learning_rate": 0.03,
  "subsample": 0.8, "colsample_bytree": 0.8,
  "reg_alpha": 1.0, "reg_lambda": 1.0, "min_child_weight": 25,
  "unified_regressor": false, "value_weighted": false
}
```
Features: 34 (13 classifier + 11 additional, all signal-carrying)
