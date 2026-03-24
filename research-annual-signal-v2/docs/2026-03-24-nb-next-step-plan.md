# NB Next-Step Plan (2026-03-24, revised)

## Status: What we know

### Opt3 (unified ML model) — current best
- Trains on ALL branches (not just dormant), LambdaRank with tiered weights [1,1,3,10]
- 13 features: da_rank_value, shadow_price_da, bf, count_active_cids, 4 density bins, rt_max, 4 top2_mean
- Training: 2018-2025 class-specific tables, expanding window
- Beats V4.4 on total SP capture at every K level in every (year × ctype)
- **On shared-branch overlap (same universe, reranked)**: Opt3 wins 9/12 quarters on avg rank of top-5 NB binders, 2-3x hit rate at K=50/100, 2-3x SP captured in dormant top-100

### Where Opt3 still struggles
- Top NB binders rank 100-500+ even in Opt3 (not top-50)
- 3 quarters where V4.4 still wins on overlap: 2025 aq2 on/off, 2025 aq3 off
- Specific branches: MNTCELO (Opt3 rk 203, V4.4 rk 145 on shared set), AUST_TAYS (539 vs 104)

### Miss taxonomy (top-20 NB binders, 2025 onpeak)
| Bucket | Count | SP | Fixable by modeling? |
|--------|:---:|---:|---|
| no_history (da_rank high, decent density) | 5 | $104K | Maybe — stronger density weighting |
| weak_density (rt_max < 0.3) | 3 | $65K | No — signal absent |
| no_hist + weak_density | 4 | $56K | No — both signals absent |
| outside_v44 | 5 | $53K | N/A — V4.4 can't see these |
| comparable | 2 | $38K | Already ok |
| v44_much_better | 1 | $19K | Target for improvement |

**~50% of top-20 NB SP is in "no_history" branches that HAVE density signal** — this is the modeling opportunity.

## Revised experiment plan

### Priority 1: Top-tail emphasis (modeling)

The model isn't paying enough attention to the dangerous top tail. 3 variants:

**A. Extreme tiered weights**: [1, 1, 10, 100] instead of [1, 1, 3, 10]
- 100x weight on tier-3 (top-third SP) binders forces the model to optimize for them

**B. Dangerous-NB binary objective**: Binary binder detection with SP-threshold weighting
- Label: 1 if dormant + SP > threshold (e.g., $10K), else 0
- Weight: `SP / threshold` for positives
- This directly optimizes "find the dangerous dormant binders"

**C. Two-stage model**:
- Stage 1: Binary — will this dormant branch bind at all?
- Stage 2: Regression on binders only — how much SP?
- Final score: P(bind) × E[SP|bind]

### Priority 2: Error analysis on residual V4.4-win branches

On the 3 quarters where V4.4 wins overlap:
- What features do those branches have?
- Is there a pattern V4.4 captures that our features contain but Opt3 doesn't extract?
- If yes → modeling fix. If no → feature gap (need new data source).

### Priority 3: Distillation (only if priority 1+2 fail)

On shared branches where V4.4 consistently outranks Opt3:
- Use V4.4 rank as soft label
- Only justified if those branches show separable patterns in our features that Opt3's objective misses

### NOT pursuing
- Hybrid (V4.4 inside / Opt3 outside) — V4.4 is opaque and not reproducible in production
- More density feature engineering — ablation proved density is at information ceiling
- Per-ctype split models — single model works better

## Evaluation contract

Every experiment must report these tables per (year × ctype):

1. **Overlap-only top-5/10 avg rank** (reranked on shared dormant set)
2. **Overlap-only Hit@30/50/100** on top-10 shared NB binders
3. **Overlap-only SP@100** on shared NB binders
4. **Native top-K SP capture** at K=50/100/200/400
5. **Deployment R30 metrics** (VC, SP, NB_SP)

Success = improve Hit@50 and SP@100 on overlap without losing >5% native SP capture.

## Files

| File | Action |
|------|--------|
| `scripts/nb_toptail_experiment.py` | Create — top-tail emphasis experiment |
| `registry/{ct}/nb_toptail/` | Create — results per variant |
| `docs/2026-03-24-nb-toptail-report.md` | Create — results |
