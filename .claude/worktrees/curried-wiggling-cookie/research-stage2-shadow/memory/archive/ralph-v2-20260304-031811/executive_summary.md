# Executive Summary — Batch ralph-v2-20260304-031811

## Outcome
**v0007 PROMOTED as new champion** (reg_lambda=1.0, mcw=25)

## Business Impact
- **EV-VC@500 +6.2%**: Top-500 constraints by expected value now capture 6.2% more actual shadow price value. This directly improves capital allocation for FTR trading.
- **EV-NDCG +0.5%**: Overall ranking quality of expected-value scores improved.
- **EV-VC@100 +1.3%**: Modest improvement at the top-100 level.
- **Spearman +0.1%**: Rank correlation essentially unchanged from baseline — preserved, not improved.
- **C-RMSE -6.9%**: Better regression calibration on binding constraints (mean improvement).

## Key Discovery: Parameter Decomposition
The batch's central finding is the decomposition of v0005's compound parameter change:

| Parameter | Effect on EV-VC | Effect on Spearman | Role |
|-----------|----------------|-------------------|------|
| reg_lambda (L2) | Improves top-K via noise reduction | **Degrades** via prediction compression | Harmful to overall promotion |
| min_child_weight | Improves @500+ via conservative leaves | **Neutral** | Pure positive lever |

**Optimal point**: reg_lambda=1.0 (v0 default), mcw=25. This captures mcw's EV-VC benefit without L2's Spearman cost.

## Iteration History

| Iter | Version | Config Change | Result | Blocker |
|------|---------|--------------|--------|---------|
| 1 | v0005 | λ=5, mcw=25 | EV-VC@100 +6.5%, Spearman -0.2% | Spearman L1 (gate calibration) |
| 2 | v0006 | + reg_alpha=1.0 | CONFIG BUG (= v0005) | Config bug + Spearman L1 |
| 3 | v0007 | λ=1, mcw=25 | EV-VC@500 +6.2%, Spearman +0.1% | **PROMOTED** |

## What We Learned

### Confirmed
1. mcw=25 is a pure EV-VC lever with no Spearman cost
2. L2 > 1.0 compresses predictions → hurts Spearman (confirmed by decomposition)
3. Regularization axis is exhausted for Spearman improvement — all tested approaches (L2, L1, depth, subsampling) either hurt Spearman or had no effect
4. Two-month screening reliably predicts full-benchmark winners (4/4 across batches)

### Failed
1. L2=5.0 reg: Spearman -0.0008 (noise, but gate blocks)
2. subsample=0.6/colsample=0.6: Signal starvation, Spearman -12%
3. depth=4: Catastrophic weak-month EV-VC@100 (-42%)
4. L1=1.0: Negligible effect (screen evidence only due to config bug)

### Process Issues
1. Gate calibration at v0 exact mean blocked 2 of 3 iterations — urgently needs HUMAN_SYNC
2. Config bug in iter 2 (benchmark used pre-change defaults) — mitigated by provenance verification in iter 3
3. Test suite was misaligned with code (13/24 vs 14/34 features) — corrected in iter 3

## Outstanding Code Debt
1. **Train-inference mismatch** in gated mode (pipeline.py)
2. **Feature importance** pipeline dead (benchmark expects it, nothing emits it)
3. **value_weighted** unwired in pipeline.py (high-potential lever for next batch)
4. **Temporal leakage audit** needed in data_loader.py
5. **business_context.md** stale (documents 13 clf/24 reg features, actual is 14/34)
6. **Gate recalibration** (floors at v0 exact mean → now should be v0007-based with headroom)

## Next Batch Priorities
1. **Gate recalibration** (HUMAN_SYNC) — floors at v0 exact mean are dysfunctional
2. **Wire value_weighted** through pipeline.py — aligns loss with business objective
3. **mcw fine-tuning** — test mcw in {15, 20, 30, 35} with reg_lambda=1.0
4. **Feature importance export** — auditability for feature contribution
5. **Tail robustness** — deep dive into 2021-11 and 2022-06 structural weakness
