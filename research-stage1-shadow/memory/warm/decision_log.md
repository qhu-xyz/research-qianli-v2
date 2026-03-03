# Decision Log

> Previous decisions (D1-D41) archived in memory/archive/feat-eng-20260303-060938/
> See memory/hot/learning.md for distilled learnings.

## D42: Promote v0007 as champion (feat-eng-2-092848, iter1)

**Context**: v0007 (19 features with shift factor + constraint metadata) produced AUC +0.0137 (12W/0L, p≈0.0002), AP +0.0455 (11W/1L, p≈0.006). All 4 Group A gates pass all 3 layers. Both reviewers independently recommend promotion.

**Decision**: PROMOTE v0007. Set as champion for Layer 3 non-regression checks going forward.

**Rationale**:
1. Exceeds all 3 human-input success criteria: AUC 0.849 > 0.840, AP 0.439 > 0.400, 12W/12 > 8/12
2. Most statistically significant result in pipeline history
3. AP bot2 trend reversed (+0.0363 vs v0, ending 6-experiment decline)
4. BRIER improved -0.0108, ending 6-experiment narrowing

**Risk accepted**: NDCG bot2 margin 0.0046 (within 0.02 tolerance). Future iterations measured against v0007 as champion.

## D43: Iter 2 direction — NDCG recovery + feature expansion (feat-eng-2-092848)

**Context**: v0007 achieves excellent AUC/AP but NDCG is flat (5W/7L) with bot2 regressing -0.0154. Late-2022 months show systematic NDCG losses. The 6 new features have low individual importance (4.66% combined) but high generalization value.

**Decision**: Iter 2 targets NDCG recovery through selective monotone constraint tuning on the new features, combined with adding distribution shape features. Do NOT sacrifice AUC/AP gains — treat v0007 as the floor.

**Rationale**:
1. NDCG is the bottleneck for future iterations (0.0046 margin on bot2)
2. v0006 showed monotone constraint structure affects NDCG strongly
3. Distribution shape features (pruned in v0006) may restore NDCG without hurting AUC
4. Human directive is aggressive feature engineering — adding more features is aligned

## D44: Gate calibration recommendations for HUMAN_SYNC

**Decision**: Recommend the following gate changes at HUMAN_SYNC:
1. **Set champion to v0007** — activates Layer 3 non-regression for all future versions
2. **VCAP@100 floor**: Tighten from -0.035 to 0.0 (currently non-binding)
3. **CAP@100/500 floors**: Consider relaxing by 0.02 — v0007's model profile shifted from threshold-dependent capture to ranking quality, making these floors misaligned with the new operating point
4. **Keep noise_tolerance at 0.02** — uniform tolerance is adequate; metric-specific tolerances add complexity without demonstrated need
5. **BRIER floor is now safe**: v0007 recovered headroom to +0.031 (was 0.0163 for v0006)
