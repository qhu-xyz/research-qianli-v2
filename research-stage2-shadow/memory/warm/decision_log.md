# Decision Log

## Batch: ralph-v2-20260304-031811

### D1: Iter 1 hypothesis selection
- **Decision**: Test L2 regularization (proven) vs L2 + subsampling (novel axis)
- **Rationale**: L2 is the strongest lever from prior batch but needs re-validation in current 6/2/34feat config. Subsampling is an unexplored complementary axis — if it helps, we get a compound improvement without the competition seen when stacking lr/trees on L2.
- **Alternatives considered**: L2 vs depth=4+L2 — rejected because depth already has 2 data points (3 and 5), while subsampling is completely untested. Subsample provides more novel information.
- **Screen months**: 2022-06 (worst across all gates) + 2022-12 (strongest EV-VC@100)

### D2: Iter 1 promotion decision
- **Decision**: Do NOT promote v0005
- **Rationale**: Spearman L1 fails (mean 0.3920 < floor 0.3928). Gate rules require all Group A gates to pass all 3 layers. The miss is 0.0008 (0.2%) and is a gate calibration artifact — floors are set at v0's exact mean, giving zero headroom. v0 itself fails its own EV-VC@100 and EV-NDCG L1 gates. However, the orchestrator must enforce gates as defined.
- **Gate change request**: Recalibrate Spearman floor (and all Group A floors) to provide variance headroom. Current gates are dysfunctional — no version can be promoted if any metric has epsilon noise below v0. Recommended: restore floors from gate_calibration.md (floor ≈ 0.87x v0 mean) or use percentile-based calibration.
- **Impact**: v0005 is blocked despite being a clear improvement on 3 of 4 Group A gates and all Group B gates.

### D3: Iter 2 direction selection
- **Decision**: Test depth=4 vs reg_alpha=1.0, both on v0005 base (L2=5, mcw=25)
- **Rationale**: depth=4 is untested middle ground between depth=5 (v0005, good mean, slight Spearman slip) and depth=3 (v0004 in prior batch, doubled bot-2 but hurt mean). reg_alpha=1.0 (L1) is an untested regularization axis that could improve Spearman by zeroing out noisy features. Both are complementary to L2 rather than competing.
- **Alternatives considered**: lr+trees — rejected, prior batch confirmed lr+trees COMPETES with L2 (worse when stacked). Moderate subsampling (0.7) — viable but deferred; aggressive 0.6 failed this iteration, unclear if 0.7 is different enough. Value-weighted — high potential but requires pipeline.py understanding, better for iter 3.
- **Screen months**: 2021-11 (worst Spearman for v0005, 0.2635) + 2022-12 (strongest EV-VC, regression canary)
