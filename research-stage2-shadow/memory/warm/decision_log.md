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

### D4: Iter 2 promotion decision
- **Decision**: Do NOT promote v0006
- **Rationale**: v0006 full benchmark is invalid — config bug caused it to run with reg_alpha=0.1 instead of 1.0, producing metrics identical to v0005. Both reviewers independently confirmed this via config.json provenance check and per-month metric comparison. Since v0006 = v0005 in practice, the same Spearman L1 failure (0.3920 < 0.3928) applies. No new information from full benchmark.
- **Config bug diagnosis**: The benchmark script likely snapshots config from RegressorConfig() defaults BEFORE the code change was applied, or the benchmark ran against the pre-commit state. The registry artifact (config.json) correctly records what was actually used (reg_alpha=0.1), contradicting the changes_summary.
- **Valid signal from screen**: Screening DID correctly apply overrides. L1=1.0 had negligible Spearman impact (+0.001 strong month, +0.0 weak month) and slightly degraded EV-VC@100 (-23% weak, -2.8% strong). Depth=4 also failed: negligible Spearman change with catastrophic EV-VC@100 loss on weak month (-42%).
- **Gate change request (reiterated)**: Spearman floor at v0 exact mean has now blocked 2 iterations. v0005's EV-VC improvements (+6.5% / +5.9%) remain stranded. HUMAN_SYNC urgently needed.

### D5: Iter 3 direction selection
- **Decision**: Test value_weighted=True vs moderate L2 relaxation (reg_lambda=2.0, mcw=15), both on current base
- **Rationale**: Regularization axis is exhausted — L2, L1, depth, subsampling all tested without recovering Spearman. Two new directions:
  - **value_weighted**: Orthogonal lever (training loss weighting). Weights high-shadow-price samples more heavily, directly aligning the loss function with what drives Spearman and EV-VC. Untested, high potential.
  - **L2 relaxation**: Interpolates between v0 (good Spearman) and v0005 (good EV-VC). At reg_lambda=2.0/mcw=15, model is more regularized than v0 but less than v0005 — should partially recover Spearman while retaining some EV-VC gain.
- **Alternatives considered**: unified_regressor — Codex suggested this, but it changes training distribution (includes non-binders) in a way that's misaligned with the business objective (predict magnitude conditional on binding). Lower lr + more trees — already tested in prior batch stacked on L2, performed WORSE. Moderate subsampling (0.7) — possibly viable but we only have one shot and the 0.6→0.7 distinction may not be meaningful enough.
- **Screen months**: 2021-11 (worst Spearman, consistent across iters) + 2022-12 (best EV-VC, regression canary, cross-iter comparable)
