# Decision Log

## Iter 1 — feat-eng-3-20260304-102111

### D1: Promote v0009 as new champion
**Rationale**: All Group A gates pass all 3 layers. EV-VC@100 +9.0% (primary business metric), EV-VC@500 +1.5%, EV-NDCG +0.5%. Spearman -0.6% is noise-level (tail improved). C-RMSE/C-MAE also improved. Note: v0009 is identical to v0008; promotion is operationally of the 39-feature config first registered as v0008.

### D2: Feature set established at 39 nominal / 34 effective
**Rationale**: 5 zero-filled features (hist_physical_interaction, overload_exceedance_product, band_severity, sf_exceed_interaction, hist_seasonal_band) remain from v0007. Effective new features are 5 (density_skewness, density_kurtosis, density_cv, season_hist_da_3, prob_below_85). Both reviewers recommend pruning zero-filled features in next iteration.

### D3: Next iteration should prune zero-filled features + HP tuning
**Rationale**: Feature set is now stabilized. Zero-filled features waste tree splits. HPs (mcw=25, depth=5, n_est=400) were tuned for 29 effective features; 34 effective features may benefit from adjustments. Batch constraint (FE-only) will be relaxed for iter 2 to allow HP changes alongside feature cleanup.

---

## Iter 1 — feat-eng-3-20260304-121042

### D4: Promote v0011 as new champion
**Rationale**: All Group A gates pass all 3 layers. EV-VC@100 +5.2% is material on the primary business metric. Spearman +0.4%. The EV-VC@500 -2.5% degradation is a genuine precision-vs-breadth tradeoff — acknowledged and accepted because the business prioritizes top-100 capital allocation over broader coverage.

**Risk factors accepted**:
- EV-VC@500 L2 at exact limit (1 tail failure, max allowed is 1)
- EV-VC@500 L3 margin only +0.0023
- EV-VC@1000 L1 margin only +0.9% (Group B, non-blocking)
- EV-VC@100 gains concentrated in few months, not broadly distributed

### D5: Feature set cleaned to 34 actual features (from 39 nominal)
**Rationale**: 5 dead features successfully pruned. Code is cleaner. Effective feature count matches nominal for the first time. flow_direction tested and rejected (lost screen against prune-only).

### D6: Relax batch constraint for iter 2 — allow HP changes
**Rationale**: Feature cleanup is complete. The feature set is stable at 34. Both reviewers and the decision log from prior batch recommend HP tuning for the 34-feature set. The HPs (n_estimators=400, lr=0.05, colsample=0.8, mcw=25) were tuned for 29 effective features. With 34 clean features, HP adjustment is the highest-value next experiment. Remaining within FE-only would limit iter 2 to marginal feature additions with unclear signal.

### D7: Iter 2 priority: recover EV-VC@500 breadth without sacrificing EV-VC@100
**Rationale**: The precision-vs-breadth tradeoff in v0011 is acceptable for now, but compounding EV-VC@500 degradation would erode the gate margin. Next iteration must target EV-VC@500 recovery. HP tuning (more trees + lower LR) is the primary lever.
