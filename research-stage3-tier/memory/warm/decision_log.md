# Decision Log

## 2026-03-04: v0 baseline established
- **Decision**: Use 5-tier multi-class XGBoost (multi:softprob) replacing two-stage pipeline
- **Rationale**: Direct tier prediction eliminates error propagation between binary classifier and regressor
- **Outcome**: v0 baseline established with 12-month benchmark. Tier-VC@100=0.0708, QWK=0.3698

## 2026-03-04: Gate calibration strategy
- **Decision**: Set floors = v0 mean, tail floors = v0 min (zero offset)
- **Rationale**: v0 is the first version; any improvement should pass gates. Zero offset means new versions must match or beat v0 on average.

## 2026-03-05: Prior batch failures — root cause and fix
- **Decision**: Commit all HUMAN-WRITE-ONLY changes before launching pipeline
- **Rationale**: 4 consecutive worker failures caused by pre-merge guard rejecting worker output due to uncommitted changes to evaluate.py and gates.json
- **Outcome**: Fixed in commit a2a38c5. Pipeline runs successfully now.

## 2026-03-05: v0005 NOT promoted — Tier-VC@100 fails L1
- **Decision**: Do not promote v0005. Tier-VC@100 mean 0.0746 fails L1 floor 0.0750 by 0.0004 (0.6%).
- **Rationale**: Gate is binary. Despite improvements across all metrics with no regressions, the money metric (Tier-VC@100) is below floor. Cannot promote a version that fails the most important gate.
- **Assessment**: v0005 is directionally positive. The interaction features (overload_x_hist, prob110_x_recent_hist, tail_x_hist) provide consistent small improvements. Bottom_2_mean improved 64%. But the improvement is insufficient to cross the floor.
- **Implication**: Iter2 should build on v0005's feature set and add more FE improvements to close the 0.0004 gap.

## 2026-03-05: Iter2 direction — add log transforms and more interactions
- **Decision**: Build on v0005 (37 features) by adding log transforms (log1p_hist_da, log1p_expected_overload) and additional interactions (overload_x_recent_hist, prob_range_high) for 37→41 features.
- **Rationale**: The 0.0004 gap is tiny. Log transforms compress long-tailed price/overload distributions, which should help the model discriminate at the high end of the ranking. Additional interactions extend the compound signal approach that showed consistent improvement in iter1.
- **Risk**: Low — additive features only. XGBoost can ignore unhelpful features via low split gain.
