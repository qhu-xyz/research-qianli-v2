# Decision Log

## 2026-03-04: v0 baseline established
- **Decision**: Use 5-tier multi-class XGBoost (multi:softprob) replacing two-stage pipeline
- **Rationale**: Direct tier prediction eliminates error propagation between binary classifier and regressor
- **Outcome**: v0 baseline established with 12-month benchmark. Tier-VC@100=0.075 (low), QWK=0.359 (moderate)

## 2026-03-04: Gate calibration strategy
- **Decision**: Set floors = v0 mean, tail floors = v0 min (zero offset)
- **Rationale**: v0 is the first version; any improvement should pass gates. Zero offset means new versions must match or beat v0 on average.

## 2026-03-05: Iter 1 worker failure — retry with simplified direction
- **Decision**: Retry the same feature engineering hypotheses in iter2 with simplified worker instructions
- **Rationale**: Worker failed to produce any artifacts despite claiming done. The hypotheses (interaction features + light pruning vs aggressive pruning) are still the right first experiments for this FE-only batch. Simplifying the direction to a single hypothesis reduces worker execution complexity and failure risk.
- **Outcome**: Pending — iter2 direction written
