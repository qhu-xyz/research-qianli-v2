# Decision Log

## 2026-03-04: v0 baseline established
- **Decision**: Use 5-tier multi-class XGBoost (multi:softprob) replacing two-stage pipeline
- **Rationale**: Direct tier prediction eliminates error propagation between binary classifier and regressor
- **Outcome**: v0 baseline established with 12-month benchmark. Tier-VC@100=0.075 (low), QWK=0.359 (moderate)

## 2026-03-04: Gate calibration strategy
- **Decision**: Set floors = v0 mean, tail floors = v0 min (zero offset)
- **Rationale**: v0 is the first version; any improvement should pass gates. Zero offset means new versions must match or beat v0 on average.
