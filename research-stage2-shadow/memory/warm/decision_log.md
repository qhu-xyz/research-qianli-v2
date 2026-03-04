# Decision Log

## Batch: ralph-v2-20260304-031811

### D1: Iter 1 hypothesis selection
- **Decision**: Test L2 regularization (proven) vs L2 + subsampling (novel axis)
- **Rationale**: L2 is the strongest lever from prior batch but needs re-validation in current 6/2/34feat config. Subsampling is an unexplored complementary axis — if it helps, we get a compound improvement without the competition seen when stacking lr/trees on L2.
- **Alternatives considered**: L2 vs depth=4+L2 — rejected because depth already has 2 data points (3 and 5), while subsampling is completely untested. Subsample provides more novel information.
- **Screen months**: 2022-06 (worst across all gates) + 2022-12 (strongest EV-VC@100)
