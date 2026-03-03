# Human Input — Feature Engineering Batch (3 iterations)

## What We've Learned (3 real-data experiments)

1. **HP tuning is a dead end.** v0 defaults are near-optimal. Deeper trees hurt AUC, help BRIER. Don't waste iterations on HP changes.
2. **Interaction features alone don't break through.** AUC +0.0000 with 3 interactions. XGBoost depth-4 already discovers most interactions.
3. **Training window expansion (10→14) is the first positive lever.** AUC +0.0013 (7W/4L), VCAP@100 +0.0034 (9W/3L). Retain 14-month window as the new default.
4. **The AUC ceiling is ~0.835 with 14 base features.** Three independent levers confirm this. Breaking through requires fundamentally new discriminative signal.
5. **2022-09 is the hardest month.** Binding rate 6.63% (lowest). Three levers all failed there. May need features we don't have yet.

## Strategy for This Batch

### Iter 1: Combine the two positive levers (H6)
- **Retain 14-month training window** (train_months=14)
- **Add 3 interaction features** (products of exceedance × severity × historical signals)
- **Keep v0 HP defaults** (proven near-optimal)
- Tests whether window expansion + interaction features are additive
- Expected: AUC 0.836-0.840 if additive, ~0.835 if not

### Iter 2-3: Depends on iter 1, but strategic priorities are:
- If H6 shows additivity → refine: try more/different interactions, or train_months=18
- If H6 shows no additivity → the current feature set has a hard ceiling. Try:
  - **Feature selection**: Drop low-signal features (shape features density_skewness/kurtosis/cv have unconstrained monotone — they may add noise). Test with 11 features instead of 14.
  - **Ratio features**: prob_exceed_110 / prob_exceed_90 (tail concentration), expected_overload / prob_exceed_100 (conditional severity)
  - **Temporal aggregation**: If the pipeline supports it, consider rolling averages of hist_da (but only if no data leakage)

### Bug fixes (do in iter 1 or 2):
- **f2p parsing crash** (Codex HIGH): `int("2p")` fails for cascade stage. Fix in benchmark.py.
- **Dual-default fragility** (Claude MEDIUM): train_months hardcoded in benchmark.py function signatures AND PipelineConfig. Single source of truth.

## Business Constraint (unchanged)
**Precision over recall.** Do NOT lower threshold, do NOT increase beta above 1.0, do NOT optimize for recall. The trading strategy needs high-confidence binding predictions, not coverage.

## What NOT to Do
- No HP tuning (proven dead end — 3 experiments confirm)
- No threshold_beta changes (keep 0.7)
- No architecture changes (XGBoost + monotone constraints is correct)
- No train/val split changes beyond window length
- Don't spend more than 1 iteration on something that shows zero signal

## Success Criteria
- **Promotion-worthy**: AUC > 0.837 AND at least 8/12 months winning AND AP > 0.396
- **Encouraging but not promotable**: AUC > 0.835 with 7+/12 wins (continue refining)
- **Dead end signal**: AUC ≤ 0.835 or fewer than 6/12 wins → pivot to different approach

## Gate Notes
- Don't change gate floors — we need more data points before calibrating
- Layer 3 (non-regression) is effectively disabled since champion=null — this is fine for now
- Promote the first version that meets promotion criteria above
