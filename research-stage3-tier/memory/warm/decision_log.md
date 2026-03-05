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
- **Outcome**: FAILED — same failure mode as iter1. Worker wrote handoff claiming done but produced no artifacts.

## 2026-03-05: Iter 2 worker failure — last chance with iter3
- **Decision**: Iter3 must use the simplest possible direction: single hypothesis, no screening phase, no A/B comparison. Just add the 3 interaction features directly and run the full 12-month benchmark.
- **Rationale**: Two consecutive worker failures with identical symptoms (handoff written, no artifacts) suggests a systematic issue. The screening phase (2-month subset) may be contributing to failure complexity. Skipping screening and going directly to full benchmark is simpler and, since this is the last iteration, we need all 12 months anyway.
- **Risk**: If the worker fails again, we end the batch with no iterations producing data. But we cannot simplify further than "add features, run benchmark."
- **Outcome**: Pending — iter3 direction to be written

## 2026-03-05: Iter 1 worker failure (batch tier-fe-2) — 3rd consecutive failure
- **Decision**: Retry in iter2 with maximally simplified direction. Single hypothesis (add 3 interaction features). No screening. No A/B comparison. Explicit instruction to NOT write handoff until benchmark pipeline completes and artifacts are verified.
- **Rationale**: 3 consecutive failures with identical symptoms across 2 batches. The worker is writing the handoff signal before doing any work — no code edits, no benchmark runs, no registry artifacts. Direction complexity is NOT the issue. The worker must be told in the strongest terms to (1) make code changes first, (2) run benchmark, (3) verify artifacts exist, (4) ONLY THEN write handoff.
- **Risk**: If worker fails again, iter3 is the last chance. May need to flag systematic worker execution bug to human.
