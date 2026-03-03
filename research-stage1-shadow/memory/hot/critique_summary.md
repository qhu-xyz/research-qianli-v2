# Critique Summary — Iteration 1 (feat-eng-20260302-194243, v0003)

## Reviewer Agreement (high confidence)
1. **Do not promote v0003** — directionally positive but not statistically significant; effect sizes <1σ
2. **Best result of 3 real-data iterations** — AUC 7W/4L/1T is stronger than HP tuning (0W/11L) and interactions (5W/6L/1T)
3. **Clean code changes** — benchmark.py plumbing fix is welcome, feature guard is correct
4. **Late-2022 weakness persists at 2022-09** — AP=0.306 (worst of all iterations), distribution shift not addressable by window alone
5. **2022-12 improved substantially** — AUC +0.0098, supporting the seasonal diversity hypothesis for this month
6. **Gains partly outlier-driven** — removing 2022-12 collapses AUC/AP mean gains; NDCG driven by 2021-01 (+0.0226)
7. **Threshold methodology debt remains** — leakage (HIGH) and `>` vs `>=` (MEDIUM) are pre-existing, deferred to HUMAN_SYNC

## Claude-Specific Insights
- Thorough statistical testing: all deltas <1σ, paired sign tests yield p>0.05 for all Group A metrics
- VCAP@100 (9W/3L, p≈0.07) is the closest to significance — top-100 value capture is the strongest signal
- VCAP@500 -0.0063 and CAP@100 -0.0117: broader ranking degraded while top-100 improved
- NDCG anomaly at 2021-08 and 2022-12: AUC improves but NDCG regresses (better discrimination, worse positive-stratum ordering)
- Dual-default fragility (MEDIUM): benchmark.py hardcodes train_months=14 alongside PipelineConfig default — lockstep updates required
- Recommends combining 14-month window + interaction features as next step (tests additivity)
- Recommends investigating 2022-09 specifically — 3 independent levers all failed to improve it
- Gate calibration appropriate — no changes recommended

## Codex-Specific Insights
- **f2p parsing crash (HIGH, new)**: `int(ptype[1:])` fails for cascade stage "f2p" → `int("2p")` crashes. Blocks stage-3 evaluation.
- Gains are outlier-driven: removing 2022-12 from AUC and 2021-01 from NDCG leaves near-zero mean improvement
- Test fixture drift (LOW): synthetic fixture is 17-wide while production config is 14 features
- Layer 3 tolerance 0.02 is very loose vs observed bottom_2 deltas (0.0002–0.0059). Metric-specific tolerances recommended.
- Recommends promoting/selecting v0 as reference champion to activate Layer 3 non-regression checking
- Recommends separate threshold tuning from evaluation (tune on val, report on holdout)

## Synthesis Assessment
- Both reviewers independently reach identical core conclusion: don't promote, retain 14-month window as default for future iterations
- Claude provides stronger statistical rigor; Codex provides stronger code correctness findings
- Key convergence: the window expansion provides real but small signal — the first lever to produce positive AUC wins
- The v0002 + v0003 combination is the obvious next test (both suggest it directly or indirectly)
- 2022-09 is the hardest month and may require fundamentally new features (binding rate 6.63%, lowest in dataset)

## Open Code Issues (cumulative)
| Issue | Severity | Source | Status |
|-------|----------|--------|--------|
| Threshold-selection leakage | HIGH | Codex (smoke-v6) | Deferred to HUMAN_SYNC |
| f2p parsing crash | HIGH | Codex (feat-eng iter1) | **Fix in iter2** |
| Threshold `>` vs `>=` mismatch | MEDIUM | Codex (smoke-v7) | Deferred to HUMAN_SYNC |
| Dual-default fragility (benchmark.py) | MEDIUM | Claude (feat-eng iter1) | **Fix in iter2** |
| Test fixture 17-wide vs 14 features | LOW | Codex (feat-eng iter1) | Include in iter2 |
| Layer 3 disabled when champion=null | MEDIUM | Codex (iter1/v0003-HP) | Accepted (v0 is reference) |
| Dead config `scale_pos_weight_auto` | LOW | Codex (smoke-v6) | Deferred |
