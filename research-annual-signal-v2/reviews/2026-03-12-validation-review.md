# Review: Annual Signal v2 Docs Validation

> **HISTORICAL — SUPERSEDED BY BRANCH-LEVEL DESIGN**
>
> This review was written against the older **constraint_id-level** prototype (one row per cid).
> The docs have since been redesigned to use **branch_name** as the row unit (see `implementer-guide.md` §5.4b, §6.2).
> Findings about universe sizes, binding rates, and LambdaRank prototype results refer to the cid-level design and should NOT be used as implementation targets.
> The core mapping mechanics findings (bridge table, naming systems, convention filter) remain valid.
> See `2026-03-12-docs-implementation-audit.md` for the current authoritative audit.

**Date**: 2026-03-12
**Scope**: Validate the current docs against local data, local `pbase` code, and runnable prototype experiments of the documented raw-density pipeline.
**Reviewed docs**:
- `/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/docs/implementer-guide.md`
- `/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/docs/bridge-table-gap-analysis.md`

## Findings

1. **CRITICAL: The holdout plan is temporally incorrect as of March 12, 2026.**

   The implementer guide defines `2025-06/aq4` as `Mar 2026, Apr 2026, May 2026`, then defines the holdout as 4 groups (`2025-06/aq1-4`). As of **March 12, 2026**, realized DA for **April 2026** and **May 2026** does not exist yet, so the documented ground-truth pipeline cannot evaluate `2025-06/aq4` today. A local check against `/home/xyz/workspace/research-qianli-v2/research-stage5-tier/data/realized_da/` confirmed that the aq4 delivery window is incomplete.

   Relevant doc locations:
   - `implementer-guide.md`: aq4 month mapping
   - `implementer-guide.md`: holdout definition

2. **HIGH: The docs describe the raw density bins with the wrong semantics.**

   The implementer guide says the 77 density columns are probabilities that flow at a threshold exceeds the limit, then applies simple positive monotonicity to them. The local parquet does not behave like cumulative exceedance probabilities:
   - a real density row sums to about `20`, not `1`
   - in sampled data, many rows had `70 < 80` and `80 < 100`
   - this means the columns do not act like tail probabilities

   This is not just wording. It affects the feature design, the monotone-constraint table, and the teammate-style "`70` filter" discussion.

   Relevant doc locations:
   - `implementer-guide.md`: density distribution description
   - `implementer-guide.md`: density bin selection
   - `implementer-guide.md`: monotone constraints

3. **HIGH: The documented `density_signal_score >= 0.001 -> ~4,500 constraints, ~7-8% binding rate` contract is not reproducible from the written collapse rules.**

   Using the guide's stated procedure:
   - max across `flow_direction`
   - mean across outage dates and market months within quarter
   - filter `density_signal_score >= 0.001`

   the local 2025-06 quarter sizes were:
   - `aq1`: `3,323`
   - `aq2`: `3,292`
   - `aq3`: `2,372`

   The local binding rates were also much higher than the guide claims:
   - `aq1`: `856 / 3323 = 25.8%`
   - `aq2`: `978 / 3292 = 29.7%`
   - `aq3`: `660 / 2372 = 27.8%`

   The guide currently uses `~4,500` and `~7-8%` as planning anchors in multiple places. Those numbers did not reproduce under the documented implementation.

   Relevant doc locations:
   - `implementer-guide.md`: project size claim
   - `implementer-guide.md`: raw-density replacement claim
   - `implementer-guide.md`: score filter flow
   - `implementer-guide.md`: elbow table
   - `implementer-guide.md`: expected dataset sizes

4. **HIGH: The bridge-table contract is incomplete; without `convention < 10`, the documented one-row-per-constraint pipeline breaks.**

   The implementer guide tells the implementer to filter the bridge table by `auction_type`, `auction_month`, `period_type`, and `class_type`. That is not sufficient for the raw-density universe.

   Local aq1 validation for `2025-06`:
   - score-filtered universe before bridge join: `3,323` constraints
   - after joining the raw bridge partition: `7,818` rows
   - bridge fanout averaged `2.32` rows per `constraint_id`

   When I added the production-style `convention < 10` filter:
   - joined rows returned to `3,323`
   - mean fanout dropped to `~1.01`

   This was an execution blocker, not a cleanliness issue. Without the `convention < 10` filter, a LightGBM LambdaRank prototype failed with:
   - `Number of rows ... exceeds upper limit of 10000 for a query`

   The bridge-table-gap doc mentions `convention < 10` in its production-code comparison, but the implementer guide never carries that requirement into the main pipeline spec.

   Relevant doc locations:
   - `implementer-guide.md`: bridge pipeline
   - `implementer-guide.md`: bridge filter example
   - `bridge-table-gap-analysis.md`: production comparison uses `convention < 10`

## Validation Experiments

I ran a reduced but real prototype of the documented raw-density pipeline using:
- raw density distribution
- density signal score
- constraint limit
- bridge join with `convention < 10`
- historical features: `da_rank_value`, `bf_6`, `bf_12`, `bf_15`
- labels from realized DA monthly cache, aggregated by quarter through the bridge table
- LightGBM LambdaRank

Training setup:
- train years: `2021-06` through `2024-06`
- test year: `2025-06`
- quarters tested: `aq1`, `aq2`, `aq3`

### Prototype holdout results

| Quarter | Test rows | Binding rows | Model VC@50 | Baseline VC@50 | Model VC@100 | Baseline VC@100 | Model NDCG | Baseline NDCG |
|--------|:---------:|:------------:|:-----------:|:--------------:|:------------:|:---------------:|:----------:|:-------------:|
| `aq1` | 3,323 | 856 | 0.1246 | 0.0013 | 0.2057 | 0.0013 | 0.7148 | 0.5366 |
| `aq2` | 3,292 | 978 | 0.0795 | 0.0000 | 0.1221 | 0.0002 | 0.6669 | 0.5602 |
| `aq3` | 2,372 | 660 | 0.1334 | 0.0000 | 0.2335 | 0.0026 | 0.6694 | 0.4965 |

Baseline = simple `-da_rank_value` ranking.

### What these experiments prove

- The raw-density design is **technically viable**.
- Once the bridge join is corrected with `convention < 10`, the prototype becomes trainable end to end.
- The prototype beats a simple DA-history-only baseline on all three tested holdout quarters.

### What they do **not** prove

- They do not validate the guide's current universe-size and binding-rate claims.
- They do not validate `aq4`, because the necessary realized DA months are not fully available on 2026-03-12.
- They do not validate the raw-bin semantic interpretation currently written in the guide.
- They do not validate offpeak BF, monthly BF, or the full proposed Phase 2 feature set.

## Bottom Line

The current docs still should **not** be treated as implementation-ready.

What is now validated:
- a raw-density annual pipeline can be built
- the expanded universe is feasible in principle
- LightGBM LambdaRank works on the corrected one-row-per-constraint dataset
- the approach shows real ranking signal on `aq1-3` holdout slices

What still blocks a trustworthy spec:
- the holdout definition includes future months for `aq4`
- the raw density bins are described incorrectly
- the score-filtered universe size and binding-rate expectations do not reproduce
- the bridge-table filter is incomplete without `convention < 10`

## Recommended Doc Fixes

1. Rewrite the holdout section using concrete dates and drop `2025-06/aq4` from any currently evaluable holdout definition until `2026-05` realized DA exists.
2. Replace the density-bin description with a schema-faithful statement and stop calling these columns direct exceedance probabilities unless that transformation is formally derived.
3. Specify one exact `density_signal_score` aggregation scope for universe definition and recompute all reported sizes from that definition.
4. Add `convention < 10` to the bridge-table contract everywhere it matters, not just in the historical pbase comparison section.
