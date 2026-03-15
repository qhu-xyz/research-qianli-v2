# Review: Phase 1 Implementation Plan

**Date**: 2026-03-12
**Scope**: Review the Phase 1 implementation plan as a handoff document for a fresh agent.
**Reviewed doc**:
- `/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/docs/superpowers/plans/2026-03-12-annual-signal-v2-phase1.md`

## Findings

1. **HIGH: The ground-truth module contract and its tests disagree about whether `ground_truth.py` returns only positive branch targets or a universe-attached table with zeros.**

   The plan defines `ground_truth.py` as a DA-cache-plus-bridge module that aggregates mapped DA to branch targets. That implies it should return only branches that actually map from DA activity, plus diagnostics. But the test at `test_gt_tiered_labels()` expects label `0` to be the majority, which only makes sense after ground truth has been joined onto the modeling universe and null targets have been filled to zero.

   That inconsistency matters because it changes module boundaries:

   - if `ground_truth.py` returns only mapped-positive branches, then `features.py` owns zero-filling after the join
   - if `ground_truth.py` returns a universe-attached branch table, then it needs the modeling universe as an input and duplicates `features.py` responsibilities

   A fresh implementer could build either interpretation and still think they are following the plan.

   Relevant locations:
   - [2026-03-12-annual-signal-v2-phase1.md](/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/docs/superpowers/plans/2026-03-12-annual-signal-v2-phase1.md#L1264)
   - [2026-03-12-annual-signal-v2-phase1.md](/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/docs/superpowers/plans/2026-03-12-annual-signal-v2-phase1.md#L1292)
   - [2026-03-12-annual-signal-v2-phase1.md](/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/docs/superpowers/plans/2026-03-12-annual-signal-v2-phase1.md#L1796)

2. **HIGH: The threshold-calibration script is calibrating on branch-level counts even though the universe rule is defined at cid level.**

   The task description says to compute `right_tail_max` for all `~13,000 cids`, sort them, and freeze a cid-level universe threshold. But the sample code immediately maps to branches and picks the elbow using branch-level counts.

   That is a material mismatch. The plan elsewhere defines the universe filter as:

   - cid-level `right_tail_max`
   - then branch collapse

   If calibration is done on branch-level counts instead, the chosen threshold may no longer match the documented target range for filtered cids, `count_active_cids`, or the branch-count planning tables.

   Relevant locations:
   - [2026-03-12-annual-signal-v2-phase1.md](/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/docs/superpowers/plans/2026-03-12-annual-signal-v2-phase1.md#L944)
   - [2026-03-12-annual-signal-v2-phase1.md](/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/docs/superpowers/plans/2026-03-12-annual-signal-v2-phase1.md#L1019)
   - [2026-03-12-annual-signal-v2-phase1.md](/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/docs/superpowers/plans/2026-03-12-annual-signal-v2-phase1.md#L1054)

3. **HIGH: The history-feature plan freezes the wrong historical mapping rule.**

   The plan says the monthly branch-binding table should use the eval planning year's annual bridge, with monthly fallback for month `M`. That is too weakly justified for historical feature generation. Bridge mappings are partition-sensitive by `auction_month` and `period_type`, and the same plan already warns that annual partitions differ.

   Using one eval-year annual bridge across all historical months can distort:

   - BF windows
   - `da_rank_value`
   - `has_hist_da`
   - cohort assignment
   - NB lookback logic

   This needs one frozen historical mapping policy before implementation starts.

   Relevant locations:
   - [2026-03-12-annual-signal-v2-phase1.md](/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/docs/superpowers/plans/2026-03-12-annual-signal-v2-phase1.md#L425)
   - [2026-03-12-annual-signal-v2-phase1.md](/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/docs/superpowers/plans/2026-03-12-annual-signal-v2-phase1.md#L1560)

4. **MED: The monthly-fallback test is too weak to catch the bug it is trying to prevent.**

   The plan adds a test meant to ensure monthly fallback uses each quarter's `market_month` as the monthly bridge `auction_month`, not the planning year. But the only unconditional numeric assertion is `monthly_recovered_cids >= 0`, which is always true. The more meaningful `monthly_recovery_detail` check is optional.

   So an implementation that incorrectly uses the wrong monthly bridge auction month can still pass the test suite.

   Relevant location:
   - [2026-03-12-annual-signal-v2-phase1.md](/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/docs/superpowers/plans/2026-03-12-annual-signal-v2-phase1.md#L1339)

5. **MED: Ground-truth diagnostics omit ambiguity accounting even though the shared bridge helper already computes it.**

   The bridge helper returns `ambiguous_cids` and `ambiguous_sp`, but the GT diagnostics contract only requires annual-mapped, monthly-recovered, and still-unmapped totals. That leaves a blind spot in coverage accounting and can distort denominator reporting for mapping-quality summaries and `Abs_SP@50`.

   Relevant locations:
   - [2026-03-12-annual-signal-v2-phase1.md](/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/docs/superpowers/plans/2026-03-12-annual-signal-v2-phase1.md#L641)
   - [2026-03-12-annual-signal-v2-phase1.md](/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/docs/superpowers/plans/2026-03-12-annual-signal-v2-phase1.md#L1316)

## Recommended Fixes

1. Freeze the `ground_truth.py` return contract.
   - Either make it return only mapped-positive branch targets plus diagnostics, or explicitly pass the modeling universe in and let it return zero-filled rows.
   - Then align the tests and `features.py` join logic to that choice.

2. Make threshold calibration use the same unit as the actual universe rule.
   - If the universe is cid-level before collapse, calibrate on cid-level filtered counts and only use branch-level collapse as a secondary reporting view.

3. Replace the historical mapping sentence with one explicit policy.
   - For example: monthly bridge-first mapping for each historical month, or a documented annual-plus-monthly strategy keyed to the historical month itself.

4. Strengthen the monthly-fallback test.
   - Require nontrivial per-month recovery detail or compare against known recovered counts for at least one slice.

5. Add ambiguity fields to GT diagnostics.
   - `ambiguous_cids`
   - `ambiguous_sp`

## Bottom Line

The plan is well structured and much closer to implementation-ready than the earlier docs. The main remaining problems are not architectural; they are execution-contract issues:

- one unresolved GT module boundary
- one threshold-calibration unit mismatch
- one under-specified historical mapping rule

Those are significant enough that I would fix them before handing the Phase 1 plan to a fresh coding agent.
