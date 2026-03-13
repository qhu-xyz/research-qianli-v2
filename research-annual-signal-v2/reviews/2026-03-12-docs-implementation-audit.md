# Audit: Annual Signal v2 Docs Implementation Readiness

**Date**: 2026-03-12
**Scope**: Re-review the current docs as a handoff spec for a fresh AI implementer. Standard used here: another AI should be able to implement the intended branch-level pipeline without inventing hidden policy decisions.
**Reviewed docs**:
- `/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/docs/implementer-guide.md`
- `/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/docs/bridge-table-gap-analysis.md`
- `/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/reviews/2026-03-12-validation-review.md`

## Findings

1. **HIGH: The raw-only universe is still not executable because the threshold is not frozen.**

   The guide has now fully moved to a raw-only universe rule based on `right_tail_max`. That is directionally correct, but the one parameter that defines the actual universe is still missing:

   - `right_tail_max >= threshold`
   - threshold `TBD by Phase 0 elbow analysis`

   That means the next implementer still cannot build the final dataset "in one go" without running extra research and making a policy choice. The guide itself now says "nothing else can proceed until the threshold is frozen," which is the right diagnosis, but it also means the handoff is not complete yet.

   This missing threshold controls:

   - which cids enter the universe
   - which branches survive Level 2
   - how `count_active_cids` is computed
   - what the branch counts and class balance will be

   Relevant locations:
   - [implementer-guide.md](/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/docs/implementer-guide.md#L298): raw-only universe rule
   - [implementer-guide.md](/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/docs/implementer-guide.md#L316): branch-count tables still tied to the old score-filter sizes
   - [implementer-guide.md](/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/docs/implementer-guide.md#L1189): Phase 1 still begins with threshold calibration as a prerequisite

2. **HIGH: The branch-level design still leaks `constraint_id` logic into NB labeling and historical-feature construction.**

   The guide is now very explicit that the training row is `(branch_name, planning_year, aq_quarter)` and that the target is branch-level SP. But §11 still defines New Binding and cohorts as if the row unit were a constraint:

   - "A New Binding constraint..."
   - `def is_new_binder(constraint_id, ...)`
   - Phase 1 says "label constraints as NB6/NB12/NB24"
   - the cohort table says signal availability for each constraint
   - Phase 1 `da_rank_value` still says "sum |shadow_price| per constraint across lookback"

   In the current branch-level design, NB and cohort assignment must be defined on **branch_name**, not raw `constraint_id`. Otherwise a fresh implementer has to invent one of several incompatible policies:

   - mark a branch NB if **any** cid is NB
   - mark a branch NB if the **mapped branch target** is NB by branch history
   - keep cid-level NB labels and somehow collapse them to one branch row

   That is not a cosmetic wording issue. It changes:

   - the primary `NB_Recall@50` promotion gate
   - cohort assignment
   - `history-zero` / `established` counts
   - any analysis of "new binders"
   - whether `da_rank_value` is branch history or cid history

   Relevant locations:
   - [implementer-guide.md](/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/docs/implementer-guide.md#L511): row unit is branch-level
   - [implementer-guide.md](/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/docs/implementer-guide.md#L1072): NB still defined as a constraint concept
   - [implementer-guide.md](/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/docs/implementer-guide.md#L1106): cohorts still described per constraint
   - [implementer-guide.md](/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/docs/implementer-guide.md#L1212): Phase 1 still says "label constraints"
   - [implementer-guide.md](/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/docs/implementer-guide.md#L1243): Phase 1 `da_rank_value` step still says "per constraint"

3. **HIGH: The Level-2 density feature contract is still not frozen enough for a handoff implementation.**

   The current guide tells the implementer all of the following:

   - use `max/min` in the main collapse contract
   - `max` was the best univariate stat
   - `std` was the best leave-one-year-out model stat
   - start with `max`
   - try `std` later
   - feature table says the second stat is `std or min`

   That is acceptable for research notes. It is not good enough for a handoff spec whose goal is "implement everything needed in one go." A fresh agent still has to decide whether the intended initial build is:

   - `max + min`
   - `max + std`
   - `max only`

   The doc now reads more like experiment notes than a frozen implementation contract.

   Relevant location:
   - [implementer-guide.md](/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/docs/implementer-guide.md#L443)
   - [implementer-guide.md](/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/docs/implementer-guide.md#L526)
   - [implementer-guide.md](/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/docs/implementer-guide.md#L611)

4. **MED: The docs still tell two different stories about the root cause of the 2025 bridge gap.**

   The main guide says the missing 2025 DA constraint_ids are "newer constraints added to the grid after the annual bridge was built." The historical bridge note says the opposite: those unmapped 2025 IDs are **not** new and likely indicate a bridge-table build issue. Both cannot be the authoritative explanation at the same time.

   This does not block implementation, but it absolutely matters for any future attempt to "fix" the mapping problem. A new agent reading both docs will not know whether the next step is:

   - accept normal refresh timing
   - escalate a probable data-build defect
   - or investigate a different root cause entirely

   Relevant location:
   - [implementer-guide.md](/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/docs/implementer-guide.md#L749)
   - [bridge-table-gap-analysis.md](/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/docs/bridge-table-gap-analysis.md#L117)

5. **MED: The branch-level rationale in §8.2 still overstates what is proved.**

   The guide still argues that sibling cids on a binding branch "SHOULD be positive." That is only true if the modeling objective is explicitly branch-level. For a pure cid-level objective, `A -> positive, B/C/D/E -> zero` can be a valid label definition. The real argument for branch-level design is objective alignment and cleaner training dynamics, not that cid-level labels are inherently wrong.

   Relevant location:
   - [implementer-guide.md](/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/docs/implementer-guide.md#L724)
   - [implementer-guide.md](/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/docs/implementer-guide.md#L1527)

6. **LOW: Some example code in §11 is wrong enough to mislead a fresh implementer.**

   The NB metric snippet uses `actual[top_k_idx & nb_binders]`, mixing integer indices with a boolean mask. That expression is not the right way to select "NB binders inside top-K" and will either error or silently compute the wrong subset depending on the array type. Since the guide is meant as an implementation handoff, broken example code should be treated as a doc bug, not just pseudocode.

   Relevant location:
   - [implementer-guide.md](/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/docs/implementer-guide.md#L1132)

7. **LOW: The implementation audit and validation review in `reviews/` still describe different stages of the design and should be labeled more explicitly.**

   The older validation note is still about the earlier constraint-row prototype and score-filtered experiments. The docs are now much more branch-level. That is fine historically, but another AI could still read the two review files as if they were equally current.

   Relevant location:
   - [2026-03-12-validation-review.md](/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/reviews/2026-03-12-validation-review.md#L82)

## What Improved

The docs are materially better than the earlier version.

- The branch-level row unit is now explicit and consistently stated.
- The raw density-bin semantics are more careful and much closer to the actual parquet behavior.
- The pipeline is now consistently raw-only; `density_signal_score` is no longer a hidden dependency.
- The combined-GT bridge path now correctly unions onpeak and offpeak bridge partitions instead of assuming onpeak-only mapping.
- The historical bridge note is now clearly labeled as a V4.4-era document.

Those changes move the docs much closer to implementation readiness.

## Additional Validation Used For This Audit

- I rechecked the current `implementer-guide.md` after its March 12 update.
- I verified that raw annual density data exists for all four quarters under the documented annual partitions.
- I reused the pre-holdout branch-level probes I ran earlier:
  - years: `2021-06` to `2024-06`
  - no `2025-06` holdout data used
  - branch-level Level-2 probe favored `std/top2` over `min`
- I rechecked annual bridge partitions for `class_type=onpeak` vs `class_type=offpeak` across all 20 annual slices (`2021-06` to `2025-06`, `aq1-aq4`):
  - the guide's new union-of-ctypes GT mapping is the safer contract
  - shared on/off constraint_ids almost always had identical branch-name sets
  - only 2 conflicting cids appeared in one slice (`2021-06/aq4`)
- I previously measured direct-vs-bridge coverage on `aq1` for `2021-06` through `2025-06`; those numbers are still the basis for the mapping comments above.

## Recommended Doc Fixes

1. Freeze the raw-only threshold and regenerate the published universe tables from that exact rule.
   - Until then, the doc is still half implementation spec, half research to-do.

2. Rewrite §11 fully in branch-level terms.
   - `NB_N` should be defined for `branch_name`, not raw `constraint_id`.
   - Phase 1 should say "label branches," not "label constraints."
   - Cohorts should also be explicitly branch-level.

3. Freeze one initial Level-2 density contract.
   - If the intended first implementation is `max+min`, say that plainly and move `std` to ablations.
   - If the intended first build is `max+std`, rewrite §6.2, §7.1, §7.2, and §12 to match.

4. Resolve the 2025 bridge-gap causal story.
   - Pick one authoritative explanation or mark the cause as unresolved.

5. Fix the sample code in §11 and mark the old validation review as historical/prototype-specific.

## Bottom Line

The docs are now **better structured**, but they are still **not safe as a clean handoff**. The remaining blockers are:

- one unfrozen raw-only universe threshold
- one unresolved branch-vs-constraint mismatch in NB/cohort logic
- one unfrozen Level-2 density feature contract

Another AI could implement a large fraction of the pipeline from these docs. Another AI still would have to invent at least one core policy decision and one core labeling rule. That is not a handoff-ready spec yet.
