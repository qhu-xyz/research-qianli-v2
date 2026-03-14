# Review: V4 Comprehensive Redesign

Reviewed file: `docs/plans/2026-03-11-v4-comprehensive-redesign.md`

## Findings

1. **HIGH: The plan does not actually explain or test the PJM-vs-MISO performance gap before redesigning around a single hypothesis.**

   The redesign is motivated by three PJM findings: formula weights are suboptimal, VC@20 was overemphasized, and `new@6` is a blind spot. That is a plausible refinement plan, but it is not a diagnosis plan for why PJM ML remains much weaker than MISO. The document does not propose any root-cause decomposition by slice, by cohort, by feature reliance, or against MISO reference behavior. Instead it jumps directly to `v0b` promotion and a new-constraint gate, then assumes that stronger structural features should dominate there. That is too narrow given the size of the cross-market discrepancy. If the real problem is elsewhere (row duplication bias, label noise, branch-level aggregation, weaker structural features, or different population mix), this plan can produce cleaner reporting without answering the project’s main question. See `docs/plans/2026-03-11-v4-comprehensive-redesign.md:8`, `docs/plans/2026-03-11-v4-comprehensive-redesign.md:14`, `docs/plans/2026-03-11-v4-comprehensive-redesign.md:124`. For contrast, MISO’s shipped result is still +43% to +92% VC@20 on holdout even though it has its own BF-zero blind spot; that means “new binders are hard” is not by itself enough to explain PJM’s weaker lift. See `/home/xyz/workspace/research-qianli-v2/research-miso-signal7/README.md:9` and `/home/xyz/workspace/research-qianli-v2/research-miso-signal7/docs/v70-new-binder-problem.md:341`.

2. **HIGH: The proposed `new@6` definition is too coarse to tell you how the model behaves on genuinely new constraints.**

   The plan defines “new constraint” as `branch_name` absent from the prior 6 months of realized DA bindings. That is operationally simple, but it collapses materially different populations into one mask: truly never-seen constraints, lapsed constraints with old history, and any join/pathology cases that look new only because of the matching rule. As a result, the proposed NewBind gates will not tell you whether the model is bad at first-time discovery, bad at dormant-reactivation, or bad at data quality edge cases. The MISO redesign work explicitly treats this as a taxonomy problem and recommends at least BF-zero vs BF-positive as the operational gate, with deeper sub-categories underneath. This PJM plan does not include any such decomposition. See `docs/plans/2026-03-11-v4-comprehensive-redesign.md:34`, `docs/plans/2026-03-11-v4-comprehensive-redesign.md:91`, `/home/xyz/workspace/research-qianli-v2/research-stage5-tier/docs/signal-quality-redesign.md:73`, `/home/xyz/workspace/research-qianli-v2/research-stage5-tier/docs/signal-quality-redesign.md:107`.

3. **MED: The plan improves metric coverage, but it still does not monitor the full metric surface it claims is important.**

   The document correctly broadens Group A from the current VC-heavy set by adding `VC@50`, `Spearman`, and NewBind metrics. That directly improves over the current setup. But it also says “all N-cutoffs matter equally” and then omits `NewBind_VC@50` from Group A even though it defines it as a first-class metric. The plan also adds only performance rates, not the monthly cohort-size context needed to interpret them, such as number of new binders or share of binding value coming from the new cohort. Without those denominator metrics, a pass/fail on new-bind performance is harder to interpret under changing prevalence. See `docs/plans/2026-03-11-v4-comprehensive-redesign.md:12`, `docs/plans/2026-03-11-v4-comprehensive-redesign.md:43`, `docs/plans/2026-03-11-v4-comprehensive-redesign.md:70`, `docs/plans/2026-03-11-v4-comprehensive-redesign.md:176`.

4. **MED: The plan does not address temporal drift in the right way; asking for walk-forward is aimed at the wrong layer.**

   If the concern is “2024-2025 may behave differently from earlier years,” the missing fix is not to switch to walk-forward. The repo already evaluates holdout with month-by-month retraining on trailing history, which is a walk-forward setup. The MISO postmortem explicitly makes the same point: walk-forward is the training protocol, not a sufficient answer to population drift or new-binder behavior. What is missing here is explicit temporal reporting and gating: year-split holdout summaries, early-vs-late holdout comparisons, rolling-origin dev windows, or drift alerts when 2025 materially differs from 2024. The current success criteria still collapse everything into aggregate holdout gates. See `scripts/run_holdout.py:145`, `scripts/run_holdout.py:178`, `docs/plans/2026-03-11-v4-comprehensive-redesign.md:164`, `/home/xyz/workspace/research-qianli-v2/research-miso-signal7/docs/v70-new-binder-problem.md:323`, `/home/xyz/workspace/research-qianli-v2/research-miso-signal7/docs/v70-new-binder-problem.md:339`.

## Answers To The Four Concerns

- **Concern 1: MISO vs PJM discrepancy**
  Partially addressed at best. The plan proposes one plausible explanation, but it does not include the diagnostic work needed to validate that explanation against the much stronger MISO lift.

- **Concern 2: behavior on new constraints / defining constraints**
  Addressed in intent, not adequately in design. The plan adds a new-constraint metric, but the definition is too coarse to answer the question with enough precision.

- **Concern 3: other metrics**
  Addressed partially. The metric surface is better than today, but still incomplete and internally inconsistent with the document’s “all cutoffs matter equally” claim.

- **Concern 4: 2024-2025 holdout and pattern shift**
  Not really addressed. The system already uses walk-forward training. The missing addition is drift-aware reporting and segmented validation, which the plan does not propose.

## Recommended Changes Before Approval

1. Add a diagnosis section specifically for the PJM-vs-MISO gap.
   Minimum outputs: per-slice lift decomposition, known-vs-new cohort lift, feature-importance comparison, and a statement of what hypothesis each check is meant to confirm or falsify.

2. Replace the single `new@6` gate with a small cohort framework.
   Minimum: `BF-positive`, `BF-zero/lapsed`, and `history-zero` or equivalent PJM-safe categories. Keep `new@6` only as a secondary summary if you still want it.

3. Expand reporting so cohort metrics are interpretable.
   Add monthly `n_new`, `new_value_share`, and `new_row_share` alongside `NewBind_*` performance metrics.

4. Add temporal segmentation to success criteria.
   At minimum, report 2024 vs 2025 separately on holdout and require no major degradation in the later year. A rolling-origin dev analysis would be better if you want to check whether lift is decaying over time.

5. Decide whether `NewBind_VC@50` matters or remove the “all N-cutoffs matter equally” claim.
   The current plan defines it, then excludes it from gating.
