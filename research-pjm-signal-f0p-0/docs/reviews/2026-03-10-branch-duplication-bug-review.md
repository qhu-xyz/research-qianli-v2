# Branch-Duplication Bug Review

## Findings

1. **HIGH**: The proposed fix of deduplicating the primary VC@20 metric by `branch_name` would change the project’s evaluation target, not just correct a bug. The PJM handoff explicitly says multiple V6.2B `constraint_id`s can map to the same `branch_name`, that they should all receive the same `realized_sp` target, and that this is intentional. The deployment contract also requires preserving the full V6.2B row universe. Because the produced signal is still a row-level constraint signal, the primary evaluation metric must stay row-level if it is meant to predict production behavior. Deduping VC@20 by branch would make research numbers no longer correspond to the output that gets written.

2. **HIGH**: There is still a real PJM-specific modeling problem here, but it is branch multiplicity bias, not an incorrect join. Once realized DA is joined at the branch level, branches with 2-3 associated constraint rows get 2-3x weight in both training and row-level evaluation. That can distort feature importance and make comparisons against MISO less apples-to-apples. If the current ML model is underperforming because frequent moderate branches dominate while rare high-value branches are underrepresented, the right fix is to add branch-aware diagnostics or weighting, not to undo the intentional branch-level target join.

3. **MED**: The current debugging conclusion over-attributes the issue to v0 “tripling” value capture. Both models are scored against the same duplicated row set, and the branch-level label duplication applies to ML training as well. The stronger claim supported by the evidence is that branch multiplicity changes the optimization landscape and the top-k composition, not that v0 alone gets an unfair accounting treatment. That distinction matters because it points toward branch-aware training/evaluation diagnostics instead of changing the label construction rule.

## Recommended Next Steps

- Keep the official target join and official VC@20 metric at the row level.
- Add a secondary diagnostic that evaluates top-k after collapsing rows to one record per `branch_name`.
- Add branch-multiplicity summaries per month: unique branches, `constraint_id` per branch distribution, and top-k overlap at row vs branch level.
- If the branch-dedup diagnostic explains the regression, test branch-aware remedies in training:
  - one-row-per-branch training views
  - sample weights inversely proportional to branch multiplicity
  - branch-level aggregation as an auxiliary analysis, not a replacement for the production metric

## Basis

- `human-input/data-gap-audit.md`: “Multiple V6.2B constraint_ids can map to the same branch_name — they'd all get the same realized_sp target. This is intentional.”
- `human-input/handoff-claude.md`: same intent, plus deployment must preserve the full V6.2B row universe.
