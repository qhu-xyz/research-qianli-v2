# Cross-Document Consistency Review — 2026-03-12

Reviewed: design spec, implementer guide, test specification, handoff prompt.

## Findings & Resolutions

### F1: Cohort definition — `has_hist_da` vs `da_rank_value = 0`

**Status**: FIXED

**Issue**: Design spec SS6.2 and handoff prompt say cohorts use `has_hist_da` flag. Implementer guide SS11.2 and test spec G5 use `da_rank_value = 0` instead. These are not equivalent — `da_rank_value` is a rank (1-based for positive-history, `n_positive + 1` for zero-history), never literally 0.

**Resolution**: Use `has_hist_da` (boolean from history_features). Updated:
- Implementer guide SS11.2: cohort table now uses `has_hist_da = False` / `has_hist_da = True`
- Test spec G5: updated cohort definitions to use `has_hist_da`

### F2: `NB_Recall@50` vs `NB12_Recall@50`

**Status**: FIXED

**Issue**: Design spec and handoff prompt require explicit "12" (`NB12_Recall@50`). Test spec L2 and implementer guide lines 1351, 1353, 1393 use `NB_Recall@50`.

**Resolution**: Updated all occurrences to `NB12_Recall@50`.

### F3: v0c formula uses `bf_12` (onpeak) vs `bf_combined_12`

**Status**: FIXED

**Issue**: Design spec SS11 and handoff prompt say `bf_combined_12` (either-ctype). Implementer guide SS12 uses `bf_12` (onpeak-only per feature table).

**Resolution**: Updated implementer guide v0c formula to use `bf_combined_12_rank_norm`. The signal is class-type agnostic — combined BF is the correct feature.

### F4: `map_cids_to_branches` period_type should be required

**Status**: FIXED

**Issue**: Design spec signature has `period_type: str | None = None` (optional). But annual bridge has aq1-aq4 partitions — period_type is required for correct partition filtering. Call examples in SS3.1 Step 4, SS4.1 Step 2, SS5.1 Step 1 omit period_type.

**Resolution**: Updated design spec:
- Signature: `period_type: str` (required, no default)
- All call examples now include explicit `period_type` parameter
- Comment clarified: `'aq1'..'aq4' for annual, 'f0' for monthly`

### F5: "LEFT JOIN" wording in design spec SS4.1 Step 2

**Status**: FIXED

**Issue**: Says "LEFT JOIN DA cids onto mapped bridge" but the shared function does an inner join with unmapped cids tracked in diagnostics.

**Resolution**: Changed to "Map DA cids via bridge (unmapped cids tracked in diagnostics for monthly fallback)".

### F6: Implementer guide `NB_Recall` in Phase 2 steps

**Status**: FIXED (same as F2)

Lines 1351, 1353, 1393 updated to `NB12_Recall@50`.

---

## Plan Fixes (post-review, pre-implementation)

### P1: Task 4 — Calibration unit relationship clarified

**Issue**: Calibration works at branch level, but UNIVERSE_THRESHOLD is applied at cid level in data_loader. The relationship wasn't documented.

**Resolution**: Added explicit "Unit relationship" note to Task 4 explaining why they're consistent: `branch_rtm = max(cid_rtm)`, so a branch passes iff at least one cid passes. Calibration at branch level correctly predicts the cid-level filter's behavior.

### P2: Task 6 — GT returns only positive-binding branches

**Issue**: GT tests assumed `label_tier=0` (non-binding) in GT output. But GT should only return mapped branches with positive DA binding. Zero-fill belongs in features.py.

**Resolution**:
- GT module boundary note added: returns only positive-binding branches + diagnostics
- GT test `test_gt_tiered_labels` updated: expects labels {1,2,3} only, asserts no label_tier=0
- GT test `test_gt_combined_ctype` updated: asserts all returned branches have realized_shadow_price > 0
- GT implementation notes updated: tiered labels are 1/2/3 only
- Task 9 (features.py) updated: explicitly documents zero-fill responsibility (left join GT, fill missing with SP=0, label=0)
- New test `test_model_table_zero_fill` added to Task 9

### P3: Task 7 — Bridge mapping rule for historical months

**Issue**: The rule "use eval PY's annual bridge for all historical months" is a deliberate design choice that could easily be implemented wrong (e.g., using the PY that contains the historical month).

**Resolution**: Added "deliberate design choice" callout explaining the rule, why it's correct (branch_name identity consistency across data_loader/GT/history_features), and the trade-off (some historical cids may not exist in eval PY's bridge).

### P4: Task 7 — period_type in bridge calls

**Issue**: History features bridge call example omitted `period_type` parameter.

**Resolution**: Updated to include explicit `period_type=aq_quarter` for annual and `period_type='f0'` for monthly fallback.
