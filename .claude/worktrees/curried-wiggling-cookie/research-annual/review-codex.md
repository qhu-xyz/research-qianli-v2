# Harsh Review: `design-planning.md`

## Score
**4.8 / 10**

## Findings

### Critical
1. **Gate definitions are not computable from the proposed metrics schema.**  
   `G8` requires `path_win_rate` (line 116), but `metrics.json` example has no per-path error deltas or win-rate field (lines 179-204). This makes promotion logic underspecified and likely impossible without reloading raw data.

2. **Metric naming is internally inconsistent and will cause implementation bugs.**  
   The design references `overall, per_py, per_class, stability, vs_baseline_h` (line 74) but example JSON uses `vs_h` (line 201). Similar mismatch exists between `dir_pct`/`dir100_pct` (line 67) and current utility output keys `dir_all`/`dir_100` in `scripts/baseline_utils.py`. This is a schema contract break waiting to happen.

3. **Comparison validity is weak despite known selection-bias risk in project docs.**  
   Gates compare quarter-level aggregate MAE/Dir directly (lines 109-117) but do not enforce matched-population head-to-head evaluation. `runbook.md` explicitly warns aggregate comparisons across different coverage are not valid. Current gate design can promote a candidate for the wrong reason.

### High
4. **CLI scope is inconsistent with implementation scope.**  
   The plan claims one file with 4 sections (line 300), but workflow uses `release` command (line 278) and release composition logic that is never specified as function/API in earlier sections. This is design drift inside the same document.

5. **`create_version` API and CLI example conflict.**  
   Function signature says `create_version(part, version_id, config: dict)` (line 86), while CLI example passes a free-text description string (line 243). This is not just stylistic; it changes input contract and error handling.

6. **Version ID format is inconsistent, which can break promotion scripts.**  
   Examples alternate between `b004` (lines 146, 147, 278) and `b004-seasonal-alpha` (lines 243, 255, 274). The plan does not define canonical ID normalization rules.

7. **No transactional safety for state changes.**  
   `promote()` updates `promoted.json` (line 99) but the design does not specify atomic writes, fsync, or rollback behavior. Interrupted writes can corrupt the single source of truth.

8. **Reproducibility metadata is incomplete.**  
   `config.json` includes `git_hash` (line 159), but no data fingerprint/version (only path strings on line 172), no Python/package versions, and no schema version. This is insufficient for true reproducibility.

### Medium
9. **Single-file constraint hurts maintainability for a long-lived pipeline.**  
   Packing metrics, storage, gates, and CLI into ~300 lines (line 20) optimizes short-term compactness but increases coupling and testing friction. This is the wrong tradeoff for a framework expected to expand to Part II.

10. **Backfill and verification steps are mostly manual and brittle.**  
   Verification section (lines 309-313) is command-driven and prose-based, with no automated tests, no fixture data, and no schema validation checks.

11. **“Existing scripts unchanged” claim is inaccurate.**  
   Workflow says scripts work unchanged (line 247) but immediately requires adding pipeline imports and `save_metrics` calls (lines 250-252).

12. **Bands section is conceptually right but operationally vague.**  
   It says Part II has independent gates (lines 288-290), but no concrete schema, threshold definitions, or compatibility checks against baseline versions.

## Open Questions That Must Be Resolved Before Implementation
1. Will promotion gates be computed strictly from stored `metrics.json`, or can gate checks load raw per-path data? This decides schema design immediately.
2. What is the canonical version identifier format (`b004` vs `b004-name`), and what parsing/validation rules apply?
3. Is matched-path head-to-head mandatory for all promotion decisions? If yes, define the exact matched key set and sampling window.
4. Should `release` be in scope for this document, or deferred to a second phase after baseline/bands versioning is stable?
5. What minimum reproducibility metadata is required (dataset hash, env lockfile hash, script args, timezone, random seed)?

## What Is Strong
1. The document correctly identifies the core operational problem (no durable experiment ledger).
2. Separate version tracks for baseline and bands is the right conceptual model.
3. Severity tiers for gates is directionally good and better than a single pass/fail metric.

## Recommended Revision Order
1. Freeze schema contracts (field names, required/optional keys, schema version).
2. Redesign gates around matched-path comparisons and explicitly computable inputs.
3. Resolve CLI/API contract mismatches and define canonical version IDs.
4. Add transactional file-write rules and basic failure semantics.
5. Add automated verification (unit tests + golden metrics fixtures) before backfill.
