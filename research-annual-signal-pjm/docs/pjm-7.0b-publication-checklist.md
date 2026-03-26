# PJM 7.0b Publication Checklist

**Date**: 2026-03-25
**Status**: Pre-implementation checklist
**Release**: `pjm 7.0b`

---

## 1. Contract Freeze

- Freeze signal prefix and round paths from [manifest.json](/home/xyz/workspace/research-qianli-v2/research-annual-signal-pjm/releases/pjm/annual/7.0b/manifest.json)
- Freeze published constraint schema
- Freeze SF row/column contract
- Freeze branch-to-constraint expansion rule
- Freeze metadata provenance rule
- Freeze tier assignment policy
- Freeze dedup / branch-cap policy

Frozen for `7.0b`:

- publish post-dedup subset only
- `5` tiers
- `200` constraints per tier
- `tier 0` strongest
- `branch_cap = 3`
- `constraint_limit` required

## 2. Publisher Implementation

- Add `ml/markets/pjm/signal_publisher.py`
- Add publish-only feature assembly path for annual `a`
- Add branch -> constraint expansion
- Add metadata attachment
- Add tier assignment
- Add SF build path with `pnode_id` parquet index
- Add schema validation before write

## 3. Required Output Checks

- Constraints parquet exists for every published cell
- SF parquet exists for every published cell
- Constraint keys are unique
- SF columns exactly match published constraint keys
- SF row index name is `pnode_id`
- No published constraint is missing SF coverage
- `constraint_limit` is populated for every published row
- `rank` direction is low-is-best
- published row count is `<= 1000`
- tiers are filled in `200`-row blocks where enough publishable rows exist
- overlap rows with released `V4.6` match on `constraint`, `equipment`, `convention`, and sign direction

## 4. Scope Checks

- All planning years `2019-06` through `2025-06`
- All rounds `R1-R4`
- All class types `onpeak`, `dailyoffpeak`, `wkndonpeak`
- `2025-06` marked publishable but not fully evaluable

## 5. Review Questions

- Are the proxy `deviation_max` / `deviation_sum` formulas acceptable for the published surface?
- Does the sign fallback ever trigger after overlap/V4.6 parity checks?
- Do we want stronger SF-similarity pruning than `branch_cap = 3` in a later release?

## 6. Artifacts

- `releases/pjm/annual/7.0b/manifest.json`
- `releases/pjm/annual/7.0b/smoke_test.json`
- registry artifacts to be added after first implementation run:
  - `spec.json`
  - `metrics.json`
  - `analysis.json`
