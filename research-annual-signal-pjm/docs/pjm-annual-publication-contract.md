# PJM Annual Signal Publication Contract

**Date**: 2026-03-25
**Status**: Draft publication contract for `pjm 7.0b` (rev 2)
**Goal**: define the exact release surface and validation gates for publishing a PJM annual signal that is benchmarked against released `V4.6`

---

## 1. Release Goal

The publication target is:

- release family: `pjm 7.0b`
- benchmark baseline: released `TEST.Signal.PJM.SPICE_ANNUAL_V4.6.R{1,2,3,4}`
- publish scope:
  - planning years `2019-06` through `2025-06`
  - rounds `R1-R4`
  - class types `onpeak`, `dailyoffpeak`, `wkndonpeak`

The current internal best challenger is `baseline_v69`. For publication discussions, treat it as the candidate model behind `pjm 7.0b`, not as the baseline itself.

---

## 2. What MISO Teaches Us

The sibling MISO repo gives the right release pattern:

- one explicit `signal_publisher.py` module owns final publish assembly
- release identity is frozen in a release manifest
- publication has a smoke-test artifact
- research artifacts and publication artifacts are separated
- output schema is contracted and validated before write
- SF rows are keyed by parquet index `pnode_id`
- published constraint IDs are the exact SF columns

Useful references:
- [signal_publisher.py](/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/ml/markets/miso/signal_publisher.py)
- [2026-03-14-project1-annual-signal-publication.md](/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/docs/superpowers/specs/2026-03-14-project1-annual-signal-publication.md)
- [7.1b manifest.json](/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/releases/miso/annual/7.1b/manifest.json)

PJM should copy this release discipline even if the final schema differs slightly.

Most important MISO lesson for PJM publication:

- **published artifact = final post-dedup subset**
- pmodel consumes the published constraint/SF set directly
- therefore branch expansion, dedup, branch caps, and tier assignment must happen before write
- not every candidate CID should be published

---

## 3. Current PJM Publish Surface to Match

Verified live `V4.6` paths:

```text
/opt/data/xyz-dataset/signal_data/pjm/
  constraints/TEST.Signal.PJM.SPICE_ANNUAL_V4.6.R{1,2,3,4}/{PY}/a/{class_type}/
  sf/TEST.Signal.PJM.SPICE_ANNUAL_V4.6.R{1,2,3,4}/{PY}/a/{class_type}/
```

### 3.1 Constraint parquet

Observed on `2024-06 / R2 / onpeak`:

- `1049` rows
- `34` data columns
- pandas parquet index = constraint key
- polars shows this index as `__index_level_0__`, for `35` visible columns

Observed columns:

- density features:
  - `{0,60,65,70,75,80,85,90,95,100}_{max,sum}`
- derived metrics:
  - `deviation_max`
  - `deviation_sum`
- metadata / rank:
  - `flow_direction`
  - `equipment`
  - `convention`
  - `shadow_sign`
  - `shadow_price`
  - `shadow_price_da`
  - `constraint`
  - `deviation_max_rank`
  - `deviation_sum_rank`
  - `shadow_rank`
  - `rank`
  - `tier`

### 3.2 SF parquet

Observed on the matching `V4.6` cell:

- shape: `11329 x 1049`
- pandas index name: `pnode_id`
- columns: published constraint keys in the form `{constraint}|{shadow_sign}|spice`

This means your requested SF rule is correct:

- **SF row key must be parquet index `pnode_id`**
- do not publish a positional integer index
- a standalone `pnode_id` data column is not necessary if the parquet index is correctly named

---

## 4. PJM 7.0b Publish Identity

Proposed release identity:

- signal prefix: `TEST.Signal.PJM.SPICE_ANNUAL_V7.0B`
- round-specific paths:
  - `TEST.Signal.PJM.SPICE_ANNUAL_V7.0B.R1`
  - `TEST.Signal.PJM.SPICE_ANNUAL_V7.0B.R2`
  - `TEST.Signal.PJM.SPICE_ANNUAL_V7.0B.R3`
  - `TEST.Signal.PJM.SPICE_ANNUAL_V7.0B.R4`

Publish grain:

- `(planning_year, market_round, class_type)`
- path period type remains annual `a`

Internal quarter logic:

- quarter-aware bridge / GT handling is internal only
- quarterly partitions must not appear in the published path

---

## 5. Proposed PJM 7.0b Output Contract

## 5.1 Constraint parquet

Base rule:

- keep the `V4.6` output shape as close as possible
- only diverge where explicitly useful and contracted

### Final schema decision

`pjm 7.0b` constraint parquet should publish:

- all `V4.6` visible data columns
- parquet index key = `"{constraint_id}|{shadow_sign}|spice"`
- one additive field: `constraint_limit`

This is an intentional schema change relative to `V4.6`, so the contract is:

- **published schema = `V4.6-compatible-plus-limit`**

Do not add `branch_name` as a published data column. Keep it as a debug / registry artifact only.

### Published columns

- density features:
  - `{0,60,65,70,75,80,85,90,95,100}_{max,sum}`
- derived metrics:
  - `deviation_max`
  - `deviation_sum`
- metadata / rank:
  - `flow_direction`
  - `equipment`
  - `convention`
  - `shadow_sign`
  - `shadow_price`
  - `shadow_price_da`
  - `constraint`
  - `deviation_max_rank`
  - `deviation_sum_rank`
  - `shadow_rank`
  - `rank`
  - `tier`
  - `constraint_limit`

Notes:

- For `7.0b`, `deviation_max`, `deviation_sum`, and their rank columns are reproducible proxy features, not PowerWorld-native values.
- The parquet index is the canonical join key into SF.

## 5.2 SF parquet

Must satisfy:

- parquet row index name = `pnode_id`
- columns exactly equal published constraint keys
- no NaN
- float64 values

---

## 6. Publication Procedure for PJM

The release procedure should mirror MISO structurally:

1. Score branches for one `(PY, R, ctype)` annual publish cell
2. Expand branch scores to candidate constraints through branch-to-CID mapping
3. Attach publish metadata
4. Build deterministic final constraint ordering
5. Assign tiers
6. Build SF matrix for exactly the published constraint set
7. Validate and write both parquets

Key PJM-specific questions that must be frozen before implementation:

### 6.1 Branch -> published constraint expansion

This is now tied down:

- publication is **post-dedup**, not all expanded CIDs
- branch scores are expanded to candidate `(branch_name, constraint_id)` rows from the annual bridge union for the exact `(planning_year, round, class_type)` cell
- a candidate constraint inherits its parent branch score
- sibling constraints under the same branch are ordered by:
  1. higher `shadow_price_da`
  2. lower `rank`
  3. lexical `constraint_id`

The published set is then selected by a deterministic walk over the expanded candidate list.

### 6.1.1 Dedup policy copied from MISO, with PJM annual adjustments

- publish only the final post-dedup subset
- `branch_cap = 3`
- exclude all-zero or missing-SF constraints
- dedup within branch / sibling groups, not globally across the full file
- deterministic tie-break: `shadow_price_da` desc, then `constraint_id` asc

Implication:

- **not all expanded CIDs will be published**
- expansion produces the candidate universe
- dedup + branch cap + tier fill produces the final published constraint set

### 6.2 Metadata provenance

This should follow the released-signal pattern, using the same simple provenance style as the DA annual families.

| Published field | Provenance | Rule |
|---|---|---|
| `constraint` | selected `constraint_id` | monitored portion before the first `:`; this matches PJM `V4.6` `constraint` semantics and PJM DA annual `monitored_line` semantics |
| parquet index | selected `constraint_id` + sign | `"{constraint_id}|{shadow_sign}|spice"` |
| `equipment` | selected bridge row + `constraint` | exact `V4.6` style: `"{branch_name},{constraint}"` |
| `convention` | selected bridge row | copy bridge `convention` exactly |
| `constraint_limit` | selected bridge / limit row | copy `limit` exactly |
| `shadow_price_da` | per-constraint historical DA aggregation | class-type-specific DA history over June `(PY-2)` through March `PY` |
| `shadow_sign` | historical DA sign | sign of the same per-constraint historical DA aggregation; if zero, fallback to overlapping `V4.6` sign if available, else `-1` |
| `flow_direction` | derived from sign | `flow_direction = -shadow_sign`, matching observed `V4.6` sign convention |
| `shadow_price` | publish sign field | set equal to `shadow_sign`, matching observed PJM `V4.6` behavior |

Validation rule:

- where a published `constraint_id` overlaps released `V4.6` for the same cell, assert parity for `constraint`, `equipment`, `convention`, and sign direction

### 6.3 Tier policy

This is now fixed:

- `5` tiers total
- `200` constraints per tier
- total target published constraints per cell = `1000`
- `tier 0` = strongest / heaviest binding bucket
- `tier 4` = weakest published bucket

Assignment rule:

- sort candidate published rows by `rank` ascending
- fill `tier 0` with the first `200`
- fill `tier 1` with the next `200`
- continue through `tier 4`
- if a cell has fewer than `1000` publishable constraints after dedup, publish the available rows and allow trailing tiers to be short

### 6.4 Dedup policy

PJM copies the MISO principle and simplifies the first release:

- post-dedup subset only
- `branch_cap = 3`
- zero-SF exclusion required
- no duplicate published parquet index keys
- deterministic tie-breaks required

For `7.0b`, there is no additional SF-correlation pruning beyond zero-SF exclusion and branch-cap selection. That keeps the first PJM publisher reproducible and reviewable.

---

## 7. What Is Already Tied Down vs Not

### Tied down

- publish grain is annual `a`
- supported ctypes are `onpeak`, `dailyoffpeak`, `wkndonpeak`
- `wkndonpeak` uses holiday-aware hour filtering
- benchmark baseline is `V4.6`
- SF row contract should be `pnode_id` index
- `constraint_limit` is required in the new published schema
- published artifact is post-dedup, not the full expanded CID universe
- tier policy is `5 x 200`, with `tier 0` strongest
- branch cap is `3`
- metadata provenance follows the table above

### Not fully tied down

- exact proxy formulas for published `deviation_max` / `deviation_sum`
- whether sign fallback to `-1` is ever exercised in practice
- whether a later PJM release should add stronger SF-similarity dedup beyond `branch_cap = 3`

So the publication contract is much tighter now, but not yet “100% done” until the first publisher run validates the metadata parity rules.

---

## 8. What You Missed

These were the main missing items beyond “publish limit” and “SF index should be pnode_id”:

1. **Constraint parquet schema decision**
   Resolved: `V4.6-compatible-plus-limit`, without published `branch_name`.

2. **Branch -> constraint publish rule**
   Resolved: expand to candidates, then publish final post-dedup subset only.

3. **Metadata provenance**
   Resolved in the table above, with parity checks against overlapping `V4.6` rows.

4. **Tier contract**
   Resolved: `5` tiers, `200` constraints each.

5. **Constraint selection / dedup**
   Resolved for `7.0b`: post-dedup only, `branch_cap = 3`, zero-SF exclusion, deterministic tie-breaks.

6. **Release artifact layer**
   Need:
   - `releases/pjm/annual/7.0b/manifest.json`
   - `releases/pjm/annual/7.0b/smoke_test.json`
   - normalized registry artifacts

7. **Publishability policy for incomplete cells**
   Now resolved per-round in `manifest.json`:
   - 2022-06: **all rounds R1-R4 publishable** (data backfilled as of 2026-03-25)
   - 2025-06 R1: **publishable** but not evaluable vs V4.6 (no V4.6 R1 for 2025-06)
   - 2025-06 R2-R4: publishable and evaluable vs V4.6
   - 2025-06 all rounds: GT incomplete (DA through Mar 2026, 10/12 months)
   - All other cells: publishable and evaluable

8. **Validation gates**
   Need explicit checks:
   - no missing SF for published constraints
   - constraint key uniqueness
   - SF columns exactly match published keys
   - no nulls in required columns
   - round/class-type path completeness

9. **Constraint ordering beyond top-K**
   Need a deterministic full ranking, not just evaluation at `K=200/400`.

10. **Downstream compatibility test**
    Need one real round-trip load through the consumer path before calling it publishable.

---

## 9. Recommended Next Step

Before implementing the publisher, only these final checks remain:

1. validate the metadata provenance table against one real publisher dry run
2. validate sign parity on overlapping `V4.6` rows
3. confirm the proxy `deviation_*` formulas are acceptable for the release surface

After that, `ml/markets/pjm/signal_publisher.py` can be implemented.
