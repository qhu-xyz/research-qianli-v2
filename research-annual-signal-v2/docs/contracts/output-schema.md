# Output Schema Contract — MISO Annual Publication

**Date**: 2026-03-24  
**Scope**: published annual constraint and SF parquets for MISO annual releases

---

## Purpose

The published annual signal needs an explicit output contract.

Until now, the schema lived implicitly in:

- `ml/signal_publisher.py`
- legacy release artifacts
- scattered verification notes

That is not sufficient. A published release must have:

- a documented output schema
- a machine-readable schema definition in code
- explicit validation in the publish path

This contract is the human-readable source for the annual published output shape.
The machine-readable counterpart lives in:

- [output_schema.py](/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2/ml/products/annual/output_schema.py)

---

## 1. Constraint parquet contract

The published annual constraint parquet must contain exactly these columns:

1. `constraint_id`
2. `flow_direction`
3. `mean_branch_max`
4. `mean_branch_max_fillna`
5. `ori_mean`
6. `branch_name`
7. `bus_key`
8. `bus_key_group`
9. `mix_mean`
10. `shadow_price_da`
11. `density_mix_rank_value`
12. `density_ori_rank_value`
13. `da_rank_value`
14. `rank_ori`
15. `density_mix_rank`
16. `rank`
17. `tier`
18. `shadow_sign`
19. `shadow_price`
20. `equipment`
21. `constraint_limit`
22. `__index_level_0__`

This matches the current `V7.0B` constraint schema shape.

### Required non-null columns

These fields are required and must never be null in the published output:

- `constraint_id`
- `branch_name`
- `flow_direction`
- `shadow_sign`
- `tier`
- `bus_key`
- `constraint_limit`

If any of these are missing, publish must fail.

### Notes

- `constraint_limit` is part of the published contract and is not optional.
- `constraint_limit` must be sourced at the CID level, not inferred from a branch-level aggregate.
- round and class type are part of the output path / release manifest, not row-level columns in the parquet.

---

## 2. SF parquet contract

The published annual SF parquet must:

- use `pnode_id` as the row identifier in the same style as `TEST.Signal.MISO.DA_ANNUAL_2YR.V3.R1`
- preserve real SPICE/DA node ids such as `AEC`, `AECI.ALTW`, etc. as the node key
- not expose a positional `RangeIndex` (`0,1,2,...`) as the effective node index
- serialize `pnode_id` as the parquet row index (`pandas` metadata `index_columns = ["pnode_id"]`)
- not duplicate `pnode_id` as a regular data column when writing the parquet
- contain one column per published constraint
- use the pipe key format:
  - `{constraint_id}|{shadow_sign}|spice`

Every published constraint must have a corresponding SF column.

### Verified presentation vs DA annual reference

Verified against:

- `/opt/data/xyz-dataset/signal_data/miso/sf/TEST.Signal.MISO.DA_ANNUAL_2YR.V3.R1/2025-06/aq1/onpeak/7102553ce81044558697ca1faa196d25-0.parquet`

The reference parquet presents SF as:

- pandas/parquet row index named `pnode_id`
- no standalone `pnode_id` data column after `pd.read_parquet(...)`
- string node ids in the index, e.g. `AECI`, `AECI.ALTW`, `AECI.AMMO`

Annual SPICE publication must preserve that row-key presentation.

If any published constraint is missing SF coverage, publish must fail.

---

## 3. Release compatibility rule

Any future annual release must:

- either preserve this output schema exactly
- or explicitly version the output schema and document the change in the release manifest

Silent schema drift is not allowed.

Examples of schema drift that require a contract update:

- adding or removing a published column
- renaming a column
- changing nullability requirements
- changing SF row-key representation (`pnode_id` index vs positional index)
- changing the SF column-key format

---

## 4. Validation requirements

The publish path must validate:

- all required columns present
- required non-null columns contain no nulls
- every published constraint has SF coverage
- SF row identifiers are real `pnode_id` values, not positional integers
- output ordering matches the contract

These checks must happen before writing the final output.

---

## 5. Relationship to other contracts

This contract is separate from:

- model spec contract
- registry schema contract
- feature recipe contract

Why:

- a model can change while the published output schema stays the same
- a release can remain compatible even if internal features or evaluation artifacts change

The output contract defines the external publication surface.
