# V7.0 Publication Verification Plan

Date: 2026-03-19

Signal under review:
- Constraints: `/opt/data/xyz-dataset/signal_data/miso/constraints/TEST.Signal.MISO.SPICE_ANNUAL_V7.0.R1/`
- SF: `/opt/data/xyz-dataset/signal_data/miso/sf/TEST.Signal.MISO.SPICE_ANNUAL_V7.0.R1/`

## Goal

Verify that the published V7.0 files are structurally valid and, more importantly, that the `DA CID -> branch -> published SPICE CID` mapping is correct enough to trust the signal.

This plan is driven by the teammate asks:

1. `把真实da merge进来存个文件我看看`
2. `我看他map的对不对`
3. `可以在training set上map一下, signal也map一下`
4. `可以存个21年的一个25年的，一个多的一个少的，看看有啥问题`

## What Is Already Known

### Output inventory

- Published files present: `54` constraint files, `54` SF files.
- A full `2019-06` to `2025-06` grid would imply `56` files per artifact type.
- Missing slices:
  - `2025-06/aq4/onpeak`
  - `2025-06/aq4/offpeak`

This must be acknowledged before running any full-set verification.

### Basic structural sweep

Across the `54` published constraint files:

- Every file has exactly `1000` rows and `21` columns.
- No nulls were found in the published constraint files.
- Unique branches per file range from `733` to `777`.
- Duplicate branch rows per file range from `223` to `267`.

Across the `54` published SF files:

- Every file has `1001` columns (`pnode_id` + `1000` constraint columns).
- No nulls were found in sampled SF files.
- Pnode counts drift materially by year: from `2032` up to `2266`.

### Known verification risks

1. Raw SPICE presence is not the same as model-universe presence.
   Example: `AUST_TAYS_1545 A` exists in raw `MISO_SPICE_CONSTRAINT_INFO`, but all mapped annual SPICE CIDs for `2025-06/aq2` are inactive in the density-based universe, so the branch never enters the model table.

2. Some realized DA CIDs never map to any branch.
   Example: `2025-06/aq2/offpeak`, DA CID `511847` carries real SP but is not recovered by the annual bridge or monthly fallback.

3. The published artifact is constraint-level, but research evaluation is branch-level.
   Verification must collapse back to branches before reporting coverage or capture metrics.

4. The mapping gap worsens sharply in late years.
   Example GT diagnostics:
   - `2021-06/aq1`: `18` still-unmapped DA CIDs
   - `2025-06/aq2`: `444` still-unmapped DA CIDs
   - `2025-06/aq3`: `193` still-unmapped DA CIDs, but `152` recovered via monthly fallback

## Verification Questions

The verification should answer these, in order:

1. Are the published files complete and schema-consistent?
2. For a given slice, when real DA is merged in, which published constraints map to which branches, and does that mapping look correct?
3. Where does value get lost?
   - DA CID never maps to branch
   - DA CID maps to branch, but branch is outside the model universe
   - branch is in the model universe, but does not survive publication
4. Does the behavior differ between an earlier cleaner year and a later degraded year?
5. Are there obvious outliers, missing columns, NaNs, fan-out issues, or drift patterns that should block whole-run verification?

## Recommended Verification Outputs

Create saved verification tables, not just console summaries.

Recommended output root:
- `registry/publication_verification/v7_mapping_review/`

For each inspected slice, save:

1. `real_da_merged.parquet`
   - One row per realized DA CID
   - Includes DA SP, class, branch mapping result, mapping source, ambiguity/unmapped status

2. `training_universe_map.parquet`
   - One row per branch in the model table
   - Includes branch features, GT, whether branch has active SPICE CIDs, published/not-published flag

3. `published_signal_map.parquet`
   - One row per published constraint
   - Includes published SPICE CID, branch, tier, rank, flow direction, real DA merged back at branch level

4. `loss_waterfall.parquet`
   - DA CID status buckets:
     - mapped to branch + published
     - mapped to branch + in model universe + not published
     - mapped to branch + outside model universe
     - still unmapped

5. `summary.json`
   - counts, SP sums, branch counts, duplicate counts, top losses

## Exact Verification Slices

Use at least these two anchor years:

1. Earlier / cleaner year:
   - `2021-06/aq1/onpeak`
   - reason: low unmapped-CID pressure, useful baseline

2. Later / degraded year:
   - `2025-06/aq2/offpeak`
   - reason: heavy unmapped-CID pressure and known failure examples

Optional third slice:
- `2025-06/aq3/onpeak`
- reason: monthly fallback actually recovers a large block (`152` CIDs via `2026-01`), so this is a good control for fallback behavior

This satisfies the “store a 21 year and a 25 year, one more and one less” request.

## Verification Sequence

### Step 1: File-level sanity

Before any mapping review:

- enumerate all expected `(planning_year, aq, class_type)` slices
- identify missing published files
- validate schema and dtypes across all published files
- validate no nulls in published constraints and SF
- validate row counts:
  - constraints should be `1000`
  - SF should be `1 + 1000` columns

Pass rule:
- no silent schema drift
- missing slices explicitly listed

### Step 2: Merge real DA into branch mapping

For each selected slice:

- start from realized DA rows
- map `DA CID -> branch` using the same bridge logic as GT
- mark mapping status:
  - annual bridge
  - monthly fallback
  - ambiguous dropped
  - still unmapped

Save this as `real_da_merged.parquet`.

This is the file the teammate wants to inspect first.

### Step 3: Compare training universe vs published signal

For each selected slice:

- build the training/model table
- attach branch-level GT
- attach whether branch has any active annual SPICE CIDs
- attach whether branch appears in the published V7 signal

This separates:
- branch exists in GT
- branch exists in model universe
- branch gets published

Save this as `training_universe_map.parquet`.

### Step 4: Compare published constraints back to branches

For each published constraint:

- attach branch name
- attach branch-level realized DA
- attach whether branch was binding in the target class
- attach mapping provenance from Step 2

Save this as `published_signal_map.parquet`.

This is where manual spot-checking of “map 对不对” should happen.

### Step 5: Build a loss waterfall

Compute SP and count loss at each stage:

1. Total class-specific DA
2. DA CIDs mapped to branches
3. Mapped branches that enter model universe
4. Model-universe branches that appear in published signal

Also break out top offenders:

- top unmapped DA CIDs by SP
- top mapped-but-outside-universe branches by SP
- top in-universe-but-not-published branches by SP

Save as `loss_waterfall.parquet` and `summary.json`.

## Specific Checks To Run

### Mapping correctness checks

- `DA CID -> branch` must be one-to-zero-or-one after ambiguity handling
- `published SPICE CID -> branch` must be one-to-one within the published file
- join coverage must be reported at each stage
- fan-out must be reported explicitly:
  - DA CID to branch
  - branch to SPICE CID
  - published constraints to unique branches

### Outlier checks

- top unmapped DA CIDs by SP
- top branches with many published sibling constraints
- branches with large DA SP but zero historical DA features
- branches present in raw `CONSTRAINT_INFO` but absent from model universe

### Drift checks

- per-year unmapped CID rate
- per-year mapped-SP coverage
- per-year unique branch count in published files
- per-year duplicate branch rows in published files
- SF pnode count drift by year (2032 in 2023-06 to 2266 in 2025-06)
  - Published SF uses raw SPICE pnode_ids WITHOUT nodal replacement applied.
  - `MisoNodalReplacement` in pbase maps `to_node → from_node` when pnodes are renamed.
  - Downstream consumers joining SF across years must apply nodal replacement themselves.
  - This is a downstream concern, not a publisher bug.

### Column checks

For saved verification tables:

- dtypes
- null counts
- distinct counts for IDs
- min/max for continuous fields
- value counts for status buckets

## Concrete Examples To Include In Manual Review

These should be included in the first verification bundle:

1. Working example:
   - branch `FORMAFORMN11_1 1`
   - DA CIDs `314718`, `394044`
   - has active annual SPICE hook and should map cleanly

2. Unmapped example:
   - DA CID `511847`
   - real SP present, but no branch hook recovered

3. Raw-SPICE-but-not-active example:
   - branch `AUST_TAYS_1545 A`
   - exists in `MISO_SPICE_CONSTRAINT_INFO`
   - annual mapping exists
   - all mapped CIDs inactive for `2025-06/aq2`
   - therefore absent from the model table

These three examples make the failure modes concrete for manual inspection.

## Recommended Execution Order

1. Verify file inventory and schema on all `54 + 54` files
2. Build the merged DA inspection file for `2025-06/aq2/offpeak`
3. Build the training-universe and published-signal mapping files for the same slice
4. Repeat the same bundle for `2021-06/aq1/onpeak`
5. Compare the two bundles side by side
6. Only then decide whether a full all-slice verification sweep is worth running

## What Would Block Sign-Off

- any missing expected slice without an explicit reason
- any schema drift across published files
- any nulls in published artifacts
- unexplained many-to-many mapping after ambiguity handling
- inability to explain top unmapped DA SP losses
- inability to distinguish:
  - unmapped DA loss
  - outside-universe loss
  - publication-stage loss

## Bottom Line

Do not start with whole-run metrics.

Start by saving a small number of inspection-friendly mapping tables for:
- one early cleaner slice
- one late degraded slice

The main thing to verify is not just whether the published signal loads; it is whether the path

`real DA CID -> branch -> model universe -> published SPICE CID`

behaves the way we think it does, and where it breaks when it does not.
