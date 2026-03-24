# Test Design: Round-Sensitive Annual Pipeline

**Date**: 2026-03-24  
**Purpose**: Verify that the round-sensitive annual implementation is correct, reproducible, and free of post-cutoff leakage.

---

## 1. Test goals

The test suite must prove all of the following:

1. annual round cutoffs are resolved correctly from the calendar source
2. history uses only data strictly before the round cutoff under the chosen granularity
3. daily DA cache aggregation is correct
4. round-aware auction-side loaders read the correct round partitions
5. round-aware model tables differ only when input data differs
6. publish outputs are internally consistent by round
7. the new code remains compatible with old wrappers during migration

It must also prove that annual artifacts are keyed at the correct base grain:

- `planning_year`
- `aq_quarter`
- `ctype`
- `market_round`

---

## 2. Core invariants

### 2.1 Cutoff invariants

- in v1, no history observation on or after the round close **date** may appear in features
- if two rounds share the same cutoff interval, their history features must match
- if one round includes more valid pre-close days than another, its history features may differ only due to those extra days

### 2.2 Cache invariants

- daily cache must be the source of truth
- monthly cache must equal aggregation of daily cache
- cache manifests must cover every required date for a run
- annual model/build caches that depend on auction-side inputs must include `market_round` in the key

### 2.3 Round loader invariants

- density loader must use requested `market_round`
- limits loader must use requested `market_round`
- density signal score / flow-direction loader must use requested `market_round`
- SF loader must use requested `market_round`
- publish metadata loader must use requested `market_round`
- no silent fallback to round 1

### 2.4 Publish invariants

- no duplicate published constraint ids
- no duplicate `constraint_id|shadow_sign|spice`
- no published constraint missing SF
- branch cap enforced
- tier counts match requested tier sizes unless candidate pool is exhausted
- output metadata/path/spec must include both `ctype` and `market_round`

---

## 3. Test layers

### 3.1 Unit tests

Small, deterministic tests for pure logic.

### 3.2 Integration tests

End-to-end tests over one or two real slices using real cached data.

### 3.3 Data-contract tests

Tests that inspect cache manifests and source partitions.

### 3.4 Golden tests

Pinned-slice checks to catch silent regressions in published outputs.

---

## 4. Unit test design

### 4.1 Calendar and cutoff resolution

Test file:

- `tests/test_annual_round_calendar.py`

Cases:

- `R1`, `R2`, `R3` resolve to expected timestamps for a known planning year
- invalid round raises
- `history_cutoff_ts` is strictly before close time

Assertions:

- exact timestamp equality
- stable serialization into spec/manifest form

### 4.2 Daily cache aggregation

Test file:

- `tests/test_realized_da_daily.py`

Cases:

- aggregate one synthetic day of raw DA rows into daily cache
- aggregate multiple daily cache files into one month
- verify onpeak/offpeak separation

Assertions:

- per-day `realized_sp` equals `abs(sum(shadow_price))` within day + ctype
- monthly aggregate equals sum of daily `realized_sp`

### 4.3 Cutoff filtering

Test file:

- `tests/test_history_cutoff_filter.py`

Cases:

- records before cutoff are included
- records on the close date are excluded
- records after the close date are excluded

Assertions:

- exact row inclusion by date

### 4.4 Month-to-date partial aggregation

Test file:

- `tests/test_history_partial_month.py`

Cases:

- partial month with 5 valid days
- partial month with 0 valid days
- full month identical to legacy aggregate

Assertions:

- BF / SPDA / recency features change exactly as expected

### 4.5 Round-aware loader routing

Test file:

- `tests/test_round_loader_paths.py`

Cases:

- density loader called with `market_round=2`
- limits loader called with `market_round=3`
- density signal score / flow-direction loader called with `market_round=2`
- SF loader called with `market_round=1`

Assertions:

- resolved path contains the requested round
- missing partition raises explicitly

### 4.6 Wrapper compatibility

Test file:

- `tests/test_compat_wrappers.py`

Cases:

- old import path still works during migration
- wrapper delegates to new implementation

Assertions:

- same output schema
- same values on a fixed slice

---

## 5. Integration test design

### 5.1 Round-aware history integration

Test file:

- `tests/test_round_history_integration.py`

Slice:

- one planning year
- one quarter
- both `onpeak` and `offpeak`
- `R1`, `R2`, `R3`

Assertions:

- feature tables build successfully
- if extra pre-close days exist, some history features differ between rounds
- if no extra valid days exist, history features match exactly

### 5.2 Round-aware auction data integration

Test file:

- `tests/test_round_auction_data_integration.py`

Assertions:

- density features differ across rounds when round partitions differ
- limits can differ across rounds
- SF matrices can differ across rounds

This test should not require that differences always exist, only that the requested round partition is actually loaded.

### 5.3 Round-aware model table integration

Test file:

- `tests/test_round_model_table_integration.py`

Assertions:

- cache key includes round
- output metadata includes `class_type` and `market_round`
- output schema includes round metadata
- two rounds with same inputs produce same table
- two rounds with different inputs produce different table in expected columns only

### 5.4 Publish integration

Test file:

- `tests/test_round_publish_integration.py`

Assertions:

- publish succeeds for one real slice per round
- constraints and SF outputs align
- output path/manifest includes round
- output path/manifest includes ctype explicitly
- dedup invariants hold
- publish fails if any selected constraint lacks SF

---

## 6. Golden test design

### 6.1 Legacy parity tests

Purpose:

- prove that old behavior is preserved when using `R1` and month-level-equivalent cutoff behavior

Test file:

- `tests/test_annual_legacy_parity.py`

Assertions:

- round-aware `R1` output matches legacy `market_round=1` output on a pinned slice, subject to intentional metadata additions

### 6.2 Round delta tests

Purpose:

- prove that round-sensitive logic can actually produce differences when data differs

Test file:

- `tests/test_annual_round_deltas.py`

Assertions:

- at least one pinned slice shows a real feature delta between rounds
- if round partitions are identical, outputs are identical

---

## 7. Leakage test design

This is the highest-risk area and must be explicit.

### 7.1 Post-cutoff exclusion test

Construct a synthetic branch where:

- a large DA event occurs after the round close
- another smaller event occurs before the round close

Assertions:

- only the pre-close event contributes to features

### 7.2 Boundary timestamp test

For v1, replace sub-day timestamp testing with close-date boundary testing.

Construct records:

- one record on the day before close
- one record on the close date
- one record on the day after close

Assertions:

- only the pre-close-date record is included

If sub-day history is introduced later, add a separate exact-timestamp boundary test.

### 7.3 Manifest coverage test

If cache manifest does not cover required dates:

- run must fail before feature building

Assertions:

- explicit failure with actionable message

---

## 8. Benchmark and metric tests

### 8.1 Rank direction tests

Any benchmark/model artifact with a rank field must declare direction.

Test file:

- `tests/test_rank_direction_contract.py`

Assertions:

- missing direction metadata fails
- direction is respected by comparison helpers

### 8.2 Metric view separation tests

Test file:

- `tests/test_metric_view_contract.py`

Assertions:

- native, overlap, deployment, and coverage views cannot be mixed silently
- ambiguous “absolute rank” helpers are disallowed

---

## 9. Cache and manifest tests

### 9.1 Daily cache manifest

Test file:

- `tests/test_daily_cache_manifest.py`

Assertions:

- manifest lists covered dates
- manifest schema version valid
- manifest source snapshot present

### 9.2 Release manifest

Test file:

- `tests/test_release_manifest.py`

Assertions:

- supported rounds are declared
- cache manifest references are present
- signal name/path consistent

---

## 10. Acceptance criteria

The implementation is considered correct only if all of these pass:

1. unit tests for cutoff logic
2. unit tests for daily cache aggregation
3. loader routing tests for round-aware partitions
4. leakage boundary tests
5. publish invariant tests
6. legacy parity tests for `R1`
7. release/manifest validation tests
8. flow-direction round-routing tests

Additionally, one manual review checklist should pass:

- inspect one slice per round
- confirm round metadata in output
- confirm no hidden fallback to round 1
- confirm no published constraint missing SF

---

## 11. Recommended rollout order for tests

1. unit tests for calendar and daily cache
2. cutoff/leakage tests
3. round-aware loader tests
4. model table integration tests
5. publish integration tests
6. golden parity/delta tests
7. manifest tests

This order catches the most dangerous bugs first.

---

## 12. Minimum pinned slices

Use at least these pinned slices for integration/golden tests:

- one older MISO annual planning year
- one recent MISO annual planning year
- one `onpeak`
- one `offpeak`
- one slice where round partitions are known to differ

These slices should be documented in the test fixture manifest so future reruns remain comparable.
