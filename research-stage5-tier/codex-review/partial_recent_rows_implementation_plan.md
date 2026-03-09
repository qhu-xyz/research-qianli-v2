# Implementation Plan: Add Partial Recent Data Back Safely

## Goal

Recover the most recent **partial** realized-DA information for `f0` without reintroducing the `M-1` leakage bug.

Primary target:

- For target month `2025-03 / f0`, at submission around `2025-02-12`, use:
  - full settled history through `2025-01`
  - partial February history through `2025-02-12`
  - no data after `2025-02-12`

Secondary target:

- Extend the same `as-of cutoff` framework to `f1`, `f2`, and quarterly products (`q2`, `q3`, `q4`, ...).

This plan spends most of its detail on `f0`, because that is the immediate production problem and the cleanest place to establish the right time model.

---

## Executive Summary

This is **not** “change everything,” but it **does** require changing the repo’s core time abstraction.

Today the code assumes:

- one row per `(auction_month, constraint_id, flow_direction)`
- full-month realized history keyed by `YYYY-MM`
- training/eval windows defined only by month

To add partial recent data back safely, the repo needs an **as-of cutoff model**:

- every target row needs a concrete `cutoff_ts`
- realized DA history must be available at daily or finer granularity
- historical features must be computed from data available **as of that cutoff**
- training rows must still stop at the last fully labeled month unless we design a separate partial-label training regime

Recommended approach:

1. Keep training rows ending at `M-2` for `f0`
2. Add partial `M-1` information back as **new as-of features**, not as mislabeled full-month rows
3. Introduce a reusable `AuctionCutoffCalendar`
4. Introduce daily/as-of realized DA caches
5. Move feature engineering from “by month” to “by cutoff”

Upstream evidence from `/research-spice-shadow-price-pred-qianli` supports this direction:

- historical DA is built with an explicit auction cutoff and a `run_at_day`
- the cutoff month is truncated to the available date, not dropped entirely
- the cutoff month is **not** naively scaled to a full-month estimate
- recency/seasonality weighting is applied after truncation

---

## What Changes and What Does Not

What does **not** need a redesign:

- model classes and LightGBM training
- ranking metrics
- most of the benchmark/report plumbing
- formula-only baselines that do not depend on realized-history features

What **does** need redesign:

- time indexing and split logic
- realized DA storage format
- all realized-history feature builders
- `binding_freq_*` and `binding_recency`
- holdout/backtest semantics
- schema for non-monthly products

---

## Current State

The current code uses full-month realized history:

- [ml/realized_da.py](/home/xyz/workspace/research-qianli-v2/research-stage5-tier/ml/realized_da.py) caches one parquet per month with:
  - `constraint_id`
  - `realized_sp`

The current safe `lag1` design fixes leakage by dropping all `M-1` realized information:

- train rows stop at `M-2`
- `binding_freq_*` for target `M` only uses data through `M-2`

That is safe, but conservative. It leaves useful partial `M-1` information on the table.

---

## Key Design Decision

### Do not add partial `M-1` back as a training row

For `f0`, this is the main trap.

Example: target `2025-03`, submission around `2025-02-12`.

- We may know **partial February history** through `2025-02-12`
- We do **not** know the final label for month `2025-02`

Therefore:

- `2025-02` should **not** become a normal training row
- partial February should come in only as **feature inputs**

That means the correct design is:

- keep supervised training rows ending at `2025-01`
- add partial-February information into the features for:
  - the target row `2025-03`
  - each historical training row `T`, using the cutoff that would have existed when `T` was scored

---

## F0 Plan

## 1. Introduce an Explicit Auction Cutoff Calendar

Create a new source of truth for decision timing.

Suggested file:

- `ml/auction_calendar.py`

Suggested schema:

```text
auction_month        str   # e.g. 2025-03
period_type          str   # f0, f1, f2, q2, ...
submission_ts        ts    # exact modeled decision timestamp
cutoff_ts            ts    # last usable timestamp for upstream inputs
last_full_month      str   # e.g. 2025-01
partial_month        str?  # e.g. 2025-02 for f0, if partial is allowed
partial_start_ts     ts?
partial_end_ts       ts?
delivery_start_month str
delivery_end_month   str
```

For `f0`, `2025-03` would map to something like:

```text
auction_month = 2025-03
period_type = f0
submission_ts = 2025-02-12T...
cutoff_ts = 2025-02-12T...
last_full_month = 2025-01
partial_month = 2025-02
partial_end_ts = 2025-02-12T...
delivery_start_month = 2025-03
delivery_end_month = 2025-03
```

Why this is required:

- month strings are no longer enough
- “safe” feature generation now depends on exact availability time, not just `M-1` vs `M-2`

Implementation note:

- if exact historical submission timestamps are not available, define a deterministic business rule such as:
  - `submission_ts = 12th calendar day of M-1 at 12:00 US/Central`
- keep the rule centralized so the backtest is reproducible

---

## 2. Add Daily or Finer Realized DA Cache

The current monthly cache is insufficient for partial-month recovery.

Current schema:

```text
constraint_id
realized_sp
```

Needed schema for raw realized DA:

Suggested directory:

- `data/realized_da_daily/`

Suggested row schema:

```text
market_date          date
peak_type            str
constraint_id        str
shadow_price_sum     float   # daily net sum before abs, or already aggregated daily
realized_sp_day      float   # abs(shadow_price_sum)
binding_flag_day     int     # 1 if realized_sp_day > 0 else 0
```

Why daily is enough for the first version:

- the user requirement is “through `2025-02-12`,” not “through 10:37am”
- daily cutoffs recover most of the missing recent signal
- daily storage is much simpler than hourly

Stretch goal:

- if submission is intraday and same-day DA availability matters, later add hourly or publish-time snapshots

Implementation tasks:

1. extend `ml/realized_da.py` with a daily fetch/cache path
2. keep current monthly cache for legacy experiments
3. add a new loader:

```python
load_realized_da_asof(cutoff_ts, peak_type="onpeak") -> pl.DataFrame
```

This loader should return per-constraint aggregates through `cutoff_ts`, including:

```text
constraint_id
realized_sp_mtd
binding_days_mtd
observed_days_mtd
last_binding_date
```

---

## 3. Create F0 As-Of Feature Definitions

Do **not** force partial `M-1` data into the old monthly `binding_freq_*` semantics without making the meaning explicit.

Recommended feature families for `f0`:

### A. Full-month history features

These use only complete months through `M-2`:

- `binding_freq_full_1`
- `binding_freq_full_3`
- `binding_freq_full_6`
- `binding_freq_full_12`
- `binding_freq_full_15`

For target `2025-03`:

- these end at `2025-01`

### B. Partial recent-month features

These summarize the current partial month `M-1` through the cutoff:

- `binding_seen_partial_m1`
- `binding_days_partial_m1`
- `realized_sp_partial_m1`
- `binding_rate_partial_m1`
- `days_observed_partial_m1`
- `days_remaining_partial_m1`

Example for `2025-03 / cutoff=2025-02-12`:

- partial month = `2025-02`
- `binding_days_partial_m1` = number of observed February onpeak days with binding through Feb 12
- `binding_rate_partial_m1` = `binding_days_partial_m1 / observed_days_partial_m1`

Important rule for v1:

- do **not** multiply these features by a completion factor like `month_days / observed_days`
- keep the raw partial numerator and denominator
- if scaling is explored later, add it as a separate experimental feature, not as the only representation

### C. Weighted as-of recency/frequency features

Optional second-stage features:

- `binding_freq_weighted_3`
- `binding_freq_weighted_6`
- `binding_freq_weighted_12`

where the newest month contributes a partial weight:

```text
effective_month_weight = observed_days_in_partial_month / total_days_in_month
```

This is more compact than using separate full-month + partial-month features, but it is harder to audit.

Recommendation:

- start with **separate** full-history and partial-month features
- only move to weighted frequencies after the feature value is proven

---

## 4. Build a New As-Of Feature Module

Suggested new module:

- `ml/asof_features.py`

Suggested API:

```python
build_asof_history_features(
    constraint_ids: list[str],
    auction_month: str,
    period_type: str,
    cutoff_ts: pd.Timestamp,
    peak_type: str = "onpeak",
) -> pl.DataFrame
```

Output schema for `f0` v1:

```text
constraint_id
binding_freq_full_1
binding_freq_full_3
binding_freq_full_6
binding_freq_full_12
binding_freq_full_15
binding_seen_partial_m1
binding_days_partial_m1
realized_sp_partial_m1
binding_rate_partial_m1
days_observed_partial_m1
binding_recency_asof
```

Important rule:

- this module must accept `cutoff_ts`
- it must never infer timing from month names alone

---

## 5. Refactor Split Logic Around Availability, Not Month Index

Current split logic in [ml/data_loader.py](/home/xyz/workspace/research-qianli-v2/research-stage5-tier/ml/data_loader.py) is month-based.

For `f0` partial-month recovery, the rule should be:

- supervised train rows are months whose final labels would already be known by the target cutoff
- for `2025-03 / cutoff=2025-02-12`, that means training ends at `2025-01`

Recommended change:

- do **not** overload the current `load_train_val_test()`
- add a new loader path, for example:

```python
load_train_val_test_asof(
    auction_month: str,
    period_type: str,
    cutoff_ts: pd.Timestamp,
    train_months: int,
    val_months: int,
)
```

For `f0`, `2025-03`, `train_months=8`:

- train rows: `2024-06 .. 2025-01`
- test row: `2025-03`

Then enrich:

- training row `T` with features as-of `cutoff_ts(T)`
- test row `M` with features as-of `cutoff_ts(M)`

This keeps labels clean while restoring partial recent signals.

---

## 6. Keep V10e-Lag1 as Control; Add New F0 Variants

Recommended experiment sequence:

1. `v10e-lag1`
   - existing safe baseline

2. `v10e-asof-f0-a`
   - `v10e-lag1` + partial `M-1` binary/day-count features

3. `v10e-asof-f0-b`
   - `a` + partial realized-sp magnitude features

4. `v10e-asof-f0-c`
   - weighted frequency features replacing separate partial features

Why this sequence:

- it isolates how much value comes from partial recent data
- it avoids mixing architecture changes with too many new features at once

Success criterion:

- improvement over `v10e-lag1`
- no row or feature leakage
- stable gain on holdout under the same `as-of` rules

---

## 7. Testing Plan for F0

Minimum unit tests:

1. cutoff test
   - for `2025-03 / cutoff=2025-02-12`, verify no realized event after Feb 12 is used

2. row availability test
   - for target `2025-03`, verify train rows stop at `2025-01`

3. historical row simulation test
   - for training row `2025-01`, verify its partial month is `2024-12`, not `2025-01`

4. feature reproducibility test
   - same cutoff and same raw data produce identical as-of features

5. audit output test
   - generate a per-row lineage report:
     - target month
     - cutoff_ts
     - last_full_month
     - partial_month
     - max source date used

Recommended integration test:

- one target month (`2025-03`) end to end
- compare:
  - `v10e` leaky
  - `v10e-lag1`
  - `v10e-asof-f0`

---

## 8. F0 Risks and Decisions

### Decision 1: Partial feature representation

Options:

- separate partial-month features
- weighted monthly frequencies

Recommendation:

- start with separate features for interpretability and auditability

### Decision 1b: Should partial `M-1` be completion-scaled?

Recommendation:

- **No** for the first production-safe version

Rationale:

- naive scaling assumes the observed part of `M-1` is representative of the unobserved remainder
- that assumption is fragile around outages, weather, and congestion bursts
- the upstream `shadow_price_prediction` repo truncates the cutoff month to `run_at` and then applies designed historical weighting, rather than full-month completion scaling

If needed later, test scaled variants explicitly:

- `realized_sp_partial_m1_scaled`
- `binding_days_partial_m1_scaled`

but keep the unscaled forms in the feature set for auditability

### Decision 2: Daily vs hourly raw cache

Options:

- daily cache
- hourly/publish-time cache

Recommendation:

- start daily
- only go finer if there is evidence that same-day timing matters materially

Implementation note from upstream:

- the upstream repo uses a month cutoff plus `run_at_day` and truncates raw DA rows within that month
- that means our first version should support at least daily filtering up to `cutoff_ts`
- if submission timing inside the day matters, hourly or publish-time snapshots can be a second-phase extension

### Decision 3: Whether to include partial `M-1` magnitude

Binary-only partial features are safer initially:

- `binding_seen_partial_m1`
- `binding_days_partial_m1`
- `binding_rate_partial_m1`

Magnitude can be added in stage 2 if it improves signal without instability.

### Decision 4: Whether to mirror upstream historical-DA weighting

Recommendation:

- yes, at least for the first historical DA as-of implementation, mirror the upstream structure before inventing new transforms

What the upstream repo currently does:

- defines `cutoff_month = auction_month - 1 month`
- truncates DA rows in the cutoff month to `run_at_day`
- computes `recent_hist_da` from recent months and divides by the number of months included
- applies a fixed `recent_da_hist_discount = 1.3`
- computes seasonal features with explicit per-year discounts and month-count normalization

Implication for stage5:

- our `binding_freq_*` family can remain separate and simpler
- but any new `hist_da`/recent-history magnitude features should be compared directly against this upstream formulation
- do not assume `generate_stat_fast`-style scaling is required unless we verify that exact path separately

---

## F1 and F2 Extension

## 9. What Changes for F1/F2

Good news:

- the **cutoff model** is the same
- the submission happens before the auction, so the “latest usable realized history” is still determined by the same cutoff calendar

What changes:

- target period mapping
- which auction months exist for each period type
- how train/eval month availability is filtered

For `f1` and `f2`, the partial recent-history machinery can be reused almost unchanged.

Recommended rule:

- keep `cutoff_ts` tied to the auction event, not the delivery horizon
- partial-history features still summarize the current `M-1` partial month as of cutoff

Example:

- if auction month is `2025-03`
- `f0` predicts March delivery
- `f1` predicts April delivery
- `f2` predicts May delivery

But if all are scored at the same pre-auction cutoff, then:

- the latest usable realized DA is still “through cutoff”
- partial recent-history features come from the same observed window

Implementation work:

1. add `period_type` to the cutoff calendar
2. extend loaders to compute `delivery_start_month` and `delivery_end_month`
3. filter unavailable auction-period combinations using the auction schedule

---

## 10. F1/F2 Schema Changes

Yes, the schema needs to grow, even if not radically.

Current implicit key:

```text
(auction_month, constraint_id, flow_direction)
```

Recommended explicit key:

```text
auction_month
period_type
delivery_start_month
delivery_end_month
cutoff_ts
constraint_id
flow_direction
```

Why:

- once we move to `as-of` features, `cutoff_ts` becomes part of the row identity
- for `f1/f2`, delivery month is no longer equal to auction month

---

## Quarterly Products

## 11. Do Quarterly Schemas Also Change?

Yes. Quarterly rows need a richer target schema.

For quarterlies, one row no longer maps to a single delivery month. It maps to a delivery range.

Recommended quarterly row schema:

```text
auction_month
period_type        # q2, q3, q4, ...
delivery_start_month
delivery_end_month
delivery_months    # optional array or derived field
cutoff_ts
constraint_id
flow_direction
```

Recommended label schema:

```text
constraint_id
target_realized_sp_total
target_realized_sp_by_month   # optional exploded or side table
target_binding_any
target_binding_count_months
```

Recommendation:

- use quarter-level aggregate labels in the first version
- keep a month-by-month decomposition side table for diagnostics

Possible target definitions:

1. sum across the quarter
2. average across the quarter
3. max month in quarter

Recommendation:

- start with **sum across the quarter**
- it is the cleanest analogue to the existing monthly total-realized-sp target

---

## 12. Quarterly Partial Recent Data

Quarterlies can still use the same `as-of` recent-history features.

What changes is the **target period**, not the availability cutoff.

So the same partial recent-history feature family can be reused:

- full-month history through last settled month
- partial `M-1` history through auction cutoff

The bigger challenge for quarterly is not leakage. It is:

- sparse sample size
- target definition
- evaluation stability

That is why quarterly can come after `f0`.

---

## Recommended Delivery Sequence

## 13. Practical Rollout

### Phase A: F0 data foundation

1. add `AuctionCutoffCalendar`
2. add daily realized DA cache
3. build `load_realized_da_asof()`
4. add audit utilities showing max source date used

### Phase B: F0 feature recovery

1. add separate full-history + partial-month features
2. implement `load_train_val_test_asof()`
3. run `v10e-asof-f0-a/b/c`
4. compare against `v10e-lag1`

### Phase C: F1/F2

1. extend row schema with `period_type`, `delivery_start_month`, `delivery_end_month`, `cutoff_ts`
2. plug the same as-of feature builder into monthly forward products
3. validate schedule filtering

### Phase D: Quarterly

1. define target aggregation
2. add quarter row schema
3. run a smaller design study before full model sweep

---

## Final Recommendation

For `f0`, the correct implementation is:

- keep training rows ending at the last fully labeled month
- add partial recent information through a new `as-of` feature layer
- never treat a partially observed month as a normal labeled training row

For `f1/f2`, the same cutoff framework should apply with only schema and period-mapping expansion.

For `q2/q3/q4`, yes, the schema should also change. Quarterlies are still compatible with the same cutoff framework, but they require richer target metadata and should be handled after `f0`.

The most important implementation decision is to move the repo from a **month-indexed** design to an **as-of-cutoff** design. That is the clean way to recover recent signal without recreating leakage.
