# Review: `docs/plan-partial-month-bf.md`

Date: 2026-03-10

## Findings

### 1. High: the implementation no longer matches the feature definition

The plan defines:

```text
bf_partial = (# days constraint bound in M-1 partial window) / (# available days in M-1 partial window)
```

This is explicitly a daily-resolution fractional feature.

But the proposed implementation changes that into a binary indicator:

- Step 1 returns only a `set[str]` of binding constraints
- Step 2 computes:

```python
freq = np.array([1.0 if cid in partial_set else 0.0 for cid in cids])
```

That collapses:

- bound once in the partial window
- bound every day in the partial window

into the same value.

This is a substantive design break, not a cosmetic implementation detail.

If the intended feature is fractional, the cache/API contract must store per-constraint binding-day counts.
If the intended feature is binary, the document should be rewritten to say that directly.

### 2. High: the leakage boundary is not proven, but the plan defaults to the aggressive side

The plan correctly identifies the risk:

- DA shadow prices for day `D` may only be available on `D+1`
- `bid_start - 2 days` is the conservative cutoff

But then it chooses `bid_start - 1 day` as the default without proving the publication timestamp relative to bid-open time.

For a temporal-causality feature, the default should remain conservative until the publication timing is verified.
Otherwise the plan risks introducing exactly the kind of small, hard-to-detect leakage that inflated earlier stage results.

### 3. Medium: training and inference use inconsistent feature semantics

The plan's overall framing is "exact partial month up to the true bid cutoff".

But for training months it chooses:

- fixed first-10-days window

while also proposing an exact-window experiment variant.

That creates a feature-definition mismatch unless both training and inference use the same rule within a given experiment.

Valid pairings would be:

- fixed-window train + fixed-window inference
- exact-window train + exact-window inference

The current draft mixes the two concepts too loosely.

### 4. Medium: the cache/API contract is incomplete

The proposed code reads:

```python
n_partial_days = partial_days.get(cutoff, 0)
```

but `partial_days` is never defined or persisted anywhere in the plan.

Related issue:

- cache naming like `2026-02_partial10.parquet` can represent only the fixed-window variant
- it cannot represent exact bid-window cutoffs, which are auction-month-specific and potentially round-specific

The storage contract needs to be specified before implementation:

- what is cached
- how exact-window caches are keyed
- whether the cache stores:
  - binding-day counts per constraint
  - total observed days
  - exact cutoff metadata

## Recommended fixes

1. Decide whether `bf_partial` is:
   - a fractional daily-binding frequency feature
   - or a binary "bound at least once in partial window" flag

2. Keep the default cutoff conservative until publication timing is verified:
   - default to `bid_start - 2 days`
   - expose tighter cutoffs only as sensitivity tests

3. Split experiments cleanly:
   - `v11a`: fixed-window train + fixed-window inference
   - `v11b`: exact-window train + exact-window inference

4. Rewrite the cache contract before coding:
   - define file keys
   - define stored columns
   - define denominator handling

## Bottom line

The idea is reasonable, but the current draft is not implementation-safe yet.

The two issues that should be fixed first are:

1. definition mismatch: fractional feature in the doc vs binary feature in the implementation sketch
2. unresolved leakage boundary: conservative cutoff identified, but not used
