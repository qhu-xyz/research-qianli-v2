# Plan: Partial-Month Binding Frequency for MISO Monthly Signal

## Problem

For auction month M, bid submission is ~mid M-1 (typically 10th-19th).
Current BF_LAG=1 cuts off at `months < M-1`, using only complete months through M-2.
This discards **8-18 days of M-1 DA data** that is already realized and available at decision time.

Example: auction month 2026-03, bid date Feb 11.
- Current: BF uses realized DA through end of January (months < 2026-02)
- Proposed: BF uses realized DA through Feb 9 (day before bid submission)

## Hypothesis

The partial month M-1 is the **most recent** binding signal available. Recency matters for BF
(bf_1 >> bf_12 in feature importance). Adding ~12 days of the freshest data should improve
short-window features (bf_1, bf_3) the most.

## Design

### New feature: `bf_partial` (single fractional feature)

Rather than modifying the existing bf_1..bf_15 windows (which use whole months), add ONE new
feature that captures the partial-month signal:

```
bf_partial = (# days constraint bound in partial window) / (# days in partial window)
```

This is a **daily-resolution fractional** binding frequency for the partial month only.
A constraint that bound 3 out of 8 available days gets bf_partial = 0.375 — NOT binary 1.0.

**Why one feature, not modify bf_1?**
- bf_1 is "fraction of last 1 complete month that bound" — clear semantic meaning
- Mixing partial months into window-based BF changes the denominator unpredictably
- A separate feature lets the model learn the partial-month signal independently
- If it helps, great. If not, the model can ignore it (importance -> 0)

### Leakage analysis

- Bid submission opens on `bid_start`. We use DA data through `bid_start - 2 days`.
- DA shadow prices for day D are published by MISO on day D+1 (next-day publication).
- **Default cutoff: `bid_start - 2 days`** (conservative). This guarantees the data is
  published and available before any bid activity begins.
- Sensitivity flag `--partial-lag-days N` to test tighter cutoffs (N=1) after verifying
  MISO publication timing relative to bid-open time.

### Training/inference consistency

Both training and inference MUST use the same partial-window rule within an experiment.
Two experiment variants, each internally consistent:

- **v11a (fixed-window)**: Both train and inference use first 8 days of M-1 as partial window.
  Fixed 8 days = conservative lower bound of observed bid dates (10th minus 2 day lag).
  No API dependency. Simple. Internally consistent.

- **v11b (exact-window)**: Both train and inference use `get_bidding_window()` per month to
  compute exact cutoff date, then subtract 2 days. Requires Ray for API access.
  More accurate but heavier. Internally consistent.

### Cache contract

#### Fixed-window cache (v11a)

File key: `{month}_partial_d{n_days}_{peak_type}.parquet`
Example: `2026-02_partial_d8_onpeak.parquet`

Stored columns:
| Column | Type | Description |
|--------|------|-------------|
| constraint_id | String | MISO constraint identifier |
| days_binding | Int32 | Number of days with abs(shadow_price) > 0 |
| days_total | Int32 | Total days in partial window (= n_days) |

Compute bf_partial = days_binding / days_total at feature-build time (not in cache).
This allows recomputing with different denominators without re-fetching.

#### Exact-window cache (v11b)

File key: `{month}_partial_exact_{cutoff_date}_{peak_type}.parquet`
Example: `2026-02_partial_exact_2026-02-09_onpeak.parquet`

Same columns plus metadata:
| Column | Type | Description |
|--------|------|-------------|
| constraint_id | String | MISO constraint identifier |
| days_binding | Int32 | Days with binding |
| days_total | Int32 | Actual days in window (1st of month to cutoff) |

Cutoff metadata stored in a sidecar `{month}_partial_exact_meta.json`:
```json
{"auction_month": "2026-03", "bid_start": "2026-02-11", "cutoff_date": "2026-02-09",
 "days_total": 9, "peak_type": "onpeak"}
```

### Data flow

```
For fixed-window (v11a):
  partial_month = prev_month(cutoff_month)   # cutoff_month = M-1 for lag=1
  partial_start = first day of partial_month
  partial_end = partial_start + n_days

For exact-window (v11b):
  get_bidding_window(auction_month, round=1, "monthly")
    -> (bid_start, bid_end)
    -> cutoff_date = bid_start - 2 days
    -> partial_month = month containing cutoff_date
    -> partial_start = first day of partial_month
    -> partial_end = cutoff_date + 1 day (exclusive)

get_da_shadow_by_peaktype(st=partial_start, et_ex=partial_end, peak_type)
  -> daily DA shadow prices
  -> per constraint_id: count distinct days with |shadow_price| > 0
  -> cache (constraint_id, days_binding, days_total)
  -> bf_partial = days_binding / days_total
```

### Implementation steps

#### Step 1: Partial-month DA fetcher

New function in `ml/realized_da.py`:

```python
def fetch_partial_month_da(
    month: str,          # e.g. "2026-02" (the partial month)
    n_days: int = 8,     # how many days from start of month
    peak_type: str = "onpeak",
    cache_dir: str = REALIZED_DA_CACHE,
) -> pl.DataFrame:
    """Fetch per-constraint binding day counts for first n_days of a month.

    Returns DataFrame with columns:
      - constraint_id (String)
      - days_binding (Int32): days with |shadow_price| > 0
      - days_total (Int32): n_days

    Caches as {month}_partial_d{n_days}_{peak_type}.parquet.
    For onpeak, uses legacy naming: {month}_partial_d{n_days}.parquet.
    """
```

#### Step 2: Build partial binding data (load once)

New function in experiment script:

```python
def load_all_partial_binding(
    n_days: int = 8,
    peak_type: str = "onpeak",
    cache_dir: str = REALIZED_DA_CACHE,
) -> dict[str, tuple[dict[str, int], int]]:
    """Load partial-month binding data.

    Returns {month: ({constraint_id: days_binding}, days_total)}.
    """
```

#### Step 3: Compute bf_partial in enrich_df

```python
def compute_bf_partial(
    cids: list[str],
    month: str,             # the partial month (= cutoff month for lag=1)
    partial_data: dict,     # from load_all_partial_binding
) -> np.ndarray:
    """Compute fractional partial-month BF."""
    entry = partial_data.get(month)
    if entry is None:
        return np.full(len(cids), np.nan)  # NaN = no partial data available
    binding_counts, days_total = entry
    if days_total == 0:
        return np.zeros(len(cids))
    return np.array([binding_counts.get(cid, 0) / days_total for cid in cids])
```

#### Step 4: New experiment script

`scripts/run_v11_partial_bf.py`:
- v11a: v10e features + bf_partial (10 features), fixed 8-day window, both train+inference
- v11b: v10e features + bf_partial (10 features), exact bid-window, both train+inference
- Compare against v10e-lag1 baseline

Feature list:
```python
V11_FEATURES = V10E_FEATURES + ["bf_partial"]  # 10 features
V11_MONOTONE = V10E_MONOTONE + [1]             # higher partial BF = more binding
```

#### Step 5: Evaluation

- Screen on 4 months first (~30s)
- If VC@20 or Spearman improves, run full 36-month dev
- Then holdout
- Report all 12 metrics + tail risk (bottom_2_mean)

## Expected outcome

- bf_partial captures "did this constraint bind recently (last ~8 days)" at daily granularity
- Most informative for volatile binders (bind some months, not others)
- Less informative for always/never-binding constraints (bf_12 already captures those)
- Expected improvement: +1-3% VC@20 if partial month carries signal beyond bf_1

## Risk

- Low risk: one additional feature. If useless, LightGBM gives it 0 importance.
- No leakage risk: conservative cutoff (bid_start - 2 days) is default.
- Cache bloat: ~107 extra partial parquet files, each tiny (~10KB). Negligible.

## Follow-up: daily BF windows (v12)

If bf_partial shows the partial-month signal is real, expand to daily-resolution BF:
- `bf_7d`: fraction of last 7 calendar days binding
- `bf_14d`: fraction of last 14 days binding
- `bf_30d`: fraction of last 30 days binding

This requires daily DA caching (not just partial-month). Separate plan if v11 is promising.

## Sequence

1. Implement partial-month DA fetcher + cache (Step 1)
2. Build partial binding loader (Step 2)
3. Add bf_partial computation (Step 3)
4. Write experiment script (Step 4)
5. Screen run (4 months, ~30s)
6. If promising -> full dev (36 months) + holdout (24 months)
7. If bf_partial helps -> plan v12 daily BF windows
