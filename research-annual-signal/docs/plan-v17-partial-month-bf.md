# V17 Plan: Partial-Month Binding Frequency for Annual Signal

## Background

### Problem statement

MISO annual FTR round 1 bids are submitted around **April 10th** each year. The current
binding frequency (BF) features use a hard cutoff of `months < YYYY-04`, meaning they
only include realized DA data through **end of March**. This discards ~8-9 days of April
DA data that is already realized and available at decision time.

For the v16 champion (holdout VC@20 = 0.3920), BF features account for ~60% of total
feature importance (bf_15 = 12%, bf_6 = 8%, bfo_12 = 29%, bfo_6 = 7%, bf_12 = 3%).
Adding the most recent ~8 days of binding data could sharpen these signals.

### Origin: monthly signal (stage5-tier) v11a experiment

This idea was first tested in `research-stage5-tier` as v11a. The monthly pipeline added
a `bf_partial` feature capturing the first 8 days of the partial month before bid
submission. Results across 4 slices:

| Slice | Baseline HO VC@20 | v11a HO VC@20 | Delta |
|-------|:--:|:--:|:--:|
| f0/onpeak (24mo) | 0.3529 | **0.3564** | +1.0% |
| f0/offpeak (24mo) | 0.3780 | **0.3794** | +0.4% |
| f1/onpeak (20mo) | 0.3677 | 0.3401 | -7.5% |
| f1/offpeak (20mo) | 0.3561 | 0.3523 | -1.1% |

Key findings:
- **f0 (front-month): consistent small improvement** — the partial month is close to
  delivery, so recent binding data is informative.
- **f1 (month+1): regresses** — the partial month is too far from f1 delivery to help;
  the extra feature adds noise.
- **bf_partial feature importance**: 3-12.6% on dev depending on slice. Strongest for
  f0/offpeak (12.6% on dev, 4.9% on holdout).

### Why annual should benefit more than monthly

1. **Annual BF windows are coarser**: bf_6 for annual covers Oct-Mar (6 months). Adding
   8 days of April is a ~4.5% data increase. For monthly bf_1 it's already 1 full month,
   so 8 extra days is proportionally less novel.

2. **Annual has a FIXED partial month**: always April for round 1. Monthly varies each
   month. Fixed = simpler implementation, no per-month bidding window lookup needed.

3. **Annual champion relies heavily on BF**: 60%+ of feature importance is BF-derived.
   Monthly v10e has more diverse features (formula score, spice6, etc.) at ~55% BF.

4. **Recency signal matters for annual**: the April partial captures the MOST RECENT
   binding behavior before a 12-month forward commitment. A constraint that bound 5 out
   of 8 days in early April is a strong signal.

## Current annual BF implementation

File: `ml/binding_freq.py`

### Cutoff logic (lines 270-271)
```python
py = int(auction_month.split("-")[0])
cutoff = f"{py}-04"   # hard-coded: months < YYYY-04
```

### Data flow
```
Realized DA cache (constraint_id, realized_sp)
  -> read from stage5-tier/data/realized_da/*.parquet
  -> filter: realized_sp > 0 -> binding set per month
  -> bridge table: constraint_id -> branch_name
     (via MISO_SPICE_CONSTRAINT_INFO, partition-filtered)
  -> binding_sets: {month: set(branch_name)}
  -> compute_binding_freq(branch_names, binding_sets, cutoff, window)
     -> count months bound / window_size -> bf_{window}
```

### Key difference from monthly (stage5-tier)

| Aspect | Monthly (stage5) | Annual |
|--------|-----------------|--------|
| Constraint ID | `constraint_id` (direct) | `branch_name` (via bridge table) |
| Cutoff | `M-1` (lag=1) | `YYYY-04` (fixed April) |
| Partial month | varies per eval month | always April |
| Champion features | 10 features | 7 features (v16) |
| BF importance | ~55% | ~60% |
| Eval groups | 36 dev + 24 holdout months | 12 dev + 3-4 holdout quarters |

### Champion (v16) feature set
```python
features = [
    "shadow_price_da",   # historical DA shadow price
    "da_rank_value",     # rank of shadow_price_da (35% importance)
    "bf_6",              # 6-month onpeak BF
    "bf_12",             # 12-month onpeak BF
    "bf_15",             # 15-month onpeak BF
    "bfo_6",             # 6-month offpeak BF
    "bfo_12",            # 12-month offpeak BF (29% importance)
]
```

## Partial-month DA cache (already exists)

The stage5-tier v11a experiment already built partial-month DA caches in:
```
/home/xyz/workspace/research-qianli-v2/research-stage5-tier/data/realized_da/
```

Files matching `*_partial_d8.parquet` (onpeak) and `*_partial_d8_offpeak.parquet`:

| File | Schema |
|------|--------|
| `{YYYY-MM}_partial_d8.parquet` | constraint_id (String), days_binding (Int32), days_total (Int32) |
| `{YYYY-MM}_partial_d8_offpeak.parquet` | Same schema |

These cover 2019-04 through 2025-11 (76+ months cached). The annual signal only needs
April files: `2019-04`, `2020-04`, ..., `2025-04`.

**No new data fetching is required** — the cache is already populated for both onpeak and
offpeak.

## Design: `bf_partial` for annual

### New feature

```
bf_partial = days_binding / days_total
```
where `days_binding` = number of days (out of first 8 of April) where `|shadow_price| > 0`
for a constraint, mapped to `branch_name` via the bridge table.

### Leakage analysis

- Annual round 1 bid submission opens ~April 10th
- Partial window: April 1-8 (conservative: bid_start - 2 days)
- DA shadow prices for day D are published by MISO on day D+1
- **April 8 DA is published April 9, one day before bids open on April 10**
- No leakage risk with the default 8-day window

### Bridge table fan-out handling

A single `constraint_id` can map to multiple `branch_name`s (and vice versa). When a
branch_name maps to multiple constraint_ids, we need to decide how to aggregate:

- **Option A (recommended)**: `max(days_binding)` per branch_name — conservative, avoids
  double-counting, keeps values in [0, 1] range
- **Option B**: Binary — if ANY mapped constraint bound on day D, branch counts as binding.
  Requires daily granularity (more complex, marginal benefit)
- **Option C**: `sum(days_binding)` — can produce bf_partial > 1.0, not recommended

Use **Option A** for simplicity. If results are promising, Option B could be a follow-up.

### Training/inference consistency

The partial month is determined by planning year, not by eval quarter:
- Planning year 2024-06 -> partial month = 2024-04 (for ALL aq1-aq4)
- Planning year 2023-06 -> partial month = 2023-04

Training uses expanding window (all prior planning years). Each training year gets its
own April partial. This is internally consistent: both train and inference use the same
rule.

### Implementation steps

#### Step 1: Add `compute_partial_bf()` to `ml/binding_freq.py`

```python
def compute_partial_bf(
    branch_names: list[str],
    auction_month: str,       # e.g. "2024-06"
    period_type: str,         # e.g. "aq1"
    n_days: int = 8,
    peak_type: str = "onpeak",
) -> np.ndarray:
    """Compute partial-month BF for first n_days of April.

    For annual round 1 submitted ~April 10, captures binding data from
    April 1-8 that is available at decision time but currently discarded
    by the months < YYYY-04 cutoff.

    Uses the stage5-tier partial DA cache (constraint_id, days_binding,
    days_total) and maps through the bridge table to branch_name.

    Returns array of shape (len(branch_names),) with values in [0, 1].
    """
    py = int(auction_month.split("-")[0])
    partial_month = f"{py:04d}-04"

    suffix = "" if peak_type == "onpeak" else f"_{peak_type}"
    partial_file = REALIZED_DA_CACHE / f"{partial_month}_partial_d{n_days}{suffix}.parquet"

    if not partial_file.exists():
        return np.zeros(len(branch_names), dtype=np.float64)

    da = pl.read_parquet(str(partial_file))
    if len(da) == 0:
        return np.zeros(len(branch_names), dtype=np.float64)

    # Map constraint_id -> branch_name via bridge table
    bridge = _load_bridge(auction_month, period_type)
    mapped = da.join(bridge, on="constraint_id", how="inner")

    if len(mapped) == 0:
        return np.zeros(len(branch_names), dtype=np.float64)

    # Aggregate per branch_name: use MAX days_binding to avoid fan-out inflation
    agg = mapped.group_by("branch_name").agg([
        pl.col("days_binding").max(),
        pl.col("days_total").first(),
    ])

    binding_map = dict(zip(
        agg["branch_name"].to_list(),
        (agg["days_binding"] / agg["days_total"]).to_list(),
    ))

    return np.array(
        [binding_map.get(bn, 0.0) for bn in branch_names],
        dtype=np.float64,
    )
```

#### Step 2: Add `enrich_with_partial_bf()` wrapper

```python
def enrich_with_partial_bf(
    df: pl.DataFrame,
    auction_month: str,
    period_type: str,
    n_days: int = 8,
    include_offpeak: bool = True,
) -> pl.DataFrame:
    """Add bf_partial (and optionally bfp_offpeak) columns."""
    branch_names = df["branch_name"].to_list()

    freq = compute_partial_bf(branch_names, auction_month, period_type, n_days, "onpeak")
    df = df.with_columns(pl.Series("bf_partial", freq))

    if include_offpeak:
        freq_off = compute_partial_bf(branch_names, auction_month, period_type, n_days, "offpeak")
        df = df.with_columns(pl.Series("bfp_offpeak", freq_off))

    return df
```

#### Step 3: Experiment script `scripts/run_v17_partial_bf.py`

Compare v16 champion vs v17 (v16 + bf_partial) and v17b (v16 + bf_partial + bfp_offpeak):

```python
V16_FEATURES = [
    "shadow_price_da", "da_rank_value",
    "bf_6", "bf_12", "bf_15", "bfo_6", "bfo_12",
]
V16_MONOTONE = [1, -1, 1, 1, 1, 1, 1]

# v17: v16 + onpeak partial (8 features)
V17_FEATURES = V16_FEATURES + ["bf_partial"]
V17_MONOTONE = V16_MONOTONE + [1]

# v17b: v16 + onpeak + offpeak partial (9 features)
V17B_FEATURES = V16_FEATURES + ["bf_partial", "bfp_offpeak"]
V17B_MONOTONE = V16_MONOTONE + [1, 1]
```

Use the same hyperparams as v16 champion:
```python
LTRConfig(
    n_estimators=200,
    learning_rate=0.03,
    num_leaves=31,
    label_mode="tiered",
    backend="lightgbm",
)
```

#### Step 4: Evaluation sequence

1. **Screen**: 4 dev groups (SCREEN_EVAL_GROUPS), ~30s
2. **Full dev**: 12 groups (3 planning years x 4 quarters), ~2min
3. **Holdout**: 3-4 groups (2025-06/aq1-aq4), only if dev is positive

Report all Group A metrics: VC@20, VC@50, VC@100, Recall@20, Recall@50, NDCG, Spearman.

## Risk assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| bf_partial adds noise (like f1 monthly) | Medium | Low | LightGBM gives it 0 importance; 1 extra feature is cheap |
| Bridge table fan-out inflates values | Low | Medium | Using max() aggregation, not sum() |
| Annual has too few eval groups to detect signal | Medium | Medium | Focus on dev (12 groups); holdout (3-4) is noisy |
| Partial cache missing for some years | Low | Low | Cache verified: 2019-04 through 2025-04 exist |

## Expected outcome

- bf_partial adds the **most recent 8 days of binding data** to a model that already
  relies heavily on BF features (~60% importance)
- Expected improvement: +1-3% VC@20 on dev if partial month carries signal beyond bf_6
- Larger effect size than monthly because:
  - Annual BF windows are coarser (fewer months)
  - April partial is closer to Round 1 decision time
  - Champion is more BF-dependent (60% vs 55%)
- If bf_partial shows <1% importance on dev, consider it a null result and move on

## Follow-up ideas (if v17 is positive)

1. **v17b with offpeak partial**: add `bfp_offpeak` — offpeak partial cache already exists
2. **Exact bid-window cutoff (v17c)**: use `get_bidding_window()` from pbase to compute
   the exact submission date per year, instead of fixed 8 days. Requires:
   ```python
   from pbase.data.dataset.ftr.market.base import get_market_info
   market_info = get_market_info("MISO")
   start, end = market_info.get_bidding_window(
       auction_month=pd.Timestamp("2024-06-01"),
       market_round=1,
       auction_type="annual",
   )
   cutoff_date = start - pd.Timedelta(days=2)
   n_days = cutoff_date.day  # dynamic instead of fixed 8
   ```
3. **Daily-resolution BF (v18)**: replace monthly BF with daily BF (bf_7d, bf_14d, bf_30d).
   Requires daily DA caching. Separate plan if v17 is promising.

## Files to modify

| File | Change |
|------|--------|
| `ml/binding_freq.py` | Add `compute_partial_bf()` and `enrich_with_partial_bf()` |
| `ml/config.py` | Add V17 feature sets (optional — can define in script) |
| `scripts/run_v17_partial_bf.py` | New experiment script |

## Dependencies

- Stage5-tier partial DA cache: `research-stage5-tier/data/realized_da/*_partial_d8*.parquet`
- Bridge table: `MISO_SPICE_CONSTRAINT_INFO.parquet` (already used by v16)
- No Ray needed (cache is pre-populated)
- No new data fetching required

## Reference: stage5-tier v11a implementation

The monthly implementation is in:
- Plan: `research-stage5-tier/docs/plan-partial-month-bf.md`
- Script: `research-stage5-tier/scripts/run_v11_partial_bf.py`
- DA fetcher: `research-stage5-tier/ml/realized_da.py` (`fetch_partial_month_da()`, `load_partial_month_da()`)
- Results: `research-stage5-tier/registry/f0/onpeak/v11a/metrics.json` (dev), `holdout/f0/onpeak/v11a/metrics.json` (holdout)

### Full monthly v11a results for reference

**Dev (in-sample):**

| Slice | Baseline | v11a | Delta |
|-------|:--:|:--:|:--:|
| f0/onpeak (36mo) | 0.4137 | **0.4399** | +6.3% |
| f0/offpeak (36mo) | 0.5200 | **0.5203** | +0.1% |
| f1/onpeak (30mo) | 0.4557 | 0.4490 | -1.5% |
| f1/offpeak (30mo) | 0.4499 | **0.4713** | +4.8% |

**Holdout (out-of-sample):**

| Slice | Baseline | v11a | Delta |
|-------|:--:|:--:|:--:|
| f0/onpeak (24mo) | 0.3529 | **0.3564** | +1.0% |
| f0/offpeak (24mo) | 0.3780 | **0.3794** | +0.4% |
| f1/onpeak (20mo) | 0.3677 | 0.3401 | -7.5% |
| f1/offpeak (20mo) | 0.3561 | 0.3523 | -1.1% |

**bf_partial feature importance (dev):**

| Slice | Importance % | Rank |
|-------|:--:|:--:|
| f0/onpeak | 3.0% | #7 of 10 |
| f0/offpeak | **12.6%** | #5 of 10 |
| f1/onpeak | 3.0% | #7 of 10 |
| f1/offpeak | 4.4% | #6 of 10 |

### Conclusion from monthly

bf_partial is a **real but modest signal for f0** (front-month). It regresses for f1
because the partial month is too far from f1's delivery. Annual is analogous to f0 in
that the partial April data is temporally close to the auction decision — so the annual
experiment is worth running.
