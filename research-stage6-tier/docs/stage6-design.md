# Stage 6 Design: Partial-Month Feature Recovery

## Problem

Stage 5's v10e-lag1 fixed temporal leakage by dropping all of month M-1.
But ~12 days of M-1 data IS available at f0 submission time (~mid of M-1).
v10e-lag1 is safe but conservative — it leaves useful partial information on the table.

Stage 6 adds that partial information back safely as **features only** (not as a
training row).

## Key Design Point

- **Partial M-1 comes back as feature information**
- **Partial M-1 does NOT come back as a labeled training row**

For target 2025-04 / f0 with look_back_days=12:
- Training rows: 2024-07 .. 2025-02 (same safe boundary as lag1)
- No 2025-03 label row used for training
- binding_freq_1 uses 2025-03-01 .. 2025-03-12 (partial)
- binding_freq_3 uses: partial 2025-03 + full 2025-02 + full 2025-01
- Older months remain full-month aggregates

## Architecture

### Daily Realized-DA Cache

Monthly cache is insufficient for partial-month recovery. Stage 6 adds a daily
cache preserving signed netting:

```
data/realized_da_daily/{YYYY-MM}.parquet
  market_date: Date
  constraint_id: String
  shadow_price_net_day: Float64  # signed daily net
  realized_sp_day: Float64       # abs of daily net
```

Monthly semantics: `realized_sp_month = abs(sum(shadow_price_net_day))`.
Caching only daily abs values would destroy cross-day cancellation and break
the 31-day equivalence check.

### Lookback History Module

`ml/lookback_history.py` computes binding_freq_* with partial-month awareness:
- All months older than M-1: full-month cache (stage 5 semantics)
- Month M-1 only: truncated to look_back_days via daily cache
- `add_binding_freq_columns_asof(df, target_month, look_back_days=12)`

### Experiment Script

`scripts/run_v10e_lookback_days.py` — same v10e model architecture with:
- Safe M-2 row boundary (from lag1)
- Partial M-1 feature rebuild (new in stage 6)
- Supports `--look-back-days N` and `--mode dev|holdout`

## Workflow

### 1. Build daily cache (requires Ray)

```bash
python scripts/cache_realized_da_daily.py --start 2020-06 --end 2025-12
```

### 2. Verify 31-day equivalence

```bash
python scripts/verify_lookback_days_31.py --start 2020-06 --end 2025-12 --look-back-days 31
```

This proves the rebuilt partial-month path reproduces old full-month binding sets
when the cutoff reaches month end. It does NOT prove the full stage 6 dataset
equals stage 5 (the safe row boundary is still in effect).

### 3. Run dev experiment

```bash
python scripts/run_v10e_lookback_days.py --look-back-days 12 --mode dev
```

Result: `registry/v10e-lookback12/metrics.json`

### 4. Run holdout experiment

```bash
python scripts/run_v10e_lookback_days.py --look-back-days 12 --mode holdout
```

Result: `holdout/v10e-lookback12/metrics.json`

### 5. Compare against v10e-lag1

The control is v10e-lag1, NOT the leaky v10e.

## Known Limitations

1. **look_back_days=31 is not a full-dataset equivalence claim.** It only
   verifies the rebuilt binding_freq_* family. Stage 6 still differs from the
   old monthly pipeline because the row boundary remains safely lagged.

2. **Daily abs is wrong.** If the cache stores only abs(daily_shadow),
   month-end reconstruction fails. Correct: `abs(sum(daily_net_shadow))`.

3. **Calendar-day cutoff may be slightly optimistic.** If production submission
   happens before end of day 12, using entire day 12 is slightly leaky. Long-term
   fix: replace look_back_days with a true cutoff_ts.

4. **look_back_days=12 is f0-specific.** Not automatically correct for f1, f2,
   or quarterlies.

## Future: As-Of Cutoff Framework

The current look_back_days scalar is a first step. The full design calls for:

- `AuctionCutoffCalendar` with per-(auction_month, period_type) cutoff_ts
- Separate full-history + partial-month features (not blended)
- `load_train_val_test_asof()` that enriches each row with as-of features
- Extension to f1/f2/quarterly via delivery-month metadata

See `docs/multi-period-extension.md` for the period-type extension plan.

## Acceptance Criteria

Stage 6 is working when:
- verify_lookback_days_31.py passes for the available monthly range
- run_v10e_lookback_days.py --look-back-days 12 completes
- Results beat v10e-lag1 or behave plausibly without reproducing the leaky v10e lift
- Inspection of one concrete month confirms no M-1 training row leak and partial
  M-1 feature usage only through the configured cutoff
