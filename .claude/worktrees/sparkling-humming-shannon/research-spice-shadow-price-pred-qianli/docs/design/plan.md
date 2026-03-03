# Plan: Generate MISO SPICE_F0P_V6.7B.R1 Signal for All Period Types & Class Types

## Background

The 6.7B signal (`TEST.TEST.Signal.MISO.SPICE_F0P_V6.7B.R1`) currently exists but only covers `f0/onpeak` for 20 months (2024-06 to 2026-01). The task is to extend it to all period types and class types so it can be used in the CIA Multistep trading workflow alongside the existing 6.2B signal.

## Current State

- **Existing 6.7B coverage**: `f0/onpeak` only (20 auction months: 2024-06 to 2026-01)
- **6.2B coverage** (reference): `f0, f1, q4` × `onpeak` (varies by auction month)
- **ML pipeline**: Produces `final_results.parquet` with columns: `constraint_id`, `flow_direction`, `branch_name`, `hist_da`, `predicted_shadow_price`, `binding_probability_scaled`, etc.

## Signal Format (from existing 6.7B data)

- **Index**: `{constraint_id}|{-flow_direction}|spice` (string)
- **Columns**: `constraint_id`, `branch_name`, `flow_direction`, `hist_da`, `predicted_shadow_price`, `binding_probability_scaled`, `prob_rank`, `hist_shadow_rank`, `pred_shadow_rank`, `rank`, `tier`, `shadow_sign`, `shadow_price`, `equipment`
- **Tier assignment**: 5 quantile bins (0-4) based on composite `rank`
- **Rank formula**: `percentile_rank(0.4 * prob_rank + 0.3 * hist_shadow_rank + 0.3 * pred_shadow_rank)`
- **Derived columns**:
  - `shadow_sign = -flow_direction`
  - `shadow_price = predicted_shadow_price * shadow_sign`
  - `equipment = branch_name`
  - `prob_rank = percentile_rank(binding_probability_scaled, descending)`
  - `pred_shadow_rank = percentile_rank(predicted_shadow_price, descending)`
  - `hist_shadow_rank = percentile_rank(hist_da, descending)`

## Target Coverage

- **Period types**: All tradable period types for each auction month (obtained via `aptools.tools.get_tradable_period_types(auction_month_str)`) — typically `f0, f1, f2, f3, q2, q3, q4` for monthly auctions
- **Class types**: `onpeak`, `offpeak`
- **Auction months**: 2024-06 to 2026-01 (matching existing f0/onpeak range)
- **Signal name**: `TEST.TEST.Signal.MISO.SPICE_F0P_V6.7B.R1`

## Implementation Steps

### Step 1: Create signal generation utility module

Create `src/shadow_price_prediction/signal_generator.py` with:

- `convert_predictions_to_signal(final_results: pd.DataFrame) -> pd.DataFrame`:
  - Takes ML pipeline `final_results` DataFrame (indexed by `(constraint_id, flow_direction)`)
  - Computes sub-ranks, composite rank, tiers
  - Adds `shadow_sign`, `shadow_price`, `equipment` columns
  - Sets index to `{constraint_id}|{-flow_direction}|spice`
  - Returns signal DataFrame ready for `ConstraintsSignal.save_data()`

- `save_signal(signal_df, rto, signal_name, period_type, class_type, auction_month)`:
  - Wraps `ConstraintsSignal.save_data()` call

### Step 2: Create signal generation notebook

Create `notebook/generate_signal_67b.ipynb` with:

1. **Init Ray** (required for pbase data loaders)
2. **Configure** parameters: auction month range, period types, class types, signal name
3. **Loop** over (auction_month, period_type, class_type):
   a. Determine market months for this (auction_month, period_type) via `aptools.tools.get_market_month_from_auction_month_and_period_trades()`
   b. Run ML pipeline for each market month, producing `final_results.parquet`
   c. For quarterly/annual: aggregate predictions across market months
   d. Convert to signal format using `convert_predictions_to_signal()`
   e. Save using `ConstraintsSignal.save_data()`
4. **Verify** saved signals can be loaded back

### Step 3: Validate

- Load a sample of generated signals and compare format with existing 6.7B f0/onpeak data
- Verify all required (auction_month, period_type, class_type) combinations have data
- Check signal quality: tier distribution, rank ranges, shadow_price stats

## Key Design Decisions

1. **Tier count**: 5 tiers (0-4), matching existing 6.7B signal (note: 6.2B also uses 5 tiers)
2. **Rank weights**: `0.4 * prob_rank + 0.3 * hist_shadow_rank + 0.3 * pred_shadow_rank` (reverse-engineered from existing data)
3. **Index suffix**: `|spice` (matching existing SPICE signal convention)
4. **Signal name**: `TEST.TEST.Signal.MISO.SPICE_F0P_V6.7B.R1` (same as existing, just adding more period_type/class_type data)

## Files to Create/Modify

1. **New**: `src/shadow_price_prediction/signal_generator.py` — Signal conversion utility
2. **New**: `notebook/generate_signal_67b.ipynb` — Signal generation notebook
3. **Modify**: `src/shadow_price_prediction/__init__.py` — Export new module
