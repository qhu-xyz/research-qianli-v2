# PJM V7.0 Data Gap Status

Date: 2026-03-10

## 1. wkndonpeak ml_pred — FILLED

Generated 614 `final_results.parquet` files for `class_type=wkndonpeak` using the existing shadow price prediction pipeline (`scripts/generate_wkndonpeak_mlpred.py`).

**Path**: `/opt/temp/tmp/pw_data/spice6/prod_f0p_model_pjm/ml_pred/auction_month={A}/market_month={M}/class_type=wkndonpeak/`

**Coverage** (all three class types now identical):

| class_type | files | auction months | range |
|------------|------:|:--------------:|-------|
| onpeak | 614 | 92 | 2018-06 to 2026-01 |
| dailyoffpeak | 614 | 92 | 2018-06 to 2026-01 |
| wkndonpeak | 614 | 92 | 2018-06 to 2026-01 |

**Schema**: Identical across all three class types. Columns: `predicted_shadow_price`, `binding_probability`, `binding_probability_scaled`, `prob_exceed_{80..110}`, `prob_below_{90,95}`, `density_skewness`, `hist_da`, `actual_shadow_price`, `actual_binding`, `error`, `abs_error`, `branch_name`, `model_used`, `threshold`, `predicted_binding`, `predicted_binding_count`, `auction_month`, `market_month`.

**Row counts**: Match across class types for the same (auction_month, market_month) pair (e.g., 12,726 rows for 2025-06/f0).

**Note for V7.0 agent**: The safe feature columns from ml_pred are `predicted_shadow_price`, `binding_probability`, `binding_probability_scaled`, `prob_exceed_*`, and `hist_da`. Do NOT use `actual_shadow_price`, `actual_binding`, `error`, or `abs_error` — these are derived from realized data.

## 2. constraint_info — NO GAP

`constraint_info` is stored only under `class_type=onpeak` by design. The data contains physical transmission topology (branch names, kV ratings, circuit info, device names) that is inherently class-type-invariant. The PJM iso_configs.py template hardcodes `class_type=onpeak` in the path.

The V7.0 agent can safely join constraint_info from `class_type=onpeak` into all three class types.

## 3. Temporal coverage — NO GAP

ml_pred covers 2018-06 to 2026-01 (92 auction months) for all three class types. V6.2B baseline covers 2017-06 to 2026-03 (105 months).

The 12-month gap at the start (2017-06 to 2018-05) is by design — the pipeline needs training lookback data. The 2-month gap at the end (2026-02/03) applies equally to all class types and is due to the pipeline run date. If future months are needed, all three class types would be extended together using the same script.

## What remains missing

- **ml_pred for 2026-02 and 2026-03**: V6.2B data exists for 2026-03 (f0/f1/f2) but ml_pred has not been generated for any class type. Density data exists for both months. A future run of the pipeline could extend coverage.
- **`constraint_limit` column**: Not present in ml_pred output (it's a V6.2B/signal column, not a pipeline output). The V7.0 agent should source `constraint_limit` from V6.2B or constraint_info, not from ml_pred.

## Generation method

Script: `scripts/generate_wkndonpeak_mlpred.py`
- Uses `@ray.remote` tasks with `ray_map_bounded(max_concurrent=12)` to run on the Ray cluster
- Each auction month is one task; results are written to disk on the worker, only status strings returned to head
- Head node memory: ~250 MB (vs ~102 GB with the old `parallel_equal_pool` approach)
- Total runtime: ~45 minutes for 614 files
