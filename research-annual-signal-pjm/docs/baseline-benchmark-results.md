# PJM Annual Signal — Baseline Benchmark Results (Corrected Again)

**Date**: 2026-03-25
**Script**: `scripts/sweep_all_ctypes.py`
**Benchmark**: `TEST.Signal.PJM.SPICE_ANNUAL_V4.6.R{round}`

**Benchmark baseline**
- `V4.6` is the released annual signal and the only true baseline to beat

**Internal challengers**
- `baseline_v69`: current best 2-component reproducible challenger = `0.3 * density_rank + 0.7 * da_rank`
- `candidate_v70`: 3-component experimental mimic = `0.3 * flow_stress_max + 0.2 * flow_stress_persistence + 0.5 * da_pressure`

**Important correction**
- Previous "we lose badly to V4.6" results were invalid.
- Root bug: the DA lookback feature was mapped with `map_da_quarterly()`, which filters rows to the evaluation PY's settlement quarters. A June-to-March historical window has no overlap with those months, so the DA feature was effectively zero.
- Current results use:
  - DA feature: class-type-filtered, June `(PY-2)` through March `PY`, mapped through the evaluation cell's monitored-line union lookup
  - GT: class-type-filtered settlement DA, mapped quarter-by-quarter and summed to annual
  - NERC holiday-aware ctype filtering for the `onpeak` / `wkndonpeak` boundary, matching `pbase.utils.hours.is_off_peak_day()`

---

## 1. Headline Summary

### `baseline_v69` vs V4.6

| Class type | @200 avg Δ | @200 wins | @400 avg Δ | @400 wins |
|-----------|-----------:|----------:|-----------:|----------:|
| onpeak | **+4.8pp** | 27/27 | **+3.5pp** | 23/27 |
| dailyoffpeak | **+1.9pp** | 19/27 | **-1.2pp** | 21/27 |
| wkndonpeak | **+2.6pp** | 23/27 | **-1.2pp** | 14/27 |

### `candidate_v70` vs V4.6

| Class type | @200 avg Δ | @200 wins | @400 avg Δ | @400 wins |
|-----------|-----------:|----------:|-----------:|----------:|
| onpeak | **+4.1pp** | 26/27 | **+3.4pp** | 23/27 |
| dailyoffpeak | **-0.6pp** | 13/27 | **-2.0pp** | 16/27 |
| wkndonpeak | **+1.5pp** | 19/27 | **-1.8pp** | 13/27 |

### `candidate_v70` vs `baseline_v69`

| Class type | @200 avg improvement | Wins | @400 avg improvement | Wins |
|-----------|----------------------:|-----:|----------------------:|-----:|
| onpeak | -0.5pp | 6/28 | -0.1pp | 9/28 |
| dailyoffpeak | -2.4pp | 4/28 | -0.9pp | 6/28 |
| wkndonpeak | -1.1pp | 4/28 | -0.6pp | 10/28 |

Current conclusion:
- `V4.6` remains the release baseline
- `baseline_v69` is the current best reproducible challenger
- `candidate_v70` does not improve on `baseline_v69`

---

## 2. Coverage Context

| Class type | GT mapping | Our GT branch recall | V4.6 GT branch recall | Avg GT branches |
|-----------|-----------:|---------------------:|----------------------:|----------------:|
| onpeak | 87.2% | 97.1% | 56.1% | 603 |
| dailyoffpeak | 83.1% | 96.9% | 63.2% | 343 |
| wkndonpeak | 84.8% | 96.7% | 63.0% | 330 |

Notes:
- `GT mapping`: mapped annual DA `|SP| / total DA |SP|`
- `GT branch recall`: fraction of GT branches present in the signal universe
- V4.6 still covers fewer GT branches, but that no longer implies better overall capture once the DA feature bug is fixed

---

## 3. What Changed

The major implementation fix was:

1. Historical DA feature windows are not settlement-quarter slices.
2. Therefore they must not be passed through `map_da_quarterly()`.
3. They must be mapped through the evaluation cell's monitored-line lookup directly.

That change makes the benchmark actually use the intended density+DA blend. Before that fix, the script was effectively evaluating density-only signals while reporting them as density+DA.

---

## 4. Interpretation

### Why `baseline_v69` is stronger than `candidate_v70`

`candidate_v70` was designed to mimic V4.6's published weighting:
- max-style flow stress
- persistence-style flow stress
- DA pressure

But with our reproducible inputs, that decomposition does not help. The likely reason is that the extra density split is too noisy:
- `flow_stress_max` and `flow_stress_persistence` both come from the same density parquet
- they do not reproduce the binary exceedance behavior of `flow_memo` / `flow_onpeak`
- splitting one weak proxy into two components adds variance without adding much new signal

The simpler `baseline_v69` appears to regularize better:
- one density proxy
- one DA-history proxy
- heavier weight on DA

### What this means for release prep

For the initial reproducible release candidate, the leading internal challenger is:
- `baseline_v69` for onpeak
- `baseline_v69` also for dailyoffpeak and wkndonpeak on average, despite mixed `@400` behavior

So the current release path should optimize around the fixed 2-component challenger first, not the 3-component hand-weighted mimic.

---

## 5. Residual Limitations

1. Density remains non-class-type-specific at the raw parquet level; ctype specificity enters through bridge mapping and DA hours.
2. 2025-06 remains an incomplete holdout year and `R1` has no V4.6 benchmark cell.
3. `candidate_v70` has only been tested as a fixed-weight formula. A learned model may still beat both `v69` and V4.6.

---

## 6. Next Step

Do not treat `candidate_v70` as the release candidate.

Next work should be:
1. freeze `baseline_v69` as the current reproducible challenger to `V4.6`
2. move to learned branch-level models using density bins + DA history instead of hand-weighted density decompositions
