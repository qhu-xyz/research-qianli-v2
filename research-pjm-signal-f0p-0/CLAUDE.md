# CLAUDE.md — research-pjm-signal-f0p-0 (PJM V7.0)

Inherits rules from parent: `/home/xyz/workspace/research-qianli-v2/CLAUDE.md`

## What This Project Does

Build a PJM V7.0b constraint-tier signal: ML-scored tiers for f0/f1, V6.2B passthrough for f2-f11. Modeled on MISO V7.0 (`research-stage5-tier/`).

## Reference Projects

| Project | Path | What to use |
|---------|------|-------------|
| MISO V7.0 ML pipeline | `research-stage5-tier/ml/` | 14 Python modules — adapt for PJM |
| MISO V7.0 scripts | `research-stage5-tier/scripts/` | run_v0, run_v10e, blend search |
| MISO V7.0 deployment | `research-miso-signal7/v70/` | inference.py, signal_writer.py |
| PJM shadow price pred | `research-spice-shadow-price-pred/` | data_loader.py (constraint mapping), iso_configs.py |

## Production Timing Lag (CRITICAL — DATA LEAKAGE TRAP)

Same rule as MISO. For f0, signal for month M is submitted ~mid(M-1). Latest complete DA is M-2.

General lag rule: for period type fN, **lag = N + 1**.

| Period | lag | Training window | binding_freq cutoff |
|--------|:---:|-----------------|---------------------|
| f0 | 1 | M-9..M-2 | months < M-1 |
| f1 | 2 | M-10..M-3 | months < M-2 |

Without lag, results are inflated 6-20%. See `research-stage5-tier/registry/f0/onpeak/v10e-lag1/NOTES.md`.

## PJM-Specific: Constraint Mapping (CRITICAL)

**Use the branch-level join, NOT the naive monitored_facility join.**

- Naive join (`constraint_id.split(":")[0]` → DA `monitored_facility`): captures only ~46% of DA value
- Branch-level join (via `constraint_info` + `map_constraints_to_branches()`): captures 96-99%

Reference implementation: `research-spice-shadow-price-pred/src/shadow_price_prediction/data_loader.py:805`

Key steps:
1. `constraint_id.split(":")[0]` → `monitored_facility` → `.upper()` → `match_str`
2. Direct match: DA `monitored_facility` (uppercased) → `match_str` → `branch_name`
3. Interface fallback: prefix-match for interface contingencies (e.g., "BED-BLA CONTINGENCY 24" → "BED-BLA")
4. Aggregate realized_sp by `branch_name`, join back to all V6.2B rows sharing that branch

pbase's `PjmTools.get_da_sp_by_equipment()` does a simpler version (no case normalization, no interface matching) — do NOT use it for ML target construction.

See `human-input/data-gap-audit.md` §7 for full coverage numbers.

## PJM-Specific: Timezone

PJM uses **US/Eastern** (not US/Central like MISO). All `pd.Timestamp` calls for DA shadow price fetching must use `tz="US/Eastern"`.

## PJM-Specific: Import Path

```python
from pbase.analysis.tools.all_positions import PjmApTools
aptools = PjmApTools()
da = aptools.tools.get_da_shadow_by_peaktype(st=start, et_ex=end, peak_type="onpeak")
```

NOT `from pbase.data.pjm.ap_tools` (does not exist).

## PJM-Specific: Class Types and Period Types

| Dimension | MISO | PJM |
|-----------|------|-----|
| Class types | onpeak, offpeak (2) | onpeak, dailyoffpeak, wkndonpeak (3) |
| ML slices | f0×2 + f1×2 = 4 | f0×3 + f1×3 = **6** |
| Passthrough | f2-f3 | f2-f11 (up to 10) |
| Period schedule | f0-f3 | f0-f11, varies by month (May: f0 only, June: all 12) |

## PJM-Specific: Data Paths

| Data | Path |
|------|------|
| V6.2B signal | `/opt/data/xyz-dataset/signal_data/pjm/constraints/TEST.TEST.Signal.PJM.SPICE_F0P_V6.2B.R1/` |
| SF | `/opt/data/xyz-dataset/signal_data/pjm/sf/TEST.TEST.Signal.PJM.SPICE_F0P_V6.2B.R1/` |
| Spice6 base | `/opt/temp/tmp/pw_data/spice6/prod_f0p_model_pjm/` |
| ml_pred | `{spice6}/ml_pred/auction_month={A}/market_month={M}/class_type={C}/final_results.parquet` |
| Density | `{spice6}/density/auction_month={A}/...` |
| constraint_info | `{spice6}/constraint_info/auction_month={A}/market_round=1/period_type={P}/class_type=onpeak/` |

constraint_info is stored only under `class_type=onpeak` — this is by design (physical topology, class-invariant).

## PJM-Specific: ml_pred Coverage

All three class types: 614 files each, 92 auction months (2018-06 to 2026-01).

**Safe feature columns**: `predicted_shadow_price`, `binding_probability`, `binding_probability_scaled`, `prob_exceed_{80..110}`, `density_skewness`, `hist_da`.

**DO NOT USE**: `actual_shadow_price`, `actual_binding`, `error`, `abs_error` — these are derived from realized data.

For months before 2018-06: fill spice6 features with 0 (pipeline needs training lookback).

## Registry Structure (MANDATORY)

```
registry/{period_type}/{class_type}/{version_id}/metrics.json
holdout/{period_type}/{class_type}/{version_id}/metrics.json
```

6 ML slices: f0/{onpeak,dailyoffpeak,wkndonpeak}, f1/{onpeak,dailyoffpeak,wkndonpeak}.
Each slice has own `gates.json` and `champion.json`. Use `ml.registry_paths` — never hardcode.

## LightGBM Threading (CRITICAL)

Always set `"num_threads": 4` in LightGBM params. Container has 64 CPUs but LightGBM deadlocks using all of them. Also use `mp_context=multiprocessing.get_context("spawn")` if using ProcessPoolExecutor.

## V6.2B Formula (Verified Exact)

```
rank_ori = 0.60 * da_rank_value + 0.30 * density_mix_rank_value + 0.10 * density_ori_rank_value
```

Same as MISO. Verified with max_abs_diff = 0.0.

## Virtual Environment

```bash
cd /home/xyz/workspace/pmodel && source .venv/bin/activate
```

Run scripts from this directory (`research-pjm-signal-f0p-0/`), not from pmodel.
