# F0 Audit

Reviewed current `f0` implementation in:

- [ml/data_loader.py](/home/xyz/workspace/research-qianli-v2/research-stage5-tier/ml/data_loader.py)
- [ml/spice6_loader.py](/home/xyz/workspace/research-qianli-v2/research-stage5-tier/ml/spice6_loader.py)
- [ml/evaluate.py](/home/xyz/workspace/research-qianli-v2/research-stage5-tier/ml/evaluate.py)
- [ml/compare.py](/home/xyz/workspace/research-qianli-v2/research-stage5-tier/ml/compare.py)
- [scripts/run_v0_formula_baseline.py](/home/xyz/workspace/research-qianli-v2/research-stage5-tier/scripts/run_v0_formula_baseline.py)
- [scripts/run_v10e_lagged.py](/home/xyz/workspace/research-qianli-v2/research-stage5-tier/scripts/run_v10e_lagged.py)
- active artifacts under [registry/f0](/home/xyz/workspace/research-qianli-v2/research-stage5-tier/registry/f0) and [holdout/f0](/home/xyz/workspace/research-qianli-v2/research-stage5-tier/holdout/f0)

## Findings

1. High: Active champion regression checks are currently disabled because the live champion artifact does not match the schema the comparison code expects. [ml/compare.py#L391](/home/xyz/workspace/research-qianli-v2/research-stage5-tier/ml/compare.py#L391) reads `champion_data.get("version")`, but the active file stores `"champion": "v10e-lag1"` instead of `"version"` in [registry/f0/onpeak/champion.json#L1](/home/xyz/workspace/research-qianli-v2/research-stage5-tier/registry/f0/onpeak/champion.json#L1). As a result, `run_comparison()` does not load champion metrics, so champion-based non-regression checks are silently skipped for `f0/onpeak`.

2. High: Re-running the `f0` baseline would overwrite the fixed gate policy with stale rules. [scripts/run_v0_formula_baseline.py#L67](/home/xyz/workspace/research-qianli-v2/research-stage5-tier/scripts/run_v0_formula_baseline.py#L67) still rebuilds gates with `Recall@100` in Group A and still includes `Tier0-AP` / `Tier01-AP`, while the live gate file explicitly removed the Tier-AP metrics and demoted `Recall@100` in [registry/f0/onpeak/gates.json#L1](/home/xyz/workspace/research-qianli-v2/research-stage5-tier/registry/f0/onpeak/gates.json#L1). This means the current code can regress governance if someone recalibrates `v0`.

3. High: The active `v10e-lag1` metadata is materially wrong. The runner trains `backend="lightgbm"` with `label_mode="tiered"` in [scripts/run_v10e_lagged.py#L128](/home/xyz/workspace/research-qianli-v2/research-stage5-tier/scripts/run_v10e_lagged.py#L128), but the saved config says `"method": "lightgbm_regression"` in [registry/f0/onpeak/v10e-lag1/config.json#L1](/home/xyz/workspace/research-qianli-v2/research-stage5-tier/registry/f0/onpeak/v10e-lag1/config.json#L1), and the live champion description repeats “LightGBM regression” in [registry/f0/onpeak/champion.json#L1](/home/xyz/workspace/research-qianli-v2/research-stage5-tier/registry/f0/onpeak/champion.json#L1). The active registry metadata does not accurately describe the actual champion training objective.

4. Medium: Offpeak `f0` governance is incomplete and the shared comparison tool will fail there. [ml/compare.py#L378](/home/xyz/workspace/research-qianli-v2/research-stage5-tier/ml/compare.py#L378) unconditionally opens both `gates.json` and `champion.json`, but the current offpeak slice only contains version files at [registry/f0/offpeak](/home/xyz/workspace/research-qianli-v2/research-stage5-tier/registry/f0/offpeak). So offpeak comparisons are currently broken even though offpeak metrics exist.

5. Medium: The earliest `f0` lagged months do not have a full 15-month history, but `binding_freq_*` silently renormalizes on whatever history exists. [scripts/run_v10e_lagged.py#L76](/home/xyz/workspace/research-qianli-v2/research-stage5-tier/scripts/run_v10e_lagged.py#L76) divides by the number of available prior months rather than the nominal lookback. Because the active dev window starts at `2020-06` in [registry/f0/onpeak/v10e-lag1/metrics.json#L1](/home/xyz/workspace/research-qianli-v2/research-stage5-tier/registry/f0/onpeak/v10e-lag1/metrics.json#L1), the first few rows cannot have a true 12/15-month history. This is not necessarily wrong, but it means early rows do not use the same feature semantics as later rows.

6. Low: Active artifact schemas are inconsistent across `f0` versions. The live `v0` artifact still carries `Tier0-AP` and `Tier01-AP` in [registry/f0/onpeak/v0/metrics.json#L1](/home/xyz/workspace/research-qianli-v2/research-stage5-tier/registry/f0/onpeak/v0/metrics.json#L1), while newer active artifacts omit them. Also, [scripts/run_v10e_lagged.py#L227](/home/xyz/workspace/research-qianli-v2/research-stage5-tier/scripts/run_v10e_lagged.py#L227) writes only `metrics.json` and no matching `config.json` / `meta.json`. That inconsistency is why the active registry metadata has drifted from the runnable code.

## What Looks Correct

- I did not find an `f0` auction-month vs market-month bug in the actual joins. For `f0`, joining realized DA on the same month in [ml/data_loader.py#L86](/home/xyz/workspace/research-qianli-v2/research-stage5-tier/ml/data_loader.py#L86) and using `market_month={auction_month}` in [ml/spice6_loader.py#L35](/home/xyz/workspace/research-qianli-v2/research-stage5-tier/ml/spice6_loader.py#L35) are conceptually sloppy but functionally correct because delivery month equals auction month.

- The `v10e-lag1` row-side and `binding_freq` timing fix is implemented correctly for `f0` in [scripts/run_v10e_lagged.py#L150](/home/xyz/workspace/research-qianli-v2/research-stage5-tier/scripts/run_v10e_lagged.py#L150) and [scripts/run_v10e_lagged.py#L96](/home/xyz/workspace/research-qianli-v2/research-stage5-tier/scripts/run_v10e_lagged.py#L96).

- The shared month cache in [ml/data_loader.py#L25](/home/xyz/workspace/research-qianli-v2/research-stage5-tier/ml/data_loader.py#L25) is correctly keyed by `(auction_month, period_type, class_type)` and does not look like a source of cross-slice contamination.

## Residual Risks

- The code still assumes the month-`M` V6.2B and Spice6 snapshots used for `f0` are genuine pre-auction artifacts. I do not see code proving that; I only see the repo assuming it.

- I could not inspect raw Spice6 parquet schemas or rerun the pipelines because the available Python environment here is missing `polars`.

## Bottom Line

The current `f0` modeling logic is mostly sound, especially the `v10e-lag1` timing fix. The biggest remaining problems are governance and artifact integrity, not core row/feature leakage: champion comparison is currently miswired, `v0` can regenerate stale gates, and the active champion metadata does not describe the actual trained model.
