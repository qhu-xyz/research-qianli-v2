# CLAUDE.md — research-qianli-v2

Inherits rules from parent: `/home/xyz/workspace/research-qianli/CLAUDE.md`

## Memory Safety (CRITICAL — Pod limit 128 GiB)

This workspace runs in a 128 GiB container with ~3 GiB baseline (Cursor, MCP servers, JupyterLab). Uncontrolled scripts WILL OOM-kill the pod.

### Rules
- **Use polars over pandas** for data loading/transforms (2-4x less memory). `polars 1.31.0` is available.
- **Use `pl.scan_parquet().filter().collect()`** (lazy scan) instead of `pl.read_parquet()` when filtering large files.
- **Never do cross-product merges** without row-count assertions before and after.
- **Never use `iterrows()`** — use vectorized joins or `.map()`.
- **Free intermediates immediately** with `del df; gc.collect()` between pipeline stages.
- **Save intermediates to parquet** between phases so scripts can resume after crash.
- **Add `--from-phase N`** flags to multi-phase scripts for partial re-runs.
- **Print `mem_mb()`** at each stage to track memory growth.
- **Shutdown Ray** (`ray.shutdown()`) as soon as all remote calls are done.
- **Never run long scripts via `claude -r`** — session resume will re-execute on pod restart, creating OOM crash loops.

### Memory budget
| Component | Budget |
|-----------|--------|
| Cursor + extensions | ~3 GiB |
| Pyright (cap at 4 GiB) | ≤4 GiB |
| Claude Code | ~1 GiB |
| Research scripts | ≤40 GiB |
| Safety margin | ~80 GiB |

### Helper pattern
```python
import resource
def mem_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
```

## Ray Usage (MANDATORY)

Any script or notebook that uses pbase data loaders MUST initialize Ray first.

Initialization — set env var BEFORE `init_ray()`:
```python
import os
os.environ["RAY_ADDRESS"] = "ray://10.8.0.36:10001"

from pbase.config.ray import init_ray
import pmodel

init_ray(extra_modules=[pmodel])  # add lightgbm if needed: extra_modules=[pmodel, lgb]
```
For notebooks, use `%env RAY_ADDRESS=ray://10.8.0.36:10001` in the first cell instead.

### What requires Ray
- `MisoCalculator.get_mcp_df()` — nodal MCPs
- `MisoApTools.get_all_cleared_trades()` — cleared trades
- `MisoDaLmpMonthlyAgg`, `MisoDaLmpDailyAgg` — DA LMP data
- `MisoNodalReplacement.load_data()` — nodal replacement mapping
- Any `pbase.data.dataset.*` loader

### Rules
- Call `init_ray()` ONCE at script start, BEFORE any data access
- **Never** use `multiprocessing`, `concurrent.futures`, `joblib`, `dask`, or `threading` — always use Ray
- Use `@ray.remote(scheduling_strategy="SPREAD")` for all parallel tasks — always include `SPREAD`
- Use `ray_map_bounded` from `pbase.utils.ray` for bounded concurrency (preferred over raw `ray.get(futures)`)
- Use `ray.put(big_obj)` for large shared data — pass the ref to `.remote()` calls to avoid repeated serialization
- Migrate away from `parallel_equal_pool` — use `@ray.remote` + `ray_map_bounded` instead (avoids deadlocks under nesting)
- See the `parallel-with-ray` skill for full patterns (composable functions, ObjectRef optimization, conversion procedure)
- The Ray cluster address `ray://10.8.0.36:10001` is the standard dev cluster

## Run Performance (MANDATORY)

When runs are slow, always investigate and optimize:
- **Profile first**: time each phase (data loading, training, evaluation) to find the real bottleneck
- **Cache expensive I/O**: large NFS parquet scans should be cached locally after first load
- **Avoid redundant work**: if multiple eval groups share training data, train once and reuse
- **Verify correctness**: any speed optimization MUST produce bit-identical results vs the unoptimized version
- **Never let long runs slide**: a 20-minute run that could be 5 minutes wastes iteration cycles
- **Report walltimes**: always print walltime for every experiment run and do full auditing of results
- **LightGBM num_threads**: This container has 64 CPUs. LightGBM auto-detects all 64 and creates massive thread contention on small data (57s → 0.1s with `num_threads=4`). **Always set `"num_threads": 4`** in LightGBM params. This applies to ALL LightGBM usage in the repo.

## Temporal Leakage: Auction Timing (CRITICAL)

**For any feature or label derived from realized market data (DA shadow prices, binding outcomes, etc.), you MUST account for the submission timing of the signal.**

- f0 (front-month) for auction month M is submitted **~mid of M-1**
- At submission time, only realized data through month **M-2** is complete
- Month M-1 is only partially observed (~12 days) and MUST NOT be used

This means: if your training or features use realized DA data, shift everything back by 1 month. Using `months < M` when you should use `months < M-1` is temporal leakage. This was discovered in stage5-tier where it inflated binding_freq results by 6-20%.

**Rule**: when building any realized-data-derived feature, always ask: "at the moment we submit this signal, is this data actually available?" If unsure, check the auction calendar.

See `research-stage5-tier/CLAUDE.md` and `research-stage5-tier/registry/f0/onpeak/v10e-lag1/NOTES.md` for the full analysis.

## MISO Annual Price Units (CRITICAL)

**For MISO annual auctions (aq1-aq4), all prices are QUARTERLY (3-month total), not monthly.**

- **`mcp`** = quarterly clearing price (the actual auction result). This is the **prediction target**.
- **`mcp_mean`** = inconsistent across rounds in `all_residuals_v2.parquet`:
  - R1: `mcp_mean = mcp` (quarterly)
  - R2/R3: `mcp_mean = mcp / 3` (monthly average)
  - **Do NOT use `mcp_mean` for R1.** Use `mcp` directly.
- **`nodal_f0`** = average of 3 monthly f0 prices = **monthly**. Must be multiplied by 3 to match quarterly `mcp`.
- **`mtm_1st_mean`** for R2/R3 = prior round's MCP / 3 = monthly. Must be multiplied by 3 to match quarterly `mcp`.

**Rule:** When building baselines or computing residuals for MISO annual:
```python
# R1: baseline is nodal_f0 scaled to quarterly
baseline_r1 = nodal_f0 * 3
residual_r1 = mcp - baseline_r1

# R2/R3: baseline is prior round's MCP (already quarterly in mcp column)
# If using mtm_1st_mean (monthly), scale up:
baseline_r2 = mtm_1st_mean * 3
residual_r2 = mcp - baseline_r2
# Or equivalently, use mcp directly from prior round's cleared trades
```

**Why:** MISO annual FTRs settle quarterly. The auction clears a single price per path per quarter. Monthly `split_month_mcp` values are accounting allocations, not separate auction outcomes. The quarterly `mcp` is the economically meaningful price.

### Current Implementation Status (2026-03-14)

**Data fix applied:** R1 `mcp_mean` in `aq*_all_baselines.parquet` and `all_residuals_v2.parquet` has been divided by 3 (backups in `.bak`). All rounds now have `mcp_mean = mcp / 3` (monthly).

**Band scripts (run_v9_bands.py, v10, v11):** Compute residuals as `mcp_mean - baseline` where both sides are monthly. This is mathematically equivalent to quarterly — coverage percentages are identical. Band widths are in monthly scale internally.

**Reporting rule (MANDATORY):** All reports and NOTES.md MUST show band widths in **quarterly scale** (monthly × 3). This is the economically meaningful number — it represents the actual bid range in $/MWh per quarter. If monthly scale is also shown, label it explicitly.

| What | Monthly | Quarterly (×3) | Which to report |
|------|--------:|---------------:|:---:|
| R1 P95 hw | 693 | 2,079 | **Quarterly** |
| R2 P95 hw | 177 | 532 | **Quarterly** |
| R3 P95 hw | 152 | 457 | **Quarterly** |
| MAE | 264 | 792 | **Quarterly** |

## Versioned Experiments (MANDATORY)

When implementing a new version (e.g., bands v2, baseline v4):
1. **Always run the experiment script** — never leave it as "ready to run"
2. **Run `validate`** after the script completes to confirm schema compliance
3. **Run `compare`** against promoted version to check gates
4. **Update NOTES.md** with actual results (not placeholders)
5. **Update mem.md** with actual numbers
6. Deliver a complete, tested version — not a stub.

### Registry Layout
Results: `registry/{period_type}/{class_type}/{version_id}/metrics.json`
Each slice has its own `gates.json` and `champion.json`.
Use `ml.registry_paths` — never hardcode paths.

## Virtual Environment

Scripts run via pmodel's venv:
```bash
cd /home/xyz/workspace/pmodel && source .venv/bin/activate
```
