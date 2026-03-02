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

## Versioned Experiments (MANDATORY)

When implementing a new version (e.g., bands v2, baseline v4):
1. **Always run the experiment script** — never leave it as "ready to run"
2. **Run `validate`** after the script completes to confirm schema compliance
3. **Run `compare`** against promoted version to check gates
4. **Update NOTES.md** with actual results (not placeholders)
5. **Update mem.md** with actual numbers
6. Deliver a complete, tested version — not a stub.

## Virtual Environment

Scripts run via pmodel's venv:
```bash
cd /home/xyz/workspace/pmodel && source .venv/bin/activate
```
