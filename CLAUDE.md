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

Any script or notebook that uses pbase data loaders MUST initialize Ray first:

```python
from pbase.config.ray import init_ray
import pmodel
init_ray(address='ray://10.8.0.36:10001', extra_modules=[pmodel])
```

### What requires Ray
- `MisoCalculator.get_mcp_df()` — nodal MCPs
- `MisoApTools.get_all_cleared_trades()` — cleared trades
- `MisoDaLmpMonthlyAgg`, `MisoDaLmpDailyAgg` — DA LMP data
- `MisoNodalReplacement.load_data()` — nodal replacement mapping
- Any `pbase.data.dataset.*` loader

### Rules
- Call `init_ray()` ONCE at script start, BEFORE any data access
- Do NOT use `ray.put()` — pbase handles serialization internally
- Add `lightgbm` to `extra_modules` only if using LightGBM: `extra_modules=[pmodel, lgb]`
- The Ray cluster address `ray://10.8.0.36:10001` is the standard dev cluster

## Virtual Environment

Scripts run via pmodel's venv:
```bash
cd /home/xyz/workspace/pmodel && source .venv/bin/activate
```
