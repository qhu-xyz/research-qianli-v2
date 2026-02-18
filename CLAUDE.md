# CLAUDE.md — research-qianli-v2

Inherits rules from parent: `/home/xyz/workspace/research-qianli/CLAUDE.md`

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
