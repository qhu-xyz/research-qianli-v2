# CLAUDE.md — research-stage5-tier

Inherits rules from parent: `/home/xyz/workspace/research-qianli-v2/CLAUDE.md`

## Experiment Runs (CRITICAL)

- When runs are slow, always investigate why and optimize whenever possible.
- Be careful about long-running scripts — check expected timing before launching.
- After any optimization, verify results are consistent with the previous (non-optimized) version before proceeding.
- NFS I/O is the main bottleneck (not CPU). Parallelization helps but has diminishing returns.
- Expected timing: ~8-10s per eval month (screen=4mo ~40s, full=12mo ~2min per variant).
- If a run takes >2min for a single variant screen, something is wrong — investigate immediately.

## LightGBM Threading (CRITICAL)

Two related deadlock issues:

1. **num_threads**: Container reports 64 CPUs but LightGBM deadlocks using them all.
   Fix: `num_threads=4` in all LightGBM params (enforced in `train.py` via `_LGB_NUM_THREADS`).

2. **ProcessPoolExecutor + fork**: `fork` copies LightGBM's broken thread pool lock state to child processes.
   Fix: `mp_context=multiprocessing.get_context("spawn")` in `benchmark.py`.
   pbase avoids this by using Ray instead of multiprocessing.

Symptom: script hangs at training with no output, processes at 0% CPU waiting on futex.

## Virtual Environment

```bash
source /home/xyz/workspace/pmodel/.venv/bin/activate
```

Always run scripts from this directory (`research-stage5-tier/`), not from pmodel, to ensure registry writes go to the correct location.
