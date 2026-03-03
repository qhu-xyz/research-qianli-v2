# CLAUDE.md — research-stage1-shadow (Agent Rules)

Inherits: `/home/xyz/workspace/research-qianli-v2/CLAUDE.md`

## For ALL agents (interactive session, orchestrator, worker, reviewers)

### Sandbox Constraints
- Only operate inside this repository directory
- NEVER run `rm -rf`
- NEVER delete any `registry/v*/` directory
- NEVER modify `registry/v0/` (baseline is immutable)
- NEVER suppress exit codes from correctness-critical subprocesses (cleanup/idempotent operations may use `|| true` but must log intent)
- All launch scripts support `--dry-run` — this flag must be preserved if modifying launchers

### HUMAN-WRITE-ONLY Files (agents must NEVER modify)
- `registry/gates.json` — promotion gate definitions (created during bootstrap; immutable to runtime agents thereafter)
- `ml/evaluate.py` — standardized evaluation harness (created during bootstrap; immutable to runtime agents thereafter)

Note: The implementation session creates these files during bootstrap (Tasks 2 and 10). The NEVER-modify constraint applies to pipeline runtime agents only (orchestrator, worker, reviewers).

### Memory Safety (from parent CLAUDE.md)
- Use polars over pandas
- Use `pl.scan_parquet().filter().collect()` (lazy scan)
- Print `mem_mb()` at each pipeline stage (including training)
- Free intermediates: `del df; gc.collect()`
- `ray.shutdown()` after data loading completes
- Gzip model files on write: `gzip registry/${VERSION_ID}/model/*.ubj`

### Virtual Environment
```bash
cd /home/xyz/workspace/pmodel && source .venv/bin/activate
```

### Ray Init (required for real data loading, NOT needed in SMOKE_TEST mode)
```python
from pbase.config.ray import init_ray
import pmodel
import ml as shadow_ml
init_ray(address='ray://10.8.0.36:10001', extra_modules=[pmodel, shadow_ml])
```

### Per-Agent Context Slices (what memory each agent MUST read)
| Agent | Required reads |
|-------|---------------|
| Orchestrator (plan) | memory/hot/ (all) + memory/warm/ (all) + memory/archive/index.md + registry/gates.json + champion metrics (if champion is null, read registry/v0/metrics.json instead) |
| Worker | memory/direction_iter{N}.md + memory/hot/champion.md + memory/hot/learning.md + memory/hot/runbook.md |
| Claude Reviewer | direction + changes_summary + comparison table + warm/experiment_log + hot/gate_calibration + warm/decision_log + gates.json + ml/ codebase |
| Codex Reviewer | Same as Claude reviewer — does NOT see Claude's review |
| Orchestrator (synth) | Both raw reviews (read independently) + comparison table + warm/ (all) |

### Artifact Naming
- All handoff/review files include `{batch_id}` and `iter{N}` to prevent stale artifact reads across batches
- Handoff JSON `artifact_path` must use relative paths (not absolute)

## Worker-Specific Rules
- Only modify files under `ml/`, `registry/${VERSION_ID}/`, and `${PROJECT_DIR}/handoff/{batch_id}/iter{N}/worker_done.json` (absolute path — handoff/ is gitignored and does not exist in the worktree)
- NEVER touch other `registry/v*/` directories
- ALWAYS commit changes before writing handoff JSON
- If tests fail 3x: write failed handoff with error summary, do NOT commit
- Read VERSION_ID from the PROJECT_DIR copy of state.json (NOT the worktree copy, which is stale): `VERSION_ID=$(jq -r '.version_id' "${PROJECT_DIR}/state.json")`

## Reviewer-Specific Rules
- Only write to `reviews/` and `handoff/` directories
- Do NOT read the other reviewer's output — independence is mandatory
- You MAY critique gates as stale or miscalibrated
- Gate changes require human approval at HUMAN_SYNC

## Orchestrator-Specific Rules
- Do NOT modify any ML code or registry/ files
- Do NOT run training
