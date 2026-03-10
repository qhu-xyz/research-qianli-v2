# Codex Review: 2026-02-26 Agentic ML Pipeline Design (v3)

## Verdict
v3 improves structure and testing coverage, but execution reliability is still blocked by reviewer artifact and timeout-path issues.

## Findings (Ordered by Severity)

1. **[CRITICAL] Claude reviewer launch still cannot produce the required review and handoff artifacts**  
   Evidence: Claude reviewer is launched with `--allowedTools "Read,Glob,Grep"` and no stdout redirect (`docs/plans/2026-02-26-agentic-ml-pipeline-design.md:362-367`).  
   Required outputs remain explicit file writes (`...:377-402`, `...:750-751`).  
   Impact: `REVIEW_CLAUDE` can deadlock until timeout because required artifacts are not guaranteed to exist.  
   Fix: Either grant write-capable tools for Claude reviewer or make launcher-owned output deterministic (stdout redirect + launcher-written handoff).

2. **[HIGH] Reviewer handoff integrity contract is still inconsistent with the declared schema**  
   Evidence: Section 8 defines handoff with `artifact_path` and `sha256` verification (`...:570-597`), but Codex launcher writes only `{"agent":"codex","status":"done"}` (`...:411-421`).  
   Impact: Controller cannot validate review artifact integrity for reviewer stages; stale/partial review completion is indistinguishable from valid completion.  
   Fix: Enforce one handoff schema for all agents and include digest of each review file.

3. **[HIGH] Timeout signaling path remains incompatible with the documented handoff contract**  
   Evidence: Handoff contract uses `handoff/{batch_id}/iter{N}/...` (`...:201-207`), while watchdog writes `handoff/{batch_id}/{iteration}/timeout_${state}.json` (`...:803`).  
   Impact: Controller polling can miss timeout artifacts; stuck states may not transition reliably.  
   Fix: Write timeout files to `handoff/${batch_id}/iter${iteration}/timeout_${state}.json` under `${PROJECT_DIR}`.

4. **[MEDIUM] Single-writer control-plane rule is contradicted by worker launch flow**  
   Evidence: Design states only controller writes `state.json` (`...:112-114`), but worker launch snippet mutates `state.json` directly (`...:300-301`).  
   Impact: CAS ownership becomes ambiguous and race-safety assumptions weaken.  
   Fix: Keep all `state.json` writes in controller; launch scripts should only emit artifacts for controller to ingest.

5. **[MEDIUM] Worktree path is still batch-agnostic and collision-prone**  
   Evidence: Worktree path remains fixed as `.claude/worktrees/iter${N}` (`...:294`) while batches can repeat iteration indices and retention may persist until HUMAN_SYNC (`...:997-998`).  
   Impact: New batch startup can fail with existing path collisions.  
   Fix: Include batch ID in worktree path (`iter${N}-${BATCH_ID}`) or prune worktrees at batch start.

6. **[LOW] Watchdog liveness probe still tracks only worker session**  
   Evidence: Watchdog reads only `worker_tmux` and reports `tmux_alive` from that session (`...:784-793`), despite multi-phase tmux execution across orchestrator/reviewer states (`...:101-109`).  
   Impact: Lower diagnosability for stalls in planning/review/synthesis phases.  
   Fix: Persist active phase session id(s) and probe liveness by current state.

## Final Assessment
Address findings 1-3 before implementation; they are still execution blockers. Findings 4-6 are important hardening items that should be completed before long unattended runs.
