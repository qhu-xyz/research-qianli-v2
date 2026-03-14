# Codex Review: 2026-02-26 Agentic ML Pipeline Design (v2)

## Verdict
This revision is substantially improved and close to implementation-safe, but three execution blockers remain.

## Findings (Ordered by Severity)

1. **[CRITICAL] Claude reviewer launch cannot produce required review/handoff artifacts**  
   Evidence: Reviewer launch grants only `Read,Glob,Grep` tools and does not redirect stdout to a file (`docs/plans/2026-02-26-agentic-ml-pipeline-design.md:337-343`).  
   Required outputs are explicit review + handoff files (`...:353-378`, `...:675-676`).  
   Impact: `REVIEW_CLAUDE` can stall until timeout because required files are not reliably produced.  
   Fix: Either (a) allow write-capable tools and keep file-writing in prompt, or (b) capture stdout to `reviews/{batch_id}_iter{N}_claude.md` and emit handoff in launcher script.

2. **[HIGH] Handoff schema is defined but not consistently enforced for reviewer stages**  
   Evidence: Section 8 requires `artifact_path` + `sha256` contract (`...:551-573`), but Codex launcher writes only `{\"agent\":\"codex\",\"status\":\"done\"}` (`...:397`).  
   Impact: Controller cannot verify reviewer artifact integrity; stale or partial outputs can pass as complete.  
   Fix: Require all handoff files (including reviewers) to include the full schema and hash the produced review artifact.

3. **[HIGH] Watchdog timeout file path does not match declared handoff directory contract**  
   Evidence: Standard pattern is `handoff/{batch_id}/iter{N}/...` (`...:199-205`, `...:425-438`), but watchdog writes `handoff/{batch_id}/{iteration}/timeout_...` (`...:728`).  
   Impact: Controller polling can miss timeout signals and hang/retry incorrectly.  
   Fix: Write timeout files under `handoff/${batch_id}/iter${iteration}/timeout_${state}.json`.

4. **[MEDIUM] Single-writer guarantee for `state.json` is contradicted by worker launch snippet**  
   Evidence: Design says only `run_single_iter.sh` writes `state.json` (`...:112-114`), while worker launch snippet updates `state.json` directly (`...:277`).  
   Impact: Reintroduces control-plane ambiguity and weakens CAS ownership.  
   Fix: Launcher should emit session name; controller performs all `state.json` mutations.

5. **[MEDIUM] Worktree path naming can collide across batches**  
   Evidence: Worker path is fixed as `.claude/worktrees/iter${N}` (`...:270`), and retention may continue until HUMAN_SYNC cleanup (`...:863-864`).  
   Impact: New batch with same iteration index can fail on existing worktree path.  
   Fix: Include batch ID in worktree path (`iter${N}-${BATCH_ID}`) or prune on batch start.

6. **[LOW] Watchdog probe tracks worker session only**  
   Evidence: Probe uses `worker_tmux` for liveness (`...:709-717`) and does not monitor orchestrator/reviewer session IDs.  
   Impact: Reduced diagnosability for stuck planning/review phases.  
   Fix: Persist per-phase tmux session IDs in state and probe by active phase.

## Resolved Since Prior Review
- Codex invocation conflict is fixed (invalid flags removed; `-p` meaning split documented).
- Review topology is now sequential and explicit (`REVIEW_CLAUDE -> REVIEW_CODEX`).
- Artifact naming includes `batch_id`, reducing stale-cross-batch collisions.
- Batch-level lock (`flock`) and start-state guard (`IDLE`/`HUMAN_SYNC`) are defined.
- Gate policy is consolidated with corrected tolerance example.
- Handshake concept with checksum verification is now documented.
- Smoke-test mode and disk-budget policy are explicitly defined.

## Final Assessment
Address findings 1-3 before implementation; those are the remaining reliability blockers. The rest are quality hardening items.
