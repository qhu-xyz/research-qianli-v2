# Critical Review v8: Agentic ML Pipeline Design

Document reviewed: `docs/plans/2026-02-26-agentic-ml-pipeline-design.md`  
Review date: 2026-02-26

## Findings (ordered by severity)

### 1) Orchestrator launch argument parsing is incompatible with documented call sites (Critical)
- Evidence:
  - Launch snippet parses only `$1`: `PHASE="${1:---phase plan}"; PHASE="${PHASE#--phase}"` (line 265).
  - Valid phases are only `plan|synthesize` in `case` (lines 266-269).
  - Controller invokes as two args: `launch_orchestrator.sh --phase plan` and `--phase synthesize` (lines 520, 545).
- Why this is a bug:
  - With two-arg invocation, `$1` is `--phase`, so parsed `PHASE` becomes empty and falls into `Unknown phase`.
  - This deterministically blocks both planning and synthesis launches.
- Required fix:
  - Use a real parser supporting `--phase plan` and `--phase=plan`, and keep call style consistent across all scripts.

### 2) Registry `rsync` before branch merge can make merge fail on untracked-path collisions (Critical)
- Evidence:
  - Controller copies worker registry artifacts into main before merge (lines 529-530).
  - Worker is required to commit `registry/${VERSION_ID}/` on its branch (lines 345-347).
  - Controller then merges that branch into main (line 536).
- Why this is a bug:
  - Pre-copying files into main creates untracked files at the same paths that merge wants to introduce as tracked, which can abort merge with overwrite protection.
  - This breaks the main continuity mechanism (`iter N` -> `iter N+1`) in the success path.
- Required fix:
  - Merge first, then verify artifacts in main from the merge result.
  - If out-of-band copy is still desired, do it only for non-git-managed artifacts and never for paths introduced by the merge commit.

### 3) Worker write constraints forbid the required worker handoff file (High)
- Evidence:
  - Worker must write `handoff/{batch_id}/iter{N}/worker_done.json` (line 348; prompt line 860).
  - Worker constraints say only modify `ml/` and `registry/${VERSION_ID}/` (lines 350-351; prompt line 863).
- Why this is a bug:
  - The handoff write is outside the allowed paths, so the contract is internally contradictory.
  - Depending on which instruction the agent prioritizes, this can cause timeout/deadlock at step 7 polling.
- Required fix:
  - Explicitly allow worker writes to its single handoff file path in both spec and prompt constraints.

### 4) Synthesis verification expects `direction_iter{N+1}.md`, but synthesis spec does not require producing it (High)
- Evidence:
  - Handoff verifier expects synthesis artifact to be `memory/direction_iter$((iter+1)).md` for iterations 1-2 (lines 715-722, 729-740).
  - Integrity test also expects synthesis to write `direction_iter2.md` (line 1130).
  - Synthesis responsibilities for iter<3 only state writing `orchestrator_synth_done.json` after memory updates, with no explicit `direction_iter{N+1}.md` write (lines 303-305).
- Why this is a bug:
  - Verification contract and agent task contract diverge, so synthesis can "succeed" from one viewpoint and fail from controller verification.
- Required fix:
  - Add an explicit iter<3 synthesis step to write `memory/direction_iter{N+1}.md` (and include it in synth prompt), or change verifier expectation to a file synthesis is guaranteed to produce.

### 5) Archive directory naming remains inconsistent (`batch-{id}` vs `${batch_id}`) (Medium)
- Evidence:
  - Synthesis writes to `memory/archive/${batch_id}/...` (lines 307-309).
  - Directory and memory contracts define `memory/archive/batch-{id}/...` (lines 232-233, 774-778, 784).
- Why this is a bug:
  - Different components can read/write different archive trees, causing missing summaries and broken traceability links.
- Required fix:
  - Normalize all archive paths to one convention via a shared helper (recommended: `batch-${BATCH_ID}`).

## Missing tests / validation gaps

1. CLI argument parsing test for `launch_orchestrator.sh --phase plan|synthesize` and `--phase=...`.
2. Integration test proving success path merge works when worker creates `registry/${VERSION_ID}` files (no untracked overwrite failure).
3. Contract test that worker constraints permit writing exactly one handoff file and nothing else outside allowed directories.
4. Iteration-1/2 synthesis test asserting `direction_iter{N+1}.md` exists and passes handoff hash verification.
5. Archive-path consistency test validating all writers/readers resolve to the same batch directory convention.

## Overall assessment

v7 fixed several prior blockers, but there are still deterministic execution failures in launch parsing and success-path merge sequencing. Findings #1-#4 should be resolved before implementation; #5 is a traceability integrity issue that will accumulate confusion quickly if left unresolved.
