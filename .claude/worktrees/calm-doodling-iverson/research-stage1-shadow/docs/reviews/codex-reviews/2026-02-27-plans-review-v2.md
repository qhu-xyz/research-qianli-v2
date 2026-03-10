# Review: docs/plans (second pass, 2026-02-27)

## Scope
- `docs/plans/2026-02-26-agentic-ml-pipeline-design.md` (v10)
- `docs/plans/2026-02-26-verification-plan.md`
- `docs/plans/2026-02-26-agentic-ml-pipeline-implementation.md`

## Findings (ordered by severity)

### 1. Critical: Implementation plan uses a non-existent `REVIEWING` state, breaking the state machine + verification specs
- Evidence:
  - Implementation skeleton: `cas_transition WORKER_RUNNING → REVIEWING` and `cas_transition REVIEWING → ORCHESTRATOR_SYNTHESIZING` (`docs/plans/2026-02-26-agentic-ml-pipeline-implementation.md:801`, `docs/plans/2026-02-26-agentic-ml-pipeline-implementation.md:805`).
  - Design state machine requires `REVIEW_CLAUDE` then `REVIEW_CODEX` (`docs/plans/2026-02-26-agentic-ml-pipeline-design.md:105`, `docs/plans/2026-02-26-agentic-ml-pipeline-design.md:106`; detailed flow `docs/plans/2026-02-26-agentic-ml-pipeline-design.md:573`, `docs/plans/2026-02-26-agentic-ml-pipeline-design.md:576`).
  - Verification explicitly tests `STATE_TO_HANDOFF` includes `REVIEW_CLAUDE` and `REVIEW_CODEX` (`docs/plans/2026-02-26-verification-plan.md:238`, `docs/plans/2026-02-26-verification-plan.md:246`, `docs/plans/2026-02-26-verification-plan.md:247`).
- Impact:
  - Any implementation following the implementation plan will necessarily diverge from the design and fail the verification plan.
- Fix:
  - Update implementation Task 17 skeleton/requirements to use `REVIEW_CLAUDE` and `REVIEW_CODEX` as the only reviewer states.

### 2. Critical: Post-iteration final state is inconsistent across docs (`IDLE` vs `ORCHESTRATOR_PLANNING`)
- Evidence:
  - Implementation Task 17 step 18: `ORCHESTRATOR_SYNTHESIZING → IDLE` for `N < 3` (`docs/plans/2026-02-26-agentic-ml-pipeline-implementation.md:784`).
  - Design integrity test: final state after 1 iteration must be `ORCHESTRATOR_PLANNING` (`docs/plans/2026-02-26-agentic-ml-pipeline-design.md:1306`, `docs/plans/2026-02-26-agentic-ml-pipeline-design.md:1309`).
  - Verification Tier 2 repeats that requirement (`docs/plans/2026-02-26-verification-plan.md:405`).
- Impact:
  - The controller logic cannot satisfy both specs at once.
- Fix:
  - Pick a single canonical “end of iter (N<3)” state and align all three docs. If you keep `ORCHESTRATOR_PLANNING`, the controller must transition immediately and the next iteration must not require a separate `run_single_iter.sh` invocation; if you keep `IDLE`, update Tier-2 pass criteria.

### 3. High: `.gitignore` in implementation plan prevents committing required bootstrap artifacts (`handoff/` and `.logs/`)
- Evidence:
  - Implementation Task 2 `.gitignore` ignores both `.logs/` and `handoff/` (`docs/plans/2026-02-26-agentic-ml-pipeline-implementation.md:131`, `docs/plans/2026-02-26-agentic-ml-pipeline-implementation.md:132`).
  - Task 2 also asks to create `handoff/.gitkeep` and commit `handoff/` (`docs/plans/2026-02-26-agentic-ml-pipeline-implementation.md:125`, `docs/plans/2026-02-26-agentic-ml-pipeline-implementation.md:200`).
  - Task 3 asks to create `.logs/audit.jsonl`, `.logs/sessions/.gitkeep`, and commit `.logs/` (`docs/plans/2026-02-26-agentic-ml-pipeline-implementation.md:215`, `docs/plans/2026-02-26-agentic-ml-pipeline-implementation.md:216`, `docs/plans/2026-02-26-agentic-ml-pipeline-implementation.md:259`).
- Impact:
  - A literal follow of the plan will not produce the committed scaffolding it claims to verify/commit.
- Fix:
  - Either stop treating these as committed scaffolding (make them runtime-created only), or unignore via explicit exceptions:
    - `!handoff/.gitkeep`
    - `!.logs/audit.jsonl` and/or `!.logs/sessions/.gitkeep`

### 4. High: Baseline/gates narrative is internally inconsistent and risks calibrating gates on smoke data
- Evidence:
  - Design: v0 baseline is a real (ported) retrain before any batch; `run_pipeline.sh` aborts if v0 absent (`docs/plans/2026-02-26-agentic-ml-pipeline-design.md:644`, `docs/plans/2026-02-26-agentic-ml-pipeline-design.md:655`).
  - Implementation: Task 26 explicitly creates v0 in `SMOKE_TEST=true` mode and then populates floors (`docs/plans/2026-02-26-agentic-ml-pipeline-implementation.md:1011`, `docs/plans/2026-02-26-agentic-ml-pipeline-implementation.md:1016`, `docs/plans/2026-02-26-agentic-ml-pipeline-implementation.md:1023`).
- Impact:
  - If `populate_v0_gates.py` uses smoke-derived v0 metrics, gate floors become meaningless for real runs.
- Fix:
  - Clarify: smoke mode is for infra tests only and must not be used to initialize promotion floors. If you need a smoke v0 for Tier tests, separate it (e.g., `registry/v0_smoke/`) or add a guard to refuse gate population when `SMOKE_TEST=true`.

### 5. Medium: Verification plan “Remaining findings (R1–R5)” reads stale relative to design v10
- Evidence:
  - Verification plan still claims “v0 baseline creation is unspecified” (R1) (`docs/plans/2026-02-26-verification-plan.md:18`), but design v10 includes an explicit v0 bootstrap section (`docs/plans/2026-02-26-agentic-ml-pipeline-design.md:644`).
  - Verification plan pre-run checklist also indicates R1–R5 are “resolved” by concrete steps (`docs/plans/2026-02-26-verification-plan.md:476`, `docs/plans/2026-02-26-verification-plan.md:480`).
- Impact:
  - Readers can’t tell what is still broken vs already fixed; “remaining findings” becomes noise.
- Fix:
  - Either move R1–R5 into a “Resolved findings” section or annotate each with “Status: Resolved in design v10” and link to the exact design lines.

### 6. Medium: “Never suppress exit codes” conflicts with documented controller/maintenance patterns
- Evidence:
  - Implementation Task 1 mandates “NEVER suppress exit codes from subprocesses” as a global rule (`docs/plans/2026-02-26-agentic-ml-pipeline-implementation.md:40`).
  - Design and implementation both include patterns that intentionally suppress failures for idempotency/cleanup (cron install uses `2>/dev/null` and `||` chain: `docs/plans/2026-02-26-agentic-ml-pipeline-design.md:1028`; worktree cleanup uses `2>/dev/null || true`: `docs/plans/2026-02-26-agentic-ml-pipeline-implementation.md:857`).
- Impact:
  - The “rule” is unenforceable as written and will be routinely violated by necessity.
- Fix:
  - Narrow the rule: “Don’t ignore exit codes for correctness-critical commands; cleanup/idempotent operations may suppress errors but must log intent.”

### 7. Medium: `SMOKE_TEST` semantics differ between design and implementation plan
- Evidence:
  - Design snippet shows `SMOKE_TEST=false` in `agents/config.sh` (`docs/plans/2026-02-26-agentic-ml-pipeline-design.md:1145`).
  - Implementation Task 3 explicitly changes to `SMOKE_TEST="${SMOKE_TEST:-false}"` to permit env override (`docs/plans/2026-02-26-agentic-ml-pipeline-implementation.md:229`).
- Impact:
  - Confusion about whether smoke mode is controlled by config file vs environment.
- Fix:
  - Pick one canonical control plane (recommended: env override supported) and update the design snippet to match.

### 8. Medium: Ray init snippet in implementation-plan CLAUDE.md doesn’t match design’s “extra_modules includes ml” guidance
- Evidence:
  - Implementation-plan CLAUDE.md snippet uses `extra_modules=[pmodel]` (`docs/plans/2026-02-26-agentic-ml-pipeline-implementation.md:66`).
  - Design’s memory safety section shows `extra_modules=[pmodel, shadow_ml]` (`docs/plans/2026-02-26-agentic-ml-pipeline-design.md:1336`).
- Impact:
  - In real-data mode, Ray workers may not have access to the local `ml` package.
- Fix:
  - Align the CLAUDE.md snippet with the design: include `ml` as an extra module.

### 9. Medium: Orchestrator “required reads champion metrics” has an undefined behavior when champion is null
- Evidence:
  - Implementation-plan CLAUDE.md requires orchestrator to read “champion metrics” (`docs/plans/2026-02-26-agentic-ml-pipeline-implementation.md:72`).
  - Bootstrap sets `registry/champion.json` to `{"version": null, ...}` (`docs/plans/2026-02-26-agentic-ml-pipeline-implementation.md:165`).
- Impact:
  - First batch plan step can fail or be ambiguous.
- Fix:
  - Specify: if champion is null, orchestrator reads `registry/v0/metrics.json` instead.

### 10. Low: Group B semantics are not fully specified where they matter (promotion/checking)
- Evidence:
  - Gates include `group: "B"` for `S1-REC` and CAP@K metrics (`docs/plans/2026-02-26-agentic-ml-pipeline-design.md:677`), but promotion logic (§7.4) doesn’t define any special handling for Group B vs non-Group-B (`docs/plans/2026-02-26-agentic-ml-pipeline-design.md:686`).
- Impact:
  - Implementers may ignore group entirely or invent inconsistent behavior in `check_gates()`.
- Fix:
  - Either remove `group` from gates.json until used, or define explicit semantics (e.g., “Group B gates are informational only” or “must pass at least 2/3 of Group B”).

## Suggested doc-level next actions
- Edit the implementation plan first: fix state names and post-iteration state contract (these are the biggest “follow-the-doc and you fail” issues).
- Then reconcile smoke vs real baseline so gate floor initialization is never based on synthetic data.
- Finally, update verification plan Part 1 to reflect what is actually unresolved in design v10.
