# Review: docs/plans (third pass, 2026-02-27)

## Scope
- `docs/plans/2026-02-26-agentic-ml-pipeline-design.md` (v10)
- `docs/plans/2026-02-26-verification-plan.md`
- `docs/plans/2026-02-26-agentic-ml-pipeline-implementation.md`

## New Findings (delta-oriented, plus any still-critical items)

### 1. Critical (still): Implementation plan references a non-existent `REVIEWING` state
- Evidence: `WORKER_RUNNING → REVIEWING → ORCHESTRATOR_SYNTHESIZING` in the Task 17 skeleton (`docs/plans/2026-02-26-agentic-ml-pipeline-implementation.md:801`, `docs/plans/2026-02-26-agentic-ml-pipeline-implementation.md:805`).
- Conflicts with: design requires `REVIEW_CLAUDE` then `REVIEW_CODEX` (`docs/plans/2026-02-26-agentic-ml-pipeline-design.md:105`, `docs/plans/2026-02-26-agentic-ml-pipeline-design.md:106`) and verification tests those states (`docs/plans/2026-02-26-verification-plan.md:238`).

### 2. Critical (new): `registry/gates.json` schema is inconsistent; design sample omits `v0_offset` even though v0-floor initialization requires it
- Evidence:
  - Design explains `populate_v0_gates.py` computes `v0_metric − offset` per pending gate (`docs/plans/2026-02-26-agentic-ml-pipeline-design.md:660`).
  - But the design’s gates.json example has `pending_v0: true` entries with no per-gate `v0_offset` (`docs/plans/2026-02-26-agentic-ml-pipeline-design.md:672`, `docs/plans/2026-02-26-agentic-ml-pipeline-design.md:676`).
  - Implementation plan calls its gates.json content “Exact content from design doc” but includes `v0_offset` everywhere needed (`docs/plans/2026-02-26-agentic-ml-pipeline-implementation.md:142`, `docs/plans/2026-02-26-agentic-ml-pipeline-implementation.md:151`, `docs/plans/2026-02-26-agentic-ml-pipeline-implementation.md:155`).
- Impact:
  - An implementer who copies the design doc’s JSON literally will lack offsets; `populate_v0_gates.py` must either hardcode defaults (not specified) or fail.
  - The implementation plan’s “exact content” claim is false, which is dangerous in bootstrap instructions.
- Fix:
  - Make the design’s gates.json canonical and include `v0_offset` fields, or explicitly define default offsets by metric name (and document that `v0_offset` is optional).

### 3. High (new): Codex CLI flag usage is inconsistent inside the design doc (`--sandbox` vs `-s`)
- Evidence:
  - Design describes Codex reviewer launch using `--sandbox read-only` (`docs/plans/2026-02-26-agentic-ml-pipeline-design.md:484`) and calls out `--sandbox` in the flag table (`docs/plans/2026-02-26-agentic-ml-pipeline-design.md:1157`).
  - But `check_clis.sh` snippet uses `-s read-only` (`docs/plans/2026-02-26-agentic-ml-pipeline-design.md:1278`).
- Impact:
  - If `-s` is not a supported shorthand in your Codex CLI, preflight will fail even though the rest of the doc uses the correct long form.
- Fix:
  - Use one form consistently (prefer `--sandbox read-only` everywhere unless you’ve validated `-s` is an alias).

### 4. Medium (new): `memory/human_input.md` and `human-input/` coexist; docs don’t define precedence/flow clearly
- Evidence:
  - Design says `human-input/` contains read-only user requirements (`docs/plans/2026-02-26-agentic-ml-pipeline-design.md:147`).
  - Design separately defines a per-batch `memory/human_input.md` channel (`docs/plans/2026-02-26-agentic-ml-pipeline-design.md:292`, `docs/plans/2026-02-26-agentic-ml-pipeline-design.md:535`).
  - Implementation plan instructs creating a stub `memory/human_input.md` while noting `human-input/` already exists and is read-only (`docs/plans/2026-02-26-agentic-ml-pipeline-implementation.md:216`, `docs/plans/2026-02-26-agentic-ml-pipeline-implementation.md:220`).
- Impact:
  - First-time operators may put input into the wrong location or assume `human-input/` is the live per-batch channel.
- Fix:
  - Add an explicit rule: `human-input/` = static requirements; `memory/human_input.md` = per-batch override; orchestrator reads both (or reads memory first then falls back).

### 5. Medium (still): Post-iteration final state mismatch (`IDLE` vs `ORCHESTRATOR_PLANNING`)
- Evidence:
  - Implementation plan ends iter with `IDLE` (`docs/plans/2026-02-26-agentic-ml-pipeline-implementation.md:784`).
  - Design + verification Tier 2 require final state `ORCHESTRATOR_PLANNING` (`docs/plans/2026-02-26-agentic-ml-pipeline-design.md:1306`, `docs/plans/2026-02-26-verification-plan.md:405`).

## Suggested priority fixes
1. Fix gates.json schema mismatch (add `v0_offset` to the design’s canonical JSON or define defaults).
2. Fix reviewer state naming in implementation plan (`REVIEWING` → `REVIEW_CLAUDE` / `REVIEW_CODEX`).
3. Decide and align the post-iter state contract.
4. Normalize Codex CLI flags (`--sandbox` vs `-s`) across the design and verification.
