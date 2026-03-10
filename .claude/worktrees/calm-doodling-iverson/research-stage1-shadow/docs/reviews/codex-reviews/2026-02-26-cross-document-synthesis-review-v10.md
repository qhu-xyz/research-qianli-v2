# Cross-Document Synthesis Review

**Documents reviewed**:
- `docs/plans/2026-02-26-agentic-ml-pipeline-design.md` (v9)
- `docs/plans/2026-02-26-verification-plan.md`

**Prior reviews cross-referenced**:
- `review-claude-v1.md` (v8 design review, C1–C4, S1–S5, M1–M5)
- `2026-02-26-agentic-ml-pipeline-design-review-v9.md` (v9 review, findings #1–#4)

**Reviewer**: Claude Sonnet 4.6
**Date**: 2026-02-26

---

## Purpose

Prior reviews were single-document. This review answers:

1. Which v8 review findings are confirmed resolved in v9?
2. Which v9 review findings are valid vs incorrectly flagged?
3. What remains open across both plan documents taken together?
4. Is the combined specification implementable as written?

---

## Resolution Status of v8 Review Findings (claude-v1.md)

| Finding | v9 Fix | Status |
|---------|--------|--------|
| C1: Shell injection in agent launches | `< "${PROMPT}"` stdin redirect throughout | RESOLVED |
| C2: Gate auto-population unspecified | `populate_v0_gates.py` assigned to controller, Section 7.3 + checklist | RESOLVED |
| C3: Codex reviewer prompt mismatch | Section 5.5 explicitly diverges: stdout-only for Codex | PARTIALLY RESOLVED — see #1 below |
| C4: run_single_iter.sh lock dependency implicit | `PIPELINE_LOCKED` guard added | RESOLVED |
| S1: Bash associative array with embedded vars | `get_expected_artifact()` dynamic function replaces declare -A | RESOLVED |
| S2: Worker commit check (sha256 after merge) | sha256 verified in worktree before merge (step 7c before 7d) | RESOLVED |
| S3: stale direction_iter*.md across batches | `rm -f memory/direction_iter*.md` in run_pipeline.sh | RESOLVED |
| S4: Codex model ID hardcoded | `CODEX_MODEL` in config.sh | RESOLVED — check_clis.sh exception (R5) |
| S5: HUMAN_SYNC notification passive | Remains open decision (#4) | DEFERRED (Open Decision) |
| M1: Orchestrator step numbering | Not explicitly fixed | MINOR — won't block implementation |
| M2: Codex handoff producer field | verify_handoff() checks path+sha only; producer not checked | EFFECTIVELY RESOLVED |
| M3: noise_tolerance on stale Group B | Protocol in Section 7.5 flags the transition | ACKNOWLEDGED, not a blocker |
| M4: all_critiques.md when Codex fails | Section 9 archive spec notes "Codex unavailable" fallback | RESOLVED |
| M5: Orchestrator crash detection lag | Asymmetry documented; orchestrator timeout is 600s anyway | ACKNOWLEDGED |

**Summary**: All critical and significant v8 findings are resolved in v9, except C3 (partially
resolved — see remaining issue #1 below) and S5 (deferred by design).

---

## Validity Assessment of v9 Review Findings

| Finding | Valid? | Rationale |
|---------|--------|-----------|
| #1: Codex read-only vs reviewer prompt WRITE instruction | VALID | Section 10.3 shows `WRITE: reviews/...` as the unified reviewer structure; no separate reviewer_codex.md spec shown. Section 5.5 says it MUST diverge, but 10.3 doesn't show how. |
| #2: verify_handoff() can't validate failed handoffs | INVALID | v9 already has status-aware branching in verify_handoff() (lines 762–767: `if [[ "$status" == "failed" ]]` branch). The reviewer cited the wrong lines. |
| #3: Orchestrator/Claude reviewer launches lack cd PROJECT_DIR | INVALID | v9 launch scripts (lines 282–285, 415–418) both have `"cd \"${PROJECT_DIR}\" && ..."` in the tmux command string. CWD anchoring was fixed in v9. |
| #4: Integrity test encodes obsolete rsync behavior | INVALID | Section 13.2 lines 1178–1179 now correctly read "sha256 of metrics.json verified against worktree branch (before merge)" and "Worker branch merged into main". No rsync step. |

**Conclusion**: Only v9 review finding #1 is valid and remains open. Findings #2–#4 were
already fixed in v9 and were incorrectly flagged by the reviewer.

---

## Remaining Open Issues (across both documents)

### O1 — Codex reviewer prompt spec gap (Critical)

**Source**: v8 finding C3 (partially resolved), v9 finding #1 (valid)

Section 5.5 states `reviewer_codex.md` MUST instruct Codex to print to stdout (not write a
file). Section 10.3 shows the reviewer prompt structure with `WRITE: reviews/...` and
`WHEN DONE: compute sha256 ... write the unified handoff JSON`. This prompt works for Claude
(who can write files), but if used verbatim for Codex (which runs in `--sandbox read-only`),
Codex will attempt file writes that the sandbox blocks.

The design acknowledges the divergence must exist but never shows what `reviewer_codex.md`
looks like. An implementer following section 10.3 will produce a broken Codex prompt.

**Fix**: Add to section 10.3 a separate `reviewer_codex.md` structure showing:
- Same 6 review sections as Claude reviewer
- `WHEN DONE: Print the full review to stdout. Do not write any files.`
- Remove the `WRITE:` and handoff-writing instructions
- Add note: "The shell wrapper (launch_reviewer_codex.sh) captures stdout and writes the
  review file and handoff JSON on your behalf."

---

### O2 — v0 baseline strategy ambiguity (High)

**Source**: Verification plan R1 fix

See verification plan review — the R1 fix says v0 is "manually extracted from source repo"
but doesn't resolve whether this means retrain-with-same-config or copy-pre-trained-weights.
This is a one-time bootstrap step but must be correct: if v0 metrics differ from the source
repo's historical baseline, all v0-relative gate floors are set against the wrong reference.

**Fix**: Explicitly state in R1 fix and in NOTES.md: "v0 is created by re-training from
scratch using the ported config, producing a fresh but config-equivalent baseline."
OR: "v0 metrics are copied directly from the source repo (registry/v0/metrics.json is
populated manually); the model weights are not used in this pipeline."

---

### O3 — WORKER_FAILED not reset between iterations (High)

**Source**: Verification plan R3 fix — cross-iteration bug

See verification plan review. A failure in iteration 1 (WORKER_FAILED=1) will persist into
iteration 2 if not explicitly reset, causing synthesis iteration 2 to use the wrong input
contract (failure path instead of success path).

**Fix**: Add `export WORKER_FAILED=0` at the start of `run_single_iter.sh`, before the
state machine begins.

---

### O4 — Handoff directory creation unspecified (Medium)

**Not previously flagged**

`handoff/{batch_id}/iter{N}/` is written to by all agents. The design never specifies who
creates this directory. If agents write their handoff files to a non-existent directory,
they will fail silently (bash redirection to a non-existent directory creates a file-not-found
error, not the file).

Who should create `handoff/{batch_id}/iter{N}/`? The natural owner is the controller
(run_single_iter.sh) at the start of each iteration, before launching any agents. This is
deterministic and requires no agent cooperation.

**Fix**: Add to step 2 of run_single_iter.sh iteration flow:
```bash
mkdir -p "${PROJECT_DIR}/handoff/${BATCH_ID}/iter${N}"
```

---

### O5 — run_pipeline.sh iteration loop not specified (Medium)

**Not previously flagged**

Section 6.1 says "Runs `run_pipeline.sh --batch-name '...'` in background" and "3-iteration
loop calls `run_single_iter.sh`". But the loop body of `run_pipeline.sh` is never shown.
Section 12.3 shows the lock acquisition and initial state check, but not the for-loop over
N=1..3. Without this, the implementer must infer:

- How N is passed to `run_single_iter.sh`
- How the loop terminates on HUMAN_SYNC (iter 3)
- How state transitions between iterations are checked (CAS at start of each iter)
- Whether `BATCH_ID` is constructed inside `run_pipeline.sh` or passed as arg

**Fix**: Add a code sketch for `run_pipeline.sh`'s iteration loop to section 12.3, similar
to how `run_single_iter.sh` is fully sketched in section 6.2.

---

### O6 — Worktree cleanup owner not specified (Low)

**Not previously flagged**

Open Decision #3 defers worktree retention, but the synthesis orchestrator's HUMAN_SYNC
distillation (Section 5.1 step 6f) says "Clean up `memory/direction_iter*.md` and other
ephemeral files" — "other ephemeral files" is vague. The worktree directories at
`PROJECT_DIR/.claude/worktrees/iter{N}-{BATCH_ID}/` are ephemeral and should be pruned
after merge. No component is assigned this responsibility.

**Fix**: Assign worktree cleanup to `run_pipeline.sh` after each successful merge (step 7d):
```bash
git -C "${PROJECT_DIR}" worktree remove "${WORKTREE}" --force
```
Or defer to HUMAN_SYNC with a `git worktree prune` step. Either way, specify the owner.

---

## Implementability Assessment

| Component | Spec completeness | Blockers |
|-----------|-------------------|---------|
| ML code (ml/) | Complete | None — config, features, gates, monotone constraints all specified |
| Shell infrastructure (agents/) | Mostly complete | O4 (mkdir handoff dir); O5 (run_pipeline.sh loop) |
| Agent prompts | Mostly complete | O1 (reviewer_codex.md must be written separately) |
| State utils (state_utils.sh) | Complete | R3 fix (WORKER_FAILED reset) |
| Watchdog | Complete | None |
| Test infrastructure | Partial | Need --dry-run flag in launch scripts; need test_arg_parser.sh, test_guards.sh |

**Overall**: The design is implementable. The remaining issues are concentrated in shell
infrastructure plumbing (not architectural) and one prompt spec gap. None require a
design revision — they are implementation guidance gaps that can be resolved with small
additions to the existing documents.

---

## Recommended Actions Before Implementation

Ordered by priority:

1. **Resolve O1**: Write `reviewer_codex.md` prompt spec in section 10.3 (or as a separate
   section), showing stdout-only output with no file-write instructions.

2. **Resolve O2**: State v0 baseline strategy explicitly (retrain vs copy).

3. **Fix R3 + O3**: Add `export WORKER_FAILED=0` reset at start of each iteration.

4. **Resolve O4**: Add `mkdir -p handoff/${BATCH_ID}/iter${N}` to controller step 2.

5. **Resolve O5**: Add `run_pipeline.sh` loop sketch to section 12.3.

6. **Fix test 3.8**: Add Brier score (lower-direction) test case.

7. **Add --dry-run flag** to launch scripts and update Part 2 checklist accordingly.

8. After all above: proceed to Tier 0 → Tier 1 → Tier 2 → Tier 3 → Tier 4 → Tier 5
   in sequence per verification plan Part 4.
