# Critical Review v10: Agentic ML Pipeline Design

Document reviewed: `docs/plans/2026-02-26-agentic-ml-pipeline-design.md` (v10)
Verification plan reviewed: `docs/plans/2026-02-26-verification-plan.md`
Review date: 2026-02-26

---

## Resolution status of prior findings

All findings from cross-document synthesis review (O1–O6) and verification plan reviews confirmed resolved:

| Finding | Fix | Status |
|---------|-----|--------|
| O1: reviewer_codex.md stdout-only spec missing from §10.3 | Section 10.4 added with explicit stdout WHEN DONE | RESOLVED |
| O2: v0 baseline strategy ambiguous | §7.3 now specifies retrain with source config + fixed random_state | RESOLVED |
| O3: WORKER_FAILED not reset between iterations | Step 0 `export WORKER_FAILED=0`; step 7a `export WORKER_FAILED=1` | RESOLVED |
| O4: handoff dir never created | Step 2 `mkdir -p handoff/${BATCH_ID}/iter${N}` | RESOLVED |
| O5: run_pipeline.sh loop body unspecified | Full loop sketch added to §12.3 | RESOLVED |
| O6: worktree cleanup owner unspecified | Deferred to Open Decision #3 (acceptable) |DEFERRED |
| Codex handoff absolute path vs relative in verify_handoff | REVIEW_FILE_REL/REVIEW_FILE_ABS split | RESOLVED |
| R5: check_clis.sh hardcoded CODEX_MODEL | Uses `${CODEX_MODEL}` now | RESOLVED |

---

## New findings

### 1) run_pipeline.sh loop does not pass `--batch-name` through to BATCH_ID (Medium)

**Where**: §12.3 `run_pipeline.sh` loop sketch, line:
```bash
BATCH_NAME="${1:---batch-$(date +%Y%m%d-%H%M%S)}"
BATCH_ID="batch-$(date +%Y%m%d-%H%M%S)"
```

`BATCH_NAME` is set from the arg but `BATCH_ID` is always auto-generated from timestamp, ignoring `BATCH_NAME`. The user-provided name (`--batch-name "lower-threshold-04"`) is captured but never used in `BATCH_ID`. This means directory names, session names, and handoff paths all get auto-generated IDs regardless of what the user specified.

**Severity**: Medium — not a correctness bug, but the `--batch-name` argument is advertised in §6.1 as a user feature and silently ignored.

**Fix**: Either:
- Use `BATCH_NAME` as the full `BATCH_ID` (with timestamp suffix for uniqueness): `BATCH_ID="${BATCH_NAME}-$(date +%Y%m%d-%H%M%S)"`
- Or drop `--batch-name` from §6.1 as an advertised feature if it's not used

---

### 2) `run_pipeline.sh` loop breaks on HUMAN_SYNC but doesn't handle early IDLE (Low)

**Where**: §12.3 loop body:
```bash
[[ "$current_state" == "HUMAN_SYNC" ]] && break
```

If `run_single_iter.sh` exits with a non-zero code (e.g., CAS failure, flock error) and the state is still IDLE or ORCHESTRATOR_PLANNING, the loop continues to the next N without investigation. The `set -euo pipefail` at the top should catch this, but only if `run_single_iter.sh` propagates its exit code correctly (a subshell call via `bash agents/run_single_iter.sh` will propagate the exit code under `set -e`).

**Severity**: Low — `set -euo pipefail` at the top of `run_pipeline.sh` handles this correctly. Noting for implementer awareness: do not suppress exit codes from `run_single_iter.sh`.

---

### 3) Section 10.4 `reviewer_codex.md` READ list references variables not substituted at prompt-write time (Low)

**Where**: §10.4 prompt spec shows:
```
- registry/${VERSION_ID}/changes_summary.md
- reports/{batch_id}_iter{N}_comparison.md
```

The prompt is a static file (`reviewer_codex.md`) written once. At read time, `${VERSION_ID}`, `{batch_id}`, `{N}` are template placeholders — they must be substituted before the prompt reaches Codex. The Codex launch script passes the prompt as a positional arg via `$(cat reviewer_codex.md)` — which reads the file verbatim.

This is the same for `reviewer_claude.md` (§10.3) and the other prompts. All prompts use `{N}`, `{batch_id}`, `${VERSION_ID}` as placeholders, but the design never specifies how these are substituted.

**Severity**: Low — this is a known pattern that the agent itself would handle by reading from `state.json` (as instructed by the prompt body context). However, the prompt spec should either: (a) note that `${VERSION_ID}` etc. are placeholders the agent must resolve from `state.json`, or (b) describe a template substitution step in the launch script.

---

## Missing test coverage in verification plan (not blocking)

The following cases exist in the design but have no corresponding test:

1. `version_counter.json` flock correctness — concurrent calls to `registry.py`'s version allocation could race. No test for this.
2. `run_pipeline.sh` v0 guard assertion — test that abort occurs when `registry/v0/metrics.json` is absent. The verification plan covers `populate_v0_gates.py` missing v0 (test 3.8 #4) but not the run_pipeline.sh guard itself.
3. Worktree branch name collision — if a crash leaves `iter1-batch-20260226-001` branch behind and the same batch_id is reused. Not testable without batch_id collision simulation, and the timestamp-based batch_id makes collision unlikely in practice.

---

## Overall assessment

v10 resolves all previously identified blockers. The design is now implementable end-to-end without ambiguity on the critical path. The three new findings above are medium/low severity and do not block implementation:

- Finding #1 (`--batch-name` ignored) should be fixed before first use (trivial)
- Finding #2 (early exit handling) is handled by `set -e` as-is
- Finding #3 (prompt placeholder substitution) should be documented but doesn't change behavior

**Recommendation**: Proceed to implementation. Address finding #1 (trivial one-liner) during implementation. Document finding #3's resolution pattern in the prompt files themselves (add a note like "Agents resolve VERSION_ID by reading state.json directly").
