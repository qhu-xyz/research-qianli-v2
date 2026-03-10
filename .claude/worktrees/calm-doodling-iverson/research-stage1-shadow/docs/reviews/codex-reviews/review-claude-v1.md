# Design Review: Agentic ML Research Pipeline (v8)

**Document reviewed**: `docs/plans/2026-02-26-agentic-ml-pipeline-design.md`
**Reviewer**: Claude Sonnet 4.6
**Date**: 2026-02-26
**Verdict**: PASS_WITH_NOTES — structurally sound; 4 issues require resolution before first batch

---

## Overall Assessment

This is a well-engineered design. Version 8 shows sustained iteration: the single-writer state machine, CAS checks, sha256 handshake verification, tiered memory, and human-write-only gates all reflect serious thinking about failure modes. The design will work as written for the happy path. The issues below are concentrated in edge cases, shell scripting details, and two specification gaps that will cause real problems if unaddressed.

---

## Critical Issues (must fix before first run)

### C1: Shell injection in agent launch scripts

**Section 5.1, 5.2, 5.3, 5.4** — all agent launches use:

```bash
claude -p "$(cat ${PROMPT})"
```

If any prompt file or memory file (which itself is `cat`-ed into a prompt) contains a double-quote, backtick, or `$()` construct, the shell will misparse the command. With LLM-generated content written into `memory/` files — especially `direction_iter{N}.md` — this is not a hypothetical. An orchestrator that generates a direction with a quoted Python string or a shell example will silently corrupt the worker's prompt on the next iteration.

**Fix**: Use a temp file or here-process-substitution:

```bash
PROMPT_FILE="${PROJECT_DIR}/agents/prompts/orchestrator_plan.md"
claude --print --model opus \
  --allowedTools "Read,Write,Edit,Glob,Grep,Bash" \
  < "${PROMPT_FILE}"
```

Or, if `claude` supports `--file` for the prompt, use that. At minimum, test that a prompt containing `"` and `$(date)` survives the launch script unchanged.

---

### C2: Gate auto-population is unspecified (Section 7.3 gap)

Section 7.3 states: "After v0 runs, the controller auto-populates v0-relative floors in `gates.json`." But the iteration loop in Section 6.2 (`run_single_iter.sh`, steps 1–18) has no step for this. It is not in `ml/registry.py`'s listed responsibilities, and it is not listed as a manual action in the Open Decisions section.

This is a blocker: no candidate version can be promoted until `null` floors are resolved, but nobody is responsible for resolving them. Promotion will silently block forever if this step is omitted.

**Fix**: Add an explicit step after iteration 0 (v0 baseline): "If `gates.json` has any `"pending_v0": true` entries, read v0's `metrics.json`, compute relative floors, write back, and set `"pending_v0": false`." Assign this to the controller (it's deterministic) or to a separate one-time script called by `run_pipeline.sh` before iteration 1.

---

### C3: Codex reviewer prompt mismatch (Section 5.5 / 10.3)

Section 10.3 shows the reviewer prompt with `WRITE: reviews/{batch_id}_iter{N}_{reviewer_id}.md`. Section 5.5 specifies the Codex launch as:

```bash
codex exec --sandbox read-only ... > ${REVIEW_FILE}
```

With `--sandbox read-only`, Codex cannot write to the filesystem. The review is captured via stdout redirect. This is technically correct — but Section 5.5 also says "Same format and mandatory sections as the Claude reviewer," implying the same prompt structure. The Claude reviewer prompt instructs the model to _write a file_. If `reviewer_codex.md` is a copy of `reviewer_claude.md` with only the file-write instruction, the model will attempt to write the file (which the sandbox will reject) rather than printing to stdout.

This will silently produce an empty or error-filled review file.

**Fix**: `reviewer_codex.md` must explicitly instruct Codex to print the review to stdout, not write a file. The current prompt design (Section 10.3) must be diverged for the two reviewers. Add a note in the design distinguishing `reviewer_claude.md` (file write) from `reviewer_codex.md` (stdout print).

---

### C4: `run_single_iter.sh` lock dependency is implicit

`run_pipeline.sh` acquires `state.lock` via `flock` (Section 12.3), then calls `run_single_iter.sh` as a subprocess. The CAS safety in `run_single_iter.sh` depends entirely on the lock being held. If someone calls `run_single_iter.sh` directly (for debugging, re-running a failed iteration), there is no lock — and two concurrent writers can occur.

**Fix**: `run_single_iter.sh` should either re-acquire the lock itself with `flock -n 9 || exit 1`, or add a guard: `[[ -n "$PIPELINE_LOCKED" ]] || { echo "ERROR: must be called via run_pipeline.sh"; exit 1; }` (with `run_pipeline.sh` exporting `PIPELINE_LOCKED=1`).

---

## Significant Issues (should fix before production)

### S1: Bash associative array with embedded variables

Section 8 (`state_utils.sh`) defines:

```bash
declare -A EXPECTED_ARTIFACT_PATTERN=(
  ["WORKER_RUNNING"]="registry/${VERSION_ID}/metrics.json"
  ...
)
```

In bash, this declaration is evaluated once at definition time. If `${VERSION_ID}` is not yet set when the array is declared (e.g., `state_utils.sh` is sourced at the top of `run_single_iter.sh` before the version_id is allocated), the value becomes `"registry//metrics.json"` — a literal empty path. The verification function will always fail.

**Fix**: Construct expected paths dynamically in `verify_handoff()` by reading `VERSION_ID` from `state.json` at call time, not from a pre-declared array.

---

### S2: Worker commit check is weak

Step 7b verifies the worker committed by checking `MAIN_HEAD != BRANCH_HEAD`. This passes if the worker committed _anything_ — including a garbage commit or a commit that deletes files. The sha256 check on `metrics.json` (step 8) is the real guard, but it happens after step 7c (the merge). If sha256 fails post-merge, we've already merged bad content into main.

**Fix**: Reorder — run sha256 check before the merge (verify `registry/${VERSION_ID}/metrics.json` in the worktree branch, not in main). Only merge if sha256 passes.

---

### S3: `direction_iter{N}.md` stale files across batches

At HUMAN_SYNC (iter 3 synthesis), the orchestrator cleans up `memory/direction_iter*.md`. If the orchestrator crashes mid-cleanup, the next batch inherits stale direction files from the prior batch. The worker reads `direction_iter{N}.md` where N is the current iteration — if `direction_iter1.md` from the prior batch exists, the new batch's iter-1 worker will read it until the orchestrator overwrites it.

**Fix**: Add a pre-batch cleanup step in `run_pipeline.sh` (before iteration 1 of each batch): `rm -f memory/direction_iter*.md`. This is a safe defensive measure — the orchestrator should overwrite anyway, but this eliminates the staleness window.

---

### S4: Codex model ID hardcoded throughout

`gpt-5.3-codex` appears hardcoded in Section 5.5 launch script. Section 16.1 flags this as an open decision. The model ID should be an environment variable in `config.sh` (e.g., `CODEX_MODEL="gpt-5.3-codex"`), so it can be changed without touching the launch script. This is especially important for an open decision that hasn't been confirmed.

---

### S5: HUMAN_SYNC notification is passive

Section 16.4 identifies this as an open decision, but it's more consequential than optional. When the pipeline reaches HUMAN_SYNC, it writes to `state.json` and prints to stdout — which is a tmux pane that the human may not be watching. A 3-iteration batch may complete while the human is away. Without active notification (Slack webhook, email, `wall` command, etc.), the pipeline silently stalls at HUMAN_SYNC until the human happens to check. Since the whole point of the design is to run autonomously, the notification mechanism is load-bearing. This should be resolved before the first real batch, not deferred.

---

## Minor Issues

### M1: Orchestrator planning phase step numbering

Section 5.1 planning steps are numbered 1, 2, 3, **5**, 6, 7 — step 4 is missing. This appears to be an editing artifact from a prior revision. Not a functional issue but will cause confusion when referencing the spec during implementation.

---

### M2: Codex handoff schema inconsistency

The Codex launch script (Section 5.5) writes the handoff JSON via a shell `jq -n` command, including a `producer` field. The canonical schema (Section 8) shows `{agent, producer, batch_id, ...}` — `producer` and `agent` are both present but always identical (both are `"codex_reviewer"`). The Claude reviewer writes its own handoff; the schema may differ subtly. Confirm that `verify_handoff()` doesn't check for `producer` (it only checks `artifact_path` and `sha256`), or drop the redundant `producer` field.

---

### M3: `noise_tolerance` applied uniformly to Group B

Section 7.4 applies `noise_tolerance = 0.02` uniformly to all gates. Group B gates (recall, CAP@K) are threshold-dependent. When the threshold optimization method changes (Section 7.5: Group B re-evaluation required), the v0-relative comparison is stale. At that transition point, applying `noise_tolerance` against the old baseline is misleading — the floor should be recomputed before tolerance is applied. The spec acknowledges this in Section 7.5 but doesn't update the promotion logic in Section 7.4 to guard against it.

---

### M4: `all_critiques.md` archive when Codex fails

Section 9 (Tier 3 archive) includes `${batch_id}/all_critiques.md` defined as "both reviewer outputs verbatim." If Codex times out or crashes for all three iterations of a batch, only Claude's review exists. The orchestrator synthesis step should handle this gracefully (write a note like "Codex unavailable" in the critiques archive rather than failing the archive step).

---

### M5: Watchdog probes Orchestrator states but doesn't kill on ORCHESTRATOR_PLANNING/SYNTHESIZING crash

Section 11.2: The immediate crash detection block is:

```bash
if [[ "$SESSION_ALIVE" == "false" ]] && [[ "$HANDOFF_EXISTS" == "false" ]] \
   && [[ "$state" == "REVIEW_CODEX" || "$state" == "REVIEW_CLAUDE" || "$state" == "WORKER_RUNNING" ]]; then
```

Orchestrator states are excluded from immediate crash detection. If the orchestrator crashes (tmux dies, auth error), the watchdog waits the full 600s timeout before writing a timeout artifact. For the REVIEW states this is 0s lag (detected immediately). For orchestrator states it's up to 600s. This is probably acceptable given the orchestrator timeout is only 600s anyway — but the design should make this asymmetry explicit.

---

## Design Strengths (for completeness)

1. **Single-writer state machine**: Clean. Eliminates the class of bugs where two components disagree on state.
2. **sha256 verification of all handshake artifacts**: Prevents stale artifact reads from prior iterations.
3. **gates.json and evaluate.py as human-write-only**: Critical for research integrity — ensures metric drift is a human decision.
4. **Tiered memory (hot/warm/archive)**: Context-window-aware and practical. The distillation step at HUMAN_SYNC is well thought out.
5. **Deterministic comparison step before AI review**: Reviewers see a pre-computed, non-AI comparison table. This prevents hallucinated metric comparisons.
6. **Smoke test mode**: First-class operational feature. Without this, pipeline debugging would require real data and real agent runs.
7. **Graceful Codex degradation**: Pipeline continues with Claude-only review on Codex failure. Correct design choice.
8. **Group B gate re-evaluation protocol (Section 7.5)**: Explicitly handles the threshold-dependency problem that most ML pipelines silently get wrong.

---

## Pre-Run Checklist

Before executing the first batch:

- [ ] Resolve C1: Test prompt embedding with shell-unsafe characters
- [ ] Resolve C2: Implement gate auto-population (assign to controller, not human)
- [ ] Resolve C3: Write diverged `reviewer_codex.md` prompt (stdout, not file write)
- [ ] Resolve C4: Add lock guard to `run_single_iter.sh`
- [ ] Fix S1: Rewrite `EXPECTED_ARTIFACT_PATTERN` as a dynamic function
- [ ] Fix S2: Move sha256 check before merge in step 7b/7c
- [ ] Fix S4: Move Codex model ID to `config.sh` as `CODEX_MODEL`
- [ ] Resolve S5: Add active HUMAN_SYNC notification (even a simple `wall` or log alert)
- [ ] Confirm Codex model ID on subscription before setting `config.sh`
- [ ] Run `test_pipeline_integrity.sh --iterations 3` (full 3-iter path, not just 1-iter smoke)

---

*Review completed: 2026-02-26*
