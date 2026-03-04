#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"
source "${SCRIPT_DIR}/state_utils.sh"

# PIPELINE_LOCKED guard
[[ "${PIPELINE_LOCKED:-}" == "true" ]] || { echo "ERROR: must be called from run_pipeline.sh (PIPELINE_LOCKED not set)"; exit 1; }

HANDOFF_DIR="${PROJECT_DIR}/handoff/${BATCH_ID}/iter${N}"
mkdir -p "$HANDOFF_DIR"

ITER_START=$SECONDS
echo "=== Iteration ${N} starting ==="

# Step 0: env vars already exported by run_pipeline.sh
export WORKER_FAILED=0

# Step 1: IDLE -> ORCHESTRATOR_PLANNING
cas_transition IDLE ORCHESTRATOR_PLANNING "{\"batch_id\":\"${BATCH_ID}\",\"iteration\":${N}}"

# Step 2: handoff dir already created above

# Step 3: launch orchestrator plan, poll
echo "[iter${N}] Launching orchestrator (plan phase)..."
SESSION="orch-${BATCH_ID}-iter${N}"
bash "${SCRIPT_DIR}/launch_orchestrator.sh" --phase plan --session "$SESSION"

set +e
poll_for_handoff "$HANDOFF_DIR" "orchestrator_plan_done.json" 420 15
POLL_RC=$?
set -e
if (( POLL_RC != 0 )); then
  kill_agent_tree "$SESSION"
  cas_transition ORCHESTRATOR_PLANNING IDLE "{\"error\":\"orchestrator plan timeout at iter ${N}\"}"
  echo "FATAL: orchestrator plan timed out at iter ${N}" >&2
  exit 1
fi

echo "[iter${N}] Orchestrator plan completed in $(( SECONDS - ITER_START ))s"
ORCH_LOG="${PROJECT_DIR}/.logs/sessions/${SESSION}.log"
[[ -f "$ORCH_LOG" ]] && echo "[iter${N}] --- last 5 lines of orchestrator plan log ---" && tail -5 "$ORCH_LOG" | sed 's/^/  | /' && echo "[iter${N}] --- end log snippet ---"

# RT-6: validate orchestrator handoff JSON
if ! jq empty "${HANDOFF_DIR}/orchestrator_plan_done.json" 2>/dev/null; then
  cas_transition ORCHESTRATOR_PLANNING IDLE "{\"error\":\"orchestrator plan handoff JSON malformed at iter ${N}\"}"
  echo "FATAL: orchestrator plan handoff JSON is malformed at iter ${N}" >&2
  exit 1
fi

# Step 4: ORCHESTRATOR_PLANNING -> WORKER_RUNNING
cas_transition ORCHESTRATOR_PLANNING WORKER_RUNNING "{\"max_seconds\":900}"

# Step 5: allocate version
VERSION_ID=$(python -c "from ml.registry import allocate_version_id; print(allocate_version_id('${PROJECT_DIR}/registry/version_counter.json'))")
# Update state with version_id
jq --arg v "$VERSION_ID" '.version_id = $v' "$STATE_FILE" > "${STATE_FILE}.tmp.$$" && mv "${STATE_FILE}.tmp.$$" "$STATE_FILE"
export VERSION_ID

# Step 5a: commit orchestrator outputs (CB-1 + HP-2)
git -C "$PROJECT_DIR" add memory/
if ! git -C "$PROJECT_DIR" diff --cached --quiet; then
  git -C "$PROJECT_DIR" commit -m "iter${N}: orchestrator plan"
fi

# Step 6: launch worker in worktree
echo "[iter${N}] Launching worker (version ${VERSION_ID})..."
WORKER_SESSION="worker-${BATCH_ID}-iter${N}"
bash "${SCRIPT_DIR}/launch_worker.sh" --session "$WORKER_SESSION"

set +e
poll_for_handoff "$HANDOFF_DIR" "worker_done.json" 3300 30
POLL_RC=$?
set -e
if (( POLL_RC != 0 )); then
  kill_agent_tree "$WORKER_SESSION"
  cas_transition WORKER_RUNNING IDLE "{\"error\":\"worker timeout at iter ${N}\"}"
  echo "FATAL: worker timed out at iter ${N}" >&2
  exit 1
fi

WORKER_ELAPSED=$(( SECONDS - ITER_START ))
echo "[iter${N}] Worker phase completed in ${WORKER_ELAPSED}s (cumulative)"
WORKER_LOG="${PROJECT_DIR}/.logs/sessions/${WORKER_SESSION}.log"
[[ -f "$WORKER_LOG" ]] && echo "[iter${N}] --- last 5 lines of worker log ---" && tail -5 "$WORKER_LOG" | sed 's/^/  | /' && echo "[iter${N}] --- end log snippet ---"

# Step 7a: check worker handoff (RT-6: validate JSON before parsing)
if ! jq empty "${HANDOFF_DIR}/worker_done.json" 2>/dev/null; then
  echo "[iter${N}] Worker handoff JSON is malformed — treating as failure" >&2
  WORKER_FAILED=1
else
  WORKER_STATUS=$(jq -r '.status' "${HANDOFF_DIR}/worker_done.json")
  if [[ "$WORKER_STATUS" == "failed" ]]; then
    echo "[iter${N}] Worker failed — skipping merge, compare, reviews"
    WORKER_FAILED=1
  fi
fi

if (( WORKER_FAILED == 0 )); then
  # RT-8: Compute worktree project subdir (monorepo: git root != PROJECT_DIR)
  WT_DIR="${PROJECT_DIR}/.claude/worktrees/iter${N}-${BATCH_ID}"
  REPO_PREFIX=$(cd "$PROJECT_DIR" && git rev-parse --show-prefix)
  WT_PROJECT="${WT_DIR}/${REPO_PREFIX%/}"

  # Step 7b: verify worker committed
  # Step 7c: sha256 verify (artifact is in worktree's project subdir, not main tree)
  if [[ -d "$WT_PROJECT" ]]; then
    pushd "$WT_PROJECT" > /dev/null
    verify_handoff "${HANDOFF_DIR}/worker_done.json" "WORKER_RUNNING"
    popd > /dev/null
  fi

  # Step 7d: pre-merge guard for HUMAN-WRITE-ONLY files + merge worktree
  if [[ -d "$WT_PROJECT" ]]; then
    # Check HUMAN-WRITE-ONLY files unchanged
    for protected in "ml/evaluate.py" "registry/gates.json"; do
      if ! diff -q "${PROJECT_DIR}/${protected}" "${WT_PROJECT}/${protected}" >/dev/null 2>&1; then
        echo "ERROR: worker modified HUMAN-WRITE-ONLY file: $protected" >&2
        WORKER_FAILED=1
        break
      fi
    done
    if (( WORKER_FAILED == 0 )); then
      # Merge worktree changes into main
      cd "$WT_DIR"
      WORKER_BRANCH=$(git rev-parse --abbrev-ref HEAD)
      cd "$PROJECT_DIR"
      # Ensure clean working tree before merge (RT-5: stale files from prior runs cause merge conflicts)
      # Exclude version_counter.json — it's incremented by allocate_version_id and must not be reset
      git -C "$PROJECT_DIR" diff --name-only | grep -v 'version_counter.json' | xargs -r git -C "$PROJECT_DIR" checkout -- 2>/dev/null || true
      if ! git merge "$WORKER_BRANCH" --no-edit -m "iter${N}: merge worker ${VERSION_ID}"; then
        echo "ERROR: merge failed for branch $WORKER_BRANCH" >&2
        git merge --abort 2>/dev/null || true
        WORKER_FAILED=1
      fi
    fi
  fi

  # Step 8: run compare.py
  if (( WORKER_FAILED == 0 )); then
    echo "[iter${N}] Running comparison..."
    mkdir -p "${PROJECT_DIR}/reports/${BATCH_ID}/iter${N}"
    cd "$PROJECT_DIR" && source "$VENV_ACTIVATE"
    set +e
    python ml/compare.py --batch-id "$BATCH_ID" --iteration "$N" \
      --output "reports/${BATCH_ID}/iter${N}/comparison.md"
    COMPARE_RC=$?
    set -e
    if (( COMPARE_RC != 0 )); then
      echo "[iter${N}] WARNING: compare.py exited with code ${COMPARE_RC} — reviews will proceed without comparison report" >&2
    fi
  fi
fi

if (( WORKER_FAILED == 0 )); then
  # Step 9: WORKER_RUNNING -> REVIEW_CLAUDE
  cas_transition WORKER_RUNNING REVIEW_CLAUDE "{\"max_seconds\":600}"

  # Step 10: launch Claude reviewer
  echo "[iter${N}] Launching Claude reviewer..."
  CLAUDE_SESSION="rev-claude-${BATCH_ID}-iter${N}"
  bash "${SCRIPT_DIR}/launch_reviewer_claude.sh" --session "$CLAUDE_SESSION"

  set +e
  poll_for_handoff "$HANDOFF_DIR" "reviewer_claude_done.json" 420 15
  POLL_RC=$?
  set -e
  if (( POLL_RC != 0 )); then
    kill_agent_tree "$CLAUDE_SESSION"
    cas_transition REVIEW_CLAUDE IDLE "{\"error\":\"Claude reviewer timeout at iter ${N}\"}"
    echo "FATAL: Claude reviewer timed out at iter ${N}" >&2
    exit 1
  fi

  echo "[iter${N}] Claude review completed in $(( SECONDS - ITER_START ))s (cumulative)"
  CLAUDE_LOG="${PROJECT_DIR}/.logs/sessions/${CLAUDE_SESSION}.log"
  [[ -f "$CLAUDE_LOG" ]] && echo "[iter${N}] --- last 5 lines of Claude reviewer log ---" && tail -5 "$CLAUDE_LOG" | sed 's/^/  | /' && echo "[iter${N}] --- end log snippet ---"

  # Step 11: verify Claude review
  verify_handoff "${HANDOFF_DIR}/reviewer_claude_done.json" "REVIEW_CLAUDE" || true

  # Step 12: REVIEW_CLAUDE -> REVIEW_CODEX
  cas_transition REVIEW_CLAUDE REVIEW_CODEX "{\"max_seconds\":600}"

  # Step 13: launch Codex reviewer
  echo "[iter${N}] Launching Codex reviewer..."
  CODEX_SESSION="rev-codex-${BATCH_ID}-iter${N}"
  bash "${SCRIPT_DIR}/launch_reviewer_codex.sh" --session "$CODEX_SESSION"

  set +e
  poll_for_handoff "$HANDOFF_DIR" "reviewer_codex_done.json" 420 15
  POLL_RC=$?
  set -e
  if (( POLL_RC != 0 )); then
    kill_agent_tree "$CODEX_SESSION"
    # Codex timeout is non-fatal — continue with synthesis
    echo "[iter${N}] Codex reviewer timed out — continuing"
  fi

  # Step 14: (poll complete or timeout)
  echo "[iter${N}] Codex review completed in $(( SECONDS - ITER_START ))s (cumulative)"
  CODEX_LOG="${PROJECT_DIR}/.logs/sessions/${CODEX_SESSION}.log"
  [[ -f "$CODEX_LOG" ]] && echo "[iter${N}] --- last 5 lines of Codex reviewer log ---" && tail -5 "$CODEX_LOG" | sed 's/^/  | /' && echo "[iter${N}] --- end log snippet ---"

  # Step 15: REVIEW_CODEX -> ORCHESTRATOR_SYNTHESIZING
  cas_transition REVIEW_CODEX ORCHESTRATOR_SYNTHESIZING "{\"max_seconds\":600}"
else
  # WORKER_FAILED path: skip reviews, go directly to synthesis
  cas_transition WORKER_RUNNING ORCHESTRATOR_SYNTHESIZING "{\"max_seconds\":600}"
fi

# Step 16: launch synthesis orchestrator
echo "[iter${N}] Launching orchestrator (synthesize phase)..."
SYNTH_SESSION="synth-${BATCH_ID}-iter${N}"
export WORKER_FAILED
bash "${SCRIPT_DIR}/launch_orchestrator.sh" --phase synthesize --session "$SYNTH_SESSION"

set +e
poll_for_handoff "$HANDOFF_DIR" "orchestrator_synth_done.json" 1200 15
POLL_RC=$?
set -e
if (( POLL_RC != 0 )); then
  kill_agent_tree "$SYNTH_SESSION"
  cas_transition ORCHESTRATOR_SYNTHESIZING IDLE "{\"error\":\"synthesis timeout at iter ${N}\"}"
  echo "FATAL: synthesis orchestrator timed out at iter ${N}" >&2
  exit 1
fi

echo "[iter${N}] Synthesis completed in $(( SECONDS - ITER_START ))s (cumulative)"
SYNTH_LOG="${PROJECT_DIR}/.logs/sessions/${SYNTH_SESSION}.log"
[[ -f "$SYNTH_LOG" ]] && echo "[iter${N}] --- last 5 lines of synthesis log ---" && tail -5 "$SYNTH_LOG" | sed 's/^/  | /' && echo "[iter${N}] --- end log snippet ---"

# RT-6: validate synthesis handoff JSON
if ! jq empty "${HANDOFF_DIR}/orchestrator_synth_done.json" 2>/dev/null; then
  echo "[iter${N}] WARNING: synthesis handoff JSON malformed — skipping promotion" >&2
fi

# Step 17: read synthesis decisions
PROMOTE=$(jq -r '.decisions.promote_version // empty' "${HANDOFF_DIR}/orchestrator_synth_done.json" 2>/dev/null || echo "")
if [[ -n "$PROMOTE" ]]; then
  echo "[iter${N}] Promoting version: $PROMOTE"
  cd "$PROJECT_DIR" && source "$VENV_ACTIVATE"
  python -c "from ml.registry import promote_version; promote_version('${PROJECT_DIR}/registry', '${PROMOTE}', '${PROJECT_DIR}/registry/champion.json')"
  # Update champion.md so next iteration's agents see current champion
  PROMOTE_METRICS=$(python -c "
import json
with open('${PROJECT_DIR}/registry/${PROMOTE}/metrics.json') as f:
    m = json.load(f)
agg = m.get('aggregate', {}).get('mean', m)
def g(k):
    v = agg.get(k, '?')
    return f'{v:.4f}' if isinstance(v, (int,float)) else str(v)
print(f'EV-VC@100={g(\"EV-VC@100\")}, EV-VC@500={g(\"EV-VC@500\")}, EV-NDCG={g(\"EV-NDCG\")}, Spearman={g(\"Spearman\")}')
" 2>/dev/null || echo "metrics unavailable")
  cat > "${PROJECT_DIR}/memory/hot/champion.md" << CHAMPEOF
# Champion

**Current champion: ${PROMOTE}** (promoted at iter${N} of batch ${BATCH_ID})

Key metrics (mean across eval months): ${PROMOTE_METRICS}

See \`registry/${PROMOTE}/metrics.json\` for full per-month breakdown.
CHAMPEOF
  echo "[iter${N}] Updated memory/hot/champion.md"
fi

# Step 17a: commit synthesis outputs (O-3 + HP-2)
git -C "$PROJECT_DIR" add memory/
if ! git -C "$PROJECT_DIR" diff --cached --quiet; then
  git -C "$PROJECT_DIR" commit -m "iter${N}: orchestrator synthesis"
fi

# Step 18: state transition
if (( N == 3 )); then
  cas_transition ORCHESTRATOR_SYNTHESIZING HUMAN_SYNC '{}'
else
  cas_transition ORCHESTRATOR_SYNTHESIZING IDLE '{}'
fi

ITER_ELAPSED=$(( SECONDS - ITER_START ))
echo "=== Iteration ${N} complete — total ${ITER_ELAPSED}s ($(( ITER_ELAPSED / 60 ))m$(( ITER_ELAPSED % 60 ))s) ==="
