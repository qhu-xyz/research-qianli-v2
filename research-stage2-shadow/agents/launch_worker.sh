#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

[[ -n "${BATCH_ID:-}" && -n "${N:-}" && -n "${VERSION_ID:-}" ]] || { echo "ERROR: BATCH_ID, N, VERSION_ID must be set"; exit 1; }

SESSION=""
DRY_RUN=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --session)   SESSION="$2"; shift 2 ;;
    --session=*) SESSION="${1#--session=}"; shift ;;
    --dry-run)   DRY_RUN=true; shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

[[ -n "$SESSION" ]] || SESSION="worker-${BATCH_ID}-iter${N}"

WT_DIR="${PROJECT_DIR}/.claude/worktrees/iter${N}-${BATCH_ID}"
WT_BRANCH="worker-iter${N}-${BATCH_ID}"
PROMPT="${PROJECT_DIR}/agents/prompts/worker.md"
LOG="${PROJECT_DIR}/.logs/sessions/${SESSION}.log"

mkdir -p "$(dirname "$WT_DIR")"
mkdir -p "$(dirname "$LOG")"

if [[ "$DRY_RUN" == "true" ]]; then
  echo "[DRY-RUN] cd ${PROJECT_DIR}"
  echo "[DRY-RUN] git worktree add ${WT_DIR} -b ${WT_BRANCH}"
  echo "[DRY-RUN] tmux new-session -d -s $SESSION"
  echo "[DRY-RUN] claude --print --model opus --dangerously-skip-permissions < $PROMPT"
  exit 0
fi

# Create worktree
cd "$PROJECT_DIR"
git worktree add "$WT_DIR" -b "$WT_BRANCH" 2>/dev/null || git worktree add "$WT_DIR" "$WT_BRANCH" 2>/dev/null || true

# RT-8: Compute worktree project subdir (monorepo: git root != PROJECT_DIR)
REPO_PREFIX=$(git rev-parse --show-prefix)  # e.g. "research-stage2-shadow/"
WT_PROJECT="${WT_DIR}/${REPO_PREFIX%/}"      # worktree's project subdirectory

# RT-14: Make HUMAN-WRITE-ONLY files read-only in worktree to prevent agent modification
for protected in "ml/evaluate.py" "registry/gates.json"; do
  if [[ -f "${WT_PROJECT}/${protected}" ]]; then
    chmod 444 "${WT_PROJECT}/${protected}"
  fi
done

# Worker runs in worktree's project subdir, needs PROJECT_DIR for absolute path access
# RT-12: timeout wrapper = OS-level hard kill if agent loops forever
# RT-16: unset CLAUDECODE to avoid "nested session" detection when launched from Claude Code
TMUX_CMD="cd '${WT_PROJECT}' && unset CLAUDECODE && export PROJECT_DIR='${PROJECT_DIR}' && export PYTHONPATH='${WT_PROJECT}' && export SMOKE_TEST='${SMOKE_TEST}' && source '${VENV_ACTIVATE}' && timeout ${TIMEOUT_WORKER} claude --print --model opus --dangerously-skip-permissions < '${PROMPT}' > '${LOG}' 2>&1; echo 'EXIT_CODE='\$? >> '${LOG}'"

tmux new-session -d -s "$SESSION" "$TMUX_CMD"
echo "[launch_worker] Started session: $SESSION (worktree=$WT_PROJECT, log=$LOG)"
