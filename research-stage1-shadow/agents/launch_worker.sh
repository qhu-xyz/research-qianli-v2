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

# Worker runs in worktree but needs PROJECT_DIR for absolute path access
TMUX_CMD="cd '${WT_DIR}' && export PROJECT_DIR='${PROJECT_DIR}' && export PYTHONPATH='${WT_DIR}' && export SMOKE_TEST='${SMOKE_TEST}' && source '${VENV_ACTIVATE}' && claude --print --model opus --dangerously-skip-permissions < '${PROMPT}' > '${LOG}' 2>&1; echo 'EXIT_CODE='\$? >> '${LOG}'"

tmux new-session -d -s "$SESSION" "$TMUX_CMD"
echo "[launch_worker] Started session: $SESSION (worktree=$WT_DIR, log=$LOG)"
