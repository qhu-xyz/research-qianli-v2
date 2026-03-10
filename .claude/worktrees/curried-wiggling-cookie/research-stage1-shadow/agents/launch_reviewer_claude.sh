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

[[ -n "$SESSION" ]] || SESSION="rev-claude-${BATCH_ID}-iter${N}"

PROMPT="${PROJECT_DIR}/agents/prompts/reviewer_claude.md"
[[ -f "$PROMPT" ]] || { echo "ERROR: prompt not found: $PROMPT"; exit 1; }

LOG="${PROJECT_DIR}/.logs/sessions/${SESSION}.log"
mkdir -p "$(dirname "$LOG")"

if [[ "$DRY_RUN" == "true" ]]; then
  echo "[DRY-RUN] cd ${PROJECT_DIR}"
  echo "[DRY-RUN] tmux new-session -d -s $SESSION"
  echo "[DRY-RUN] claude --print --model opus --dangerously-skip-permissions < $PROMPT"
  exit 0
fi

# RT-12: timeout wrapper = OS-level hard kill if agent loops forever
TMUX_CMD="cd '${PROJECT_DIR}' && export PYTHONPATH='${PROJECT_DIR}' && source '${VENV_ACTIVATE}' && timeout ${TIMEOUT_REVIEWER_CLAUDE} claude --print --model opus --dangerously-skip-permissions < '${PROMPT}' > '${LOG}' 2>&1; echo 'EXIT_CODE='\$? >> '${LOG}'"

tmux new-session -d -s "$SESSION" "$TMUX_CMD"
echo "[launch_reviewer_claude] Started session: $SESSION (log=$LOG)"
