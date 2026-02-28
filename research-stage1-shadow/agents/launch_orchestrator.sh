#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

[[ -n "${BATCH_ID:-}" && -n "${N:-}" ]] || { echo "ERROR: BATCH_ID and N must be set"; exit 1; }

PHASE="plan"
SESSION=""
DRY_RUN=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --phase)     PHASE="$2"; shift 2 ;;
    --phase=*)   PHASE="${1#--phase=}"; shift ;;
    --session)   SESSION="$2"; shift 2 ;;
    --session=*) SESSION="${1#--session=}"; shift ;;
    --dry-run)   DRY_RUN=true; shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

[[ "$PHASE" == "plan" || "$PHASE" == "synthesize" ]] || { echo "ERROR: --phase must be 'plan' or 'synthesize'"; exit 1; }
[[ -n "$SESSION" ]] || SESSION="orch-${BATCH_ID}-iter${N}"

PROMPT="${PROJECT_DIR}/agents/prompts/orchestrator_${PHASE}.md"
[[ -f "$PROMPT" ]] || { echo "ERROR: prompt not found: $PROMPT"; exit 1; }

LOG="${PROJECT_DIR}/.logs/sessions/${SESSION}.log"
mkdir -p "$(dirname "$LOG")"

if [[ "$DRY_RUN" == "true" ]]; then
  echo "[DRY-RUN] cd ${PROJECT_DIR}"
  echo "[DRY-RUN] tmux new-session -d -s $SESSION"
  if [[ "$PHASE" == "synthesize" ]]; then
    echo "[DRY-RUN] { echo WORKER_FAILED=${WORKER_FAILED:-0}; cat $PROMPT; } | claude --print --model opus --dangerously-skip-permissions"
  else
    echo "[DRY-RUN] claude --print --model opus --dangerously-skip-permissions < $PROMPT"
  fi
  exit 0
fi

# Build the tmux command with full env setup
# RT-12: timeout wrapper = OS-level hard kill if agent loops forever
if [[ "$PHASE" == "synthesize" ]]; then
  HARD_TIMEOUT="${TIMEOUT_SYNTHESIZER}"
  TMUX_CMD="cd '${PROJECT_DIR}' && export PYTHONPATH='${PROJECT_DIR}' && source '${VENV_ACTIVATE}' && { echo 'WORKER_FAILED=${WORKER_FAILED:-0}'; cat '${PROMPT}'; } | timeout ${HARD_TIMEOUT} claude --print --model opus --dangerously-skip-permissions > '${LOG}' 2>&1; echo 'EXIT_CODE='\$? >> '${LOG}'"
else
  HARD_TIMEOUT="${TIMEOUT_ORCHESTRATOR}"
  TMUX_CMD="cd '${PROJECT_DIR}' && export PYTHONPATH='${PROJECT_DIR}' && source '${VENV_ACTIVATE}' && timeout ${HARD_TIMEOUT} claude --print --model opus --dangerously-skip-permissions < '${PROMPT}' > '${LOG}' 2>&1; echo 'EXIT_CODE='\$? >> '${LOG}'"
fi

tmux new-session -d -s "$SESSION" "$TMUX_CMD"
echo "[launch_orchestrator] Started session: $SESSION (phase=$PHASE, log=$LOG)"
