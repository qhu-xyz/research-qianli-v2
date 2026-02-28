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

[[ -n "$SESSION" ]] || SESSION="rev-codex-${BATCH_ID}-iter${N}"

PROMPT_TEMPLATE="${PROJECT_DIR}/agents/prompts/reviewer_codex.md"
[[ -f "$PROMPT_TEMPLATE" ]] || { echo "ERROR: prompt not found: $PROMPT_TEMPLATE"; exit 1; }

# Use envsubst for targeted variable substitution (HP-5)
PROMPT_TEXT=$(BATCH_ID="${BATCH_ID}" N="${N}" VERSION_ID="${VERSION_ID}" \
  envsubst '${BATCH_ID} ${N} ${VERSION_ID}' \
  < "$PROMPT_TEMPLATE")

LOG="${PROJECT_DIR}/.logs/sessions/${SESSION}.log"
mkdir -p "$(dirname "$LOG")"

if [[ "$DRY_RUN" == "true" ]]; then
  echo "[DRY-RUN] cd ${PROJECT_DIR}"
  echo "[DRY-RUN] tmux new-session -d -s $SESSION"
  echo "[DRY-RUN] codex exec --model ${CODEX_MODEL} --sandbox read-only '<prompt>'"
  exit 0
fi

# Write expanded prompt to a temp file (codex takes positional arg, not stdin)
PROMPT_FILE="${PROJECT_DIR}/.logs/sessions/${SESSION}_prompt.md"
echo "$PROMPT_TEXT" > "$PROMPT_FILE"

# RT-9: Codex needs workspace-write to create handoff/review files (read-only prevents writes)
# Codex needs the prompt as a positional argument; use the file content
TMUX_CMD="cd '${PROJECT_DIR}' && codex exec --model '${CODEX_MODEL}' --sandbox workspace-write \"\$(cat '${PROMPT_FILE}')\" > '${LOG}' 2>&1; echo 'EXIT_CODE='\$? >> '${LOG}'"

tmux new-session -d -s "$SESSION" "$TMUX_CMD"
echo "[launch_reviewer_codex] Started session: $SESSION (log=$LOG)"
