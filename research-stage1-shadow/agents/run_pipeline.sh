#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"
source "${SCRIPT_DIR}/state_utils.sh"

# Parse args
BATCH_NAME=""
MAX_ITER=3
while [[ $# -gt 0 ]]; do
  case "$1" in
    --batch-name)   BATCH_NAME="$2"; shift 2 ;;
    --batch-name=*) BATCH_NAME="${1#--batch-name=}"; shift ;;
    --max-iter)     MAX_ITER="$2"; shift 2 ;;
    --max-iter=*)   MAX_ITER="${1#--max-iter=}"; shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

# Validate batch name
if [[ -n "$BATCH_NAME" ]] && ! [[ "$BATCH_NAME" =~ ^[a-zA-Z0-9_-]+$ ]]; then
  echo "ERROR: --batch-name must be alphanumeric, hyphens, or underscores only (got: '$BATCH_NAME')" >&2
  exit 1
fi

if [[ -n "$BATCH_NAME" ]]; then
  BATCH_ID="${BATCH_NAME}-$(date +%Y%m%d-%H%M%S)"
else
  BATCH_ID="batch-$(date +%Y%m%d-%H%M%S)"
fi
export BATCH_ID

echo "=== Pipeline starting: ${BATCH_ID} ==="

# Acquire exclusive lock
exec 200>"${PROJECT_DIR}/state.lock"
flock -n 200 || { echo "ERROR: another pipeline is running (state.lock held)"; exit 1; }
export PIPELINE_LOCKED=true

# HUMAN_SYNC reset (CB-4)
current_state=$(jq -r '.state' "$STATE_FILE")
if [[ "$current_state" == "HUMAN_SYNC" ]]; then
  cas_transition HUMAN_SYNC IDLE '{"batch_id":null,"iteration":0}'
fi

# v0 guard
if [[ ! -d "${PROJECT_DIR}/registry/v0" ]]; then
  echo "ERROR: v0 baseline not found. Run Task 26 first." >&2
  exit 1
fi

# Defensive cleanup: remove stale direction files from prior crashed batch (design §12.3)
rm -f "${PROJECT_DIR}/memory/direction_iter"*.md

# Set up Python environment for the entire pipeline
cd "$PROJECT_DIR"
export PYTHONPATH="${PROJECT_DIR}"
source "$VENV_ACTIVATE"

# Check gates populated
PENDING=$(jq '[.gates | to_entries[] | select(.value.pending_v0 == true)] | length' "${PROJECT_DIR}/registry/gates.json")
if (( PENDING > 0 )); then
  echo "Running populate_v0_gates.py..."
  python ml/populate_v0_gates.py
fi

# Iteration loop (default 3, override with --max-iter)
for N in $(seq 1 "$MAX_ITER"); do
  export N
  echo ""
  echo "=============================="
  echo "=== Starting iteration ${N} ==="
  echo "=============================="
  bash "${SCRIPT_DIR}/run_single_iter.sh"
done

# Post-loop cleanup
for i in $(seq 1 "$MAX_ITER"); do
  WT="${PROJECT_DIR}/.claude/worktrees/iter${i}-${BATCH_ID}"
  [[ -d "$WT" ]] && git worktree remove "$WT" --force 2>/dev/null || true
done

echo "=== Pipeline complete: ${BATCH_ID} ==="
echo "Final state: $(jq -r '.state' "$STATE_FILE")"
