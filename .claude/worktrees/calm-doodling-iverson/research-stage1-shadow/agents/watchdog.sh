#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"
source "${SCRIPT_DIR}/state_utils.sh"

# RT-12: Watchdog with recovery logic.
# Detects stuck states, kills orphaned agent tmux sessions, resets state.
# Safe to run from cron every 5 minutes, or manually.

STATE=$(jq -r '.state' "$STATE_FILE")
if [[ "$STATE" == "IDLE" || "$STATE" == "HUMAN_SYNC" ]]; then
  exit 0
fi

ENTERED_AT=$(jq -r '.entered_at' "$STATE_FILE")
MAX_S=$(jq -r '.max_seconds // 0' "$STATE_FILE")
BATCH_ID_VAL=$(jq -r '.batch_id // empty' "$STATE_FILE")
ITER=$(jq -r '.iteration // 0' "$STATE_FILE")

if [[ -z "$ENTERED_AT" || "$MAX_S" == "0" ]]; then
  exit 0
fi

NOW=$(date -u +%s)
ENTERED_EPOCH=$(date -u -d "$ENTERED_AT" +%s 2>/dev/null || echo 0)
ELAPSED=$(( NOW - ENTERED_EPOCH ))

# Check if handoff already exists (SB-3 guard)
EXPECTED_ARTIFACT=$(get_expected_artifact "$STATE")
HANDOFF_DIR="${PROJECT_DIR}/handoff/${BATCH_ID_VAL}/iter${ITER}"
HANDOFF_EXISTS="false"
if [[ -n "$EXPECTED_ARTIFACT" && -f "${HANDOFF_DIR}/${EXPECTED_ARTIFACT}" ]]; then
  HANDOFF_EXISTS="true"
fi

TIMEOUT_FILE="${HANDOFF_DIR}/timeout_${STATE}.json"

# Phase 1: Detect timeout and write timeout artifact (for poll_for_handoff to pick up)
if (( ELAPSED > MAX_S )) && [[ "$HANDOFF_EXISTS" == "false" ]]; then
  if [[ ! -f "$TIMEOUT_FILE" ]]; then
    mkdir -p "$HANDOFF_DIR"
    NOW_ISO=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    echo "{\"state\":\"$STATE\",\"elapsed_s\":$ELAPSED,\"max_s\":$MAX_S,\"detected_at\":\"$NOW_ISO\"}" > "$TIMEOUT_FILE"
    echo "[watchdog] Timeout detected: state=$STATE elapsed=${ELAPSED}s max=${MAX_S}s" >&2
  fi
fi

# Phase 2: Recovery — if stuck for 2x max_seconds, the controller is probably dead.
# Kill orphaned agent tmux sessions and reset state to IDLE.
RECOVERY_THRESHOLD=$(( MAX_S * 2 ))
if (( ELAPSED > RECOVERY_THRESHOLD )); then
  echo "[watchdog] RECOVERY: state=$STATE stuck for ${ELAPSED}s (threshold=${RECOVERY_THRESHOLD}s)" >&2

  # Kill any tmux sessions matching this batch
  for pattern in "orch-${BATCH_ID_VAL}" "worker-${BATCH_ID_VAL}" "rev-claude-${BATCH_ID_VAL}" "rev-codex-${BATCH_ID_VAL}" "synth-${BATCH_ID_VAL}" "pipeline-"; do
    for session in $(tmux ls 2>/dev/null | grep -F "$pattern" | cut -d: -f1); do
      tmux kill-session -t "$session" 2>/dev/null && echo "[watchdog] Killed orphan session: $session" >&2 || true
    done
  done

  # Reset state to IDLE with error record
  NOW_ISO=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
  jq --arg s "IDLE" --arg t "$NOW_ISO" --arg err "watchdog recovery: state=$STATE stuck ${ELAPSED}s" \
    '.state = $s | .entered_at = $t | .error = $err' "$STATE_FILE" > "${STATE_FILE}.tmp.$$"
  mv "${STATE_FILE}.tmp.$$" "$STATE_FILE"

  # Release lock if held
  rm -f "${PROJECT_DIR}/state.lock"

  echo "[watchdog] State reset to IDLE. Manual intervention required to restart batch." >&2
fi

# Audit log
mkdir -p "${PROJECT_DIR}/.logs"
NOW_ISO=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
echo "{\"ts\":\"$NOW_ISO\",\"state\":\"$STATE\",\"elapsed_s\":$ELAPSED,\"handoff_exists\":$HANDOFF_EXISTS}" >> "${PROJECT_DIR}/.logs/audit.jsonl"
