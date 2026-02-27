#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

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
source "${SCRIPT_DIR}/state_utils.sh"
EXPECTED_ARTIFACT=$(get_expected_artifact "$STATE")
HANDOFF_DIR="${PROJECT_DIR}/handoff/${BATCH_ID_VAL}/iter${ITER}"
HANDOFF_EXISTS="false"
if [[ -n "$EXPECTED_ARTIFACT" && -f "${HANDOFF_DIR}/${EXPECTED_ARTIFACT}" ]]; then
  HANDOFF_EXISTS="true"
fi

TIMEOUT_FILE="${HANDOFF_DIR}/timeout_${STATE}.json"

if (( ELAPSED > MAX_S )) && [[ "$HANDOFF_EXISTS" == "false" ]]; then
  if [[ ! -f "$TIMEOUT_FILE" ]]; then
    mkdir -p "$HANDOFF_DIR"
    NOW_ISO=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    echo "{\"state\":\"$STATE\",\"elapsed_s\":$ELAPSED,\"max_s\":$MAX_S,\"detected_at\":\"$NOW_ISO\"}" > "$TIMEOUT_FILE"
    echo "[watchdog] Timeout detected: state=$STATE elapsed=${ELAPSED}s max=${MAX_S}s" >> "${PROJECT_DIR}/.logs/audit.jsonl"
  fi
fi

# Audit log
mkdir -p "${PROJECT_DIR}/.logs"
NOW_ISO=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
echo "{\"ts\":\"$NOW_ISO\",\"state\":\"$STATE\",\"elapsed_s\":$ELAPSED,\"handoff_exists\":$HANDOFF_EXISTS}" >> "${PROJECT_DIR}/.logs/audit.jsonl"
