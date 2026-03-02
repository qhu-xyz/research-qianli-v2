#!/usr/bin/env bash
# State machine utilities for the agentic ML pipeline.
# Usage: source agents/state_utils.sh
# Self-test: bash agents/state_utils.sh test

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

# 6 valid states
VALID_STATES=(IDLE ORCHESTRATOR_PLANNING WORKER_RUNNING REVIEW_CLAUDE REVIEW_CODEX ORCHESTRATOR_SYNTHESIZING HUMAN_SYNC)

# State-to-handoff mapping
declare -A STATE_TO_HANDOFF=(
  [ORCHESTRATOR_PLANNING]="orchestrator_plan_done.json"
  [WORKER_RUNNING]="worker_done.json"
  [REVIEW_CLAUDE]="reviewer_claude_done.json"
  [REVIEW_CODEX]="reviewer_codex_done.json"
  [ORCHESTRATOR_SYNTHESIZING]="orchestrator_synth_done.json"
  [HUMAN_SYNC]=""
)

cas_transition() {
  # Compare-and-swap state transition
  # CRITICAL: temp file MUST be ${STATE_FILE}.tmp.$$ (same filesystem for atomic mv)
  local expected="$1" new_state="$2" new_fields="$3"
  local current
  current=$(jq -r '.state' "$STATE_FILE")
  if [[ "$current" != "$expected" ]]; then
    echo "CAS FAILED: expected=$expected actual=$current" >&2
    return 1
  fi
  local tmpfile="${STATE_FILE}.tmp.$$"
  local now
  now=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
  # Build new state JSON
  local _default_fields='{}'
  local _fields="${new_fields:-$_default_fields}"
  jq --arg s "$new_state" --arg t "$now" --argjson f "$_fields" \
    '.state = $s | .entered_at = $t | . + $f' "$STATE_FILE" > "$tmpfile"
  mv "$tmpfile" "$STATE_FILE"
  echo "STATE: ${expected} -> ${new_state} [${now}]"
}

get_expected_artifact() {
  local state="$1"
  echo "${STATE_TO_HANDOFF[$state]:-}"
}

verify_handoff() {
  # Verify handoff file integrity
  local handoff_file="$1" state="$2"
  if [[ ! -f "$handoff_file" ]]; then
    echo "VERIFY FAILED: handoff file not found: $handoff_file" >&2
    return 1
  fi
  local status
  status=$(jq -r '.status' "$handoff_file")
  if [[ "$status" == "failed" ]]; then
    echo "VERIFY: handoff reports failure" >&2
    return 0  # failed is a valid handoff status
  fi
  # For success: verify artifact_path and sha256
  local artifact_path sha256_expected
  artifact_path=$(jq -r '.artifact_path // empty' "$handoff_file")
  sha256_expected=$(jq -r '.sha256 // empty' "$handoff_file")
  if [[ -n "$artifact_path" && -n "$sha256_expected" ]]; then
    local sha256_actual
    sha256_actual=$(sha256sum "$artifact_path" 2>/dev/null | cut -d' ' -f1)
    if [[ "$sha256_actual" != "$sha256_expected" ]]; then
      echo "VERIFY FAILED: sha256 mismatch for $artifact_path" >&2
      return 1
    fi
  fi
  return 0
}

poll_for_handoff() {
  # Poll for handoff file or timeout artifact
  # Returns 0 for normal handoff, 1 for timeout/error
  local handoff_dir="$1" filename="$2" timeout_s="$3" interval_s="${4:-30}"
  local elapsed=0
  local handoff_path="${handoff_dir}/${filename}"
  # Derive state from context for timeout artifact check
  local current_state
  current_state=$(jq -r '.state' "$STATE_FILE")
  local timeout_file="${handoff_dir}/timeout_${current_state}.json"

  echo "POLL: waiting for ${filename} (timeout=${timeout_s}s, interval=${interval_s}s)"
  while (( elapsed < timeout_s )); do
    if [[ -f "$handoff_path" ]]; then
      echo "POLL: ${filename} received after ${elapsed}s"
      return 0
    fi
    if [[ -f "$timeout_file" ]]; then
      echo "POLL: timeout artifact detected: $timeout_file" >&2
      return 1
    fi
    sleep "$interval_s"
    elapsed=$((elapsed + interval_s))
    echo "POLL: ... ${elapsed}s/${timeout_s}s waiting for ${filename}"
  done
  echo "POLL: timed out after ${timeout_s}s waiting for $filename" >&2
  return 1
}

# Self-test mode
if [[ "${1:-}" == "test" ]]; then
  PASS=0
  FAIL=0

  run_test() {
    local name="$1" expected="$2" actual="$3"
    if [[ "$expected" == "$actual" ]]; then
      echo "PASS: $name"
      PASS=$((PASS + 1))
    else
      echo "FAIL: $name (expected=$expected, actual=$actual)"
      FAIL=$((FAIL + 1))
    fi
  }

  # Test get_expected_artifact for all 6 states
  run_test "artifact_ORCH_PLAN" "orchestrator_plan_done.json" "$(get_expected_artifact ORCHESTRATOR_PLANNING)"
  run_test "artifact_WORKER" "worker_done.json" "$(get_expected_artifact WORKER_RUNNING)"
  run_test "artifact_REVIEW_CLAUDE" "reviewer_claude_done.json" "$(get_expected_artifact REVIEW_CLAUDE)"
  run_test "artifact_REVIEW_CODEX" "reviewer_codex_done.json" "$(get_expected_artifact REVIEW_CODEX)"
  run_test "artifact_SYNTH" "orchestrator_synth_done.json" "$(get_expected_artifact ORCHESTRATOR_SYNTHESIZING)"
  run_test "artifact_HUMAN_SYNC" "" "$(get_expected_artifact HUMAN_SYNC)"

  # Test CAS transition
  TMPDIR=$(mktemp -d)
  STATE_FILE="${TMPDIR}/state.json"
  echo '{"state":"IDLE","entered_at":null,"batch_id":null,"iteration":0}' > "$STATE_FILE"

  cas_transition IDLE ORCHESTRATOR_PLANNING '{"batch_id":"test-batch","iteration":1}'
  run_test "cas_success" "ORCHESTRATOR_PLANNING" "$(jq -r '.state' "$STATE_FILE")"
  run_test "cas_batch_id" "test-batch" "$(jq -r '.batch_id' "$STATE_FILE")"

  # Test CAS failure
  cas_transition IDLE WORKER_RUNNING '{}' 2>/dev/null
  CAS_RC=$?
  run_test "cas_fail_rc" "1" "$CAS_RC"
  run_test "cas_fail_state" "ORCHESTRATOR_PLANNING" "$(jq -r '.state' "$STATE_FILE")"

  # Test verify_handoff with good file
  HANDOFF_FILE="${TMPDIR}/test_handoff.json"
  echo '{"status":"done","artifact_path":"","sha256":""}' > "$HANDOFF_FILE"
  verify_handoff "$HANDOFF_FILE" "WORKER_RUNNING" 2>/dev/null
  run_test "verify_good" "0" "$?"

  # Test verify_handoff with missing file
  verify_handoff "${TMPDIR}/nonexistent.json" "WORKER_RUNNING" 2>/dev/null
  run_test "verify_missing" "1" "$?"

  # Test verify_handoff with failed status
  echo '{"status":"failed","error":"test error"}' > "$HANDOFF_FILE"
  verify_handoff "$HANDOFF_FILE" "WORKER_RUNNING" 2>/dev/null
  run_test "verify_failed_status" "0" "$?"

  rm -rf "$TMPDIR"

  echo ""
  echo "Results: $PASS passed, $FAIL failed"
  if (( FAIL > 0 )); then exit 1; fi
fi
