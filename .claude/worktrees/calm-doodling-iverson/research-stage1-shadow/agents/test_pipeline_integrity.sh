#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

ITERATIONS=1
INJECT_WORKER_FAILURE=false
INJECT_CODEX_TIMEOUT=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --iterations)               ITERATIONS="$2"; shift 2 ;;
    --inject-worker-failure)    INJECT_WORKER_FAILURE=true; shift ;;
    --inject-codex-timeout)     INJECT_CODEX_TIMEOUT=true; shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

PASS=0; FAIL=0
assert() {
  local name="$1" condition="$2"
  if eval "$condition"; then
    echo "PASS: $name"
    PASS=$((PASS+1))
  else
    echo "FAIL: $name"
    FAIL=$((FAIL+1))
  fi
}

echo "=== Pipeline Integrity Test (${ITERATIONS} iterations) ==="

# Run pipeline
SMOKE_TEST=true bash "${SCRIPT_DIR}/run_pipeline.sh" --batch-name "integrity-test"

# Assertions
assert "state.json exists" "[[ -f '${STATE_FILE}' ]]"

FINAL_STATE=$(jq -r '.state' "$STATE_FILE")
if (( ITERATIONS == 3 )); then
  assert "final state is HUMAN_SYNC" "[[ '$FINAL_STATE' == 'HUMAN_SYNC' ]]"
else
  assert "final state is IDLE" "[[ '$FINAL_STATE' == 'IDLE' ]]"
fi

# Check for orphaned tmux sessions
ORPHANED=$(tmux ls 2>/dev/null | grep -c "integrity-test" || true)
assert "no orphaned tmux sessions" "(( ORPHANED == 0 ))"

echo ""
echo "Results: $PASS passed, $FAIL failed"
if (( FAIL > 0 )); then exit 1; fi
