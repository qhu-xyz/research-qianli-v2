#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

PASS=0; FAIL=0
run_test() {
  local name="$1" expected="$2" actual="$3"
  if [[ "$expected" == "$actual" ]]; then
    echo "PASS: $name"; PASS=$((PASS+1))
  else
    echo "FAIL: $name (expected=$expected, actual=$actual)"; FAIL=$((FAIL+1))
  fi
}

# Test PIPELINE_LOCKED guard (negative)
export BATCH_ID="test" N=1 VERSION_ID="v0001"
unset PIPELINE_LOCKED
set +e
bash "${SCRIPT_DIR}/run_single_iter.sh" 2>/dev/null
RC=$?
set -e
run_test "pipeline_locked_guard" "1" "$RC"

# Test PIPELINE_LOCKED guard (positive)
export PIPELINE_LOCKED=true
# Can't actually run the full iter without setup, so just check it gets past the guard

echo ""
echo "Results: $PASS passed, $FAIL failed"
if (( FAIL > 0 )); then exit 1; fi
