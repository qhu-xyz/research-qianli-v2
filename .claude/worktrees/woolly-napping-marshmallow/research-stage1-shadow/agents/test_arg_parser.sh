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

# Test orchestrator --phase plan
export BATCH_ID="test-batch" N=1 VERSION_ID="v0001"
OUT=$(bash "${SCRIPT_DIR}/launch_orchestrator.sh" --phase plan --dry-run 2>&1)
run_test "orch-plan-dry-run" "0" "$?"

# Test orchestrator --phase=synthesize
export WORKER_FAILED=0
OUT=$(bash "${SCRIPT_DIR}/launch_orchestrator.sh" --phase=synthesize --dry-run 2>&1)
run_test "orch-synth-dry-run" "0" "$?"

# Test worker --dry-run
OUT=$(bash "${SCRIPT_DIR}/launch_worker.sh" --dry-run 2>&1)
run_test "worker-dry-run" "0" "$?"

# Test Claude reviewer --dry-run
OUT=$(bash "${SCRIPT_DIR}/launch_reviewer_claude.sh" --dry-run 2>&1)
run_test "claude-reviewer-dry-run" "0" "$?"

# Test Codex reviewer --dry-run
OUT=$(bash "${SCRIPT_DIR}/launch_reviewer_codex.sh" --dry-run 2>&1)
run_test "codex-reviewer-dry-run" "0" "$?"

echo ""
echo "Results: $PASS passed, $FAIL failed"
if (( FAIL > 0 )); then exit 1; fi
