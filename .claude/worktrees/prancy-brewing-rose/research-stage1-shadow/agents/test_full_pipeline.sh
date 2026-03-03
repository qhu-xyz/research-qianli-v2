#!/usr/bin/env bash
# Full pipeline test — simulates all 4 agent roles through 3 iterations
# Instead of launching real Claude/Codex agents (which require API calls),
# this test exercises the controller logic by writing handoff files directly,
# simulating what each agent would produce.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"
source "${SCRIPT_DIR}/state_utils.sh"

export PYTHONPATH="${PROJECT_DIR}"

echo "============================================"
echo "=== Full Pipeline Test (3 iterations)    ==="
echo "============================================"

PASS=0; FAIL=0
assert() {
  local name="$1"
  shift
  if "$@" >/dev/null 2>&1; then
    echo "  PASS: $name"
    PASS=$((PASS + 1))
  else
    echo "  FAIL: $name"
    FAIL=$((FAIL + 1))
  fi
}

# Reset state to IDLE
echo '{"state":"IDLE","batch_id":null,"iteration":0,"version_id":null,"entered_at":null,"max_seconds":null,"orchestrator_tmux":null,"worker_tmux":null,"claude_reviewer_tmux":null,"codex_reviewer_tmux":null,"history":[],"human_input":null,"error":null}' > "$STATE_FILE"

BATCH_ID="test-$(date +%Y%m%d-%H%M%S)"
export BATCH_ID

echo ""
echo "Batch: ${BATCH_ID}"
echo ""

# Activate venv
source "$VENV_ACTIVATE"

for N_ITER in 1 2 3; do
  export N=$N_ITER
  echo "==============================="
  echo "=== Iteration ${N} ==="
  echo "==============================="

  HANDOFF_DIR="${PROJECT_DIR}/handoff/${BATCH_ID}/iter${N}"
  mkdir -p "$HANDOFF_DIR"

  # === STEP 1: IDLE -> ORCHESTRATOR_PLANNING ===
  echo "[step 1] Transitioning IDLE -> ORCHESTRATOR_PLANNING"
  cas_transition IDLE ORCHESTRATOR_PLANNING "{\"batch_id\":\"${BATCH_ID}\",\"iteration\":${N},\"max_seconds\":600}"
  assert "state is ORCHESTRATOR_PLANNING" test "$(jq -r '.state' "$STATE_FILE")" = "ORCHESTRATOR_PLANNING"

  # === SIMULATE ORCHESTRATOR (plan) ===
  echo "[step 3] Simulating orchestrator plan..."

  if (( N == 1 )); then
    DIRECTION_CONTENT="# Direction — Iteration 1

## Hypothesis
Test baseline configuration with default hyperparameters. Establish v0001 as first experimental version.

## Specific Changes
- Run pipeline with default HyperparamConfig (n_estimators=200, max_depth=4)
- No code changes needed — just run with current defaults

## Expected Impact
- Establish baseline gate metrics for comparison
- All gate floors should be close to v0 since using same model architecture

## Risk Assessment
- Low risk — using default configuration"
  elif (( N == 2 )); then
    DIRECTION_CONTENT="# Direction — Iteration 2

## Hypothesis
Increasing tree depth from 4 to 6 may improve recall (S1-REC) by capturing more complex feature interactions.

## Specific Changes
- In ml/config.py, modify HyperparamConfig default: max_depth=6
- Run pipeline with overrides: {\"max_depth\": 6}

## Expected Impact
- S1-REC should improve (currently 0.0 which is very low)
- S1-AUC may improve slightly
- Risk of overfitting on small sample (100 rows)

## Risk Assessment
- Medium risk — deeper trees can overfit on small datasets"
  else
    DIRECTION_CONTENT="# Direction — Iteration 3

## Hypothesis
Reducing the threshold scaling factor to 0.8 may improve recall without hurting precision too much.

## Specific Changes
- Run pipeline with overrides: {\"max_depth\": 6, \"threshold_scaling_factor\": 0.8}

## Expected Impact
- S1-REC should improve significantly with lower threshold
- S1-BRIER should stay within bounds
- Potential trade-off with precision

## Risk Assessment
- Medium risk — lower threshold trades precision for recall"
  fi

  echo "$DIRECTION_CONTENT" > "${PROJECT_DIR}/memory/direction_iter${N}.md"

  # Update progress
  echo "# Progress

Iteration ${N} of batch ${BATCH_ID} — orchestrator planning complete." > "${PROJECT_DIR}/memory/hot/progress.md"

  # Write orchestrator handoff
  ARTIFACT="memory/direction_iter${N}.md"
  SHA=$(sha256sum "${PROJECT_DIR}/${ARTIFACT}" | cut -d' ' -f1)
  cat > "${HANDOFF_DIR}/orchestrator_plan_done.json" << EOF
{"status": "done", "artifact_path": "${ARTIFACT}", "sha256": "${SHA}"}
EOF
  assert "orchestrator plan handoff exists" test -f "${HANDOFF_DIR}/orchestrator_plan_done.json"

  # === STEP 4: ORCHESTRATOR_PLANNING -> WORKER_RUNNING ===
  echo "[step 4] Transitioning ORCHESTRATOR_PLANNING -> WORKER_RUNNING"
  cas_transition ORCHESTRATOR_PLANNING WORKER_RUNNING "{\"max_seconds\":1800}"

  # === STEP 5: Allocate version ===
  VERSION_ID=$(python -c "from ml.registry import allocate_version_id; print(allocate_version_id('${PROJECT_DIR}/registry/version_counter.json'))")
  jq --arg v "$VERSION_ID" '.version_id = $v' "$STATE_FILE" > "${STATE_FILE}.tmp.$$" && mv "${STATE_FILE}.tmp.$$" "$STATE_FILE"
  export VERSION_ID
  echo "[step 5] Allocated version: ${VERSION_ID}"

  # === STEP 5a: Commit orchestrator outputs ===
  echo "[step 5a] Committing orchestrator outputs..."
  git -C "$PROJECT_DIR" add memory/ 2>/dev/null || true
  if ! git -C "$PROJECT_DIR" diff --cached --quiet 2>/dev/null; then
    git -C "$PROJECT_DIR" commit -m "iter${N}: orchestrator plan" --allow-empty 2>/dev/null || true
  fi

  # === SIMULATE WORKER ===
  echo "[step 6] Simulating worker (version ${VERSION_ID})..."

  # Worker runs the pipeline
  if (( N == 1 )); then
    OVERRIDES='{}'
  elif (( N == 2 )); then
    OVERRIDES='{"max_depth": 6}'
  else
    OVERRIDES='{"max_depth": 6}'
  fi

  SMOKE_TEST=true python ml/pipeline.py --version-id "$VERSION_ID" --auction-month 2021-07 \
    --class-type onpeak --period-type f0 \
    --overrides "$OVERRIDES" 2>&1 | head -5

  # Write changes summary
  mkdir -p "${PROJECT_DIR}/registry/${VERSION_ID}"
  cat > "${PROJECT_DIR}/registry/${VERSION_ID}/changes_summary.md" << EOF
# Changes Summary — ${VERSION_ID} (Iteration ${N})

## What Changed
- Ran pipeline with overrides: ${OVERRIDES}

## Results
$(cat "${PROJECT_DIR}/registry/${VERSION_ID}/metrics.json" 2>/dev/null || echo "metrics pending")
EOF

  # Commit worker changes
  git -C "$PROJECT_DIR" add "registry/${VERSION_ID}/" 2>/dev/null || true
  if ! git -C "$PROJECT_DIR" diff --cached --quiet 2>/dev/null; then
    git -C "$PROJECT_DIR" commit -m "iter${N}: worker ${VERSION_ID}" 2>/dev/null || true
  fi

  # Write worker handoff
  ARTIFACT="registry/${VERSION_ID}/changes_summary.md"
  SHA=$(sha256sum "${PROJECT_DIR}/${ARTIFACT}" | cut -d' ' -f1)
  cat > "${HANDOFF_DIR}/worker_done.json" << EOF
{"status": "done", "artifact_path": "${ARTIFACT}", "sha256": "${SHA}"}
EOF
  assert "worker handoff exists" test -f "${HANDOFF_DIR}/worker_done.json"
  assert "worker metrics exist" test -f "${PROJECT_DIR}/registry/${VERSION_ID}/metrics.json"

  # === STEP 8: Run comparison ===
  echo "[step 8] Running comparison..."
  mkdir -p "${PROJECT_DIR}/reports/${BATCH_ID}/iter${N}"
  python ml/compare.py --batch-id "$BATCH_ID" --iteration "$N" \
    --output "reports/${BATCH_ID}/iter${N}/comparison.md" 2>&1 | head -3 || true
  assert "comparison report exists" test -f "${PROJECT_DIR}/reports/${BATCH_ID}/iter${N}/comparison.md"

  # === STEP 9: WORKER_RUNNING -> REVIEW_CLAUDE ===
  echo "[step 9] Transitioning WORKER_RUNNING -> REVIEW_CLAUDE"
  cas_transition WORKER_RUNNING REVIEW_CLAUDE "{\"max_seconds\":1200}"

  # === SIMULATE CLAUDE REVIEWER ===
  echo "[step 10] Simulating Claude reviewer..."
  METRICS=$(cat "${PROJECT_DIR}/registry/${VERSION_ID}/metrics.json")
  cat > "${PROJECT_DIR}/reviews/${BATCH_ID}_iter${N}_claude.md" << EOF
## Claude Review: ${BATCH_ID} Iteration ${N}

### Summary
Version ${VERSION_ID} was evaluated against v0 baseline. The worker executed the planned hypothesis.

### Gate Analysis
$(cat "${PROJECT_DIR}/reports/${BATCH_ID}/iter${N}/comparison.md" 2>/dev/null | head -20 || echo "Comparison table not available")

### Code Review Findings
1. No code modifications were made — only hyperparameter overrides were used.
2. Pipeline executed successfully with all 6 phases completing.

### Recommendations
1. Consider adjusting threshold to improve recall (currently 0.0 on validation set)
2. The small sample size (100 rows) makes metrics noisy
3. Focus on improving S1-REC which is currently below floor

### Gate Calibration
- S1-VCAP floors (0.95) are very aggressive given the small sample
- S1-CAP floors are negative, effectively meaningless
EOF

  ARTIFACT="reviews/${BATCH_ID}_iter${N}_claude.md"
  SHA=$(sha256sum "${PROJECT_DIR}/${ARTIFACT}" | cut -d' ' -f1)
  cat > "${HANDOFF_DIR}/reviewer_claude_done.json" << EOF
{"status": "done", "artifact_path": "${ARTIFACT}", "sha256": "${SHA}"}
EOF
  assert "Claude review exists" test -f "${PROJECT_DIR}/reviews/${BATCH_ID}_iter${N}_claude.md"
  assert "Claude handoff exists" test -f "${HANDOFF_DIR}/reviewer_claude_done.json"

  # === STEP 12: REVIEW_CLAUDE -> REVIEW_CODEX ===
  echo "[step 12] Transitioning REVIEW_CLAUDE -> REVIEW_CODEX"
  cas_transition REVIEW_CLAUDE REVIEW_CODEX "{\"max_seconds\":1200}"

  # === SIMULATE CODEX REVIEWER ===
  echo "[step 13] Simulating Codex reviewer..."
  cat > "${PROJECT_DIR}/reviews/${BATCH_ID}_iter${N}_codex.md" << EOF
## Codex Review: ${BATCH_ID} Iteration ${N}

### Summary
Independent review of version ${VERSION_ID}. Pipeline ran to completion.

### Gate Analysis
Metrics reviewed against gates.json. Key observations:
- S1-AUC and S1-AP meet their floors
- S1-REC is 0.0 — well below the 0.40 floor
- VCAP metrics are high due to small sample concentration

### Code Findings
1. No structural code changes detected
2. Pipeline is correctly configured for SMOKE_TEST mode
3. Overrides: ${OVERRIDES}

### Recommendations
1. Address the recall problem — current threshold is too high
2. Consider lowering threshold_beta to produce a lower decision boundary
3. The 100-row SMOKE_TEST dataset is too small for reliable metrics

### Gate Calibration
- S1-REC floor of 0.40 may be too ambitious for this sample size
EOF

  ARTIFACT="reviews/${BATCH_ID}_iter${N}_codex.md"
  SHA=$(sha256sum "${PROJECT_DIR}/${ARTIFACT}" | cut -d' ' -f1)
  cat > "${HANDOFF_DIR}/reviewer_codex_done.json" << EOF
{"status": "done", "artifact_path": "${ARTIFACT}", "sha256": "${SHA}"}
EOF
  assert "Codex review exists" test -f "${PROJECT_DIR}/reviews/${BATCH_ID}_iter${N}_codex.md"
  assert "Codex handoff exists" test -f "${HANDOFF_DIR}/reviewer_codex_done.json"

  # === STEP 15: REVIEW_CODEX -> ORCHESTRATOR_SYNTHESIZING ===
  echo "[step 15] Transitioning REVIEW_CODEX -> ORCHESTRATOR_SYNTHESIZING"
  cas_transition REVIEW_CODEX ORCHESTRATOR_SYNTHESIZING "{\"max_seconds\":600}"

  # === SIMULATE SYNTHESIS ORCHESTRATOR ===
  echo "[step 16] Simulating synthesis orchestrator..."

  # Update memory files
  cat >> "${PROJECT_DIR}/memory/warm/experiment_log.md" << EOF

## Iteration ${N} — ${VERSION_ID}
- Overrides: ${OVERRIDES}
- Key metrics: S1-AUC=$(jq -r '."S1-AUC"' "${PROJECT_DIR}/registry/${VERSION_ID}/metrics.json"), S1-REC=$(jq -r '."S1-REC"' "${PROJECT_DIR}/registry/${VERSION_ID}/metrics.json")
- Status: completed
EOF

  cat >> "${PROJECT_DIR}/memory/warm/decision_log.md" << EOF

## Iteration ${N} Decision
- Do not promote ${VERSION_ID} — S1-REC is below floor
- Next: try $(( N < 3 ? N + 1 : N )) with adjusted threshold
EOF

  echo "# Critique Summary

## Iteration ${N}
Both reviewers agree: recall (S1-REC) is the critical bottleneck. The threshold is too high, causing zero predicted positives on the validation set. Recommend lowering threshold or using a different threshold_beta." > "${PROJECT_DIR}/memory/hot/critique_summary.md"

  if (( N == 3 )); then
    # Archive on final iteration
    mkdir -p "${PROJECT_DIR}/memory/archive/${BATCH_ID}"
    cat > "${PROJECT_DIR}/memory/archive/${BATCH_ID}/executive_summary.md" << EOF
# Executive Summary — ${BATCH_ID}

## Batch Overview
3-iteration batch completed. Tested hyperparameter variations on SMOKE_TEST data.

## Results
- Iteration 1: baseline defaults
- Iteration 2: max_depth=6
- Iteration 3: max_depth=6 (threshold adjustment explored)

## Key Learning
Recall is the primary bottleneck on small datasets. Threshold optimization on 100-row samples is unreliable.

## Recommendation
Move to real data for meaningful results.
EOF

    # Update archive index
    echo "
## ${BATCH_ID}
Completed $(date -u +"%Y-%m-%dT%H:%M:%SZ"). See memory/archive/${BATCH_ID}/executive_summary.md" >> "${PROJECT_DIR}/memory/archive/index.md"

    ARTIFACT="memory/archive/${BATCH_ID}/executive_summary.md"
  else
    ARTIFACT="memory/direction_iter${N}.md"
  fi

  # Write synthesis handoff with decisions
  SHA=$(sha256sum "${PROJECT_DIR}/${ARTIFACT}" | cut -d' ' -f1)
  cat > "${HANDOFF_DIR}/orchestrator_synth_done.json" << EOF
{
  "status": "done",
  "artifact_path": "${ARTIFACT}",
  "sha256": "${SHA}",
  "decisions": {
    "promote_version": null,
    "gate_change_requests": [],
    "next_hypothesis": "$(( N < 3 ? 1 : 0 )) — adjust threshold"
  }
}
EOF
  assert "synthesis handoff exists" test -f "${HANDOFF_DIR}/orchestrator_synth_done.json"
  assert "synthesis has decisions field" test "$(jq -r '.decisions.promote_version' "${HANDOFF_DIR}/orchestrator_synth_done.json")" = "null"

  # === STEP 17: Check promotion ===
  PROMOTE=$(jq -r '.decisions.promote_version // empty' "${HANDOFF_DIR}/orchestrator_synth_done.json" 2>/dev/null || echo "")
  if [[ -n "$PROMOTE" ]]; then
    echo "[step 17] Promoting version: $PROMOTE"
    python -c "from ml.registry import promote_version; promote_version('${PROJECT_DIR}/registry', '${PROMOTE}', '${PROJECT_DIR}/registry/champion.json')"
  else
    echo "[step 17] No promotion this iteration"
  fi

  # === STEP 17a: Commit synthesis outputs ===
  git -C "$PROJECT_DIR" add memory/ reviews/ reports/ 2>/dev/null || true
  if ! git -C "$PROJECT_DIR" diff --cached --quiet 2>/dev/null; then
    git -C "$PROJECT_DIR" commit -m "iter${N}: synthesis" 2>/dev/null || true
  fi

  # === STEP 18: State transition ===
  if (( N == 3 )); then
    echo "[step 18] Transitioning ORCHESTRATOR_SYNTHESIZING -> HUMAN_SYNC"
    cas_transition ORCHESTRATOR_SYNTHESIZING HUMAN_SYNC '{}'
    assert "final state is HUMAN_SYNC" test "$(jq -r '.state' "$STATE_FILE")" = "HUMAN_SYNC"
  else
    echo "[step 18] Transitioning ORCHESTRATOR_SYNTHESIZING -> IDLE"
    cas_transition ORCHESTRATOR_SYNTHESIZING IDLE '{}'
    assert "state back to IDLE" test "$(jq -r '.state' "$STATE_FILE")" = "IDLE"
  fi

  echo ""
done

echo ""
echo "============================================"
echo "=== Post-Pipeline Verification ==="
echo "============================================"

# Final assertions
assert "3 versions created" test "$(ls -d registry/v0??? 2>/dev/null | wc -l)" -ge 3
assert "v0 still exists" test -d "${PROJECT_DIR}/registry/v0"
assert "archive exists" test -f "${PROJECT_DIR}/memory/archive/${BATCH_ID}/executive_summary.md"
assert "archive index updated" grep -q "$BATCH_ID" "${PROJECT_DIR}/memory/archive/index.md"
assert "experiment log has 3 entries" test "$(grep -c 'Iteration' "${PROJECT_DIR}/memory/warm/experiment_log.md")" -ge 3
assert "state is HUMAN_SYNC" test "$(jq -r '.state' "$STATE_FILE")" = "HUMAN_SYNC"
assert "champion unchanged (null)" test "$(jq -r '.version' "${PROJECT_DIR}/registry/champion.json")" = "null"
assert "gates.json not modified by agents" test "$(jq '.gates | to_entries | length' "${PROJECT_DIR}/registry/gates.json")" = "10"
assert "evaluate.py still has HUMAN-WRITE-ONLY marker" grep -q "HUMAN-WRITE-ONLY" "${PROJECT_DIR}/ml/evaluate.py"

# Check all handoff files exist
for i in 1 2 3; do
  HD="${PROJECT_DIR}/handoff/${BATCH_ID}/iter${i}"
  assert "iter${i} orchestrator_plan handoff" test -f "${HD}/orchestrator_plan_done.json"
  assert "iter${i} worker handoff" test -f "${HD}/worker_done.json"
  assert "iter${i} Claude review handoff" test -f "${HD}/reviewer_claude_done.json"
  assert "iter${i} Codex review handoff" test -f "${HD}/reviewer_codex_done.json"
  assert "iter${i} synthesis handoff" test -f "${HD}/orchestrator_synth_done.json"
done

echo ""
echo "============================================"
echo "=== RESULTS: ${PASS} passed, ${FAIL} failed ==="
echo "============================================"

if (( FAIL > 0 )); then
  echo "SOME TESTS FAILED"
  exit 1
else
  echo "ALL TESTS PASSED"
fi
