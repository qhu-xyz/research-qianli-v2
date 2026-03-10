#!/usr/bin/env bash
# Clean up after a failed or interrupted pipeline run.
#
# Usage: bash agents/cleanup_failed_run.sh [--dry-run]
#
# Cleans: stale worktrees, handoff dirs, direction files, tmux sessions, state.json.
# Does NOT touch: registry/v*/, git commits, reviews/, reports/
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

dry() {
  if [[ "$DRY_RUN" == "true" ]]; then
    echo "  [DRY-RUN] $*"
  else
    echo "  $*"
    eval "$@"
  fi
}

echo "=== Pipeline Cleanup ==="
echo ""

# 1. Kill stale pipeline tmux sessions
echo "1. Stale tmux sessions:"
STALE_SESSIONS=$(tmux ls 2>/dev/null | grep -oE '(pipeline|orch|worker|rev-claude|rev-codex|synth)-[^ :]+' || true)
if [[ -n "$STALE_SESSIONS" ]]; then
  while IFS= read -r sess; do
    dry "tmux kill-session -t '$sess' 2>/dev/null || true"
  done <<< "$STALE_SESSIONS"
else
  echo "  None found"
fi

# 2. Remove stale worktrees
echo ""
echo "2. Stale worktrees:"
WORKTREES_DIR="${PROJECT_DIR}/.claude/worktrees"
if [[ -d "$WORKTREES_DIR" ]]; then
  for wt in "$WORKTREES_DIR"/iter*; do
    [[ -d "$wt" ]] || continue
    echo "  Found: $(basename "$wt")"
    dry "git worktree remove '$wt' --force 2>/dev/null || rm -rf '$wt'"
  done
  # Prune any orphaned worktree references
  dry "git -C '$PROJECT_DIR' worktree prune"
else
  echo "  None found"
fi

# 3. Remove stale direction files
echo ""
echo "3. Stale direction files:"
DIRECTIONS=$(ls "${PROJECT_DIR}"/memory/direction_iter*.md 2>/dev/null || true)
if [[ -n "$DIRECTIONS" ]]; then
  for df in ${DIRECTIONS}; do
    echo "  Found: $(basename "$df")"
    dry "rm -f '$df'"
  done
else
  echo "  None found"
fi

# 4. Reset state.json to clean IDLE
echo ""
echo "4. State machine:"
CURRENT_STATE=$(jq -r '.state' "$STATE_FILE")
CURRENT_ERROR=$(jq -r '.error // "null"' "$STATE_FILE")
echo "  Current state: $CURRENT_STATE (error: $CURRENT_ERROR)"
if [[ "$CURRENT_STATE" != "IDLE" || "$CURRENT_ERROR" != "null" ]]; then
  dry "jq '{state:\"IDLE\",batch_id:null,iteration:0,version_id:null,entered_at:null,max_seconds:600,orchestrator_tmux:null,worker_tmux:null,claude_reviewer_tmux:null,codex_reviewer_tmux:null,history:[],human_input:null,error:null}' <<< '{}' > '${STATE_FILE}'"
  echo "  Reset to clean IDLE"
else
  echo "  Already clean"
fi

# 5. Release lock if held
echo ""
echo "5. State lock:"
if flock -n 200 2>/dev/null; then
  echo "  Not held"
else
  echo "  Lock held — cannot release from cleanup script"
  echo "  If no pipeline is running: rm ${PROJECT_DIR}/state.lock"
fi 200>"${PROJECT_DIR}/state.lock"

# 6. Summary of artifacts NOT cleaned (for human review)
echo ""
echo "6. Artifacts preserved (not cleaned):"
echo "  - Git commits (revert manually if needed: git log --oneline -5)"
echo "  - registry/v*/ directories (delete manually if incomplete)"
echo "  - handoff/ directories (historical, gitignored)"
echo "  - reviews/ (historical)"
echo "  - reports/ (historical)"
echo "  - .logs/sessions/ (historical)"

# 7. Check for orphaned registry versions (allocated but no metrics)
echo ""
echo "7. Orphaned registry versions (no metrics.json):"
ORPHANS=0
for vdir in "${PROJECT_DIR}"/registry/v*/; do
  [[ -d "$vdir" ]] || continue
  if [[ ! -f "${vdir}/metrics.json" ]]; then
    echo "  $(basename "$vdir"): no metrics.json (worker may not have finished)"
    ORPHANS=$((ORPHANS + 1))
  fi
done
[[ $ORPHANS -eq 0 ]] && echo "  None found"

echo ""
if [[ "$DRY_RUN" == "true" ]]; then
  echo "=== DRY RUN — no changes made. Run without --dry-run to apply. ==="
else
  echo "=== Cleanup complete ==="
fi
