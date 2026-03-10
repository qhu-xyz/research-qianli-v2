#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

echo "Checking CLIs..."
FAIL=0

if command -v claude >/dev/null 2>&1; then
  echo "PASS: claude CLI found"
else
  echo "FAIL: claude CLI not found"
  FAIL=1
fi

if command -v codex >/dev/null 2>&1; then
  echo "PASS: codex CLI found"
else
  echo "FAIL: codex CLI not found"
  FAIL=1
fi

if command -v jq >/dev/null 2>&1; then
  echo "PASS: jq found"
else
  echo "FAIL: jq not found"
  FAIL=1
fi

if command -v tmux >/dev/null 2>&1; then
  echo "PASS: tmux found"
else
  echo "FAIL: tmux not found"
  FAIL=1
fi

if (( FAIL > 0 )); then
  echo "Some CLI checks failed"
  exit 1
fi
echo "All CLI checks passed"
