#!/bin/bash
# Demo: Using Claude Code subscription in a bash script
# Prereq: Run `claude login` once to authenticate via OAuth

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Test 1: Ask Claude what 2+2 is ==="
claude -p "What is 2+2? Reply with just the number."

echo ""
echo "=== Test 2: Ask Claude to read and summarize a sentence ==="
claude -p "Read this sentence and tell me what animal it mentions:" < "$SCRIPT_DIR/sentence.txt"

echo ""
echo "=== Test 3: JSON output ==="
claude -p "What is 2+2? Reply with just the number." --output-format json
