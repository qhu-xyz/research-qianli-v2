#!/usr/bin/env bash
# All environment-specific settings. Source this at top of every script.
PROJECT_DIR="/home/xyz/workspace/research-qianli-v2/research-stage1-shadow"
RAY_ADDRESS="ray://10.8.0.36:10001"
DATA_ROOT="/opt/temp/tmp/pw_data/spice6"
VENV_ACTIVATE="/home/xyz/workspace/pmodel/.venv/bin/activate"
SMOKE_TEST="${SMOKE_TEST:-false}"
REGISTRY_DISK_LIMIT_MB=10240
CODEX_MODEL="gpt-5.3-codex"
STATE_FILE="${PROJECT_DIR}/state.json"

# Hard timeout limits (seconds) — OS-level kill, defense against agent infinite loops.
# These are the absolute max wall-clock time an agent process can run.
# poll_for_handoff timeouts are shorter; these are the last-resort kill switch.
TIMEOUT_ORCHESTRATOR=900     # 15 min (poll: 10 min)
TIMEOUT_WORKER=2400          # 40 min (poll: 30 min)
TIMEOUT_REVIEWER_CLAUDE=1500 # 25 min (poll: 20 min)
TIMEOUT_REVIEWER_CODEX=1500  # 25 min (poll: 20 min)
TIMEOUT_SYNTHESIZER=900      # 15 min (poll: 10 min)
