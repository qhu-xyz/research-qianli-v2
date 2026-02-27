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
